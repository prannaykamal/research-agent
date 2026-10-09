import ipaddress
import json
import socket
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit

import arxiv
import httpx
import wikipedia
from bs4 import BeautifulSoup
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_tavily import TavilySearch


MAX_TOOL_OUTPUT_CHARS = 12_000
MAX_SCRAPE_BYTES = 2_000_000
MAX_SCRAPE_REDIRECTS = 3
SCRAPE_TIMEOUT_SECONDS = 15.0
USER_AGENT = "research-agent/0.1 (+https://github.com/prannaykamal/research-agent)"

# Wikimedia rate-limits the wikipedia library's generic, shared User-Agent (HTTP 429);
# its API policy asks clients to identify themselves.
wikipedia.set_user_agent(USER_AGENT)


def _bounded(value: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    return str(value).strip()[:limit]


def _source_payload(
    *, title: str, url: str, excerpt: str, source_type: str, limit: int = MAX_TOOL_OUTPUT_CHARS
) -> dict[str, str]:
    return {
        "source_title": title or "Untitled source",
        "source_url": url,
        "excerpt": _bounded(excerpt, limit),
        "source_type": source_type,
    }


def _excerpt_limit(count: int) -> int:
    """Share the per-tool output cap across every returned document."""
    return MAX_TOOL_OUTPUT_CHARS // max(1, count)


class BoundedTavilySearch(TavilySearch):
    """Standard TavilySearch whose output is serialized and bounded like other tools."""

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return _bounded(json.dumps(super()._run(*args, **kwargs), default=str))


class WikipediaEvidenceTool(WikipediaQueryRun):
    """Standard WikipediaQueryRun with source metadata preserved for evidence."""

    name: str = "wikipedia"

    def _run(self, query: str, run_manager=None) -> str:
        try:
            documents = self.api_wrapper.load(query)
            limit = _excerpt_limit(len(documents))
            payload = [
                _source_payload(
                    title=document.metadata.get("title", "Wikipedia"),
                    url=document.metadata.get("source", ""),
                    excerpt=document.page_content,
                    source_type="wikipedia",
                    limit=limit,
                )
                for document in documents
            ]
            return json.dumps(payload)
        except Exception as exc:
            return f"Wikipedia search failed: {exc}"


class ArxivEvidenceTool(ArxivQueryRun):
    """ArxivQueryRun searching through the current arxiv client API.

    langchain-community's wrapper still calls ``Search.results()``, which arxiv
    4.x removed, so the search runs through ``arxiv.Client`` here. A client per
    call means parallel analysts never queue behind its request delay.
    """

    name: str = "arxiv"

    def _run(self, query: str, run_manager=None) -> str:
        try:
            wrapper = self.api_wrapper
            if wrapper.is_arxiv_identifier(query):
                search = arxiv.Search(id_list=query.split(), max_results=wrapper.top_k_results)
            else:
                search = arxiv.Search(
                    query[: wrapper.ARXIV_MAX_QUERY_LENGTH], max_results=wrapper.top_k_results
                )
            client = arxiv.Client(page_size=wrapper.top_k_results, num_retries=1)
            results = list(client.results(search))
            limit = _excerpt_limit(len(results))
            payload = [
                _source_payload(
                    title=result.title or "arXiv paper",
                    url=result.entry_id,
                    excerpt=result.summary,
                    source_type="arxiv",
                    limit=limit,
                )
                for result in results
            ]
            return json.dumps(payload)
        except Exception as exc:
            return f"arXiv search failed: {exc}"


class PubMedEvidenceTool(PubmedQueryRun):
    """Standard PubmedQueryRun with canonical PubMed references in its output."""

    name: str = "pubmed"

    def _run(self, query: str, run_manager=None) -> str:
        try:
            articles = [article for article in self.api_wrapper.load(query) if article.get("uid")]
            limit = _excerpt_limit(len(articles))
            payload = [
                _source_payload(
                    title=str(article.get("Title", "PubMed article")),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{article['uid']}/",
                    excerpt=str(article.get("Summary", "")),
                    source_type="pubmed",
                    limit=limit,
                )
                for article in articles
            ]
            return json.dumps(payload)
        except Exception as exc:
            return f"PubMed search failed: {exc}"


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
    )


def _validate_public_http_url(url: str) -> str:
    """Reject non-HTTP(S) URLs and any host that resolves to a non-public address."""
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be an absolute HTTP(S) URL.")
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Local URLs cannot be scraped.")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(hostname, parsed.port or None)}
    except socket.gaierror as exc:
        raise ValueError(f"Host could not be resolved: {hostname}") from exc
    for address in addresses:
        if _is_blocked_address(ipaddress.ip_address(address.split("%", 1)[0])):
            raise ValueError("Private network URLs cannot be scraped.")
    return url.strip()


def _fetch_public_page(url: str) -> tuple[str, str]:
    """Fetch a page, validating every redirect hop; return (final URL, HTML)."""
    current = _validate_public_http_url(url)
    with httpx.Client(
        follow_redirects=False,
        timeout=SCRAPE_TIMEOUT_SECONDS,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        for _ in range(MAX_SCRAPE_REDIRECTS + 1):
            with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "")
                    current = _validate_public_http_url(urljoin(current, location))
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if "html" not in content_type and "text" not in content_type:
                    raise ValueError(f"Unsupported content type: {content_type or 'unknown'}")
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) >= MAX_SCRAPE_BYTES:
                        break
                return current, body.decode(response.encoding or "utf-8", errors="replace")
    raise ValueError("Too many redirects.")


@tool
def scrape_webpage(url: str) -> str:
    """Fetch one public webpage and return its title, URL, and bounded text content."""
    try:
        final_url, html = _fetch_public_page(url)
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        title = soup.title.get_text(strip=True) if soup.title else final_url
        text = " ".join(soup.get_text(" ").split())
        if not text:
            return "Webpage scrape returned no content."
        payload = _source_payload(
            title=title, url=final_url, excerpt=text, source_type="scraped_page"
        )
        return json.dumps(payload)
    except Exception as exc:
        return f"Webpage scrape failed: {exc}"


def build_research_tools(allowed: Iterable[str] | None = None) -> list[Any]:
    """Build the tools exposed to the Researcher node, optionally restricted by name."""
    tools = [
        BoundedTavilySearch(max_results=3),
        WikipediaEvidenceTool(api_wrapper=WikipediaAPIWrapper()),
        ArxivEvidenceTool(),
        PubMedEvidenceTool(),
        scrape_webpage,
    ]
    if allowed is None:
        return tools
    names = set(allowed)
    return [tool for tool in tools if tool.name in names]
