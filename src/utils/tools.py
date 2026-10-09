import ipaddress
import json
import re
import socket
from typing import Any, Iterable, Literal
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


# Source trust tiers, mirroring the provenance tiers the report is judged by:
# "trusted" needs no verification, "verify" warrants a spot-check, and "low"
# (user-generated, content-farm or predatory) should never carry a headline.
SourceTier = Literal["trusted", "verify", "low"]
TIER_RANK: dict[str, int] = {"trusted": 0, "verify": 1, "low": 2}
TRUSTED_SUFFIXES = (
    ".gov", ".mil", ".edu", ".int", ".europa.eu",
    ".gov.uk", ".ac.uk", ".nhs.uk", ".gov.au", ".edu.au", ".gc.ca", ".ac.jp", ".go.jp",
)
TRUSTED_DOMAINS = frozenset(
    {
        # Preprint archives, identifiers and indexes.
        "arxiv.org", "biorxiv.org", "medrxiv.org", "doi.org", "ssrn.com", "nber.org",
        "jstor.org",
        # Peer-reviewed publishers and journals.
        "nature.com", "science.org", "sciencedirect.com", "springer.com", "wiley.com",
        "ieee.org", "acm.org", "nejm.org", "thelancet.com", "bmj.com", "cell.com",
        "pnas.org", "plos.org", "oup.com", "cambridge.org", "tandfonline.com",
        "sagepub.com", "annualreviews.org", "aps.org", "iop.org", "acs.org", "rsc.org",
        # Intergovernmental bodies, standards bodies and research institutes.
        "un.org", "oecd.org", "worldbank.org", "imf.org", "ipcc.ch", "iea.org",
        "iso.org", "ietf.org", "w3.org", "rand.org",
        # Established professional press.
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "ft.com", "economist.com",
    }
)
LOW_DOMAINS = frozenset(
    {
        # User-generated video and social platforms.
        "youtube.com", "youtu.be", "tiktok.com", "facebook.com", "instagram.com",
        "x.com", "twitter.com", "reddit.com", "quora.com", "pinterest.com",
        # Self-published blogs and simplified references.
        "medium.com", "blogspot.com", "wordpress.com", "simple.wikipedia.org",
        # Content farms and predatory publishers that carried claims in past runs.
        "brewminate.com", "omicsonline.org", "symbiosisonlinepublishing.com",
    }
)
# Never worth a search result: the Researcher should not spend calls on them.
EXCLUDED_SEARCH_DOMAINS = [
    "simple.wikipedia.org", "youtube.com", "tiktok.com", "facebook.com",
    "instagram.com", "pinterest.com",
]

# Tools here report failure as text instead of raising, so detect it from the result.
TOOL_FAILURE_PATTERN = re.compile(r"^\s*[\w -]+ (?:search|scrape) failed:", re.IGNORECASE)
NO_RESULTS_MARKER = "No search results found"
AUTH_FAILURE_PATTERN = re.compile(
    r"\b40[13]\b|unauthori[sz]ed|forbidden|api[ _-]?key", re.IGNORECASE
)
# Only keyed services: a scraped site answering 403 is blocking bots, not a bad key.
KEYED_TOOLS = frozenset({"tavily_search"})


class ToolAuthError(RuntimeError):
    """A keyed research service rejected its credentials."""

    def __init__(self, tool: str, detail: str) -> None:
        super().__init__(f"{tool} rejected its API key: {detail}")
        self.tool = tool


def _host(url: str) -> str:
    host = (urlsplit(url.strip()).hostname or "").lower()
    return host.removeprefix("www.")


def _in_domains(host: str, domains: Iterable[str]) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def source_tier(url: str) -> SourceTier:
    """Classify a source by its domain alone; deterministic, never an LLM judgement."""
    host = _host(url)
    if not host or _in_domains(host, LOW_DOMAINS):
        return "low"
    if host.endswith(TRUSTED_SUFFIXES) or _in_domains(host, TRUSTED_DOMAINS):
        return "trusted"
    return "verify"


def tool_failure(content: Any, status: str | None = None) -> str | None:
    """Return the failure text when a tool result reports one, else ``None``.

    An empty search is a normal outcome, not a failure.
    """
    text = str(content).strip()
    if status == "error":
        return None if NO_RESULTS_MARKER in text else text
    return text if TOOL_FAILURE_PATTERN.match(text) else None


def auth_failure(tool: str, content: Any, status: str | None = None) -> str | None:
    """Return the failure text when a keyed service rejected its credentials."""
    failure = tool_failure(content, status)
    if tool in KEYED_TOOLS and failure and AUTH_FAILURE_PATTERN.search(failure):
        return failure
    return None


def _bounded(value: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    return str(value).strip()[:limit]


def _source_payload(
    *,
    title: str,
    url: str,
    excerpt: str,
    source_type: str,
    limit: int = MAX_TOOL_OUTPUT_CHARS,
    published: str = "",
) -> dict[str, str]:
    payload = {
        "source_title": title or "Untitled source",
        "source_url": url,
        "excerpt": _bounded(excerpt, limit),
        "source_type": source_type,
    }
    if published:
        # Lets the extractor date the finding, so stale status claims can be flagged.
        payload["published"] = published
    return payload


def _iso_date(value: Any) -> str:
    """A datetime as YYYY-MM-DD; empty for a missing or placeholder date (arxiv uses datetime.min)."""
    if value is None or getattr(value, "year", 0) < 1900:
        return ""
    return value.date().isoformat()


def _excerpt_limit(count: int) -> int:
    """Share the per-tool output cap across every returned document."""
    return MAX_TOOL_OUTPUT_CHARS // max(1, count)


class BoundedTavilySearch(TavilySearch):
    """Standard TavilySearch whose output is serialized and bounded like other tools.

    TavilySearch returns API errors, including a rejected key, as an
    ``{"error": ...}`` result; report them in the same form as the other tools'
    failures so they are counted rather than read as evidence.
    """

    def _run(self, *args: Any, **kwargs: Any) -> str:
        result = super()._run(*args, **kwargs)
        if isinstance(result, dict) and "error" in result:
            return f"Tavily search failed: {result['error']}"
        return _bounded(json.dumps(result, default=str))


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
                    published=_iso_date(result.published),
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
                    published=str(article.get("Published", "") or ""),
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
        BoundedTavilySearch(max_results=3, exclude_domains=EXCLUDED_SEARCH_DOMAINS),
        WikipediaEvidenceTool(api_wrapper=WikipediaAPIWrapper()),
        ArxivEvidenceTool(),
        PubMedEvidenceTool(),
        scrape_webpage,
    ]
    if allowed is None:
        return tools
    names = set(allowed)
    return [tool for tool in tools if tool.name in names]
