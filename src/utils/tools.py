import ipaddress
import json
from typing import Any
from urllib.parse import urlsplit

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_tavily import TavilySearch


MAX_TOOL_OUTPUT_CHARS = 12_000


def _bounded(value: Any) -> str:
    return str(value).strip()[:MAX_TOOL_OUTPUT_CHARS]


def _source_payload(
    *, title: str, url: str, excerpt: str, source_type: str
) -> dict[str, str]:
    return {
        "source_title": title or "Untitled source",
        "source_url": url,
        "excerpt": _bounded(excerpt),
        "source_type": source_type,
    }


class WikipediaEvidenceTool(WikipediaQueryRun):
    """Standard WikipediaQueryRun with source metadata preserved for evidence."""

    name: str = "wikipedia"

    def _run(self, query: str, run_manager=None) -> str:
        try:
            documents = self.api_wrapper.load(query)
            payload = [
                _source_payload(
                    title=document.metadata.get("title", "Wikipedia"),
                    url=document.metadata.get("source", ""),
                    excerpt=document.page_content,
                    source_type="wikipedia",
                )
                for document in documents
            ]
            return json.dumps(payload)
        except Exception as exc:
            return f"Wikipedia search failed: {exc}"


class ArxivEvidenceTool(ArxivQueryRun):
    """Standard ArxivQueryRun with result metadata retained in tool output."""

    name: str = "arxiv"

    def _run(self, query: str, run_manager=None) -> str:
        try:
            documents = self.api_wrapper.get_summaries_as_docs(query)
            payload = [
                _source_payload(
                    title=document.metadata.get("Title", "arXiv paper"),
                    url=document.metadata.get("Entry ID", ""),
                    excerpt=document.page_content,
                    source_type="arxiv",
                )
                for document in documents
            ]
            return json.dumps(payload)
        except Exception as exc:
            return f"arXiv search failed: {exc}"


class PubMedEvidenceTool(PubmedQueryRun):
    """Standard PubmedQueryRun with canonical PubMed references in its output."""

    name: str = "pubmed"

    def _run(self, query: str, run_manager=None) -> str:
        try:
            articles = self.api_wrapper.load(query)
            payload = [
                _source_payload(
                    title=str(article.get("Title", "PubMed article")),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{article['uid']}/",
                    excerpt=str(article.get("Summary", "")),
                    source_type="pubmed",
                )
                for article in articles
                if article.get("uid")
            ]
            return json.dumps(payload)
        except Exception as exc:
            return f"PubMed search failed: {exc}"


def _validate_public_http_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be an absolute HTTP(S) URL.")
    hostname = parsed.hostname.lower()
    if hostname == "localhost":
        raise ValueError("Local URLs cannot be scraped.")
    try:
        address = ipaddress.ip_address(hostname)
        if address.is_private or address.is_loopback or address.is_link_local:
            raise ValueError("Private network URLs cannot be scraped.")
    except ValueError as exc:
        if "cannot be scraped" in str(exc):
            raise
    return url


@tool
def scrape_webpage(url: str) -> str:
    """Fetch one public webpage and return its title, URL, and bounded text content."""
    try:
        safe_url = _validate_public_http_url(url)
        documents = WebBaseLoader(web_paths=(safe_url,)).load()
        if not documents:
            return "Webpage scrape returned no content."
        document = documents[0]
        payload = _source_payload(
            title=document.metadata.get("title", safe_url),
            url=document.metadata.get("source", safe_url),
            excerpt=document.page_content,
            source_type="scraped_page",
        )
        return json.dumps(payload)
    except Exception as exc:
        return f"Webpage scrape failed: {exc}"


def build_research_tools() -> list[Any]:
    """Build the only tools exposed to the Researcher node."""
    return [
        TavilySearch(max_results=3),
        WikipediaEvidenceTool(api_wrapper=WikipediaAPIWrapper()),
        ArxivEvidenceTool(),
        PubMedEvidenceTool(),
        scrape_webpage,
    ]
