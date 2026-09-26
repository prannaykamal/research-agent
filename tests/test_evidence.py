from src.analyst_research import canonicalize_url, deduplicate_findings, evidence_key
from src.utils.objects import ResearchFinding


def finding(url: str, claim: str = "Claim", excerpt: str = "Evidence") -> ResearchFinding:
    return ResearchFinding(
        sub_question="What happened?",
        claim=claim,
        source_title="Source",
        source_url=url,
        excerpt=excerpt,
        source_type="web",
    )


def test_canonicalize_url_removes_fragments_and_tracking() -> None:
    assert canonicalize_url("HTTPS://Example.COM/path/?utm_source=x&b=2#part") == "https://example.com/path?b=2"


def test_deduplicate_findings_uses_canonical_url_and_evidence() -> None:
    original = finding("https://example.com/report?utm_source=mail")
    duplicate = finding("https://EXAMPLE.com/report#section")
    distinct = finding("https://example.com/report", claim="Different claim")

    additions = deduplicate_findings([original], [duplicate, distinct])

    assert additions == [distinct.model_copy(update={"source_url": "https://example.com/report"})]
    assert evidence_key(original) == evidence_key(duplicate)
