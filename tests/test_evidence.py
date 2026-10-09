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


def test_source_tier_flags_the_sources_behind_past_correctness_failures() -> None:
    from src.utils.tools import source_tier

    for url in (
        "https://brewminate.com/the-industrial-revolution",
        "https://simple.wikipedia.org/wiki/Organic_chemistry",
        "https://www.youtube.com/watch?v=abc",
        "https://symbiosisonlinepublishing.com/vaccines/paper.php",
    ):
        assert source_tier(url) == "low", url
    for url in (
        "https://arxiv.org/abs/2401.00001",
        "https://doi.org/10.1038/nature12345",
        "https://pubmed.ncbi.nlm.nih.gov/123/",
        "https://www.nasa.gov/missions",
        "https://link.springer.com/article/1",
    ):
        assert source_tier(url) == "trusted", url
    # Tertiary references and unknown sites warrant a spot-check, not trust.
    assert source_tier("https://en.wikipedia.org/wiki/Fusion_power") == "verify"
    assert source_tier("https://example.com/post") == "verify"


def test_deduplication_prefers_trusted_sources_at_capacity() -> None:
    from src.utils.guardrails import MAX_FINDINGS_PER_ANALYST

    existing = [finding(f"https://example.com/{index}", claim=f"Old {index}") for index in range(MAX_FINDINGS_PER_ANALYST - 1)]
    weak = finding("https://brewminate.com/claim", claim="Weak")
    strong = finding("https://arxiv.org/abs/1", claim="Strong")

    assert deduplicate_findings(existing, [weak, strong]) == [strong]
    # With room for both, extraction order is kept.
    assert deduplicate_findings([], [weak, strong]) == [weak, strong]
