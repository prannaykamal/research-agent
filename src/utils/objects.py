from typing import List, Literal

from pydantic import BaseModel, Field

from src.utils.guardrails import MAX_FINDINGS_PER_PASS, MAX_REQUIREMENTS


class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(
        description=(
            "Three or four sentences: expertise and vantage point; the specific sub-topics, "
            "periods, and evidence pursued for each owned requirement; trusted source types; "
            "and the claims the analyst is sceptical of."
        )
    )
    requirements: List[str] = Field(
        default_factory=list,
        description="The question requirements this analyst owns, copied verbatim from the supplied list.",
    )

    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\nRole: {self.role}\n"
            f"Affiliation: {self.affiliation}\nDescription: {self.description}\n"
        )


class QuestionRequirements(BaseModel):
    requirements: List[str] = Field(
        min_length=1,
        max_length=MAX_REQUIREMENTS,
        description="Separate, non-overlapping requirements a complete answer must cover.",
    )


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description=(
            "Exactly the requested number of distinct analysts with roles and affiliations."
        )
    )


class ResearchFinding(BaseModel):
    sub_question: str
    claim: str
    source_title: str
    source_url: str
    excerpt: str
    source_type: str
    requirement: str = Field(
        default="",
        description="The analyst requirement this finding supports, copied verbatim; empty if none.",
    )
    source_date: str = Field(
        default="",
        description=(
            "Publication or as-of date of the source (YYYY, YYYY-MM or YYYY-MM-DD) from the "
            "tool's `published` field, the URL path, or the text; empty if unknown."
        ),
    )


class ResearchFindingBatch(BaseModel):
    findings: List[ResearchFinding] = Field(default_factory=list, max_length=MAX_FINDINGS_PER_PASS)


class ResearchPlan(BaseModel):
    sub_questions: List[str] = Field(min_length=2, max_length=3)


class RequirementStatus(BaseModel):
    requirement: str = Field(description="An owned requirement, copied verbatim.")
    status: Literal["supported", "thin", "missing"]


class ResearchEvaluation(BaseModel):
    is_complete: bool
    coverage_gaps: List[str] = Field(default_factory=list)
    feedback: str
    requirement_status: List[RequirementStatus] = Field(
        default_factory=list,
        description="One status for every requirement the analyst owns.",
    )
