from typing import List

from pydantic import BaseModel, Field


class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives."
    )

    @property
    def persona(self) -> str:
        return (
            f"Name: {self.name}\nRole: {self.role}\n"
            f"Affiliation: {self.affiliation}\nDescription: {self.description}\n"
        )


class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with roles and affiliations."
    )


class ResearchFinding(BaseModel):
    sub_question: str
    claim: str
    source_title: str
    source_url: str
    excerpt: str
    source_type: str


class ResearchFindingBatch(BaseModel):
    findings: List[ResearchFinding] = Field(default_factory=list, max_length=4)


class ResearchPlan(BaseModel):
    sub_questions: List[str] = Field(min_length=2, max_length=3)


class ResearchEvaluation(BaseModel):
    is_complete: bool
    coverage_gaps: List[str] = Field(default_factory=list)
    feedback: str
