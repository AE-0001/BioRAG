from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Modality = Literal["text", "figure"]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str
    source: str
    modality: Modality = "text"
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Document:
        return cls(**value)


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float
    lexical_score: float
    semantic_score: float

    def citation(self) -> str:
        page = f", p. {self.document.page}" if self.document.page else ""
        return f"{self.document.title} ({self.document.source}{page})"


@dataclass
class AgentTrace:
    agent: str
    action: str
    detail: str


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[str]
    evidence: list[SearchHit]
    grounded: bool
    trace: list[AgentTrace]
