"""Small data structures shared across the pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A retrievable piece of a document, plus whatever score the last stage gave it."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))
