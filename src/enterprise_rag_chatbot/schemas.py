"""Portable chunks and scored retrieval results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
