"""Atomic, namespace-scoped corpus snapshots for local lexical retrieval."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.schemas import Chunk


@dataclass
class Snapshot:
    corpus_id: str
    signature: dict
    chunks: list[Chunk]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(asdict(self), stream, ensure_ascii=False)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: Path) -> "Snapshot":
        if not path.exists():
            raise FileNotFoundError(f"No corpus snapshot at {path}. Run ingestion first.")
        value = json.loads(path.read_text())
        return cls(value["corpus_id"], value["signature"], [Chunk(**c) for c in value["chunks"]])


def signature(settings: Settings) -> dict:
    return {
        "format": 1,
        "index": settings.index_name,
        "namespace": settings.namespace,
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def snapshot_path(settings: Settings, local_only: bool = False) -> Path:
    scope = f"{settings.index_name}:{settings.namespace}"
    name = hashlib.sha256(scope.encode()).hexdigest()[:20]
    suffix = ".local.json" if local_only else ".json"
    return settings.cache_dir / (name + suffix)
