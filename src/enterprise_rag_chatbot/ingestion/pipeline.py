"""Build section-aware snapshots and publish a corpus for hybrid retrieval."""

import argparse
import hashlib
import json
import re
import warnings
from collections.abc import Iterable
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.utils import get_tokenizer

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.ingestion.document_registry import Snapshot, signature, snapshot_path
from enterprise_rag_chatbot.ingestion.loaders import load_documents
from enterprise_rag_chatbot.schemas import Chunk


SECTION_HEADING = re.compile(r"(?m)^(\d+(?:\.\d+)*)[.)]\s+([^\n]+)$")
SAMPLE_HEADER = re.compile(r"AQ-[A-Z]+-\d+\s*\|\s*v[\d.]+\s*\|\s*\d+")
SAMPLE_NOTICE = "FICTIONAL TRAINING / TEST DOCUMENT - NOT LEGAL ADVICE"


def _clean_page(document: Document) -> tuple[str, dict]:
    """Remove known sample-pack furniture; retain source qualifications."""
    metadata = dict(document.metadata)
    text = document.text
    if not text.strip():
        raise ValueError(f"Empty document page: {document.id_}; check extraction/OCR")
    if SAMPLE_NOTICE in text:
        metadata["source_notice"] = SAMPLE_NOTICE
        text = text.split("Contents and citation guide", 1)[0]
        text = "\n".join(
            line for line in text.splitlines()
            if line.strip() != SAMPLE_NOTICE
            and not SAMPLE_HEADER.fullmatch(line.strip())
        )
    metadata.setdefault("document_id", metadata.get("source", metadata.get("file_name", document.id_)))
    metadata.setdefault("version", "unversioned")
    return text.strip(), metadata


def _sections(text: str) -> Iterable[tuple[str, str, str]]:
    """Yield section number, heading, and body within a page."""
    headings = list(SECTION_HEADING.finditer(text))
    if not headings:
        yield "", "", text
        return
    preamble = text[:headings[0].start()].strip()
    if preamble:
        yield "", "", preamble
    for position, heading in enumerate(headings):
        end = headings[position + 1].start() if position + 1 < len(headings) else len(text)
        body = text[heading.end():end].strip()
        if body:
            yield heading[1], heading[2].strip(), body


def build_snapshot(documents: Iterable[Document], settings: Settings) -> Snapshot:
    """Build bounded chunks and stable IDs without embedding/service calls."""
    tokenizer = get_tokenizer()
    prepared: list[tuple[str, dict]] = []
    for document in documents:
        text, metadata = _clean_page(document)
        for number, title, body in _sections(text):
            prefix = f"{number}. {title}\n" if number else ""
            # Repeat headings on continuation chunks without exceeding the cap.
            budget = settings.chunk_size - len(tokenizer(prefix)) - 2
            if budget <= settings.chunk_overlap:
                raise ValueError("CHUNK_SIZE is too small for the heading and overlap")
            splitter = SentenceSplitter(chunk_size=budget, chunk_overlap=settings.chunk_overlap)
            for position, part in enumerate(splitter.split_text(body)):
                content = prefix + part
                if len(tokenizer(content)) > settings.chunk_size:
                    raise ValueError("Chunk exceeds configured token limit")
                prepared.append((content, {
                    **metadata,
                    "section_number": number,
                    "section_title": title,
                    "chunk_position": position,
                    "namespace": settings.namespace,
                }))
    if not prepared:
        raise ValueError("No searchable text chunks produced")
    config = signature(settings)
    payload = json.dumps([config, prepared], sort_keys=True, ensure_ascii=False)
    corpus_id = hashlib.sha256(payload.encode()).hexdigest()
    chunks = [
        Chunk(
            id=hashlib.sha256(f"{corpus_id}:{position}".encode()).hexdigest(),
            text=text,
            metadata={**metadata, "corpus_id": corpus_id},
        )
        for position, (text, metadata) in enumerate(prepared)
    ]
    return Snapshot(corpus_id=corpus_id, signature=config, chunks=chunks)


def ingest(
    directory: str | Path,
    settings: Settings,
    *,
    local_only: bool = False,
    dry_run: bool = False,
) -> Snapshot:
    """Replace the complete corpus; use one writer per index/namespace.

    Uploads finish before the new local snapshot is published. Failed uploads
    leave the previous generation active. Only previous known IDs are deleted;
    corpus filtering excludes any unused vectors from interrupted uploads.
    Pinecone search visibility can lag successful upserts.
    """
    snapshot = build_snapshot(load_documents(directory), settings)
    if dry_run:
        return snapshot
    path = snapshot_path(settings, local_only)
    if local_only:
        snapshot.save(path)
        return snapshot
    settings.require_services()
    # Offline preparation does not need service clients.
    from enterprise_rag_chatbot.retrieval.pinecone_store import PineconeStore

    previous = Snapshot.load(path) if path.exists() else None
    store = PineconeStore(settings, create=True)
    store.upsert(snapshot.chunks, settings.embedding())
    snapshot.save(path)
    if previous and previous.corpus_id != snapshot.corpus_id:
        try:
            store.delete([chunk.id for chunk in previous.chunks])
        except Exception as exc:
            warnings.warn(
                f"New corpus active; old vector cleanup failed ({type(exc).__name__}).",
                RuntimeWarning,
                stacklevel=2,
            )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="data/sample/alder_quay")
    parser.add_argument("--dry-run", action="store_true", help="Prepare without writes or API calls")
    parser.add_argument("--local-only", action="store_true", help="Save a separate offline BM25 corpus")
    args = parser.parse_args()
    snapshot = ingest(
        args.directory, Settings.from_env(), local_only=args.local_only, dry_run=args.dry_run,
    )
    print(f"Prepared {len(snapshot.chunks)} chunks; corpus {snapshot.corpus_id[:12]}")
    if args.dry_run:
        print("Dry run: no corpus written or uploaded")
    elif args.local_only:
        print("Saved local BM25 corpus")
    else:
        print("Uploaded to Pinecone and saved local corpus snapshot")
    print(snapshot.chunks[0].metadata)
    print(snapshot.chunks[0].text[:300])


if __name__ == "__main__":
    main()
