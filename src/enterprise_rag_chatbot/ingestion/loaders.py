
"""Load local PDF documents with source and page metadata."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader
from llama_index.readers.file import PDFReader


def load_documents(directory: str | Path) -> list[Document]:
    """Read PDFs recursively, returning one Document per page."""
    directory = Path(directory)

    if not directory.is_dir():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    def file_metadata(filename: str) -> dict:
        path = Path(filename)
        manifest_path = path.parent / "manifest.json"
        entries = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
        matches = [entry for entry in entries if entry["filename"] == path.name]
        if len(matches) > 1:
            raise ValueError(f"Duplicate manifest entry: {path.name}")
        entry = matches[0] if matches else {}
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if entry.get("sha256") and entry["sha256"] != checksum:
            raise ValueError(f"Manifest checksum mismatch: {path.name}")
        source = path.resolve().relative_to(directory.resolve()).as_posix()
        metadata = {key: entry[key] for key in (
            "document_id", "title", "owner", "version", "effective_date", "status", "classification",
        ) if key in entry}
        metadata.setdefault("document_id", source)
        metadata.setdefault("version", "unversioned")
        metadata.update(source=source, file_name=path.name, sha256=checksum)
        return metadata

    reader = SimpleDirectoryReader(
        input_dir=str(directory),
        recursive=True,
        required_exts=[".pdf"],
        file_extractor={".pdf": PDFReader(return_full_document=False)},
        filename_as_id=True,
        raise_on_error=True,
        file_metadata=file_metadata,
    )

    documents = reader.load_data()
    pages = Counter()
    for document in documents:
        source = document.metadata["source"]
        pages[source] += 1
        document.metadata["page"] = pages[source]
        document.id_ = f"{source}:page:{pages[source]}"
        if not document.text.strip():
            raise ValueError(f"Empty PDF page: {source}, page {pages[source]}; check OCR")
    return documents


if __name__ == "__main__":
    documents = load_documents("data/sample/alder_quay")

    print(f"Loaded {len(documents)} document pages")

    if documents:
        first = documents[0]
        print("Metadata:", first.metadata)
        print("Text preview:", first.text[:500])
