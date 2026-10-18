"""Generate answers with source-bound citations and inspectable evidence quotes."""

import argparse
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from enterprise_rag_chatbot.generation.prompts import ANSWER_SYSTEM
from enterprise_rag_chatbot.schemas import SearchResult


class Support(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    supports: list[Support]


class DraftAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient", "refused"]
    claims: list[Claim]


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    version: str
    page: str
    section: str
    source: str
    quote: str

    @property
    def label(self) -> str:
        return f"[{self.document_id} v{self.version}, page {self.page}, section {self.section}]"


class Answer(BaseModel):
    status: Literal["answered", "insufficient", "refused"]
    text: str
    citations: list[Citation]
    reason: str
    usage: dict
    diagnostics: list[str] = Field(default_factory=list)


def abstain(reason: str = "insufficient_evidence", usage: dict | None = None,
            diagnostics: list[str] | None = None) -> Answer:
    text = "I don't have enough evidence in the retrieved documents to answer that."
    if reason == "invalid_citation":
        text = "I found documents, but couldn't verify the generated answer's citations. Please retry."
    return Answer(status="insufficient", text=text, citations=[], reason=reason,
                  usage=usage or {}, diagnostics=diagnostics or [])


def prepare_evidence(results: list[SearchResult]) -> tuple[list[dict], dict]:
    """Give exact source excerpts short IDs; the model selects, rather than copies."""
    evidence, references = [], {}
    for source_number, result in enumerate(results, start=1):
        source_id = f"S{source_number}"
        excerpts = []
        for excerpt_number, text in enumerate(re.split(r"(?<=[.!?])\s+", result.chunk.text), start=1):
            quote = text.strip()
            if not quote:
                continue
            evidence_id = f"{source_id}.E{excerpt_number}"
            references[evidence_id] = (result.chunk, quote)
            excerpts.append({"evidence_id": evidence_id, "text": quote})
        evidence.append({"source_id": source_id, "metadata": result.chunk.metadata, "excerpts": excerpts})
    return evidence, references


class AnswerGenerator:
    def __init__(self, model):
        self.structured = model.with_structured_output(
            DraftAnswer, method="json_schema", strict=True, include_raw=True,
        )

    def generate(self, question: str, results: list[SearchResult]) -> Answer:
        if not question.strip():
            raise ValueError("Question must not be empty")
        if not results:
            return abstain()
        evidence, references = prepare_evidence(results)
        response = self.structured.invoke([
            ("system", ANSWER_SYSTEM),
            ("human", json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False)),
        ])
        raw = response.get("raw")
        usage = getattr(raw, "usage_metadata", None) or {}
        if getattr(raw, "additional_kwargs", {}).get("refusal"):
            return Answer(status="refused", text="I can't help disclose secrets or bypass access controls.",
                          citations=[], reason="model_refusal", usage=usage)
        if response.get("parsing_error"):
            raise ValueError("Chat model returned an invalid structured answer") from response["parsing_error"]
        draft = response.get("parsed")
        if draft is None:
            raise ValueError("Chat model returned no structured answer")
        draft = DraftAnswer.model_validate(draft)
        if draft.status == "refused":
            return Answer(status="refused", text="I can't help disclose secrets or bypass access controls.",
                          citations=[], reason="model_refusal", usage=usage)
        if draft.status == "insufficient" or not draft.claims:
            return abstain(usage=usage)
        citations, lines = [], []
        for claim_number, claim in enumerate(draft.claims, start=1):
            if not claim.text.strip() or not claim.supports or re.search(r"\[[^\]]+\]", claim.text):
                return abstain("invalid_citation", usage, [f"Claim {claim_number}: missing text/support or model-added citation label"])
            labels = []
            for support in claim.supports:
                reference = references.get(support.evidence_id)
                if reference is None:
                    return abstain("invalid_citation", usage, [f"Claim {claim_number}: unknown evidence ID"])
                chunk, quote = reference
                m = chunk.metadata
                citation = Citation(
                    chunk_id=chunk.id, document_id=str(m["document_id"]),
                    version=str(m.get("version", "unversioned")), page=str(m.get("page", m.get("page_label", "unknown"))),
                    section=str(m.get("section_number") or "cover/unsectioned"),
                    source=str(m.get("source", m.get("file_name", ""))), quote=quote,
                )
                if citation not in citations:
                    citations.append(citation)
                if citation.label not in labels:
                    labels.append(citation.label)
            lines.append(claim.text.strip() + " " + " ".join(labels))
        return Answer(status="answered", text="\n\n".join(lines), citations=citations, reason="", usage=usage)


def main() -> None:
    from enterprise_rag_chatbot.config import Settings
    from enterprise_rag_chatbot.workflows.chat_graph import ChatSession

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Local BM25 retrieval; chat still calls OpenAI")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--document-id", action="append")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not args.question and not args.interactive:
        parser.error("Supply a question or --interactive")
    session = ChatSession.from_settings(
        Settings.from_env(), local_only=args.local_only, mode=args.mode, rerank=not args.no_rerank,
        allowed_document_ids=set(args.document_id) if args.document_id is not None else None,
    )

    def show(question):
        answer = session.ask(question)
        print(answer.model_dump_json(indent=2) if args.json else answer.text)

    if args.question:
        show(args.question)
    if args.interactive:
        print("Ask follow-up questions; type /reset or /quit.")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question == "/quit":
                break
            if question == "/reset":
                session.reset()
            elif question:
                show(question)


if __name__ == "__main__":
    main()
