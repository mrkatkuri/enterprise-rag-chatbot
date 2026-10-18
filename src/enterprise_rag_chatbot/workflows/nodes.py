"""Bounded workflow steps; access scope is bound by the application, not the model."""

import json

from pydantic import BaseModel, ConfigDict

from enterprise_rag_chatbot.generation.answer import AnswerGenerator, abstain
from enterprise_rag_chatbot.generation.prompts import REWRITE_SYSTEM
from enterprise_rag_chatbot.workflows.state import ChatState


class RewrittenQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str


class ChatNodes:
    def __init__(self, retriever, model, *, history_turns=6, mode="hybrid", rerank=True,
                 allowed_document_ids: set[str] | None = None):
        self.retriever = retriever
        self.generator = AnswerGenerator(model)
        self.rewriter = model.with_structured_output(
            RewrittenQuestion, method="json_schema", strict=True, include_raw=True,
        )
        self.history_turns, self.mode, self.rerank = history_turns, mode, rerank
        self.allowed_document_ids = frozenset(allowed_document_ids) if allowed_document_ids is not None else None

    def resolve(self, state: ChatState) -> dict:
        question = state["question"]
        usage = {}
        history = state.get("history", [])[-2 * self.history_turns:]
        if history:
            response = self.rewriter.invoke([
                ("system", REWRITE_SYSTEM),
                ("human", json.dumps({"history": history, "question": question})),
            ])
            if response.get("parsing_error") or response.get("parsed") is None:
                raise ValueError("Could not resolve the follow-up question; try a standalone question")
            question = RewrittenQuestion.model_validate(response["parsed"]).question.strip()
            if not question or len(question) > 8000:
                raise ValueError("Resolved question is empty or too long")
            usage = getattr(response.get("raw"), "usage_metadata", None) or {}
        return {"resolved_question": question, "results": [], "rewrite_usage": usage}

    def retrieve(self, state: ChatState) -> dict:
        results = self.retriever.search(
            state["resolved_question"], mode=self.mode, rerank=self.rerank,
            allowed_document_ids=self.allowed_document_ids,
        )
        return {"results": results}

    def evidence_route(self, state: ChatState) -> str:
        return "generate" if state.get("results") else "abstain"

    def generate(self, state: ChatState) -> dict:
        # Conversation history helps resolve references, but is not policy evidence.
        answer = self.generator.generate(state["resolved_question"], state["results"])
        return {"answer": answer}

    def abstain(self, state: ChatState) -> dict:
        return {"answer": abstain()}

    def finish(self, state: ChatState) -> dict:
        answer = state["answer"]
        answer = answer.model_copy(update={"usage": {
            "generation": answer.usage, "rewrite": state.get("rewrite_usage", {}),
        }})
        history = state.get("history", []) + [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer.text},
        ]
        return {"answer": answer, "history": history[-2 * self.history_turns:]}
