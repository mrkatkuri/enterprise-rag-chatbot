"""Checkpointed per-conversation state."""

from typing import TypedDict

from enterprise_rag_chatbot.generation.answer import Answer
from enterprise_rag_chatbot.schemas import SearchResult


class ChatState(TypedDict, total=False):
    question: str
    resolved_question: str
    history: list[dict[str, str]]
    results: list[SearchResult]
    answer: Answer
    rewrite_usage: dict
