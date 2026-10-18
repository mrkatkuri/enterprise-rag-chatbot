"""An acyclic, checkpointed conversation graph and application-facing session."""

from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.generation.answer import Answer
from enterprise_rag_chatbot.workflows.nodes import ChatNodes
from enterprise_rag_chatbot.workflows.state import ChatState


def build_chat_graph(nodes: ChatNodes, checkpointer=None):
    graph = StateGraph(ChatState)
    for name in ("resolve", "retrieve", "generate", "abstain", "finish"):
        graph.add_node(name, getattr(nodes, name))
    graph.add_edge(START, "resolve")
    graph.add_edge("resolve", "retrieve")
    graph.add_conditional_edges("retrieve", nodes.evidence_route,
                                {"generate": "generate", "abstain": "abstain"})
    graph.add_edge("generate", "finish")
    graph.add_edge("abstain", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


class ChatSession:
    """One process-local conversation with a fixed retriever and access scope.

    Session IDs must not be supplied by untrusted clients. A production server
    should construct sessions from authenticated identities and use durable storage.
    """

    def __init__(self, retriever, model, *, history_turns=6, mode="hybrid", rerank=True,
                 allowed_document_ids: set[str] | None = None):
        self.checkpointer = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[
            ("enterprise_rag_chatbot.schemas", "Chunk"),
            ("enterprise_rag_chatbot.schemas", "SearchResult"),
            ("enterprise_rag_chatbot.generation.answer", "Answer"),
        ]))
        self.nodes = ChatNodes(retriever, model, history_turns=history_turns, mode=mode,
                               rerank=rerank, allowed_document_ids=allowed_document_ids)
        self.graph = build_chat_graph(self.nodes, self.checkpointer)
        self.thread_id = uuid4().hex

    @classmethod
    def from_settings(cls, settings: Settings, *, local_only=False, mode="hybrid", rerank=True,
                      allowed_document_ids: set[str] | None = None):
        from enterprise_rag_chatbot.generation.models import create_chat_model
        from enterprise_rag_chatbot.ingestion.document_registry import Snapshot, snapshot_path
        from enterprise_rag_chatbot.retrieval.retriever import HybridRetriever

        snapshot = Snapshot.load(snapshot_path(settings, local_only))
        return cls(
            HybridRetriever(settings, snapshot), create_chat_model(settings),
            history_turns=settings.history_turns, mode="bm25" if local_only else mode,
            rerank=False if local_only else rerank, allowed_document_ids=allowed_document_ids,
        )

    def ask(self, question: str) -> Answer:
        if not question.strip() or len(question) > 8000:
            raise ValueError("Enter a question between 1 and 8000 characters")
        result = self.graph.invoke(
            {"question": question.strip()},
            config={"configurable": {"thread_id": self.thread_id}, "recursion_limit": 12},
        )
        return Answer.model_validate(result["answer"])

    def reset(self) -> None:
        self.checkpointer.delete_thread(self.thread_id)
        self.thread_id = uuid4().hex
