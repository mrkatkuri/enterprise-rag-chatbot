"""Environment-backed settings; credentials are never written to snapshots."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default="", repr=False)
    pinecone_api_key: str = field(default="", repr=False)
    index_name: str = "enterprise-rag-chatbot"
    namespace: str = "local-development"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    chunk_size: int = 512
    chunk_overlap: int = 50
    dense_top_k: int = 20
    bm25_top_k: int = 20
    final_top_k: int = 6
    rrf_k: int = 60
    rerank_model: str = "bge-reranker-v2-m3"
    cache_dir: Path = Path("data/index")
    cloud: str = "aws"
    region: str = "us-east-1"
    chat_model: str = "gpt-4o-mini"
    chat_max_tokens: int = 2000
    history_turns: int = 6

    def __post_init__(self) -> None:
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Require 0 <= CHUNK_OVERLAP < CHUNK_SIZE")
        for name in ("embedding_dimensions", "dense_top_k", "bm25_top_k", "final_top_k", "rrf_k", "chat_max_tokens", "history_turns"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.namespace or not self.index_name or not self.embedding_model:
            raise ValueError("Index, namespace, and embedding model must not be empty")
        if not self.cloud.strip() or not self.region.strip():
            raise ValueError("PINECONE_CLOUD and PINECONE_REGION must not be empty")

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            index_name=os.getenv("PINECONE_INDEX_NAME", "enterprise-rag-chatbot"),
            namespace=os.getenv("PINECONE_NAMESPACE", "local-development"),
            embedding_model=os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",
            embedding_dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            dense_top_k=int(os.getenv("DENSE_TOP_K", "20")),
            bm25_top_k=int(os.getenv("BM25_TOP_K", "20")),
            final_top_k=int(os.getenv("FINAL_TOP_K", "6")),
            rrf_k=int(os.getenv("RRF_K", "60")),
            rerank_model=os.getenv("RERANK_MODEL", "bge-reranker-v2-m3"),
            cache_dir=Path(os.getenv("INDEX_CACHE_DIR", "data/index")),
            cloud=(os.getenv("PINECONE_CLOUD") or "").strip() or "aws",
            region=(os.getenv("PINECONE_REGION") or "").strip() or "us-east-1",
            chat_model=(os.getenv("CHAT_MODEL") or "").strip() or "gpt-4o-mini",
            chat_max_tokens=int(os.getenv("CHAT_MAX_TOKENS", "2000")),
            history_turns=int(os.getenv("CHAT_HISTORY_TURNS", "6")),
        )

    def require_services(self) -> None:
        if not self.openai_api_key or not self.pinecone_api_key:
            raise ValueError("Set OPENAI_API_KEY and PINECONE_API_KEY in .env")

    def embedding(self):
        from llama_index.embeddings.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
            api_key=self.openai_api_key,
            timeout=30,
            max_retries=3,
        )
