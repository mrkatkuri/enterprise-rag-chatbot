"""Dense Pinecone storage and hosted reranking, with explicit service access."""

from pinecone import Pinecone, ServerlessSpec

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.schemas import Chunk, SearchResult


class PineconeStore:
    def __init__(self, settings: Settings, *, create: bool = False):
        settings.require_services()
        self.settings = settings
        self.client = Pinecone(api_key=settings.pinecone_api_key, timeout=30)
        if not self.client.has_index(settings.index_name):
            if not create:
                raise ValueError("Pinecone index does not exist; run remote ingestion first")
            self.client.create_index(
                name=settings.index_name, dimension=settings.embedding_dimensions,
                metric="cosine", spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
                timeout=120,
            )
        description = self.client.describe_index(settings.index_name)
        if description.dimension != settings.embedding_dimensions or description.metric != "cosine":
            raise ValueError("Pinecone index must match configured dimensions and cosine metric")
        self.index = self.client.Index(host=description.host)

    def upsert(self, chunks: list[Chunk], embedding) -> None:
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset:offset + 64]
            vectors = embedding.get_text_embedding_batch([chunk.text for chunk in batch])
            if len(vectors) != len(batch) or any(len(v) != self.settings.embedding_dimensions for v in vectors):
                raise ValueError("Embedding output does not match configured dimensions or batch size")
            self.index.upsert(
                namespace=self.settings.namespace,
                vectors=[{"id": chunk.id, "values": vector,
                          "metadata": {**chunk.metadata, "text": chunk.text}}
                         for chunk, vector in zip(batch, vectors, strict=True)],
            )

    def query(self, vector, top_k: int, filters: dict) -> list[SearchResult]:
        response = self.index.query(
            namespace=self.settings.namespace, vector=vector, top_k=top_k,
            include_metadata=True, filter=filters,
        )
        return [SearchResult(Chunk(match.id, match.metadata["text"],
                                  {k: v for k, v in match.metadata.items() if k != "text"}),
                             float(match.score)) for match in response.matches]

    def delete(self, ids: list[str]) -> None:
        for offset in range(0, len(ids), 500):
            self.index.delete(ids=ids[offset:offset + 500], namespace=self.settings.namespace)

    def rerank(self, query: str, candidates: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not candidates:
            return []
        response = self.client.inference.rerank(
            model=self.settings.rerank_model, query=query,
            documents=[{"id": item.chunk.id, "text": item.chunk.text} for item in candidates],
            rank_fields=["text"], top_n=min(top_n, len(candidates)), return_documents=False,
        )
        return [SearchResult(candidates[item.index].chunk, float(item.score)) for item in response.data]
