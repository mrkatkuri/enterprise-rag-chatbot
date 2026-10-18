"""Dense/BM25 retrieval, reciprocal rank fusion, and optional reranking."""

import argparse
import json

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.ingestion.document_registry import Snapshot, signature, snapshot_path
from enterprise_rag_chatbot.retrieval.bm25 import BM25
from enterprise_rag_chatbot.schemas import SearchResult


def reciprocal_rank_fusion(rankings: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores, chunks = {}, {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks[chunk_id] = item.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (k + rank)
    return [SearchResult(chunks[key], scores[key])
            for key in sorted(scores, key=lambda key: (-scores[key], key))]


class HybridRetriever:
    def __init__(self, settings: Settings, snapshot: Snapshot, *, store=None, embedding=None):
        if snapshot.signature != signature(settings):
            raise ValueError("Corpus settings changed; re-ingest with current model/chunk configuration")
        self.settings, self.snapshot = settings, snapshot
        self.store, self.embedding = store, embedding
        self.chunks = {chunk.id: chunk for chunk in snapshot.chunks}
        self.bm25 = BM25(snapshot.chunks)

    def search(self, query: str, *, mode: str = "hybrid", rerank: bool = True,
               allowed_document_ids: set[str] | None = None) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        if mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError("Mode must be dense, bm25, or hybrid")
        if allowed_document_ids is not None and not allowed_document_ids:
            return []
        use_reranker = rerank and bool(self.settings.rerank_model)
        if mode != "bm25" or use_reranker:
            if self.store is None:
                from enterprise_rag_chatbot.retrieval.pinecone_store import PineconeStore

                self.store = PineconeStore(self.settings)
        rankings = []
        if mode in {"dense", "hybrid"}:
            filters = {"corpus_id": {"$eq": self.snapshot.corpus_id}}
            if allowed_document_ids is not None:
                filters["document_id"] = {"$in": sorted(allowed_document_ids)}
            if self.embedding is None:
                self.embedding = self.settings.embedding()
            vector = self.embedding.get_query_embedding(query)
            if len(vector) != self.settings.embedding_dimensions:
                raise ValueError("Query embedding has incorrect dimensions")
            dense = self.store.query(vector, self.settings.dense_top_k, filters)
            # Validate candidates against local authoritative text before reranking.
            dense = [SearchResult(self.chunks[item.chunk.id], item.score) for item in dense
                     if item.chunk.id in self.chunks
                     and (allowed_document_ids is None
                          or self.chunks[item.chunk.id].metadata["document_id"] in allowed_document_ids)]
            rankings.append(dense)
        if mode in {"bm25", "hybrid"}:
            rankings.append(self.bm25.search(query, self.settings.bm25_top_k, allowed_document_ids))
        candidates = reciprocal_rank_fusion(rankings, self.settings.rrf_k) if mode == "hybrid" else rankings[0]
        if use_reranker:
            return self.store.rerank(query, candidates, self.settings.final_top_k)
        return candidates[:self.settings.final_top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Use offline BM25 corpus; disables reranking")
    parser.add_argument("--document-id", action="append", help="Restrict both branches to these document IDs")
    args = parser.parse_args()
    settings = Settings.from_env()
    snapshot = Snapshot.load(snapshot_path(settings, args.local_only))
    retriever = HybridRetriever(settings, snapshot)
    results = retriever.search(
        args.query, mode="bm25" if args.local_only else args.mode,
        rerank=not (args.no_rerank or args.local_only),
        allowed_document_ids=set(args.document_id) if args.document_id is not None else None,
    )
    print(json.dumps([{"id": item.chunk.id, "score": item.score,
                       "text": item.chunk.text, "metadata": item.chunk.metadata}
                      for item in results], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
