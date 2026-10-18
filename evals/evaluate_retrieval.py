"""Measure supporting-document recall; never index the answer key."""

import argparse
import json
import time
from pathlib import Path

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.ingestion.document_registry import Snapshot, snapshot_path
from enterprise_rag_chatbot.retrieval.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("evals/alder_quay/rag_test_questions.jsonl"))
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    snapshot = Snapshot.load(snapshot_path(settings, args.local_only))
    retriever = HybridRetriever(settings, snapshot)
    rows = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    recalls = []
    elapsed = []
    for row in rows:
        # Retrieval alone cannot evaluate refusal or abstention correctness.
        if row["test_type"] in {"abstain", "adversarial"}:
            continue
        expected = set(row["document_ids"])
        start = time.perf_counter()
        results = retriever.search(row["question"], mode="bm25" if args.local_only else args.mode,
                                   rerank=not (args.no_rerank or args.local_only))
        elapsed.append(time.perf_counter() - start)
        found = {item.chunk.metadata["document_id"] for item in results}
        recall = len(expected & found) / len(expected)
        recalls.append(recall)
        print(json.dumps({"test_id": row["test_id"], "document_recall": recall,
                          "sources": [{key: item.chunk.metadata.get(key) for key in
                                       ("document_id", "version", "page", "section_number")}
                                      for item in results]}))
    if not recalls:
        raise ValueError("No eligible retrieval evaluation cases")
    print(json.dumps({"evaluated": len(recalls), "top_k": settings.final_top_k,
                      "mean_document_recall": sum(recalls) / len(recalls),
                      "mean_latency_seconds": sum(elapsed) / len(elapsed)}))


if __name__ == "__main__":
    main()
