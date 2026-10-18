"""Run independent RAG cases and save answers for human grounding assessment."""

import argparse
import json
import time
from pathlib import Path

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.workflows.chat_graph import ChatSession


def evaluate_case(session: ChatSession, row: dict) -> dict:
    session.reset()  # Benchmark questions are independent, not follow-ups.
    start = time.perf_counter()
    answer = session.ask(row["question"])
    latency = time.perf_counter() - start
    expected_status = {"abstain": "insufficient", "adversarial": "refused"}.get(row["test_type"], "answered")
    expected_ids = set(row["document_ids"]) - {"none"}
    found_ids = {citation.document_id for citation in answer.citations}
    state = session.graph.get_state({"configurable": {"thread_id": session.thread_id}})
    retrieved_context = [
        {"chunk_id": result.chunk.id, "text": result.chunk.text, "metadata": result.chunk.metadata}
        for result in state.values.get("results", [])
    ]
    return {
        "test_id": row["test_id"], "question": row["question"],
        "reference_answer": row["expected_answer"], "expected_status": expected_status,
        "status_match": answer.status == expected_status,
        "citation_document_recall": len(expected_ids & found_ids) / len(expected_ids) if expected_ids else None,
        "latency_seconds": latency, "answer": answer.model_dump(),
        "retrieved_context": retrieved_context,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("evals/alder_quay/rag_test_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluations/answers.jsonl"))
    parser.add_argument("--mode", choices=["dense", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--local-only", action="store_true", help="Local BM25 retrieval; chat still calls OpenAI")
    args = parser.parse_args()
    settings = Settings.from_env()
    session = ChatSession.from_settings(settings, local_only=args.local_only, mode=args.mode, rerank=not args.no_rerank)
    rows = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Question set is empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open("w") as stream:
        for row in rows:
            result = evaluate_case(session, row)
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{row['test_id']}: {result['answer']['status']}")
    usage = {"input_tokens": 0, "output_tokens": 0}
    for result in results:
        for stage in result["answer"]["usage"].values():
            for key in usage:
                usage[key] += stage.get(key, 0)
    recalls = [r["citation_document_recall"] for r in results if r["citation_document_recall"] is not None]
    print(json.dumps({
        "cases": len(results), "status_match_rate": sum(r["status_match"] for r in results) / len(results),
        "mean_citation_document_recall": sum(recalls) / len(recalls) if recalls else None,
        "mean_latency_seconds": sum(r["latency_seconds"] for r in results) / len(results),
        "chat_token_usage": usage, "output": str(args.output),
        "note": "Status and document recall do not establish grounded correctness. Review claims, quotes, sections, and exceptions manually. Embedding/reranking usage is excluded.",
    }, indent=2))


if __name__ == "__main__":
    main()
