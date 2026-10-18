# Tests

Run offline ingestion and retrieval checks from the project root:

```bash
uv run python -m unittest discover -s tests -v
```

Tests use the synthetic sample PDFs and mocked service clients. They cover
stable IDs, token limits, document filters, rank fusion, tenant snapshot scoping,
adapter payloads, and preservation of the active corpus when an upload fails.
They do not make embedding, Pinecone search, or hosted reranking API calls.

Chat tests additionally cover exact source-quote validation, structured-output
failures/refusals, checkpointed follow-ups, conversation reset and isolation,
mocked Streamlit interaction, safe upload paths, and evaluation answer-key isolation.
Chat model responses are mocked; live generation quality requires the benchmark
runner and review of its saved evidence.
