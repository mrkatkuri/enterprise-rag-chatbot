# Enterprise RAG Chatbot

A Python project scaffold for document-grounded conversations using LlamaIndex,
Pinecone, LangChain, LangGraph, and Streamlit.

## Current status

PDF ingestion, section-aware chunking, OpenAI embeddings, Pinecone dense search,
local BM25, reciprocal rank fusion, and optional Pinecone-hosted reranking are
implemented. Cited answer generation, a LangGraph conversation workflow, and
Streamlit chat with uploads and feedback are available. Authentication, durable
conversation storage, and production deployment remain future work.

## Setup

Requires Python 3.12 and uv.

```bash
uv sync --locked
cp -n .env.example .env
```

Fill in credentials in `.env` before remote ingestion/search. Existing `.env`
values are preserved by the setup command; use `.env.example` to add new settings.
Settings load from `.env`, with shell environment values taking precedence.
Never commit `.env` or private enterprise documents.

Ask a question (requires a previously indexed corpus and an OpenAI key):

```bash
uv run enterprise-rag-chatbot "What are the annual leave carry-forward rules?"
```

Launch the chat interface:

```bash
uv run streamlit run src/enterprise_rag_chatbot/ui/streamlit_app.py
```

## Project structure

```text
src/enterprise_rag_chatbot/
├── __init__.py                 # Existing command entry point
├── config.py                   # Settings and environment configuration
├── schemas.py                  # Shared data models
├── ingestion/
│   ├── loaders.py              # Document readers
│   ├── pipeline.py             # LlamaIndex ingestion
│   └── document_registry.py    # Document versions and status
├── retrieval/
│   ├── pinecone_store.py        # Vector storage operations
│   └── retriever.py             # Permission-aware retrieval
├── generation/
│   ├── models.py                # LangChain model integrations
│   ├── prompts.py               # Grounding and citation prompts
│   └── answer.py                # Generation, citation validation, and CLI
├── workflows/
│   ├── state.py                 # Conversation state
│   ├── nodes.py                 # Workflow steps
│   └── chat_graph.py            # LangGraph assembly
└── ui/
    └── streamlit_app.py         # Streamlit entry point

tests/                          # Unit and integration tests
evals/                          # Quality evaluation datasets and scripts
data/sample/                    # Non-sensitive sample documents
.env.example                    # Configuration template
```

Each package directory also contains an `__init__.py` file.

## Answer generation and conversations

`CHAT_MODEL` defaults to `gpt-4o-mini`; choose an available OpenAI model that
supports structured outputs. `CHAT_MAX_TOKENS` defaults to 2000. `CHAT_HISTORY_TURNS`
defaults to six completed exchanges. Existing `.env` files with a blank CHAT_MODEL
use the default automatically; embedding/index configuration is unchanged.

```bash
uv run enterprise-rag-chatbot "What are the annual leave carry-forward rules?"
uv run enterprise-rag-chatbot "What are the annual leave carry-forward rules?" --json
uv run enterprise-rag-chatbot --interactive
```

In interactive mode, use `/reset` to clear history and `/quit` to exit. The same
CLI is available through `python -m enterprise_rag_chatbot.generation.answer`.
Use `--mode dense|bm25|hybrid`, `--no-rerank`, and repeatable `--document-id` filters
to compare retrieval. `--local-only` uses the separate local BM25 snapshot, but
answer generation and follow-up resolution still make OpenAI API calls.

The LangGraph workflow is resolve follow-up → retrieve → check for evidence →
generate or abstain → record history. Evidence sufficiency is assessed by the
generation model; an empty retrieval skips generation. The graph has no retry
loops; the chat client's transport retries are capped at two with a 45-second
timeout per request. History is used only to resolve question references, never
as policy evidence. Threads have server-generated IDs and process-local in-memory
checkpoints; sessions are not shared through Streamlit's global resource cache.
History and checkpoints disappear when the process/session is replaced. Production
needs authenticated session ownership, retention limits, and durable storage.

The model returns factual claims and selects short evidence IDs from source
excerpts. The application validates those IDs, obtains exact quotes from the
original chunks, and constructs document/version/page/section citations from stored
metadata. Unknown IDs cause citation rejection; the message distinguishes this
from missing evidence and `--json` includes diagnostics. Source text and long hash
IDs do not need to be reproduced by the model. These checks prove source membership,
not that every claim logically follows from its quote. Prompts require conditions, exceptions, synthetic
source qualifications, and refusal of secret disclosure; these behaviors still
need benchmark and adversarial evaluation. No universal reranker score threshold
is used. API/parsing failures surface as errors rather than successful answers.

The Streamlit sidebar supports retrieval comparisons, conversation reset, PDF
uploads, and corpus rebuilding. Uploads are capped at 10 MB each and saved under
`data/uploads/` using generated names. Rebuild indexes every PDF under `data/`,
including sample documents and existing uploads, as a complete corpus replacement.
The first UI rebuild can re-embed existing documents because their source paths
are relative to `data/`. A corpus or retrieval-setting change resets chat history.
Feedback stores rating, status, and chunk IDs in ignored `data/feedback/` files.
This is a local development interface with access to all documents in its configured
namespace; do not expose it as an authenticated enterprise application.

## Ingestion and retrieval

Defaults: `text-embedding-3-small`, 1536 dimensions, cosine similarity, section-aware
chunks capped at 512 tokens with 50-token overlap when sections need splitting.
Short sections remain short; headings are retained in each chunk. Page boundaries
are preserved for citations. The splitter uses LlamaIndex's default tokenizer.

Inspect chunks without API calls or file writes:

```bash
uv run python -m enterprise_rag_chatbot.ingestion.pipeline --dry-run
```

Build and search a separate offline BM25 corpus:

```bash
uv run python -m enterprise_rag_chatbot.ingestion.pipeline --local-only
uv run python -m enterprise_rag_chatbot.retrieval.retriever "What are the carry-forward rules?" --local-only
```

Index in Pinecone (calls OpenAI embeddings and uploads PDF chunks), then retrieve:

```bash
uv run python -m enterprise_rag_chatbot.ingestion.pipeline
uv run python -m enterprise_rag_chatbot.retrieval.retriever "What are the carry-forward rules?"
```

Ingestion creates the configured Pinecone index if absent and validates cosine
similarity and dimensions if present. Cloud/region are configurable. By default,
search fuses the top 20 dense and top 20 BM25 hits using reciprocal rank fusion,
then sends the merged candidates to Pinecone's hosted `bge-reranker-v2-m3` and
returns six results. Hosted reranking is a separate API call. Set `RERANK_MODEL=`
or pass `--no-rerank` to disable it; service failures are surfaced, not silently
converted into successful searches. The first tokenizer use may download its data.

Compare retrieval modes on the same indexed corpus:

```bash
uv run python -m enterprise_rag_chatbot.retrieval.retriever "AQ-HR-001 annual leave" --mode dense --no-rerank
uv run python -m enterprise_rag_chatbot.retrieval.retriever "AQ-HR-001 annual leave" --mode bm25 --no-rerank
uv run python -m enterprise_rag_chatbot.retrieval.retriever "AQ-HR-001 annual leave" --mode hybrid --no-rerank
uv run python -m enterprise_rag_chatbot.retrieval.retriever "AQ-HR-001 annual leave" --mode hybrid
```

Use `--document-id AQ-HR-001` (repeatable) to restrict both retrieval branches.
Library callers can pass `allowed_document_ids`; an empty set returns no results.
This is a filtering interface, not authentication. A future authenticated server
must derive the namespace and permitted document IDs from trusted session state.
The current CLI defaults to all documents in its configured namespace.

Only PDFs enter the evidence corpus. Optional adjacent `manifest.json` files supply
document IDs, versions, classifications, and checksums; checksum mismatches and
empty pages fail ingestion. Known sample-pack page headers are removed and the
fictional-use notice is retained in metadata. README and evaluation answers are
excluded. The sample pack's table of contents is removed to prevent heading-only
chunks from outranking substantive policy evidence; cover metadata remains.
Missing manifest entries receive path-based document IDs.

Snapshots under ignored `data/index/` contain full extracted text for BM25. Each
index/namespace has its own snapshot. Treat this local directory as document data.
Run one ingestion writer per index/namespace and ingest the complete desired corpus:
this is a replacement operation, not an append operation. Generation IDs include
content and configuration; failed uploads leave the previous snapshot active.
After successful publication, vectors from the previous known generation are
deleted by ID. Interrupted uploads may leave unused vectors; corpus filtering
excludes them. Pinecone is eventually consistent, so newly uploaded vectors may
take time to become searchable. Back up snapshots with their matching index.
Changing the model/chunk settings requires re-ingestion; changing dimensions also
requires a compatible index. Local-only snapshots never replace remote snapshots.

## Verification

```bash
uv run python -m unittest discover -s tests -v
uv run python evals/evaluate_retrieval.py --local-only
uv run python evals/evaluate_retrieval.py --mode dense --no-rerank
uv run python evals/evaluate_retrieval.py --mode bm25 --no-rerank
uv run python evals/evaluate_retrieval.py --mode hybrid --no-rerank
uv run python evals/evaluate_retrieval.py --mode hybrid
```

The evaluation runner reports supporting-document recall at the configured final
top-k and latency, plus retrieved section/page metadata for inspection. It excludes
abstention/adversarial cases because these need an answer-generation evaluator.
Document recall does not establish section relevance or answer correctness, and
the runner does not estimate API cost. Its question/answer files are never indexed.

Evaluate generated answers (calls OpenAI and, in remote mode, Pinecone):

```bash
uv run python evals/evaluate_answers.py --mode hybrid
uv run python evals/evaluate_answers.py --mode hybrid --no-rerank --output data/evaluations/hybrid-no-rerank.jsonl
uv run python evals/evaluate_answers.py --local-only --output data/evaluations/bm25.jsonl
```

Each case starts a fresh conversation. The runner saves answers, verified source
quotes, the full retrieved context, reference answers, status matches, citation document recall, latency, and
chat token counts. Reference answers are never passed to the model. Inspect the
saved claims against the references and cited sections for correctness, exceptions,
and multi-document reasoning. Metrics alone do not prove grounding or security.
Token counts exclude embedding/reranking usage and are not total dollar costs.

## Remaining production work

Authenticate users, derive tenant/document scope from trusted identity, add durable
conversation and document lifecycle storage, evaluate grounding and refusal under
adversarial inputs, and configure operational monitoring and deployment.
