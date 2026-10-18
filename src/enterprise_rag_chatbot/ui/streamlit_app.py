"""Session-scoped chat, source inspection, corpus uploads, and local feedback."""

import json
from pathlib import Path
from uuid import uuid4

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.generation.answer import Answer
from enterprise_rag_chatbot.ingestion.document_registry import Snapshot, snapshot_path
from enterprise_rag_chatbot.workflows.chat_graph import ChatSession


def render_answer(st, answer: Answer, key: str) -> None:
    st.markdown(answer.text)
    if answer.citations:
        with st.expander("Sources and supporting quotes"):
            for citation in answer.citations:
                st.text(citation.label)
                st.caption(citation.source)
                st.text(citation.quote)
    if answer.usage:
        with st.expander("Chat token usage"):
            st.json(answer.usage)
    rating = st.feedback("thumbs", key=f"feedback-{key}")
    if rating is not None and key not in st.session_state.setdefault("feedback_saved", set()):
        directory = Path("data/feedback")
        directory.mkdir(parents=True, exist_ok=True)
        # One event per file avoids interleaved writes from independent sessions.
        (directory / f"{uuid4().hex}.json").write_text(json.dumps({
            "message_id": key, "rating": rating, "status": answer.status,
            "chunk_ids": [c.chunk_id for c in answer.citations],
        }))
        st.session_state.feedback_saved.add(key)


def save_uploads(uploads, directory: Path) -> int:
    """Store PDFs using generated names; never trust an uploaded filename as a path."""
    import hashlib

    directory.mkdir(parents=True, exist_ok=True)
    for upload in uploads:
        content = upload.getvalue()
        if not content.startswith(b"%PDF-") or len(content) > 10 * 1024 * 1024:
            raise ValueError("Upload valid PDFs no larger than 10 MB each")
        name = hashlib.sha256(content).hexdigest() + ".pdf"
        (directory / name).write_bytes(content)
    return len(uploads)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Enterprise RAG Chatbot", page_icon="📚")
    st.title("Enterprise RAG Chatbot")
    st.caption("Answers grounded in your indexed documents, with inspectable citations.")
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Retrieval")
        local_only = st.checkbox("Use local BM25 corpus", help="Chat still uses OpenAI")
        mode = st.selectbox("Search mode", ["hybrid", "dense", "bm25"], disabled=local_only)
        rerank = st.checkbox("Rerank results", value=True, disabled=local_only)
        st.caption(f"Chat model: {settings.chat_model}")
        reset = st.button("Reset conversation")
        st.divider()
        st.header("Document ingestion")
        uploads = st.file_uploader("Add PDFs (10 MB maximum each)", type=["pdf"], accept_multiple_files=True)
        st.caption("Rebuild indexes all PDFs under data/, including samples and saved uploads. Remote mode uploads their text to OpenAI and Pinecone.")
        if st.button("Save uploads and rebuild corpus"):
            from enterprise_rag_chatbot.ingestion.pipeline import ingest

            try:
                with st.spinner("Loading, chunking, and indexing documents…"):
                    if uploads:
                        save_uploads(uploads, Path("data/uploads"))
                    snapshot = ingest("data", settings, local_only=local_only)
                st.success(f"Indexed {len(snapshot.chunks)} chunks")
                st.session_state.pop("chat_session", None)
                st.session_state.messages = []
            except Exception:
                st.error("Ingestion failed. Check PDF contents, configuration, credentials, and service availability. The previous active corpus is preserved if upload failed.")

    try:
        snapshot = Snapshot.load(snapshot_path(settings, local_only))
        scope = (snapshot.corpus_id, settings.chat_model, settings.chat_max_tokens,
                 settings.history_turns, local_only, mode, rerank)
        if reset or st.session_state.get("chat_scope") != scope:
            old = st.session_state.pop("chat_session", None)
            if old:
                old.reset()
            st.session_state.messages = []
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = ChatSession.from_settings(
                settings, local_only=local_only, mode=mode, rerank=rerank,
            )
        st.session_state.chat_scope = scope
    except (ValueError, FileNotFoundError) as exc:
        st.info(str(exc))
        st.stop()

    st.session_state.setdefault("messages", [])
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["text"])
            else:
                render_answer(st, Answer.model_validate(message["answer"]), message["id"])
    question = st.chat_input("Ask a question about your documents", max_chars=8000)
    if question:
        with st.chat_message("user"):
            st.text(question)
        try:
            with st.spinner("Retrieving evidence and preparing an answer…"):
                answer = st.session_state.chat_session.ask(question)
        except Exception:
            st.error("The request failed. Check your model access, credentials, and service availability, then retry. Reset the conversation if the problem persists.")
        else:
            st.session_state.messages.extend([
                {"role": "user", "text": question},
                {"role": "assistant", "answer": answer.model_dump(), "id": uuid4().hex},
            ])
            st.rerun()


if __name__ == "__main__":
    main()
