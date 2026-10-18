"""Offline checks of corpus boundaries, ranking, and ingestion failure behavior."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llama_index.core import Document
from llama_index.core.utils import get_tokenizer

from enterprise_rag_chatbot.config import Settings
from enterprise_rag_chatbot.ingestion.document_registry import Snapshot, snapshot_path
from enterprise_rag_chatbot.ingestion.loaders import load_documents
from enterprise_rag_chatbot.ingestion.pipeline import build_snapshot, ingest
from enterprise_rag_chatbot.retrieval.bm25 import BM25
from enterprise_rag_chatbot.retrieval.retriever import HybridRetriever, reciprocal_rank_fusion
from enterprise_rag_chatbot.retrieval.pinecone_store import PineconeStore
from enterprise_rag_chatbot.schemas import Chunk, SearchResult


def page(text, document_id="AQ-HR-001"):
    return Document(text=text, metadata={"document_id": document_id, "page": 2, "version": "1"})


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(embedding_dimensions=3)
        self.snapshot = build_snapshot([
            page("1. Annual leave\nEmployees receive annual leave and holiday carry-forward."),
            page("1. Security\nReport an incident with error code ERR-401.", "AQ-CMP-002"),
        ], self.settings)

    def test_identifier_search_and_no_match(self):
        index = BM25(self.snapshot.chunks)
        self.assertEqual(index.search("ERR-401", 5)[0].chunk.metadata["document_id"], "AQ-CMP-002")
        self.assertEqual(index.search("AQ-HR-001", 5)[0].chunk.metadata["document_id"], "AQ-HR-001")
        self.assertEqual(index.search("xyzunseenword", 5), [])

    def test_rrf_rewards_agreement_without_duplicate_votes(self):
        a, b = [SearchResult(chunk, 1) for chunk in self.snapshot.chunks]
        results = reciprocal_rank_fusion([[a, b], [b, b]])
        self.assertEqual(results[0].chunk.id, b.chunk.id)
        self.assertAlmostEqual(results[0].score, 1 / 62 + 1 / 61)

    def test_permissions_filter_both_branches_before_reranking(self):
        store = Mock()
        store.query.return_value = [SearchResult(c, 1) for c in self.snapshot.chunks]
        store.rerank.side_effect = lambda query, candidates, top_n: candidates[:top_n]
        embedding = Mock()
        embedding.get_query_embedding.return_value = [0, 1, 0]
        retriever = HybridRetriever(self.settings, self.snapshot, store=store, embedding=embedding)
        results = retriever.search("annual leave security", allowed_document_ids={"AQ-HR-001"})
        self.assertTrue(results)
        self.assertTrue(all(r.chunk.metadata["document_id"] == "AQ-HR-001" for r in results))
        self.assertEqual(store.query.call_args.args[2]["document_id"], {"$in": ["AQ-HR-001"]})
        store.reset_mock()
        self.assertEqual(retriever.search("leave", allowed_document_ids=set()), [])
        store.query.assert_not_called()

    def test_stale_dense_ids_are_discarded(self):
        store = Mock()
        store.query.return_value = [SearchResult(Chunk("stale", "private", {}), 1)]
        embedding = Mock()
        embedding.get_query_embedding.return_value = [0, 0, 1]
        retriever = HybridRetriever(self.settings, self.snapshot, store=store, embedding=embedding)
        self.assertEqual(retriever.search("leave", mode="dense", rerank=False), [])

    def test_model_configuration_change_requires_reingestion(self):
        with self.assertRaisesRegex(ValueError, "re-ingest"):
            HybridRetriever(replace(self.settings, embedding_model="text-embedding-3-large"), self.snapshot)

    def test_deterministic_ids_and_token_cap(self):
        document = page("2. Carry-forward\n" + "A policy rule with exceptions. " * 700)
        a = build_snapshot([document], self.settings)
        b = build_snapshot([document], self.settings)
        self.assertEqual(a, b)
        self.assertGreater(len(a.chunks), 1)
        tokenizer = get_tokenizer()
        for chunk in a.chunks:
            self.assertLessEqual(len(tokenizer(chunk.text)), self.settings.chunk_size)
            self.assertTrue(chunk.text.startswith("2. Carry-forward\n"))
            self.assertEqual(chunk.metadata["page"], 2)
        changed = build_snapshot([document], replace(self.settings, chunk_size=400))
        self.assertNotEqual(a.corpus_id, changed.corpus_id)

    def test_sample_pack_pages_metadata_and_evidence_only(self):
        documents = load_documents("data/sample/alder_quay")
        self.assertEqual(len(documents), 36)
        self.assertEqual(len({d.metadata["document_id"] for d in documents}), 9)
        self.assertTrue(all(d.metadata["file_name"].endswith(".pdf") for d in documents))
        snapshot = build_snapshot(documents, Settings())
        self.assertTrue(any(c.metadata["section_number"] == "3" for c in snapshot.chunks))
        self.assertTrue(all("source_notice" in c.metadata for c in snapshot.chunks))
        self.assertTrue(all(len(get_tokenizer()(c.text)) <= 512 for c in snapshot.chunks))
        self.assertTrue(all(c.metadata["section_number"] == "" for c in snapshot.chunks if c.metadata["page"] == 1))
        results = BM25(snapshot.chunks).search("What are the carry-forward rules?", 6, {"AQ-HR-001"})
        self.assertEqual(results[0].chunk.metadata["page"], 3)
        self.assertEqual(results[0].chunk.metadata["section_number"], "3")

    def test_namespaces_use_distinct_snapshots_and_generation_ids(self):
        other = replace(self.settings, namespace="another-tenant")
        self.assertNotEqual(snapshot_path(self.settings), snapshot_path(other))
        self.assertNotEqual(snapshot_path(self.settings), snapshot_path(self.settings, local_only=True))
        a = build_snapshot([page("Same text")], self.settings)
        b = build_snapshot([page("Same text")], other)
        self.assertNotEqual(a.corpus_id, b.corpus_id)

    def test_failed_upload_preserves_active_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = replace(self.settings, cache_dir=Path(temporary), openai_api_key="test", pinecone_api_key="test")
            old = build_snapshot([page("Old approved text")], settings)
            path = snapshot_path(settings)
            old.save(path)
            store = Mock()
            store.upsert.side_effect = RuntimeError("upload failed")
            with patch("enterprise_rag_chatbot.ingestion.pipeline.load_documents", return_value=[page("New text")]), \
                 patch("enterprise_rag_chatbot.retrieval.pinecone_store.PineconeStore", return_value=store), \
                 patch.object(Settings, "embedding", return_value=Mock()):
                with self.assertRaisesRegex(RuntimeError, "upload failed"):
                    ingest("unused", settings)
            self.assertEqual(Snapshot.load(path), old)
            store.delete.assert_not_called()

    def test_successful_update_cleans_only_previous_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = replace(self.settings, cache_dir=Path(temporary), openai_api_key="test", pinecone_api_key="test")
            old = build_snapshot([page("Old text")], settings)
            old.save(snapshot_path(settings))
            store = Mock()
            with patch("enterprise_rag_chatbot.ingestion.pipeline.load_documents", return_value=[page("New text")]), \
                 patch("enterprise_rag_chatbot.retrieval.pinecone_store.PineconeStore", return_value=store), \
                 patch.object(Settings, "embedding", return_value=Mock()):
                new = ingest("unused", settings)
            self.assertEqual(Snapshot.load(snapshot_path(settings)), new)
            store.delete.assert_called_once_with([c.id for c in old.chunks])

    def test_pinecone_rejects_incompatible_index(self):
        settings = replace(self.settings, openai_api_key="test", pinecone_api_key="test")
        client = Mock()
        client.has_index.return_value = True
        client.describe_index.return_value = SimpleNamespace(dimension=999, metric="cosine")
        with patch("enterprise_rag_chatbot.retrieval.pinecone_store.Pinecone", return_value=client):
            with self.assertRaisesRegex(ValueError, "dimensions"):
                PineconeStore(settings)
        client.Index.assert_not_called()

    def test_pinecone_adapter_batches_text_and_maps_rerank_indices(self):
        settings = replace(self.settings, openai_api_key="test", pinecone_api_key="test")
        client = Mock()
        client.has_index.return_value = True
        client.describe_index.return_value = SimpleNamespace(dimension=3, metric="cosine", host="test-host")
        client.inference.rerank.return_value = SimpleNamespace(data=[SimpleNamespace(index=1, score=0.9)])
        with patch("enterprise_rag_chatbot.retrieval.pinecone_store.Pinecone", return_value=client):
            store = PineconeStore(settings)
        embedding = Mock()
        embedding.get_text_embedding_batch.return_value = [[1, 0, 0], [0, 1, 0]]
        store.upsert(self.snapshot.chunks, embedding)
        payload = store.index.upsert.call_args.kwargs
        self.assertEqual(payload["namespace"], settings.namespace)
        self.assertEqual(payload["vectors"][0]["metadata"]["text"], self.snapshot.chunks[0].text)
        candidates = [SearchResult(c, 1) for c in self.snapshot.chunks]
        results = store.rerank("question", candidates, 1)
        self.assertEqual(results[0].chunk.id, candidates[1].chunk.id)
        self.assertEqual(results[0].score, 0.9)
        self.assertEqual(client.inference.rerank.call_args.kwargs["model"], "bge-reranker-v2-m3")

    def test_invalid_embedding_output_is_never_uploaded(self):
        store = PineconeStore.__new__(PineconeStore)
        store.settings = self.settings
        store.index = Mock()
        embedding = Mock()
        embedding.get_text_embedding_batch.return_value = [[1, 0]]
        with self.assertRaisesRegex(ValueError, "Embedding output"):
            store.upsert(self.snapshot.chunks, embedding)
        store.index.upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
