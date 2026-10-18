"""Offline generation, graph memory, and Streamlit checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from enterprise_rag_chatbot.generation.answer import AnswerGenerator, DraftAnswer, prepare_evidence
from enterprise_rag_chatbot.schemas import Chunk, SearchResult
from enterprise_rag_chatbot.ui.streamlit_app import save_uploads
from enterprise_rag_chatbot.workflows.chat_graph import ChatSession
from enterprise_rag_chatbot.workflows.nodes import RewrittenQuestion


def evidence():
    return SearchResult(Chunk("chunk-1", "Up to five days may be carried forward. Protected exceptions apply.", {
        "document_id": "AQ-HR-001", "version": "2.1", "page": 3,
        "section_number": "3", "source": "policy.pdf",
    }), 0.9)


def draft(**overrides):
    value = {"status": "answered", "claims": [{
        "text": "The synthetic policy allows up to five ordinary days to be carried forward.",
        "supports": [{"evidence_id": "S1.E1"}],
    }]}
    value.update(overrides)
    return DraftAnswer.model_validate(value)


class FakeModel:
    def __init__(self, answer=None):
        self.answer = draft() if answer is None else answer
        self.inputs = []

    def with_structured_output(self, schema, **kwargs):
        def invoke(messages):
            payload = json.loads(messages[-1][1])
            self.inputs.append((schema, payload))
            parsed = RewrittenQuestion(question="What about protected carry-forward after sickness?") if schema is RewrittenQuestion else self.answer
            return {"parsed": parsed, "raw": SimpleNamespace(usage_metadata={"input_tokens": 20, "output_tokens": 10}), "parsing_error": None}
        return SimpleNamespace(invoke=invoke)


class GenerationTests(unittest.TestCase):
    def test_valid_citations_are_built_from_metadata(self):
        answer = AnswerGenerator(FakeModel()).generate("carry-forward?", [evidence()])
        self.assertEqual(answer.status, "answered")
        self.assertIn("[AQ-HR-001 v2.1, page 3, section 3]", answer.text)
        self.assertEqual(answer.citations[0].source, "policy.pdf")
        self.assertEqual(answer.usage["input_tokens"], 20)

    def test_invented_evidence_ids_or_labels_abstain(self):
        for support in ({"evidence_id": "S99.E1"}, {"evidence_id": "S1.E99"}, {"evidence_id": ""}):
            with self.subTest(support=support):
                bad = draft(claims=[{"text": "A claim", "supports": [support]}])
                answer = AnswerGenerator(FakeModel(bad)).generate("question", [evidence()])
                self.assertEqual(answer.status, "insufficient")
                self.assertEqual(answer.reason, "invalid_citation")
                self.assertEqual(answer.citations, [])
        bad = draft(claims=[{"text": "A claim [fake citation]", "supports": draft().claims[0].supports}])
        self.assertEqual(AnswerGenerator(FakeModel(bad)).generate("question", [evidence()]).status, "insufficient")

    def test_quotes_are_selected_from_source_and_never_generated(self):
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            draft(claims=[{"text": "Claim", "supports": [{"evidence_id": "S1.E1", "quote": "Invented wording"}]}])
        source = evidence()
        source = SearchResult(Chunk(source.chunk.id, "Policy heading\nFirst rule.\nA second rule without punctuation", source.chunk.metadata), 1)
        payload, references = prepare_evidence([source])
        self.assertEqual(set(references), {"S1.E1", "S1.E2"})
        selected = draft(claims=[{"text": "A second rule", "supports": [{"evidence_id": "S1.E2"}]}])
        answer = AnswerGenerator(FakeModel(selected)).generate("question", [source])
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.citations[0].quote, "A second rule without punctuation")
        self.assertIn(answer.citations[0].quote, source.chunk.text)

    def test_validation_failure_message_does_not_claim_evidence_is_missing(self):
        bad = draft(claims=[{"text": "Claim", "supports": [{"evidence_id": "unknown"}]}])
        answer = AnswerGenerator(FakeModel(bad)).generate("question", [evidence()])
        self.assertIn("couldn't verify", answer.text)
        self.assertNotIn("enough evidence", answer.text)
        self.assertTrue(answer.diagnostics)

    def test_no_evidence_skips_model(self):
        model = FakeModel()
        answer = AnswerGenerator(model).generate("question", [])
        self.assertEqual(answer.status, "insufficient")
        self.assertEqual(model.inputs, [])

    def test_refusal_and_insufficient_discard_claims(self):
        for status in ("refused", "insufficient"):
            answer = AnswerGenerator(FakeModel(draft(status=status))).generate("question", [evidence()])
            self.assertEqual(answer.status, status)
            self.assertEqual(answer.citations, [])

    def test_parse_failure_is_not_reported_as_success(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = {
            "parsed": None, "parsing_error": ValueError("invalid"), "raw": None,
        }
        with self.assertRaisesRegex(ValueError, "invalid structured"):
            AnswerGenerator(model).generate("question", [evidence()])

    def test_provider_refusal_handles_parser_failure(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = {
            "parsed": None, "parsing_error": ValueError("refusal"),
            "raw": SimpleNamespace(additional_kwargs={"refusal": "denied"}, usage_metadata={}),
        }
        self.assertEqual(AnswerGenerator(model).generate("question", [evidence()]).status, "refused")


class ConversationTests(unittest.TestCase):
    def test_followups_use_history_and_fixed_access_scope(self):
        retriever = Mock()
        retriever.search.return_value = [evidence()]
        model = FakeModel()
        session = ChatSession(retriever, model, allowed_document_ids={"AQ-HR-001"}, history_turns=1)
        self.assertEqual(session.ask("What can I carry forward?").status, "answered")
        self.assertEqual(session.ask("What about sickness?").status, "answered")
        rewrite_inputs = [payload for schema, payload in model.inputs if schema is RewrittenQuestion]
        self.assertEqual(len(rewrite_inputs), 1)
        self.assertEqual(rewrite_inputs[0]["history"][0]["content"], "What can I carry forward?")
        self.assertEqual(retriever.search.call_args.args[0], "What about protected carry-forward after sickness?")
        self.assertEqual(retriever.search.call_args.kwargs["allowed_document_ids"], {"AQ-HR-001"})
        generation_inputs = [payload for schema, payload in model.inputs if schema is DraftAnswer]
        self.assertNotIn("history", generation_inputs[-1])
        session.reset()
        session.ask("A new standalone question")
        self.assertEqual(len([1 for schema, _ in model.inputs if schema is RewrittenQuestion]), 1)

    def test_empty_retrieval_routes_to_abstention(self):
        retriever = Mock()
        retriever.search.return_value = []
        model = FakeModel()
        session = ChatSession(retriever, model)
        self.assertEqual(session.ask("Unanswerable question").status, "insufficient")
        self.assertEqual(model.inputs, [])

    def test_separate_sessions_do_not_share_history(self):
        retriever = Mock()
        retriever.search.return_value = [evidence()]
        model = FakeModel()
        a, b = ChatSession(retriever, model), ChatSession(retriever, model)
        a.ask("First user's question")
        b.ask("Second user's question")
        self.assertNotEqual(a.thread_id, b.thread_id)
        self.assertFalse(any(schema is RewrittenQuestion for schema, _ in model.inputs))

    def test_service_errors_propagate_and_do_not_return_stale_answer(self):
        retriever = Mock()
        retriever.search.side_effect = [[evidence()], RuntimeError("service offline"), []]
        session = ChatSession(retriever, FakeModel())
        session.ask("First question")
        with self.assertRaisesRegex(RuntimeError, "service offline"):
            session.ask("Second question")
        self.assertEqual(session.ask("Third question").status, "insufficient")


class UITests(unittest.TestCase):
    def test_upload_filenames_cannot_escape_upload_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            upload = SimpleNamespace(name="../../private.pdf", getvalue=lambda: b"%PDF-1.4\nexample")
            save_uploads([upload], Path(temporary))
            files = list(Path(temporary).glob("*.pdf"))
            self.assertEqual(len(files), 1)
            self.assertEqual(len(files[0].stem), 64)
            with self.assertRaisesRegex(ValueError, "valid PDFs"):
                save_uploads([SimpleNamespace(getvalue=lambda: b"not a pdf")], Path(temporary))

    def test_streamlit_startup_and_chat_with_mocked_services(self):
        from streamlit.testing.v1 import AppTest
        from enterprise_rag_chatbot.generation.answer import abstain

        snapshot = SimpleNamespace(corpus_id="test-corpus")
        session = Mock()
        session.ask.return_value = abstain()
        with patch("enterprise_rag_chatbot.ui.streamlit_app.Snapshot.load", return_value=snapshot), \
             patch("enterprise_rag_chatbot.ui.streamlit_app.ChatSession.from_settings", return_value=session):
            script = Path(__file__).resolve().parents[1] / "src/enterprise_rag_chatbot/ui/streamlit_app.py"
            app = AppTest.from_file(str(script)).run()
            self.assertEqual(len(app.exception), 0)
            app.chat_input[0].set_value("What can I carry forward?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("enough evidence" in item.value for item in app.markdown))
            session.ask.assert_called_once_with("What can I carry forward?")


class EvaluationTests(unittest.TestCase):
    def test_reference_answers_are_not_passed_to_generation(self):
        from evals.evaluate_answers import evaluate_case

        retriever = Mock()
        retriever.search.return_value = [evidence()]
        model = FakeModel()
        session = ChatSession(retriever, model)
        result = evaluate_case(session, {
            "test_id": "example", "question": "What can I carry forward?",
            "expected_answer": "ANSWER KEY MUST NOT ENTER THE PROMPT",
            "test_type": "answer", "document_ids": ["AQ-HR-001"],
        })
        self.assertTrue(result["status_match"])
        self.assertEqual(result["citation_document_recall"], 1.0)
        self.assertNotIn("ANSWER KEY", json.dumps([payload for _, payload in model.inputs]))

    def test_abstention_cases_are_scored_without_fake_document_ids(self):
        from evals.evaluate_answers import evaluate_case

        retriever = Mock()
        retriever.search.return_value = []
        result = evaluate_case(ChatSession(retriever, FakeModel()), {
            "test_id": "abstain", "question": "What is the fund price?",
            "expected_answer": "Not in corpus", "test_type": "abstain", "document_ids": ["none"],
        })
        self.assertTrue(result["status_match"])
        self.assertIsNone(result["citation_document_recall"])


if __name__ == "__main__":
    unittest.main()
