"""Streamlit integration tests with mocked storage, embeddings and generation."""

import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

from utils import ai, db, rag, grounding


class StreamlitRagTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        self.rows = [
            dict(id=1, date=pd.Timestamp(date.today() - timedelta(days=1)),
                 weight=70.0, protein=0.0, sleep_hours=0.0, workout=0),
            dict(id=2, date=pd.Timestamp(date.today()),
                 weight=70.0, protein=100.0, sleep_hours=6.0, workout=1),
        ]
        self.documents = rag.prep_documents(rag.load_passages())
        self.sleep_doc = next(doc for doc in self.documents if doc.metadata["id"] == "sleep_02")
        self.store = Mock()
        self.store.similarity_search_with_score.return_value = [(self.sleep_doc, 0.28)]

        self.patch(db, "init_db")
        self.patch(db, "load_data", side_effect=lambda: pd.DataFrame(
            self.rows, columns=["id", "date", "weight", "protein", "sleep_hours", "workout"],
        ))
        self.patch(db, "export_csv", return_value=b"date,weight\n")
        self.embedding = self.patch(rag, "get_embeddings_function", return_value=Mock())
        self.build = self.patch(rag, "build_vector_store", return_value=self.store)
        self.generate = self.patch(ai, "draft_and_review", return_value={
            "answer": "Keep consistent sleep times. [sleep_02]",
            "claims": [{"source_id": "sleep_02"}], "unknown_ids": [],
            "mechanical_issues": [], "review_issues": [], "support_status": "model_reviewed",
        })
        self.resolver = self.patch(grounding, "resolve_question", side_effect=lambda question, history, **kw: {
            "query": question if not history else question + " (resolved from session)", "clarification": "",
        })
        self.checked = self.patch(ai, "generate_checked_result", wraps=ai.generate_checked_result)
        self.patch(ai, "nutrirecall_ai", side_effect=AssertionError("Legacy pipeline called"))
        self.patch(ai, "store_memory", side_effect=AssertionError("Legacy memory write"))
        self.patch(ai, "retrieve_memory", side_effect=AssertionError("Legacy memory read"))
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))

    def patch(self, module, name, **kwargs):
        patcher = patch.object(module, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def open_ai(self, key=True):
        self.app.run(timeout=30)
        self.app.radio[0].set_value("✨ AI Assistant").run(timeout=30)
        if key:
            self.app.text_input[0].set_value("test-key-not-real").run(timeout=30)
        self.assertEqual(len(self.app.exception), 0)

    def submit(self, question):
        self.app.chat_input[0].set_value(question).run(timeout=30)
        self.assertEqual(len(self.app.exception), 0)

    def button(self, label):
        return next(b for b in self.app.button if b.label == label)

    def test_page_is_lazy_has_one_chat_and_no_technical_modes(self):
        self.open_ai()
        self.embedding.assert_not_called()
        self.generate.assert_not_called()
        self.assertEqual(len(self.app.selectbox), 0)
        self.assertEqual(len(self.app.checkbox), 0)
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertTrue(any("About answer quality" == e.label for e in self.app.expander))

    def test_default_chat_includes_fresh_logs_and_expandable_evidence(self):
        self.open_ai()
        self.submit("What sleep habits help?")
        summary = self.checked.call_args.kwargs["log_summary"]
        self.assertEqual(summary["avg_protein_g"], 50.0)
        self.assertEqual(summary["avg_sleep_hours"], 3.0)
        self.assertEqual(summary["logged_days"], 2)
        self.assertTrue(any(e.label == "Exact evidence" for e in self.app.expander))
        self.assertTrue(any(e.label == "Sources and review notes" for e in self.app.expander))
        self.assertNotIn("test-key-not-real", str(self.app.session_state["rag_chat_history"]))

    def test_read_only_summary_works_without_key_retrieval_or_model(self):
        self.open_ai(key=False)
        self.generate.side_effect = grounding.draft_and_review
        with patch.object(grounding, "call_json", side_effect=AssertionError("No API call for record arithmetic")):
            self.submit("Summarise my logged protein and sleep. Which dates are you using, and how complete is my week?")
        result = self.app.session_state["rag_chat_history"][1]["result"]
        self.assertIn("50 g of protein", result["answer"])
        self.assertIn("3 hours of sleep", result["answer"])
        self.assertIn("Coverage: 2 of 7 days", result["answer"])
        self.assertEqual(result["source_footer"], "")
        self.assertEqual(result["grounding"]["support_review"], "deterministic_log_summary")
        self.assertTrue(any(e.label == "Logs used for this answer" for e in self.app.expander))
        self.assertFalse(result["log_usage"]["fresh_summary_sent_to_model"])
        self.embedding.assert_not_called()
        self.resolver.assert_not_called()

    def test_read_only_summary_respects_log_sharing_optout(self):
        self.open_ai(key=False)
        self.generate.side_effect = grounding.draft_and_review
        next(t for t in self.app.toggle if t.label == "Include my recent logs").set_value(False).run()
        with patch.object(grounding, "call_json", side_effect=AssertionError("No API call needed")):
            self.submit("Show my average protein and sleep this week")
        result = self.app.session_state["rag_chat_history"][1]["result"]
        self.assertIn("No personal log summary was supplied", result["answer"])
        self.assertNotIn("50 g", result["answer"])
        self.embedding.assert_not_called()

    def test_switch_from_foods_to_records_and_refresh_window_without_model(self):
        self.open_ai(key=False)
        self.generate.side_effect = grounding.draft_and_review
        self.app.session_state["rag_chat_history"] = [
            {"role": "user", "content": "What foods provide protein without powder?"},
            {"role": "assistant", "result": {"answer": "Beans and nuts. [protein_03]", "response_mode": "normal"}},
        ]
        with patch.object(grounding, "call_json", side_effect=AssertionError("No API call for explicit records")):
            self.submit("Summarise my protein and sleep now.")
            first = self.app.session_state["rag_chat_history"][-1]["result"]
            self.assertIn("50 g of protein", first["answer"])
            self.assertIn("3 hours of sleep", first["answer"])
            next(d for d in self.app.date_input if d.label == "Summary ending on").set_value(date.today() - timedelta(days=1)).run()
            self.submit("And now, using the selected reporting dates?")
        fresh = self.app.session_state["rag_chat_history"][-1]["result"]
        self.assertIn("0 g of protein", fresh["answer"])
        self.assertIn("0 hours of sleep", fresh["answer"])
        self.assertIn("Coverage: 1 of 7 days", fresh["answer"])
        self.assertIn((date.today() - timedelta(days=1)).isoformat(), fresh["log_header"])
        self.assertNotIn("50 g", fresh["answer"])
        self.assertEqual(fresh["grounding"]["support_review"], "deterministic_log_summary")
        self.embedding.assert_not_called()
        self.resolver.assert_not_called()

    def test_followup_uses_bounded_session_context_and_reretrieves(self):
        self.open_ai()
        self.submit("Protein foods?")
        self.submit("What about plant options?")
        history = self.checked.call_args.kwargs["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["content"], "Protein foods?")
        self.assertIn("resolved from session", self.store.similarity_search_with_score.call_args.args[0])
        self.assertEqual(self.generate.call_count, 2)
        self.embedding.assert_called_once()
        self.build.assert_called_once()

    def test_changed_logs_override_old_answer_snapshot(self):
        self.open_ai()
        self.submit("Protein summary?")
        self.rows[1]["protein"] = 140.0
        self.submit("And now?")
        self.assertEqual(self.checked.call_args.kwargs["log_summary"]["avg_protein_g"], 70.0)
        old = self.app.session_state["rag_chat_history"][1]["result"]["log_header"]
        self.assertIn("2/7", old)

    def test_privacy_optout_clears_old_context_and_excludes_summary(self):
        self.open_ai()
        self.submit("My logs?")
        next(t for t in self.app.toggle if t.label == "Include my recent logs").set_value(False).run()
        self.assertEqual(self.app.session_state["rag_chat_history"], [])
        self.submit("Sleep habits?")
        self.assertIsNone(self.checked.call_args.kwargs["log_summary"])
        self.assertEqual(self.checked.call_args.kwargs["history"], [])

    def test_router_has_no_used_logs_card_when_no_fresh_summary_was_sent(self):
        self.open_ai()
        self.generate.side_effect = grounding.draft_and_review
        response = {"intent": "unsupported", "claims": [], "log_fields": [], "follow_up": "", "issues": []}
        with patch.object(grounding, "call_json", return_value=response):
            self.submit("How do I configure a wireless router?")
        result = self.app.session_state["rag_chat_history"][-1]["result"]
        self.assertEqual(result["log_header"], "")
        self.assertFalse(any(e.label == "Logs used for this answer" for e in self.app.expander))
        self.assertTrue(any("Available logs:" in item.value for item in self.app.markdown))

    def test_default_mixed_request_keeps_figures_even_when_model_omits_fields(self):
        self.open_ai()
        self.generate.side_effect = grounding.draft_and_review
        selection = {"intent": "unsupported", "claims": [], "log_fields": [], "follow_up": "", "issues": []}
        with patch.object(grounding, "call_json", return_value=selection) as call:
            self.submit("Can my logged protein explain recovery?")
        self.assertTrue(any(e.label == "Logs used for this answer" for e in self.app.expander))
        self.assertFalse(any(e.label == "Log context sent with this request" for e in self.app.expander))
        result = self.app.session_state["rag_chat_history"][-1]["result"]
        self.assertIn("50 g of protein", result["answer"])
        self.assertIn("Coverage: 2 of 7", result["answer"])
        self.assertEqual(result["request_flow"]["model_stages"], ["evidence_selection"])
        self.assertEqual(call.call_count, 1)

    def test_legacy_log_snapshot_does_not_claim_known_usage(self):
        self.open_ai(key=False)
        self.app.session_state["rag_chat_history"] = [{"role": "user", "content": "An old question"},
            {"role": "assistant", "result": {"answer": "An older reply", "log_header": "### Personal log coverage\n- Coverage: 2/7"}}]
        self.app.run()
        self.assertTrue(any(e.label == "Log coverage snapshot (usage not recorded)" for e in self.app.expander))
        self.assertFalse(any(e.label == "Logs used for this answer" for e in self.app.expander))

    def test_explicit_help_confirmation_preserves_warning_without_key(self):
        self.open_ai(key=False)
        help_section = next(e for e in self.app.sidebar.expander if e.label == "Help & safety")
        self.assertFalse(help_section.proto.expanded)
        self.assertTrue(any(b.label == "Sudden or unexplained weight loss" for b in help_section.button))
        self.assertFalse(any(b.label in ("Unexpected weight loss?", "Sudden or unexplained weight loss")
                             for b in self.app.main.button))
        self.assertEqual(self.app.session_state["rag_chat_history"], [])
        self.generate.assert_not_called()
        self.build.assert_not_called()
        self.button("Sudden or unexplained weight loss").click().run()
        self.assertTrue(any(b.label == "Show curated guidance" for b in self.app.button))
        self.assertEqual(self.app.session_state["rag_chat_history"], [])
        self.button("Show curated guidance").click().run()
        self.generate.assert_not_called()
        self.build.assert_not_called()
        result = self.app.session_state["rag_chat_history"][1]["result"]
        self.assertEqual(result["response_mode"], "confirmed_safety")
        self.assertEqual(result["safety_source_ids"], ["weight_04"])
        self.assertIn("[weight_04]", result["answer"])
        self.assertIn("overdue", result["source_footer"])
        self.assertFalse(self.app.session_state["rag_weight_help"])

    def test_cancel_help_does_not_force_normal_question_into_safety(self):
        self.open_ai()
        self.button("Sudden or unexplained weight loss").click().run()
        self.button("Cancel").click().run()
        self.submit("I have not lost weight unexpectedly. What protein foods are there?")
        self.assertFalse(self.checked.call_args.kwargs.get("confirmed_unexplained_weight_loss", False))

    def test_missing_key_does_not_initialize_or_generate(self):
        self.open_ai(key=False)
        self.submit("Sleep habits?")
        self.build.assert_not_called()
        self.generate.assert_not_called()
        self.assertTrue(any("Add your Groq API key" in item.value for item in self.app.error))

    def test_ambiguous_followup_asks_clarification_without_generation(self):
        self.open_ai()
        self.app.session_state["rag_chat_history"] = [
            {"role": "user", "content": "Protein foods?"},
            {"role": "assistant", "content": "An earlier answer"},
        ]
        self.resolver.side_effect = None
        self.resolver.return_value = {"query": "", "clarification": "Which topic do you mean?"}
        self.submit("What about that?")
        self.generate.assert_not_called()
        result = self.app.session_state["rag_chat_history"][-1]["result"]
        self.assertEqual(result["response_mode"], "clarification")

    def test_standalone_new_topic_skips_resolution_despite_history(self):
        self.open_ai()
        self.app.session_state["rag_chat_history"] = [
            {"role": "user", "content": "Protein foods?"},
            {"role": "assistant", "content": "An earlier answer"},
        ]
        self.generate.side_effect = grounding.draft_and_review
        selection = {"intent": "knowledge", "claims": [{"source_id": "sleep_02", "unit_index": 0, "item_indices": []}],
                     "log_fields": [], "follow_up": "", "issues": []}
        with patch.object(grounding, "call_json", return_value=selection) as call:
            self.submit("What habits could help me sleep better?")
        self.resolver.assert_not_called()
        self.assertEqual(call.call_count, 1)
        result = self.app.session_state["rag_chat_history"][-1]["result"]
        self.assertEqual(result["request_flow"]["reference_resolution"], "standalone")
        self.assertEqual(result["request_flow"]["model_stages"], ["evidence_selection"])

    def test_errors_do_not_expose_provider_details(self):
        class RateLimited(Exception):
            status_code = 429
        self.open_ai()
        self.checked.side_effect = RateLimited("private-provider-details test-key-not-real")
        self.submit("Sleep habits?")
        self.assertTrue(any("rate limit" in item.value for item in self.app.error))
        self.assertNotIn("private-provider-details", str(self.app.session_state["rag_chat_history"]))

    def test_groq_timeout_has_clear_message_without_exposing_provider_details(self):
        from openai import APITimeoutError
        import httpx

        self.open_ai()
        self.checked.side_effect = APITimeoutError(request=httpx.Request(
            "POST", "https://example.invalid/private-provider-details",
        ))
        self.submit("Sleep habits?")
        message = self.app.session_state["rag_chat_history"][-1]["content"]
        self.assertIn("timed out", message)
        self.assertIn("10-second timeout per API call", message)
        self.assertNotIn("private-provider-details", message)
        self.assertTrue(self.app.session_state["rag_chat_history"][-1]["error"])
        self.checked.assert_called_once()

    def test_validation_error_displays_safe_stage_and_diagnostic(self):
        self.open_ai()
        error = ValueError("Model did not return valid structured output.")
        error.nutrirecall_stage = "evidence_selection"
        self.checked.side_effect = error
        self.submit("Sleep habits?")
        message = self.app.session_state["rag_chat_history"][-1]["content"]
        self.assertIn("evidence selection", message)
        self.assertIn("invalid_json", message)
        self.assertNotIn("please retry", message.lower())

    def test_new_conversation_resets_context_without_changing_logs(self):
        self.open_ai()
        self.submit("Protein foods?")
        self.button("New conversation").click().run()
        self.assertEqual(self.app.session_state["rag_chat_history"], [])
        self.submit("Sleep habits?")
        self.assertEqual(self.checked.call_args.kwargs["history"], [])
        self.assertEqual(len(self.rows), 2)

    def test_local_model_tag_is_forwarded(self):
        self.open_ai(key=False)
        next(t for t in self.app.toggle if t.label == "Use local Ollama model").set_value(True).run()
        next(t for t in self.app.text_input if t.label == "Ollama model tag").set_value("llama3.1:8b").run()
        self.submit("Sleep habits?")
        self.assertTrue(self.checked.call_args.kwargs["use_local"])
        self.assertEqual(self.checked.call_args.kwargs["local_model"], "llama3.1:8b")

    def test_empty_logs_do_not_block_general_knowledge(self):
        self.rows = []
        self.open_ai()
        self.submit("Sleep habits?")
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(self.checked.call_args.kwargs["log_summary"]["logged_days"], 0)


if __name__ == "__main__":
    unittest.main()
