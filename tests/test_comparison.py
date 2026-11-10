"""Offline comparison mechanics, not scores for real model performance."""

import copy
import json
import io
from contextlib import redirect_stderr
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.compare_rag import compare_case, load_cases, main
from utils import ai, grounding as g, rag


def plan(*choices, intent="knowledge", fields=None):
    return {"intent": intent, "claims": list(choices), "log_fields": fields or [], "follow_up": "", "issues": []}


def choice(source_id, unit=0, items=None):
    return {"source_id": source_id, "unit_index": unit, "item_indices": items or []}


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = rag.prep_documents(rag.load_passages())
        cls.matches = [(doc, 0.3) for doc in cls.docs]
        cls.summary = ai.summarize_logs([
            dict(date="2026-10-01", protein=0, sleep_hours=0, workout=0),
            dict(date="2026-10-05", protein=100, sleep_hours=6, workout=1),
        ], "2026-10-05")

    def test_default_single_and_opt_in_review_make_one_and_two_calls(self):
        for kwargs, count in [({}, 1), ({"review_evidence": True}, 2)]:
            with self.subTest(kwargs=kwargs), patch.object(g, "call_json", return_value=plan(choice("protein_03")) ) as call:
                result = ai.generate_result("What foods provide protein?", self.matches, **kwargs)
            self.assertEqual(call.call_count, count)
            self.assertIn("not automatically necessary", result["answer"])
            self.assertIn("source_units_v1", result["grounding"]["selection_contract"])
            self.assertEqual(result["grounding"]["support_review"],
                             "extractive_selection_unreviewed" if count == 1 else "extractive_selection_reviewed")

    def test_runner_requires_live_flag_before_loading_secrets_or_models(self):
        with patch("sys.argv", ["compare_rag.py", "--case", "rag_02", "--output", "unused.json"]), \
                patch("dotenv.load_dotenv") as secrets, patch.object(rag, "get_embeddings_function") as embeddings, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main()
        self.assertEqual(caught.exception.code, 2)
        secrets.assert_not_called()
        embeddings.assert_not_called()

    def test_runner_refuses_to_overwrite_previous_results(self):
        with patch("sys.argv", ["compare_rag.py", "--live", "--case", "rag_02", "--output", "existing.json"]), \
                patch("scripts.compare_rag.Path.exists", return_value=True), \
                patch("dotenv.load_dotenv") as secrets, redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as caught:
            main()
        self.assertEqual(caught.exception.code, 2)
        secrets.assert_not_called()

    def test_direct_logs_and_explicit_warning_remain_zero_calls_in_both_modes(self):
        for review in (False, True):
            with self.subTest(review=review), patch.object(g, "call_json", side_effect=AssertionError("No model needed")):
                summary = ai.generate_result("Summarise my protein and sleep now.", [], self.summary, review_evidence=review)
                warning = ai.generate_checked_result("Read warning", [], True, documents=self.docs, review_evidence=review)
            self.assertIn("50 g of protein", summary["answer"])
            self.assertIn("3 hours of sleep", summary["answer"])
            self.assertIn("2 of 7", summary["answer"])
            self.assertIn("[weight_04]", warning["answer"])

    def test_single_selection_has_no_fresh_values_but_code_renders_current_summary(self):
        before = copy.deepcopy(self.summary)
        with patch.object(g, "call_json", return_value=plan(intent="mixed", fields=["avg_protein_g"])) as call:
            result = ai.generate_result("How does my logged protein relate to recovery?", self.matches,
                                        self.summary, review_evidence=False)
        self.assertNotIn("personal_log_summary", call.call_args.args[1])
        self.assertIn("50 g of protein", result["answer"])
        self.assertFalse(result["log_usage"]["fresh_summary_sent_to_model"])
        self.assertEqual(self.summary, before)

    def test_medical_boundary_and_invalid_indices_in_both_modes(self):
        for review in (False, True):
            with self.subTest(review=review), patch.object(g, "call_json", return_value=plan(choice("protein_02"), intent="medical_boundary")):
                result = ai.generate_result("Prescribe protein for my kidney disease at 70 kg.", self.matches, review_evidence=review)
            self.assertIn("can't prescribe", result["answer"])
            self.assertNotIn("84–140", result["answer"])
            with patch.object(g, "call_json", return_value=plan(choice("protein_03", 999))):
                result = ai.generate_result("Protein foods?", self.matches, review_evidence=review)
            self.assertEqual(result["cited_ids"], [])
            self.assertEqual(result["grounding"]["support_review"], "invalid_selection")

    def test_single_food_subset_retains_qualifications_and_review_warning(self):
        with patch.object(g, "call_json", return_value=plan(choice("protein_03", 0, [5, 6]), choice("weight_02", 2, [0, 1]))):
            result = ai.generate_result("Plant options?", self.matches, review_evidence=False)
        self.assertIn("beans and nuts", result["answer"])
        self.assertIn("beans and pulses", result["answer"])
        self.assertNotIn("eggs", result["answer"])
        self.assertIn("Limitations:", result["answer"])
        self.assertIn("overdue", result["source_review_status"])

    def test_pair_freezes_retrieval_and_does_not_change_inputs(self):
        case = {"id": "synthetic", "question": "Protein foods?"}
        before = copy.deepcopy(case)
        retrieve = Mock(return_value=self.matches)
        with patch.object(g, "call_json", return_value=plan(choice("protein_03"))) as call:
            result = compare_case(case, self.docs, retrieve)
        retrieve.assert_called_once_with("Protein foods?")
        self.assertEqual(call.call_count, 3)
        self.assertEqual([len(v["requests"]) for v in result["variants"]], [1, 2])
        self.assertEqual(result["manual_review"], "pending")
        self.assertEqual(result["comparison_status"], "awaiting_human_review")
        self.assertEqual(case, before)
        self.assertEqual(call.call_args_list[0].args[1], call.call_args_list[1].args[1])

    def test_history_resolution_is_shared_and_reverse_order_is_supported(self):
        case = next(case for case in load_cases() if case["id"] == "plant_followup")
        retrieve = Mock(return_value=self.matches)
        with patch.object(g, "call_json", side_effect=[
            {"query": "Plant protein foods?", "clarification": ""},
            plan(choice("protein_03", 0, [5, 6])), plan(choice("protein_03", 0, [5, 6])),
            plan(choice("protein_03", 0, [5, 6])),
        ]) as call:
            result = compare_case(case, self.docs, retrieve, first="review")
        self.assertEqual(call.call_count, 4)
        self.assertEqual(len(result["shared_requests"]), 1)
        self.assertEqual([v["variant"] for v in result["variants"]], ["review", "single"])
        retrieve.assert_called_once_with("Plant protein foods?")

    def test_ambiguous_resolution_stops_after_one_shared_call(self):
        case = next(case for case in load_cases() if case["id"] == "ambiguous_followup")
        retrieve = Mock()
        with patch.object(g, "call_json", return_value={"query": "", "clarification": "Which topic?"}) as call:
            result = compare_case(case, self.docs, retrieve)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["variants"], [])
        retrieve.assert_not_called()

    def test_rate_limit_stops_pair_without_retry_or_exposing_identifiers(self):
        class Limited(RuntimeError):
            status_code = 429
            body = {"error": {"message": "gsk_private org_private tokens per day: Limit 200000, Used 199000, Requested 2000. Try again in 10s."}}
        with patch.object(g, "call_json", side_effect=Limited()) as call:
            result = compare_case({"id": "synthetic", "question": "Protein foods?"}, self.docs, Mock(return_value=self.matches))
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result["comparison_status"], "incomplete_pair")
        self.assertNotIn("gsk_private", str(result))
        self.assertNotIn("org_private", str(result))
        self.assertEqual(result["variants"][0]["requests"][0]["status"], "error")

    def test_groq_metrics_use_actual_counts_and_do_not_store_messages(self):
        metrics = []
        with patch("openai.OpenAI") as constructor:
            client = constructor.return_value.__enter__.return_value
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(plan())))],
                usage=SimpleNamespace(prompt_tokens=123, completion_tokens=45, total_tokens=168),
            )
            g.call_json_for_stage("evidence_selection", "private prompt", {}, g.ANSWER_SCHEMA,
                                 request_metrics=metrics, groq_api_key="gsk_private")
        self.assertEqual(metrics[0]["tokens"], {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168})
        self.assertGreaterEqual(metrics[0]["response_seconds"], 0)
        self.assertNotIn("private", str(metrics))
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_unavailable_usage_is_null_and_truncated_response_is_not_lost(self):
        metrics = []
        with patch("openai.OpenAI") as constructor:
            client = constructor.return_value.__enter__.return_value
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="{}"))],
            )
            with self.assertRaises(ValueError):
                g.call_json_for_stage("evidence_selection", "system", {}, g.ANSWER_SCHEMA,
                                     request_metrics=metrics, groq_api_key="test-key")
        self.assertTrue(all(value is None for value in metrics[0]["tokens"].values()))
        self.assertEqual(metrics[0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
