"""Offline call budgets, reference routing and personal-data boundaries."""

import copy
import unittest
from unittest.mock import patch

from utils import ai, grounding as g, rag


def selection(intent="knowledge", claims=None, fields=None):
    return {"intent": intent, "claims": claims or [], "log_fields": fields or [], "follow_up": "", "issues": []}


class RequestFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = rag.prep_documents(rag.load_passages())
        cls.matches = [(doc, 0.3) for doc in cls.docs]
        cls.summary = ai.summarize_logs([
            dict(date="2026-10-01", weight=70, protein=0, sleep_hours=0, workout=0),
            dict(date="2026-10-05", weight=70, protein=100, sleep_hours=6, workout=1),
        ], "2026-10-05")
        cls.history = [{"role": "user", "content": "What foods contain protein?"},
                       {"role": "assistant", "content": "Food options were discussed."}]

    def test_standalone_questions_do_not_trigger_a_resolver_after_history(self):
        for question in ["What habits could help me sleep better?", "Protein foods?",
                         "How does my logged protein relate to athlete guidance?",
                         "What is a diet that supports recovery?", "Is it safe to eat eggs?",
                         "I have kidney disease. Prescribe my exact daily protein intake.",
                         "How do I configure a wireless router?", "Show my sleep averages for this week"]:
            with self.subTest(question=question), patch.object(g, "call_json") as call:
                self.assertEqual(g.resolve_question(question, self.history)["query"], question)
            call.assert_not_called()

    def test_contextual_followup_uses_one_resolver_then_one_selection(self):
        for question in ["What about plant options?", "Does that apply to me?", "Is it enough?", "Why?"]:
            metrics = []
            with self.subTest(question=question), patch.object(g, "call_json", side_effect=[
                {"query": "Plant protein foods?", "clarification": ""}, selection(),
            ]) as call:
                resolved = g.resolve_question(question, self.history, request_metrics=metrics)
                ai.generate_result(question, self.matches, history=self.history, request_metrics=metrics)
            self.assertEqual(resolved["query"], "Plant protein foods?")
            self.assertEqual(call.call_count, 2)
            self.assertEqual([m["stage"] for m in metrics], ["follow_up_resolution", "evidence_selection"])

    def test_requested_mixed_fields_survive_empty_plan_and_incomplete_coverage(self):
        before = copy.deepcopy(self.summary)
        for intent in ("knowledge", "mixed", "unsupported", "clarify"):
            with self.subTest(intent=intent), patch.object(g, "call_json", return_value=selection(intent)) as call:
                result = ai.generate_result("How do my logged protein and sleep relate to recovery?", self.matches,
                                            self.summary, history=self.history)
            self.assertEqual(call.call_count, 1)
            self.assertIn("50 g of protein", result["answer"])
            self.assertIn("3 hours of sleep", result["answer"])
            self.assertIn("Coverage: 2 of 7", result["answer"])
            self.assertIn("5 days are missing", result["answer"])
            self.assertIn("2026-10-01, 2026-10-05", result["answer"])
            self.assertFalse(result["log_usage"]["fresh_summary_sent_to_model"])
            self.assertNotIn("personal_log_summary", call.call_args.args[1])
        self.assertEqual(self.summary, before)

    def test_athlete_source_is_separate_and_not_assumed_to_apply(self):
        claim = {"source_id": "protein_02", "unit_index": 0, "item_indices": []}
        with patch.object(g, "call_json", return_value=selection(claims=[claim])):
            result = ai.generate_result("How does my logged protein relate to athlete guidance?", self.matches, self.summary)
        self.assertIn("Your recorded data", result["answer"])
        self.assertIn("General guidance", result["answer"])
        self.assertIn("not a universal target", result["answer"])
        self.assertIn("logs do not establish athlete status", result["answer"])
        self.assertNotIn("84–140", result["answer"])
        self.assertNotIn("deficit", result["answer"])

    def test_no_unrequested_metrics_or_optout_leak_even_if_model_selects_them(self):
        for question in ["How do I configure my wireless router?", "What foods contain protein?",
                         "Do not show my logged protein; explain the general guidance.",
                         "Explain protein guidance without using my logs."]:
            with self.subTest(question=question), patch.object(g, "call_json", return_value=selection("mixed", fields=["avg_protein_g"])):
                result = ai.generate_result(question, self.matches, self.summary)
            self.assertEqual(result["log_header"], "")
            self.assertNotIn("50 g", result["answer"])

    def test_unavailable_periods_and_statistics_do_not_get_current_averages(self):
        for question in ["How did my protein compare last month?", "What is my total protein this week?",
                         "How was my sleep yesterday?", "Compare my protein on 2026-10-01"]:
            with self.subTest(question=question), patch.object(g, "call_json", return_value=selection("mixed", fields=["avg_protein_g"])):
                result = ai.generate_result(question, self.matches, self.summary)
            self.assertNotIn("50 g", result["answer"])
            self.assertIn("requested period or statistic cannot be calculated", result["answer"])
            self.assertIn("Coverage: 2 of 7", result["answer"])

    def test_empty_or_disabled_logs_never_reuse_old_numbers(self):
        history = [{"role": "user", "content": "My logged protein?"},
                   {"role": "assistant", "content": "Earlier average was 999 g."}]
        for summary in (None, ai.summarize_logs([], "2026-10-05")):
            with self.subTest(summary=summary), patch.object(g, "call_json", return_value=selection("mixed")):
                result = ai.generate_result("How does my logged protein relate to recovery?", self.matches, summary, history)
            self.assertNotIn("999 g", result["answer"])
            self.assertNotIn("0 g of protein", result["answer"])
            self.assertIn("unavailable" if summary else "No personal log summary", result["answer"])

    def test_refresh_after_mixed_request_inherits_fields_but_uses_new_records(self):
        history = [{"role": "user", "content": "How does my logged protein relate to athlete guidance?"},
                   {"role": "assistant", "content": "Earlier protein average was 999 g."}]
        fresh = ai.summarize_logs([dict(date="2026-10-06", protein=120, sleep_hours=8, workout=0)], "2026-10-06")
        for question in ("And now?", "And now, using the selected reporting dates?"):
            with self.subTest(question=question), patch.object(g, "call_json", return_value=selection("mixed")):
                result = ai.generate_result(question, self.matches, fresh, history)
            self.assertIn("120 g of protein", result["answer"])
            self.assertIn("2026-09-30 to 2026-10-06", result["answer"])
            self.assertNotIn("999 g", result["answer"])
        unrelated = history + [{"role": "user", "content": "Configure a router"},
                               {"role": "assistant", "content": "No evidence"}]
        self.assertEqual(g.requested_log_fields("And now?", unrelated), [])
        continued = history + [{"role": "user", "content": "And now?"},
                               {"role": "assistant", "content": "Earlier average was 888 g."}]
        self.assertIn("avg_protein_g", g.requested_log_fields("And now, using the selected reporting dates?", continued))

    def test_medical_boundary_does_not_turn_known_weight_into_prescription(self):
        claim = {"source_id": "protein_02", "unit_index": 0, "item_indices": []}
        with patch.object(g, "call_json", return_value=selection("medical_boundary", claims=[claim])) as call:
            result = ai.generate_result("Prescribe protein for my kidney disease using my logged weight.", self.matches, self.summary)
        self.assertEqual(call.call_count, 1)
        self.assertIn("can't prescribe", result["answer"])
        self.assertEqual(result["log_header"], "")
        self.assertNotIn("84–140", result["answer"])

    def test_invalid_output_and_provider_failures_never_trigger_extra_calls(self):
        for response in [{"answer": "wrong contract"}, ValueError("Model did not return valid structured output."), TimeoutError()]:
            with self.subTest(response=response), patch.object(g, "call_json") as call:
                if isinstance(response, Exception):
                    call.side_effect = response
                else:
                    call.return_value = response
                with self.assertRaises(Exception):
                    ai.generate_result("Protein food examples?", self.matches)
            self.assertEqual(call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
