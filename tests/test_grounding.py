"""Mechanical guarantees and orchestration tests, not model-quality scores."""
import copy
import unittest
from unittest.mock import patch

from utils import grounding as g, rag, ai


def plan(claims=None, intent="knowledge", fields=None, follow_up="", issues=None):
    return {"intent": intent, "claims": claims or [], "log_fields": fields or [],
            "follow_up": follow_up, "issues": issues or []}


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.docs = rag.prep_documents(rag.load_passages())
        self.sleep = next(doc for doc in self.docs if doc.metadata["id"] == "sleep_02")
        self.matches = [(self.sleep, 0.28)]
        self.claim = {"source_id": "sleep_02", "unit_index": 0, "item_indices": []}

    def test_review_can_remove_irrelevant_but_valid_selection(self):
        bad = dict(self.claim)
        with patch.object(g, "call_json", side_effect=[plan([bad]), plan([], intent="unsupported", issues=["Claim not supported"])]) as call:
            result = g.draft_and_review("sleep", self.matches, review_evidence=True)
        self.assertEqual(call.call_count, 2)
        self.assertNotIn("cures", result["answer"])
        self.assertIn("Claim not supported", result["review_issues"])

    def test_selection_notes_keep_their_stage_separate_from_rendered_fields(self):
        summary = ai.summarize_logs([dict(date="2026-11-10", protein=100,
            weight=65, sleep_hours=8, workout=1)], "2026-11-10")
        response = plan([self.claim], intent="mixed", follow_up="reporting_period",
            issues=["missing logged values", "missing coverage", "Missing sleep quality data", "Claim not supported"])
        with patch.object(g, "call_json", return_value=response) as call:
            result = ai.generate_result("Compare my logged protein and weight with general guidance", self.matches, summary)
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("Which reporting dates", result["answer"])
        self.assertEqual(result["grounding"]["review_issues"], [])
        self.assertEqual(result["grounding"]["model_issues"], response["issues"])
        self.assertEqual(result["grounding"]["model_issue_stage"], "evidence_selection")
        self.assertEqual(result["grounding"]["rendered_log_field_status"]["avg_protein_g"], "rendered")
        self.assertFalse(result["log_usage"]["fresh_summary_sent_to_model"])

    def test_unavailable_and_zero_values_have_distinct_render_status(self):
        empty = ai.summarize_logs([], "2026-11-10")
        fields = ["coverage", "avg_protein_g", "avg_sleep_hours"]
        self.assertEqual(g.rendered_log_field_status(empty, fields),
            {"coverage": "rendered", "avg_protein_g": "unavailable", "avg_sleep_hours": "unavailable"})
        zeros = ai.summarize_logs([dict(date="2026-11-10", protein=0, sleep_hours=0, workout=0)], "2026-11-10")
        self.assertTrue(all(value == "rendered" for value in g.rendered_log_field_status(zeros, fields).values()))
        self.assertEqual(g.rendered_log_field_status(empty, []), {})
        self.assertEqual(g.rendered_log_field_status(None, fields), {})

    def test_optional_review_notes_are_not_mislabelled_selection_notes(self):
        with patch.object(g, "call_json", side_effect=[
            plan([self.claim], issues=["selection doubt"]),
            plan([self.claim], issues=["review doubt"]),
        ]) as call:
            result = g.draft_and_review("Sleep advice?", self.matches, review_evidence=True)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result["model_issues"], ["selection doubt"])
        self.assertEqual(result["review_issues"], ["review doubt"])

    def test_stated_age_labels_selected_source_group_without_health_verdict(self):
        sleep = next(doc for doc in self.docs if doc.metadata["id"] == "sleep_01")
        claim = {"source_id": "sleep_01", "unit_index": 0, "item_indices": []}
        for age, group in [(25, "18–60"), (60, "18–60"), (61, "61–64"),
                           (64, "61–64"), (65, "65 and above")]:
            with self.subTest(age=age), patch.object(g, "call_json", return_value=plan([claim], follow_up="age")) as call:
                result = g.draft_and_review(f"I'm {age}. Does that prove my sleep is healthy?", [(sleep, .3)])
            self.assertIn(f"**{group} years**. [sleep_01]", result["answer"])
            self.assertIn("duration alone does not assess sleep quality", result["answer"])
            self.assertNotIn("Which adult age group", result["answer"])
            self.assertEqual(call.call_count, 1)

    def test_age_label_never_uses_assistant_numbers_or_absent_evidence(self):
        for text in ["My friend is 25", "If I am 25, what then?", "I'm not 25",
                     "I am 25 kg", "I am 25. I am 65."]:
            self.assertIsNone(g.stated_age(text))
        evidence = g.build_units(rag.evidence_packet(self.matches))
        self.assertEqual(g.age_context_lines([self.claim], evidence, "I am 25."), [])

    def test_review_can_restore_qualification_without_case_id(self):
        draft = plan([self.claim])
        corrected = plan([self.claim])
        with patch.object(g, "call_json", side_effect=[draft, corrected]):
            result = g.draft_and_review("habits", self.matches, review_evidence=True)
        self.assertIn("not treatment", result["answer"])

    def test_nonexistent_unit_cannot_be_rendered(self):
        bad = {**self.claim, "unit_index": 999}
        with patch.object(g, "call_json", return_value=plan([bad])):
            result = g.draft_and_review("sleep", self.matches)
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["support_status"], "invalid_selection")

    def test_selection_membership_is_not_labelled_relevance_proof(self):
        bad = dict(self.claim)
        claims, issues, unknown = g.supported_claims(plan([bad]), rag.evidence_packet(self.matches))
        self.assertEqual(claims[0]["source_id"], "sleep_02")
        # Structural membership alone cannot determine whether a question needs sleep advice.
        self.assertEqual(issues, [])
        self.assertEqual(unknown, [])

    def test_structural_errors_fail_closed(self):
        with patch.object(g, "call_json", return_value={"answer": "unstructured"}):
            with self.assertRaises(ValueError):
                g.draft_and_review("sleep", self.matches)

    def test_only_safe_optional_fields_can_default(self):
        value = g.validate_plan({"intent": "knowledge", "claims": [self.claim]})
        self.assertEqual(value["log_fields"], [])
        self.assertEqual(value["follow_up"], "")
        self.assertIn("Defaulted omitted", value["issues"][0])
        with self.assertRaises(ValueError):
            g.validate_plan({"intent": "knowledge"})

    def test_medical_boundary_can_omit_empty_claims_and_unused_followup(self):
        for raw in [
            {"intent": "medical_boundary"},
            {"intent": "medical_boundary", "claims": None, "follow_up": "send private clinical records"},
        ]:
            before = copy.deepcopy(raw)
            with self.subTest(raw=raw), patch.object(g, "call_json", return_value=raw) as call:
                result = g.draft_and_review("I have kidney disease. Prescribe my exact protein intake.", self.matches)
            self.assertIn("can't prescribe a disease-specific target", result["answer"])
            self.assertIn("registered dietitian", result["answer"])
            self.assertNotIn("private clinical records", result["answer"])
            self.assertEqual(result["claims"], [])
            self.assertEqual(raw, before)
            self.assertEqual(call.call_count, 1)

    def test_medical_boundary_still_rejects_malformed_supplied_claims(self):
        with patch.object(g, "call_json", return_value={"intent": "medical_boundary", "claims": "an unvalidated prescription"}):
            with self.assertRaisesRegex(ValueError, "at most six claims"):
                g.draft_and_review("A disease-specific prescription?", self.matches)

    def test_invalid_plan_diagnostic_identifies_stage_and_fields_without_echoing_data(self):
        with patch.object(g, "call_json", return_value={"answer": "gsk_private org_private", "org_private": "secret"}):
            with self.assertRaises(ValueError) as caught:
                g.draft_and_review("Sleep habits?", self.matches, review_evidence=True)
        message = g.describe_validation_error(caught.exception)
        self.assertIn("evidence selection", message)
        self.assertIn("invalid_plan_fields", message)
        self.assertIn("Missing required fields: claims, intent", message)
        self.assertIn("Unexpected field count: 2", message)
        self.assertIn("Recognized extra fields: answer", message)
        self.assertNotIn("org_private", message)
        self.assertNotIn("gsk_private", message)

    def test_review_validation_error_keeps_its_stage(self):
        invalid = plan(intent="not_a_supported_intent")
        with patch.object(g, "call_json", side_effect=[plan([self.claim]), invalid]):
            with self.assertRaises(ValueError) as caught:
                g.draft_and_review("Sleep habits?", self.matches, review_evidence=True)
        self.assertIn("evidence review", g.describe_validation_error(caught.exception))
        self.assertIn("invalid_intent", g.describe_validation_error(caught.exception))

    def test_truncated_provider_json_is_rejected_and_reported_without_another_request(self):
        import json
        from types import SimpleNamespace
        with patch("openai.OpenAI") as constructor:
            client = constructor.return_value.__enter__.return_value
            client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="length", message=SimpleNamespace(content=json.dumps(plan([self.claim]))),
            )])
            with self.assertRaises(ValueError) as caught:
                g.call_json_for_stage("evidence_selection", "system", {}, g.ANSWER_SCHEMA, groq_api_key="test-key")
        message = g.describe_validation_error(caught.exception)
        self.assertIn("output_truncated", message)
        self.assertIn("evidence selection", message)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_invalid_json_diagnostic_never_echoes_the_provider_content(self):
        from types import SimpleNamespace
        with patch("openai.OpenAI") as constructor:
            client = constructor.return_value.__enter__.return_value
            client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content="gsk_private org_private"),
            )])
            with self.assertRaises(ValueError) as caught:
                g.call_json_for_stage("evidence_review", "system", {}, g.ANSWER_SCHEMA, groq_api_key="test-key")
        message = g.describe_validation_error(caught.exception)
        self.assertIn("invalid_json", message)
        self.assertNotIn("gsk_private", message)
        self.assertNotIn("org_private", message)

    def test_unknown_data_error_diagnostic_does_not_blame_or_expose_model_content(self):
        message = g.describe_validation_error(ValueError("gsk_private org_private"))
        self.assertIn("request_data_invalid", message)
        self.assertNotIn("private", message)

    def test_source_scope_is_separate_in_packet(self):
        packet = rag.evidence_packet(self.matches)["sleep_02"]
        self.assertIn("Adults", packet["population"])
        self.assertIn("not treatment", packet["limitations"])

    def test_fresh_log_values_are_rendered_by_code_not_old_ai_text(self):
        summary = ai.summarize_logs([dict(date="2026-10-05", protein=120, sleep_hours=8, workout=1)], "2026-10-05")
        history = [{"role": "user", "content": "My logged protein summary?"},
                   {"role": "assistant", "content": "Protein average was 999g."}]
        with patch.object(g, "call_json", return_value=plan(intent="logs", fields=["avg_protein_g"])) as call:
            result = g.draft_and_review("And now?", [], summary, history)
        self.assertIn("120 g", result["answer"])
        self.assertNotIn("999", result["answer"])
        call.assert_not_called()

    def test_explicit_refresh_after_food_discussion_uses_fresh_records_without_model(self):
        history = [{"role": "user", "content": "What foods provide protein without powder?"},
                   {"role": "assistant", "content": "Beans and nuts. [protein_03]"}]
        summary = ai.summarize_logs([dict(date="2026-10-05", protein=120, sleep_hours=8, workout=1)], "2026-10-05")
        for question in ["Summarise my protein and sleep now.",
                         "Please refresh my protein and sleep averages.",
                         "Show me an updated summary of my logged protein and sleep.",
                         "Summarize my protein and sleep again."]:
            with self.subTest(question=question), patch.object(g, "call_json", side_effect=AssertionError("No model needed")):
                result = g.draft_and_review(question, self.matches, summary, history)
            self.assertIn("120 g of protein", result["answer"])
            self.assertIn("8 hours of sleep", result["answer"])
            self.assertEqual(result["support_status"], "deterministic_log_summary")

    def test_reporting_refresh_inherits_fields_but_not_old_numbers(self):
        history = [{"role": "user", "content": "Summarise my protein and sleep now."},
                   {"role": "assistant", "content": "Protein 999 g; sleep 24 hours."}]
        summary = ai.summarize_logs([dict(date="2026-10-06", protein=140, sleep_hours=7, workout=0)], "2026-10-06")
        with patch.object(g, "call_json", side_effect=AssertionError("No model needed")):
            result = g.draft_and_review("And now, using the selected reporting dates?", [], summary, history)
        self.assertIn("140 g of protein", result["answer"])
        self.assertIn("7 hours of sleep", result["answer"])
        self.assertIn("2026-09-30 to 2026-10-06", result["answer"])
        self.assertNotIn("999", result["answer"])

    def test_refresh_needs_clear_record_context_and_does_not_skip_an_intervening_topic(self):
        records = [{"role": "user", "content": "Summarise my logged protein and sleep."},
                   {"role": "assistant", "content": "A record summary"}]
        foods = [{"role": "user", "content": "What foods provide protein?"},
                 {"role": "assistant", "content": "A food answer"}]
        for question in ["And now?", "And now, using the selected reporting dates?"]:
            with self.subTest(question=question):
                self.assertIsNone(g.log_summary_fields(question))
                self.assertIsNone(g.log_summary_fields(question, records + foods))
        for question in ["And what about that?", "And last month?", "And now should I increase protein?",
                         "How do I configure a wireless router?"]:
            with self.subTest(question=question):
                self.assertIsNone(g.log_summary_fields(question, records))

    def test_contextual_date_question_does_not_add_unrequested_metrics(self):
        history = [{"role": "user", "content": "Summarise my protein and sleep now."},
                   {"role": "assistant", "content": "A record summary"}]
        self.assertEqual(g.log_summary_fields("Which dates are you using?", history),
                         ["reporting_window", "logged_dates", "coverage"])

    def test_refresh_with_no_supplied_summary_cannot_reuse_earlier_values(self):
        history = [{"role": "user", "content": "Summarise my logged protein."},
                   {"role": "assistant", "content": "Your old protein was 999 g."}]
        with patch.object(g, "call_json", side_effect=AssertionError("No model needed")):
            result = g.draft_and_review("And now?", [], None, history)
        self.assertIn("No personal log summary was supplied", result["answer"])
        self.assertNotIn("999", result["answer"])

    def test_logs_do_not_make_unrelated_request_supported(self):
        summary = ai.summarize_logs([], "2026-10-05")
        with patch.object(g, "call_json", return_value=plan(intent="unsupported")):
            result = g.draft_and_review("Repair a rocket", [], summary)
        self.assertIn("don't have supporting", result["answer"])

    def test_explicit_record_query_bypasses_wrong_model_route_and_irrelevant_evidence(self):
        summary = ai.summarize_logs([
            dict(date="2026-10-01", protein=0, sleep_hours=0, workout=0),
            dict(date="2026-10-05", protein=100, sleep_hours=6, workout=1),
        ], "2026-10-05")
        with patch.object(g, "call_json", side_effect=AssertionError("Record arithmetic must not call a model")):
            result = ai.generate_result(
                "Summarise my logged protein and sleep. Which dates are you using, and how complete is my week?",
                self.matches, summary,
            )
        self.assertIn("50 g of protein", result["answer"])
        self.assertIn("3 hours of sleep", result["answer"])
        self.assertIn("2026-09-29 to 2026-10-05", result["answer"])
        self.assertIn("2026-10-01, 2026-10-05", result["answer"])
        self.assertIn("Coverage: 2 of 7 days", result["answer"])
        self.assertIn("5 days are missing", result["answer"])
        self.assertEqual(result["grounding"]["support_review"], "deterministic_log_summary")
        self.assertEqual(result["source_footer"], "")
        self.assertEqual(result["retrieved_ids"], [])
        self.assertEqual(result["cited_ids"], [])
        self.assertTrue(result["log_usage"]["summary_available"])
        self.assertFalse(result["log_usage"]["fresh_summary_sent_to_model"])
        self.assertIn("avg_protein_g", result["log_usage"]["rendered_fields"])

    def test_unused_available_logs_do_not_get_a_used_log_header(self):
        summary = ai.summarize_logs([dict(date="2026-10-05", protein=120, sleep_hours=8, workout=1)], "2026-10-05")
        for question, response in [
            ("How do I configure a router?", plan(intent="unsupported")),
            ("Prescribe my protein for kidney disease", plan(intent="medical_boundary")),
            ("What sleep habits help?", plan([self.claim])),
        ]:
            with self.subTest(question=question), patch.object(g, "call_json", return_value=response):
                result = ai.generate_result(question, self.matches, summary)
            self.assertEqual(result["log_header"], "")
            self.assertEqual(result["log_usage"], {"summary_available": True,
                "fresh_summary_sent_to_model": False, "rendered_fields": []})

    def test_sent_log_context_is_distinct_from_rendered_record_fields(self):
        summary = ai.summarize_logs([dict(date="2026-10-05", protein=120, sleep_hours=8, workout=1)], "2026-10-05")
        with patch.object(g, "call_json", side_effect=[
            plan(intent="mixed", fields=["avg_protein_g"]), plan(intent="unsupported"),
        ]) as call:
            result = ai.generate_result("Can my logs explain recovery?", self.matches, summary, review_evidence=True)
        self.assertEqual(call.call_args.args[1]["personal_log_summary"], summary)
        self.assertTrue(result["log_usage"]["fresh_summary_sent_to_model"])
        self.assertEqual(result["log_usage"]["rendered_fields"], [])
        self.assertIn("2026-10-05", result["log_header"])
        self.assertNotIn("120", result["answer"])

    def test_curated_warning_has_no_log_usage_even_when_records_are_available(self):
        summary = ai.summarize_logs([], "2026-10-05")
        with patch.object(g, "call_json", side_effect=AssertionError("No model needed")):
            result = ai.generate_checked_result("Show curated guidance", [], True, summary, documents=self.docs)
        self.assertEqual(result["log_header"], "")
        self.assertEqual(result["log_usage"], {"summary_available": True,
            "fresh_summary_sent_to_model": False, "rendered_fields": []})

    def test_old_conversation_numbers_are_not_reported_as_a_fresh_summary_attachment(self):
        summary = ai.summarize_logs([], "2026-10-05")
        history = [{"role": "user", "content": "My protein?"},
                   {"role": "assistant", "content": "Earlier logged protein: 120 g."}]
        with patch.object(g, "call_json", return_value=plan(intent="unsupported")) as call:
            result = g.draft_and_review("How do I configure a router?", [], summary, history)
        self.assertIn("120 g", str(call.call_args.args[1]["conversation_context"]))
        self.assertFalse(result["log_usage"]["fresh_summary_sent_to_model"])
        self.assertEqual(result["log_usage"]["rendered_fields"], [])

    def test_record_query_paraphrases_and_missing_summary(self):
        for question in ["Show my average sleep for this week", "Please summarize my protein intake",
                         "List my logged dates", "Summarise my health logs and week completeness"]:
            with self.subTest(question=question), patch.object(g, "call_json", side_effect=AssertionError("No model needed")):
                result = g.draft_and_review(question, self.matches)
            self.assertIn("No personal log summary was supplied", result["answer"])

    def test_empty_record_query_does_not_invent_zero_averages(self):
        with patch.object(g, "call_json", side_effect=AssertionError("No model needed")):
            result = g.draft_and_review("Summarise my logged protein and sleep", [], ai.summarize_logs([], "2026-10-05"))
        self.assertIn("Coverage: 0 of 7 days", result["answer"])
        self.assertIn("unavailable", result["answer"])
        self.assertNotIn("0 g of protein", result["answer"])

    def test_summary_rounds_presentation_without_changing_values(self):
        summary = ai.summarize_logs([dict(date="2026-10-05", protein=106.6666667, sleep_hours=7.1333333, workout=0)], "2026-10-05")
        before = copy.deepcopy(summary)
        answer = g.render_log_summary(summary, ["avg_protein_g", "avg_sleep_hours", "coverage"])
        self.assertIn("106.7 g of protein", answer)
        self.assertIn("7.1 hours of sleep", answer)
        self.assertIn("1 logged day", answer)
        self.assertIn("Next step:", answer)
        self.assertEqual(summary, before)

    def test_gapped_dates_are_not_presented_as_continuous(self):
        summary = ai.summarize_logs([dict(date=day, protein=100, sleep_hours=7, workout=0)
            for day in ("2026-10-01", "2026-10-03", "2026-10-05")], "2026-10-05")
        answer = g.render_log_summary(summary, ["logged_dates"])
        self.assertIn("2026-10-01, 2026-10-03, 2026-10-05", answer)
        self.assertNotIn("through", answer)

    def test_complete_zero_logs_keep_valid_zero_averages(self):
        summary = ai.summarize_logs([dict(date=f"2026-10-{day:02}", protein=0, sleep_hours=0, workout=0)
            for day in range(1, 8)], "2026-10-07")
        answer = g.render_log_summary(summary, ["avg_protein_g", "avg_sleep_hours", "coverage"])
        self.assertIn("0 g of protein", answer)
        self.assertIn("0 hours of sleep", answer)
        self.assertIn("all days are logged", answer)
        self.assertNotIn("unavailable", answer)

    def test_rate_limit_reports_only_safe_type_and_duration(self):
        error = RuntimeError("gsk_private org_private")
        error.body = {"error": {"message": "tokens per day (TPD) in org_private gsk_private. Please try again in 12m37.296s. Upgrade at private-url"}}
        message = g.describe_rate_limit(error, "Groq")
        self.assertIn("daily token limit", message)
        self.assertIn("12m37.296s from this request", message)
        for private in ("org_private", "gsk_private", "private-url"):
            self.assertNotIn(private, message)

    def test_rate_limit_uses_numeric_retry_header_without_guessing(self):
        from types import SimpleNamespace
        error = RuntimeError("private provider details")
        error.response = SimpleNamespace(headers={"retry-after": "30"})
        self.assertIn("30 seconds", g.describe_rate_limit(error))
        self.assertNotIn("private", g.describe_rate_limit(error))

    def test_rate_limit_missing_details_has_no_invented_retry_time(self):
        message = g.describe_rate_limit(RuntimeError("gsk_private org_private"))
        self.assertIn("No retry time was provided", message)
        self.assertNotIn("private", message)

    def test_rate_limit_exposes_only_numeric_counters_and_known_stage(self):
        error = RuntimeError("gsk_private org_private")
        error.body = {"error": {"message": "tokens per day in org_private. Limit 200000, Used 199855, Requested 1898. Please try again in 11m35.952s."}}
        error.nutrirecall_stage = "evidence_review"
        message = g.describe_rate_limit(error, "Groq")
        self.assertIn("after a completed selection call", message)
        self.assertIn("199,855 of 200,000 tokens used", message)
        self.assertIn("1,898 tokens", message)
        self.assertNotIn("org_private", message)
        self.assertNotIn("gsk_private", message)
        error.nutrirecall_stage = "gsk_private"
        self.assertNotIn("gsk_private", g.describe_rate_limit(error))

    def test_limit_stage_is_recorded_without_retrying(self):
        class Limited(RuntimeError):
            status_code = 429
        for stage, responses, expected_calls in [
            ("evidence_selection", [Limited()], 1),
            ("evidence_review", [plan([self.claim]), Limited()], 2),
        ]:
            with self.subTest(stage=stage), patch.object(g, "call_json", side_effect=responses) as call:
                with self.assertRaises(Limited) as caught:
                    g.draft_and_review("Sleep habits?", self.matches, review_evidence=stage == "evidence_review")
            self.assertEqual(caught.exception.nutrirecall_stage, stage)
            self.assertEqual(call.call_count, expected_calls)

    def test_followup_limit_stage_is_recorded(self):
        class Limited(RuntimeError):
            status_code = 429
        history = [{"role": "user", "content": "Protein foods?"}, {"role": "assistant", "content": "An answer"}]
        with patch.object(g, "call_json", side_effect=Limited()) as call:
            with self.assertRaises(Limited) as caught:
                g.resolve_question("What about plant foods?", history)
        self.assertEqual(caught.exception.nutrirecall_stage, "follow_up_resolution")
        call.assert_called_once()

    def test_mixed_advice_negation_totals_and_other_periods_are_not_read_only_summary_queries(self):
        for question in ["Summarise my protein and suggest foods", "Do not summarise my logs",
                         "Summarise my protein last month", "What is my total protein this week?",
                         "What did my sleep look like yesterday?", "How do I configure a wireless router?",
                         "Summarise my kidney disease and prescribe my protein intake"]:
            with self.subTest(question=question):
                self.assertIsNone(g.log_summary_fields(question))

    def test_router_cannot_render_records_even_if_model_emits_logs_intent(self):
        summary = ai.summarize_logs([dict(date="2026-10-05", protein=120, sleep_hours=8, workout=1)], "2026-10-05")
        with patch.object(g, "call_json", return_value=plan(
            intent="logs", fields=["avg_protein_g", "avg_sleep_hours"], follow_up="reporting_period",
        )) as call:
            result = g.draft_and_review("How do I configure a wireless router?", [], summary)
        self.assertNotIn("personal_log_summary", call.call_args.args[1])
        self.assertNotIn("120", result["answer"])
        self.assertNotIn("Sleep", result["answer"])
        self.assertIn("do not support", result["answer"])
        self.assertNotIn("Which reporting dates", result["answer"])

    def test_log_only_schema_excludes_health_claim_channel(self):
        summary = ai.summarize_logs([], "2026-10-05")
        with patch.object(g, "call_json", return_value={
            key: value for key, value in plan(intent="logs", fields=["avg_sleep_hours"]).items()
            if key != "claims"
        }) as call:
            result = g.draft_and_review("My sleep?", [], summary)
        self.assertNotIn("claims", call.call_args.args[2]["properties"])
        self.assertIn("unavailable", result["answer"])

    def test_empty_selection_abstains_without_inventing_a_clarification(self):
        summary = ai.summarize_logs([], "2026-10-05")
        with patch.object(g, "call_json", return_value=plan()):
            result = g.draft_and_review("Configure my router", [], summary)
        self.assertIn("do not support", result["answer"])
        self.assertNotIn("clarify", result["answer"])

    def test_field_selection_does_not_receive_values_and_knowledge_does_not_render_them(self):
        summary = ai.summarize_logs([dict(date="2026-10-05", protein=120, sleep_hours=8, workout=1)], "2026-10-05")
        with patch.object(g, "call_json", return_value=plan(fields=["avg_protein_g"])) as call:
            result = g.draft_and_review("Configure a router", [], summary)
        self.assertNotIn("personal_log_summary", call.call_args_list[0].args[1])
        self.assertEqual(call.call_count, 1)
        self.assertNotIn("120", result["answer"])

    def test_date_and_coverage_fields_are_rendered_from_fresh_summary(self):
        summary = ai.summarize_logs([dict(date="2026-10-05", protein=120, sleep_hours=8, workout=1)], "2026-10-05")
        answer = "\n".join(g.render_log_values(summary, ["reporting_window", "logged_dates", "coverage"]))
        self.assertIn("2026-09-29 to 2026-10-05", answer)
        self.assertIn("1/7 days recorded; 6 missing", answer)

    def test_bounded_completed_history_only(self):
        history = []
        for index in range(10):
            history += [{"role": "user", "content": f"Question {index}"},
                        {"role": "assistant", "result": {"answer": "x" * 3000}}]
        history += [{"role": "user", "content": "Failed request"},
                    {"role": "assistant", "content": "Error", "error": True}]
        before = copy.deepcopy(history)
        bounded = g.bounded_history(history)
        self.assertLessEqual(len(bounded), 6)
        self.assertLessEqual(sum(len(m["content"]) for m in bounded), 6000)
        self.assertFalse(any("Failed request" in m["content"] for m in bounded))
        self.assertEqual(history, before)

    def test_referents_are_resolved_without_reusing_ai_as_evidence(self):
        history = [{"role": "user", "content": "Protein foods?"}, {"role": "assistant", "content": "An answer"}]
        with patch.object(g, "call_json", return_value={"query": "Plant protein food options?", "clarification": ""}) as call:
            result = g.resolve_question("What about plant options?", history)
        self.assertIn("Plant protein", result["query"])
        self.assertIn("not evidence", call.call_args.args[0])
        with patch.object(g, "call_json", return_value={"query": "", "clarification": "Which topic?"}):
            self.assertEqual(g.resolve_question("that?", history)["clarification"], "Which topic?")

    def test_empty_history_needs_no_resolution_call(self):
        with patch.object(g, "call_json") as call:
            self.assertEqual(g.resolve_question("sleep", []), {"query": "sleep", "clarification": ""})
        call.assert_not_called()

    def test_support_review_is_not_reported_as_proof(self):
        with patch.object(g, "call_json", return_value=plan([self.claim])):
            result = ai.generate_result("sleep", self.matches)
        self.assertTrue(result["grounding"]["citation_ids_valid"])
        self.assertEqual(result["grounding"]["support_review"], "extractive_selection_unreviewed")
        self.assertNotIn("grounding_passed", str(result))

    def test_model_paraphrases_are_rejected_by_the_contract(self):
        wrong = {**self.claim, "text": "Dark rooms and blue light cure insomnia."}
        with patch.object(g, "call_json", return_value=plan([wrong])):
            with self.assertRaisesRegex(ValueError, "Invalid claim fields"):
                g.draft_and_review("Sleep habits?", self.matches)

    def test_medical_boundary_cannot_be_downgraded_to_clinical_clarification(self):
        with patch.object(g, "call_json", side_effect=[plan(intent="medical_boundary"), plan(intent="clarify", follow_up="topic")]):
            result = g.draft_and_review("Prescribe for my condition", self.matches, review_evidence=True)
        self.assertIn("can't prescribe", result["answer"])
        self.assertNotIn("Which topic", result["answer"])

    def test_followup_text_cannot_request_arbitrary_clinical_data(self):
        with self.assertRaises(ValueError):
            g.validate_plan(plan(follow_up="Send your lab tests so I can prescribe."))

    def test_clarification_retains_supported_guidance_and_citations(self):
        with patch.object(g, "call_json", return_value=plan(
            [self.claim], intent="clarify", follow_up="age",
        )):
            result = g.draft_and_review("Sleep advice for me?", self.matches)
        self.assertIn("[sleep_02]", result["answer"])
        self.assertIn("Which adult age group", result["answer"])

    def test_numeric_range_calculation_is_deterministic_and_not_a_prescription(self):
        protein = next(doc for doc in self.docs if doc.metadata["id"] == "protein_02")
        claim = {"source_id": "protein_02", "unit_index": 0, "item_indices": []}
        with patch.object(g, "call_json", return_value=plan([claim])):
            result = g.draft_and_review("An adult athlete at 70 kg?", [(protein, 0.3)])
        self.assertIn("84–140 g/day", result["answer"])
        self.assertIn("not an individual prescription", result["answer"])

    def test_selected_source_review_status_is_not_lost_in_rendering(self):
        weight = next(doc for doc in self.docs if doc.metadata["id"] == "weight_02")
        claim = {"source_id": "weight_02", "unit_index": 4, "item_indices": []}
        with patch.object(g, "call_json", return_value=plan([claim])):
            result = ai.generate_result("Is the source recently reviewed?", [(weight, 0.3)])
        self.assertIn("Source review:", result["source_review_status"])
        self.assertIn("overdue when checked on 2026-10-10", result["source_review_status"])
        self.assertNotIn("Recheck before release", result["source_review_status"])

    def test_review_status_does_not_depend_on_model_mention_and_is_deduplicated(self):
        docs = [doc for doc in self.docs if doc.metadata["id"] in ("weight_01", "weight_02")]
        matches = [(doc, 0.3) for doc in docs]
        status = rag.build_source_review_status(matches, [doc.metadata["id"] for doc in docs])
        self.assertEqual(status.count("Source review:"), 1)
        self.assertIn("overdue", status)

    def test_summary_rejects_invalid_values_without_changing_inputs(self):
        row = dict(date="2026-10-05", protein=float("nan"), sleep_hours=8, workout=1)
        with self.assertRaises(ValueError):
            ai.summarize_logs([row], "2026-10-05")


if __name__ == "__main__":
    unittest.main()
