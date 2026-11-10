"""Offline tests: curated public KB, synthetic logs and mocked models/stores.

No health database, real vector index, downloads or model API calls are used.
These checks do not establish claim-level grounding or clinical safety.
"""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

from utils import ai, rag
from utils import grounding


class RagPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passages = rag.load_passages()
        cls.documents = rag.prep_documents(cls.passages)
        cls.by_id = {doc.metadata["id"]: doc for doc in cls.documents}
        cls.sleep_matches = [(cls.by_id["sleep_02"], 0.282)]
        cases = json.loads(
            (rag.ROOT / "tests" / "rag_cases.json").read_text(encoding="utf-8")
        )
        cls.fixture = next(c["context"] for c in cases if c["id"] == "rag_08")

    def test_kb_has_16_unique_passages_and_citations(self):
        self.assertEqual(len(self.passages), 16)
        self.assertEqual(len(self.by_id), 16)
        self.assertEqual(self.by_id["sleep_02"].metadata["source_url"],
                         "https://www.cdc.gov/sleep/about/")
        self.assertEqual(rag.ROOT, Path(__file__).resolve().parents[1])

    def test_loader_rejects_duplicate_ids(self):
        contents = json.dumps([self.passages[0], self.passages[0]])
        with patch.object(Path, "open", mock_open(read_data=contents)):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                rag.load_passages("unused.json")

    def test_loader_rejects_empty_or_malformed_kb(self):
        for contents in ["[]", "{}", '[{"id": "missing_fields"}]']:
            with self.subTest(contents=contents):
                with patch.object(Path, "open", mock_open(read_data=contents)):
                    with self.assertRaises(ValueError):
                        rag.load_passages("unused.json")

    def test_embedding_setup_is_explicit_and_normalized(self):
        constructor = Mock()
        with patch.dict(sys.modules, {
            "langchain_huggingface": SimpleNamespace(HuggingFaceEmbeddings=constructor),
        }):
            rag.get_embeddings_function()
        constructor.assert_called_once_with(
            model_name=rag.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def test_store_uses_notebook_collection_and_cosine(self):
        constructor = Mock()
        embeddings = Mock()
        with patch.dict(sys.modules, {
            "langchain_chroma": SimpleNamespace(Chroma=constructor),
        }):
            rag.open_vector_store(embeddings)
        constructor.assert_called_once_with(
            collection_name="neutricall_knowledge_v1",
            persist_directory=str(rag.CHROMA_DIR),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def test_index_build_uses_stable_ids(self):
        store, embeddings = Mock(), Mock()
        with patch.object(rag, "open_vector_store", return_value=store):
            result = rag.build_vector_store(self.documents, embeddings)
        self.assertIs(result, store)
        store.add_documents.assert_called_once_with(
            documents=self.documents,
            ids=[doc.metadata["id"] for doc in self.documents],
        )

    def test_index_build_rejects_empty_or_duplicate_documents(self):
        with patch.object(rag, "open_vector_store") as open_store:
            for docs in [[], [self.documents[0], self.documents[0]]]:
                with self.assertRaises(ValueError):
                    rag.build_vector_store(docs, Mock())
            open_store.assert_not_called()

    def test_distance_filter_and_provisional_defaults(self):
        store = Mock()
        store.similarity_search_with_score.return_value = [
            (self.by_id["activity_03"], 0.579),
            (self.by_id["activity_02"], 0.605),
        ]
        matches = rag.retrieve_relevant_passages("exercise", store)
        self.assertEqual([doc.metadata["id"] for doc, _ in matches], ["activity_03"])
        store.similarity_search_with_score.assert_called_once_with("exercise", k=6)

    def test_blank_query_does_not_search(self):
        store = Mock()
        self.assertEqual(rag.retrieve_relevant_passages("  ", store), [])
        store.similarity_search_with_score.assert_not_called()

    def test_invalid_retrieval_arguments(self):
        for k in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                rag.retrieve_passages("sleep", Mock(), k=k)
        with self.assertRaises(ValueError):
            rag.retrieve_relevant_passages("sleep", Mock(), max_distance=-0.1)

    def test_context_preserves_qualifications_and_omits_distance(self):
        context = rag.build_context(self.sleep_matches)
        self.assertIn("[sleep_02]", context)
        self.assertIn("Limitations:", context)
        self.assertNotIn("0.282", context)

    def test_footer_preserves_nhs_warning_and_deduplicates_ids(self):
        match = (self.by_id["weight_04"], None)
        footer = rag.build_source_footer([match, match])
        self.assertEqual(footer.count("- [weight_04]"), 1)
        self.assertIn("2023-03-28", footer)
        self.assertIn("2026-03-28", footer)
        self.assertIn("overdue", footer)

    def test_evidence_only_shows_cited_supplied_passages(self):
        evidence = rag.build_evidence_used(self.sleep_matches, ["sleep_02", "sleep_99"])
        self.assertIn(self.by_id["sleep_02"].page_content, evidence)
        self.assertNotIn("sleep_99", evidence)
        self.assertEqual(rag.build_source_footer([]), "")
        self.assertEqual(rag.build_citation_warning([]), "")

    def test_summary_uses_logged_dates_and_valid_zeros(self):
        result = ai.summarize_logs(self.fixture["logs"], self.fixture["as_of"])
        self.assertEqual(result["logged_dates"], ["2026-10-01", "2026-10-05"])
        self.assertEqual(result["logged_days"], 2)
        self.assertEqual(result["missing_days"], 5)
        self.assertEqual(result["avg_protein_g"], 50.0)
        self.assertEqual(result["avg_sleep_hours"], 3.0)
        self.assertEqual(result["workout_days"], 1)

    def test_empty_summary_has_unknown_not_zero_values(self):
        result = ai.summarize_logs([], "2026-10-05")
        self.assertEqual(result["missing_days"], 7)
        for field in ["avg_protein_g", "avg_sleep_hours", "workout_days"]:
            self.assertIsNone(result[field])

    def test_summary_window_boundaries_and_duplicate_rejection(self):
        logs = [dict(date=day, protein=100.0, sleep_hours=8.0, workout=1)
                for day in ["2026-09-28", "2026-09-29", "2026-10-05", "2026-10-06"]]
        result = ai.summarize_logs(logs, "2026-10-05")
        self.assertEqual(result["logged_dates"], ["2026-09-29", "2026-10-05"])
        with self.assertRaises(ValueError):
            ai.summarize_logs([logs[1], logs[1]], "2026-10-05")

    def test_summary_header_and_message_are_separate_from_knowledge(self):
        summary = ai.summarize_logs(self.fixture["logs"], self.fixture["as_of"])
        header = ai.build_log_header(summary)
        self.assertIn("2026-09-29 to 2026-10-05", header)
        self.assertIn("2/7", header)
        payload = json.loads(ai.build_user_message("summarise", [], summary))
        self.assertEqual(payload["personal_log_summary"], summary)
        self.assertEqual(ai.build_log_header(None), "")
        with self.assertRaises(ValueError):
            ai.build_user_message("summarise", [], {"bad": float("nan")})

    def test_no_evidence_or_summary_skips_api_call(self):
        with patch("openai.OpenAI") as client:
            answer = ai.generate_answer("spacecraft", [])
        self.assertIn("supporting information", answer)
        client.assert_not_called()

    def test_missing_key_raises_without_api_call(self):
        with patch.dict("os.environ", {}, clear=True), patch("openai.OpenAI") as client:
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                ai.generate_answer("sleep", self.sleep_matches)
            client.assert_not_called()

    def test_groq_receives_system_message_and_synthetic_summary(self):
        summary = ai.summarize_logs(self.fixture["logs"], self.fixture["as_of"])
        with patch("openai.OpenAI") as constructor:
            client = constructor.return_value.__enter__.return_value
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                    "intent": "logs", "claims": [], "log_fields": ["avg_protein_g"],
                    "follow_up": "", "issues": [],
                }))) ]
            )
            answer = ai.generate_answer("How can my logged protein support recovery?", [], summary, groq_api_key="test-key")
        self.assertIn("50 g of protein", answer)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        kwargs = client.chat.completions.create.call_args_list[0].kwargs
        self.assertTrue(kwargs["messages"][0]["content"].startswith(ai.SYSTEM_PROMPT))
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertIn("JSON schema:", kwargs["messages"][0]["content"])
        draft_payload = json.loads(kwargs["messages"][1]["content"])
        self.assertNotIn("personal_log_summary", draft_payload)
        self.assertIn("available_log_fields", draft_payload)
        self.assertEqual(constructor.call_args.kwargs["base_url"], "https://api.groq.com/openai/v1")
        self.assertEqual(constructor.call_args.kwargs["timeout"], 10)
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)

    def test_optional_local_backend_is_mocked(self):
        local = Mock()
        local.chat.return_value = {"message": {"content": json.dumps({
            "intent": "knowledge", "claims": [{"source_id": "sleep_02", "unit_index": 0, "item_indices": []}],
            "log_fields": [], "follow_up": "", "issues": [],
        })}}
        with patch.dict(sys.modules, {"ollama": local}):
            answer = ai.generate_answer("sleep", self.sleep_matches, use_local=True)
        self.assertIn("Helpful habits", answer)
        self.assertEqual(local.chat.call_count, 1)
        schema = local.chat.call_args.kwargs["format"]
        self.assertEqual(schema["properties"]["claims"]["items"]["properties"]["source_id"]["enum"], ["sleep_02"])
        self.assertEqual(schema["properties"]["claims"]["maxItems"], 6)
        self.assertEqual(local.chat.call_args.kwargs["messages"][0]["role"], "system")

    def test_result_flags_unknown_ids_without_claiming_grounding(self):
        plan = {"intent": "knowledge", "claims": [
            {"source_id": "sleep_02", "unit_index": 0, "item_indices": []},
            {"source_id": "sleep_99", "unit_index": 0, "item_indices": []},
        ], "log_fields": [], "follow_up": "", "issues": []}
        with patch.object(grounding, "call_json", return_value=plan):
            result = ai.generate_result("sleep", self.sleep_matches)
        self.assertEqual(result["unknown_citation_ids"], ["sleep_99"])
        self.assertIn("sleep_99", result["citation_warning"])
        self.assertIn("Practical sleep habits", result["evidence_used"])
        self.assertNotIn("Invented advice", result["answer"])
        self.assertNotIn("grounding_passed", result)

    def test_safety_and_normal_results_have_same_fields(self):
        reviewed = {"answer": "Advice [sleep_02]", "claims": [{"source_id": "sleep_02"}],
                    "unknown_ids": [], "mechanical_issues": [], "review_issues": [],
                    "support_status": "model_reviewed"}
        with patch.object(ai, "draft_and_review", return_value=reviewed) as generate:
            normal = ai.generate_result("sleep", self.sleep_matches)
            generate.reset_mock()
            safety = ai.generate_checked_result(
                "confirmed loss", [], True, documents=self.documents,
            )
            generate.assert_not_called()
        self.assertEqual(set(normal), set(safety))
        self.assertEqual(safety["retrieved_ids"], [])
        self.assertEqual(safety["safety_source_ids"], ["weight_04"])
        self.assertIn("overdue", safety["source_footer"])
        self.assertTrue(safety["source_footer"].startswith("### Safety source"))

    def test_safety_routes_only_by_explicit_confirmation(self):
        with patch.object(ai, "generate_result", return_value={"response_mode": "normal"}) as normal:
            for question in ["protein foods", "not losing weight", "what is weight loss"]:
                self.assertEqual(ai.generate_checked_result(question, []), {"response_mode": "normal"})
            self.assertEqual(normal.call_count, 3)
        with self.assertRaises(ValueError):
            ai.generate_checked_result("loss", [], "False")

    def test_safety_missing_passage_fails_without_generating(self):
        with patch.object(ai, "generate_answer") as generate:
            with self.assertRaisesRegex(ValueError, "weight_04"):
                ai.generate_checked_result("loss", [], True, documents=[])
            generate.assert_not_called()

    def test_wrapper_forwards_summary_and_backend_options(self):
        with patch.object(ai, "generate_result") as normal:
            ai.generate_checked_result("summary", [], log_summary={"test": 1}, use_local=True)
        normal.assert_called_once_with("summary", [], log_summary={"test": 1}, history=None, use_local=True)


if __name__ == "__main__":
    unittest.main()
