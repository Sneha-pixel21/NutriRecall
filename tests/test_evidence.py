"""Contract and rendering checks. These do not measure live model relevance."""

import unittest
from unittest.mock import patch

from utils import ai, grounding as g, rag
from utils.evidence import build_units, model_packet, render_selections


def selection(source_id, unit_index=0, items=None):
    return {"source_id": source_id, "unit_index": unit_index, "item_indices": items or []}


def plan(*choices, intent="knowledge"):
    return {"intent": intent, "claims": list(choices), "log_fields": [], "follow_up": "", "issues": []}


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = rag.prep_documents(rag.load_passages())
        cls.matches = [(doc, 0.3) for doc in cls.docs]
        cls.sources = build_units(rag.evidence_packet(cls.matches))

    def result(self, *choices, question="Food examples?"):
        with patch.object(g, "call_json", return_value=plan(*choices)) as call:
            result = ai.generate_result(question, self.matches)
        self.assertEqual(call.call_count, 1)
        return result

    def test_plant_subset_keeps_supplement_qualification_without_animal_foods(self):
        result = self.result(selection("protein_03", 0, [5, 6]),
                             selection("weight_02", 2, [0, 1]), question="What about plant options?")
        answer = result["answer"]
        self.assertIn("Protein food examples: beans and nuts.", answer)
        self.assertIn("Protein food examples: beans and pulses.", answer)
        self.assertIn("not automatically necessary", answer)
        self.assertIn("Adults for whom gaining weight is appropriate", answer)
        self.assertIn("allergies", answer)
        for word in ("eggs", "dairy", "cheese", "fish", "lean meat", "Smaller"):
            self.assertNotIn(word, answer)
        self.assertIn("overdue", result["source_review_status"])
        self.assertIn("Recheck before release", result["source_footer"])
        self.assertIn("Nuts, seeds and cheese can add calories", result["evidence_used"])
        self.assertIn("selected_evidence", result["grounding"])
        self.assertNotIn("claims", result["grounding"])

    def test_calorie_relationship_cannot_be_relabelled_as_protein(self):
        result = self.result(selection("weight_02", 1, [0, 1, 2]))
        self.assertIn("Calorie additions: Nuts, seeds and cheese.", result["answer"])
        self.assertNotIn("Protein food examples", result["answer"])
        self.assertEqual(result["grounding"]["selected_evidence"][0]["text"],
                         "Calorie additions: Nuts, seeds and cheese.")

    def test_nonfood_context_preserves_uncertainty_and_population(self):
        result = self.result(selection("protein_04", 0))
        self.assertIn("Evidence on narrowly timed", result["answer"])
        self.assertIn("is mixed", result["answer"])
        self.assertIn("does not establish failed recovery", result["answer"])
        self.assertIn("Athletes and resistance-training adults", result["answer"])

    def test_missing_or_out_of_range_indices_are_not_rendered(self):
        for choice in (selection("protein_03", 999), selection("protein_03", 0, [99]),
                       selection("protein_04", 0, [0])):
            with self.subTest(choice=choice):
                result = self.result(choice)
                self.assertEqual(result["cited_ids"], [])
                self.assertTrue(result["grounding"]["provenance_issues"])
                # A valid source ID does not establish a valid evidence selection.
                self.assertTrue(result["grounding"]["citation_ids_valid"])

    def test_indices_are_integers_not_booleans_or_negative_offsets(self):
        for choice in (selection("protein_03", True), selection("protein_03", -1),
                       selection("protein_03", 0, [False]), selection("protein_03", 0, [-1]),
                       selection("protein_03", "0")):
            with self.subTest(choice=choice), self.assertRaises(ValueError):
                g.validate_plan(plan(choice))

    def test_written_claim_or_quote_is_not_an_allowed_selection_field(self):
        for field in ("text", "quote"):
            choice = {**selection("protein_03"), field: "Seeds provide protein and cure disease."}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Invalid claim fields"):
                g.validate_plan(plan(choice))

    def test_unknown_source_diagnostic_does_not_echo_arbitrary_strings(self):
        result = self.result(selection("gsk_private_org_private"))
        self.assertNotIn("gsk_private", str(result))
        self.assertFalse(result["grounding"]["citation_ids_valid"])

    def test_medical_boundary_never_renders_selected_athlete_target(self):
        with patch.object(g, "call_json", return_value=plan(selection("protein_02"), intent="medical_boundary")):
            result = ai.generate_result("Prescribe protein for my kidney disease at 70 kg.", self.matches)
        self.assertNotIn("84", result["answer"])
        self.assertNotIn("1.2", result["answer"])
        self.assertIn("Do not use as a disease-specific diet", result["answer"])
        self.assertIn("registered dietitian", result["answer"])

    def test_empty_items_select_full_list_and_duplicates_do_not_repeat_food(self):
        accepted, issues, _ = g.supported_claims(plan(selection("protein_03")), self.sources)
        self.assertEqual(accepted[0]["item_indices"], list(range(7)))
        accepted, issues, _ = g.supported_claims(plan(selection("protein_03", 0, [5, 5])), self.sources)
        lines = render_selections(accepted, self.sources, "knowledge")
        self.assertEqual(lines[0], "- Protein food examples: beans. [protein_03]")
        self.assertEqual(issues, [])

    def test_provider_contract_has_no_generated_health_text_channel(self):
        packet = model_packet(self.sources)
        self.assertNotIn("text", packet["protein_03"])
        self.assertEqual(packet["protein_03"]["units"][0]["items"]["5"], "beans")
        self.assertEqual(set(g.CLAIM_SCHEMA["properties"]), {"source_id", "unit_index", "item_indices"})


if __name__ == "__main__":
    unittest.main()
