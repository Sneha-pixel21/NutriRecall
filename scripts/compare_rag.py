"""One-case, opt-in live comparison; never reads the health database.

Both candidates share retrieval and a fixed synthetic conversation. Resolution
is measured once as shared overhead. This is an ablation, not an end-to-end
session benchmark. Human review remains necessary. No automatic retry.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_cases():
    cases = json.loads((ROOT / "tests" / "rag_cases.json").read_text(encoding="utf-8"))
    fixture = next(case["context"] for case in cases if case["category"] == "personal_summary")
    history = [{"role": "user", "content": "What foods provide protein without powder?"},
               {"role": "assistant", "content": "Food options were discussed. [protein_03]"}]
    cases.extend([
        {"id": "plant_followup", "question": "What about plant options?", "history": history,
         "must_include": ["Beans, nuts or pulses with source support; focus on plant options"],
         "must_not_include": ["Animal-food recommendations", "Calories relabelled as protein"]},
        {"id": "new_topic_with_history", "question": "What habits could help me sleep better?", "history": history,
         "must_include": ["Sleep guidance, without importing the previous food topic"]},
        {"id": "ambiguous_followup", "question": "What about that?", "history": history,
         "must_include": ["Clarification if the reference cannot be established"]},
        {"id": "mixed_logs", "question": "How does my logged protein relate to the athlete guidance?",
         "context": fixture,
         "must_include": ["Fresh synthetic log values; preserve athlete population and limits"],
         "must_not_include": ["Universal personal protein prescription"]},
        {"id": "unrelated_with_logs", "question": "How do I configure a wireless router?", "context": fixture,
         "must_include": ["Abstention without displaying unrelated personal figures"]},
        {"id": "unconfirmed_warning", "question": "My weight is dropping unexpectedly despite my usual meals and activity. What should I do?",
         "must_include": ["Relevant warning and professional assessment; no inferred diagnosis"],
         "must_not_include": ["Protein or calories presented as sufficient treatment"]},
    ])
    return cases


def safe_error(error):
    from utils import grounding

    status = getattr(error, "status_code", None)
    if status == 429:
        message = grounding.describe_rate_limit(error, "Groq")
    elif isinstance(error, ValueError):
        message = grounding.describe_validation_error(error)
    else:
        message = "Provider or connection failure; no automatic retry."
    return {"status_code": status if type(status) is int else None, "diagnostic": message}


def compare_case(case, documents, retrieve, summary=None, *, first="single", **options):
    """Freeze inputs; keep errors and incomplete pairs instead of scoring them."""
    from utils import ai, grounding

    report = {"case": case["id"], "question": case["question"],
              "criteria": {key: case.get(key, []) for key in ("must_include", "must_not_include")},
              "manual_review": "pending", "shared_requests": [], "variants": []}
    question, history = case["question"], case.get("history", [])
    confirmed = case.get("category") == "safety"
    direct = grounding.log_summary_fields(question, history) is not None
    try:
        if confirmed or direct:
            matches = []
        else:
            resolved = grounding.resolve_question(question, history, request_metrics=report["shared_requests"], **options)
            report["resolved"] = resolved
            if not resolved["query"].strip():
                report["clarification"] = resolved["clarification"]
                report["comparison_status"] = "shared_clarification_only"
                return report
            matches = retrieve(resolved["query"])
        report["frozen_retrieved_ids"] = [doc.metadata["id"] for doc, _ in matches]
        for label in (["single", "review"] if first == "single" else ["review", "single"]):
            entry = {"variant": label, "requests": [], "manual_review": "pending"}
            report["variants"].append(entry)
            try:
                entry["result"] = ai.generate_checked_result(
                    question, matches, confirmed, summary, documents=documents, history=history,
                    review_evidence=label == "review", request_metrics=entry["requests"], **options,
                )
            except Exception as error:
                entry["error"] = safe_error(error)
                report["comparison_status"] = "incomplete_pair"
                return report  # Stop on failure, including 429/timeouts; no repair call.
        report["comparison_status"] = "awaiting_human_review"
    except Exception as error:
        report["error"] = safe_error(error)
        report["comparison_status"] = "preparation_failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--case", choices=[case["id"] for case in load_cases()])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--first", choices=["single", "review"], default="single")
    parser.add_argument("--interval", type=float, default=20, help="Seconds between stages, 0–60; no retries")
    args = parser.parse_args()
    if not args.live or not args.case or args.output is None:
        parser.error("Provide --live, one --case and a new --output path to authorize provider calls.")
    if not 0 <= args.interval <= 60:
        parser.error("--interval must be between 0 and 60 seconds.")
    destination = ROOT / args.output
    if destination.exists():
        parser.error("Output already exists; choose a new path to preserve previous results.")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY", "").strip():
        parser.error("GROQ_API_KEY is missing; comparison not run.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from utils import ai, rag, grounding
    from unittest.mock import patch

    case = next(case for case in load_cases() if case["id"] == args.case)
    summary = ai.summarize_logs(case["context"]["logs"], case["context"]["as_of"]) if case.get("context") else None
    docs = rag.prep_documents(rag.load_passages())
    # Retrieval initializes only for a case that reaches it; no persistent index.
    def retrieve(query):
        import chromadb
        from langchain_chroma import Chroma
        embeddings = rag.get_embeddings_function()
        store = Chroma(client=chromadb.EphemeralClient(), collection_name="nutrirecall_comparison",
                       embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})
        store.add_documents(docs, ids=[doc.metadata["id"] for doc in docs])
        return rag.retrieve_relevant_passages(query, store)

    original_call, last_call = grounding.call_json, [None]
    def paced_call(*call_args, **kwargs):
        if last_call[0] is not None:
            time.sleep(max(0, args.interval - (time.monotonic() - last_call[0])))
        last_call[0] = time.monotonic()
        return original_call(*call_args, **kwargs)

    report = {"model": ai.API_MODEL, "data": "public KB and synthetic fixtures only",
              "scope": "Frozen-input selection ablation; shared resolution is not duplicated between variants.",
              "caveat": "Same-model review is not independent factual verification. No quality scores are automatic.",
              "pipeline_version": "single_default_v2",
              "candidate_difference": "Single selection sees field names, not fresh values. Explicit review may see fresh values. Both now preserve explicitly requested record fields and coverage through code; earlier saved comparisons used the prior renderer. Reference resolution is gated for standalone questions.",
              "measurement": "Stage attempts and provider-reported tokens; unavailable counts are null. response_seconds measures client setup through receipt, excluding pacing; elapsed_seconds includes pacing and validation. Shared retrieval/initialization time is excluded.",
              "order": args.first, "manual_review": "pending"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    with patch.object(grounding, "call_json", side_effect=paced_call):
        report["comparison"] = compare_case(case, docs, retrieve, summary, first=args.first)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved comparison; manual review pending. No automatic pass/fail score.")


if __name__ == "__main__":
    main()
