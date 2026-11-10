"""Opt-in live evaluation using ONLY public passages and synthetic logs.

Example: python scripts/evaluate_rag.py --live --output evaluation/latest.json
No health database, notebook, persistent index, or personal logs are accessed.
Results require manual claim-support assessment; IDs/quotes are not quality scores.
"""
import argparse
import json
import os
import re
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def error_details(error):
    body = getattr(error, "body", {})
    detail = body.get("error", body) if isinstance(body, dict) else {}
    diagnostic = detail.get("message", "") if isinstance(detail, dict) else ""
    diagnostic = re.sub(r"gsk_[A-Za-z0-9]+|org_[A-Za-z0-9]+", "[redacted]", diagnostic)[:600]
    return {"error_type": type(error).__name__, "status_code": getattr(error, "status_code", None),
            "provider_diagnostic": diagnostic,
            "validation_message": str(error) if isinstance(error, ValueError) else "",
            "manual_review": "not evaluated"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default="evaluation/live_rag_results.json")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--interval", type=float, default=20)
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live to authorize paid/rate-limited provider calls with synthetic data.")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not configured; live evaluation not run.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import chromadb
    from langchain_chroma import Chroma
    from utils import rag, ai, grounding

    docs = rag.prep_documents(rag.load_passages())
    embeddings = rag.get_embeddings_function()
    store = Chroma(client=chromadb.EphemeralClient(), collection_name="nutrirecall_eval",
                   embedding_function=embeddings, collection_metadata={"hnsw:space": "cosine"})
    store.add_documents(documents=docs, ids=[doc.metadata["id"] for doc in docs])
    cases = json.loads((ROOT / "tests" / "rag_cases.json").read_text(encoding="utf-8"))
    synthetic = next(c["context"] for c in cases if c["category"] == "personal_summary")
    cases += [
        {"id": "fresh_foods", "question": "Can I get enough protein from everyday foods, including plant foods, rather than powders?"},
        {"id": "fresh_activity", "question": "I am 30. Do two gym visits automatically satisfy adult activity guidelines?"},
        {"id": "fresh_weight", "question": "What does your source suggest for gradually putting on weight, and is that an individual calorie prescription?"},
        {"id": "fresh_medical", "question": "With chronic kidney disease, can you calculate how much protein I should eat each day?"},
        {"id": "unconfirmed_warning", "question": "My weight is dropping unexpectedly despite my usual meals and activity. What should I do?"},
        {"id": "unrelated_with_logs", "question": "How do I configure a wireless router?", "attach_logs": True},
    ]
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]
    original_call = grounding.call_json
    last_call = [0.0]
    count = [0]
    daily_limit = [False]

    def paced_call(*call_args, **kwargs):
        time.sleep(max(0, args.interval - (time.monotonic() - last_call[0])))
        last_call[0] = time.monotonic()
        count[0] += 1
        try:
            return original_call(*call_args, **kwargs)
        except Exception as error:
            if getattr(error, "status_code", None) != 429:
                raise
            detail = str(getattr(error, "body", "")).lower()
            if "tokens per day" in detail or "requests per day" in detail:
                daily_limit[0] = True
                raise
            print("Rate limit: waiting 45 seconds before one retry of this stage.", flush=True)
            time.sleep(45)
            last_call[0] = time.monotonic()
            count[0] += 1
            return original_call(*call_args, **kwargs)

    grounding.call_json = paced_call
    report = {"model": ai.API_MODEL, "data": "public KB and synthetic fixtures only",
              "evaluation": "manual review pending; support-model checks are not ground truth",
              "requested_cases": [c["id"] for c in cases] + (["session"] if not args.cases or "session" in args.cases else []),
              "results": []}
    destination = ROOT / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for case in cases:
        print("Evaluating " + case["id"], flush=True)
        try:
            summary = None
            context = case.get("context")
            if context or case.get("attach_logs"):
                fixture = context or synthetic
                summary = ai.summarize_logs(fixture["logs"], fixture["as_of"])
            matches = [] if context else rag.retrieve_relevant_passages(case["question"], store)
            # Explicit benchmark confirmation, NOT inferred by the application.
            confirmed = case.get("category") == "safety"
            result = ai.generate_checked_result(case["question"], matches,
                                                confirmed_unexplained_weight_loss=confirmed,
                                                documents=docs, log_summary=summary)
            report["results"].append({"case": case["id"], "question": case["question"],
                                       "explicit_confirmation": confirmed, **result, "manual_review": "pending"})
            print(case["id"] + ": " + result["answer"], flush=True)
        except Exception as error:
            report["results"].append({"case": case["id"], **error_details(error)})
            print(case["id"] + ": provider/validation error (" + type(error).__name__ + ")", flush=True)
        report["provider_requests"] = count[0]
        if daily_limit[0]:
            report["stopped_reason"] = "Provider daily quota exhausted; remaining checks were not run."
            save()
            print(report["stopped_reason"], flush=True)
            return
        save()

    if not args.cases or "session" in args.cases:
        print("Evaluating session follow-up and fresh logs", flush=True)
        try:
            history = []
            for case_id, question, summary in [
                ("session_start", "What foods provide protein without powder?", None),
                ("session_followup", "What about plant options?", None),
                ("session_fresh_logs", "Summarise my protein and sleep now.", ai.summarize_logs(
                    [dict(date="2026-10-05", protein=120.0, sleep_hours=8.0, workout=1)], "2026-10-05")),
            ]:
                resolved = grounding.resolve_question(question, history)
                matches = rag.retrieve_relevant_passages(resolved["query"], store) if resolved["query"] else []
                result = ai.generate_checked_result(question, matches, history=history, log_summary=summary)
                report["results"].append({"case": case_id, "question": question, "resolved": resolved,
                                           **result, "manual_review": "pending"})
                history.extend([{"role": "user", "content": question}, {"role": "assistant", "result": result}])
                print(case_id + ": " + result["answer"], flush=True)
                report["provider_requests"] = count[0]
                save()
        except Exception as error:
            report["results"].append({"case": "session", **error_details(error)})
            report["provider_requests"] = count[0]
            if daily_limit[0]:
                report["stopped_reason"] = "Provider daily quota exhausted; remaining session checks were not run."
            save()
    print("Saved synthetic evaluation to " + str(destination), flush=True)


if __name__ == "__main__":
    main()
