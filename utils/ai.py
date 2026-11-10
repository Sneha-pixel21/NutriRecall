"""
NutriRecall AI pipeline
Source-backed Streamlit generation plus legacy notebook-compatible helpers.
The app uses generate_checked_result; legacy TF-IDF/memory helpers are retained
for compatibility but are not called by the current Streamlit AI page.
"""

import os
import json
import math
from datetime import date, timedelta
from statistics import mean
from utils.grounding import SYSTEM_PROMPT, draft_and_review, bounded_history
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

from utils.rag import (
    build_context, build_source_footer, build_evidence_used,
    load_passages, prep_documents, build_source_review_status,
)

# ─── Knowledge base ───────────────────────────────────────────────────────────

KNOWLEDGE = [
    "Protein intake for muscle growth should be 1.6 to 2.2 grams per kg body weight. Optimal per meal: 0.3–0.5 g/kg.",
    "Sleeping less than 6 hours significantly reduces recovery and muscle protein synthesis.",
    "Consistent resistance training at least 3–4 times per week is required to build muscle.",
    "Low protein intake is the most common reason people fail to gain muscle despite training.",
    "High-quality protein sources: eggs, chicken, fish, dairy (curd, paneer, milk), lentils, chickpeas, tofu.",
    "Ideal sleep for recovery: 7–9 hours per night.",
    "If protein intake is below 1.6 g/kg body weight, muscle gain will be limited regardless of training.",
    "If sleep is below 6 hours, warn about significant recovery drop and cortisol spike.",
    "Recovery days are as important as training days — muscles grow during rest, not during the workout.",
    "Caloric surplus of 200–300 kcal above maintenance is needed for lean muscle gain.",
    "Dehydration of even 2% body weight reduces performance and recovery quality.",
    "Creatine monohydrate 3–5g/day is one of the most evidence-backed supplements for strength and muscle.",
]

_vectorizer = TfidfVectorizer()
_knowledge_matrix = _vectorizer.fit_transform(KNOWLEDGE)


def retrieve_knowledge(query: str, top_k: int = 2) -> str:
    query_vec = _vectorizer.transform([query])
    scores = cosine_similarity(query_vec, _knowledge_matrix)[0]
    top_indices = scores.argsort()[-top_k:][::-1]
    return "\n".join(KNOWLEDGE[i] for i in top_indices)


# ─── ChromaDB persistent memory ───────────────────────────────────────────────

def _get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="NutriRecall_Memory")


def store_memory(insight_text: str, memory_id: str):
    try:
        col = _get_collection()
        col.upsert(documents=[insight_text], ids=[memory_id])
    except Exception:
        pass


def retrieve_memory(query: str, n_results: int = 2) -> str:
    try:
        col = _get_collection()
        count = col.count()
        if count == 0:
            return "No past memory stored yet."
        results = col.query(query_texts=[query], n_results=min(n_results, count))
        docs = results["documents"][0]
        return "\n".join(docs)
    except Exception:
        return "Memory unavailable."


# ─── Health scoring ───────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["protein_required"] = df["weight"] * 1.6
    df["protein_score"] = (df["protein"] / df["protein_required"]).clip(0, 1)
    df["sleep_score"] = (df["sleep_hours"] / 8).clip(0, 1)
    df["workout_score"] = df["workout"].astype(float)
    df["health_score"] = (
        0.4 * df["protein_score"]
        + 0.3 * df["sleep_score"]
        + 0.3 * df["workout_score"]
    )
    df["health_score_10"] = (df["health_score"] * 9) + 1
    return df


def detect_patterns(df: pd.DataFrame) -> list:
    recent = df.tail(7)
    issues = []
    if recent["protein"].mean() < recent["protein_required"].mean():
        issues.append("Consistently low protein intake")
    if recent["sleep_hours"].mean() < 6:
        issues.append("Poor sleep pattern (avg < 6 hrs)")
    if recent["workout"].sum() < 3:
        issues.append("Low workout consistency (< 3 days/week)")
    if recent["health_score"].mean() < 0.6:
        issues.append("Overall lifestyle needs improvement")
    return issues


def generate_insights(df: pd.DataFrame) -> str:
    recent = df.tail(7)
    issues = detect_patterns(df)
    avg_protein = recent["protein"].mean()
    avg_req = recent["protein_required"].mean()
    avg_sleep = recent["sleep_hours"].mean()
    workout_days = int(recent["workout"].sum())
    health = recent["health_score_10"].mean()

    return f"""Last 7 Days Summary:
Avg Protein: {avg_protein:.1f}g (target: {avg_req:.1f}g)
Avg Sleep: {avg_sleep:.1f} hrs
Workout Days: {workout_days}/7
Health Score: {health:.1f}/10

Key Issues: {", ".join(issues) if issues else "No major issues — keep it up!"}"""


# ─── LLM call ─────────────────────────────────────────────────────────────────

def generate_response(prompt: str, use_local: bool = False, groq_api_key: str = "") -> str:
    if use_local:
        try:
            import ollama
            response = ollama.chat(
                model="phi3:mini",
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Ollama error: {e}. Is Ollama running locally?"
    else:
        try:
            from openai import OpenAI
            key = groq_api_key or os.getenv("GROQ_API_KEY", "")
            if not key:
                return "No GROQ_API_KEY found. Add it to your .env file or paste it in the sidebar."
            client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
            response = client.chat.completions.create(
                #model="llama-3.1-8b-instant",
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Groq API error: {e}"


def nutrirecall_ai(query: str, df: pd.DataFrame, use_local: bool = False, groq_api_key: str = "") -> str:
    insight = generate_insights(df)
    knowledge = retrieve_knowledge(query)
    memory = retrieve_memory(query)

    prompt = f"""You are a smart fitness and nutrition assistant. Be direct, practical, and evidence-based.

User's recent data:
{insight}

Relevant nutrition/fitness knowledge:
{knowledge}

Past context from memory:
{memory}

User question: {query}

Give a short, personalized answer in 3-5 bullet points. Be specific to their numbers."""

    store_memory(insight, memory_id="latest_insight")
    return generate_response(prompt, use_local=use_local, groq_api_key=groq_api_key)

# ─── Experimental source-backed pipeline (used by Streamlit) ───
# Legacy entry points above remain available for compatibility only.


API_MODEL = "openai/gpt-oss-20b"
LOCAL_MODEL = "phi3:mini"


def build_user_message(question, matches, log_summary=None, history=None):
    return json.dumps(
        {
            "question": question,
            "personal_log_summary": log_summary,
            "knowledge_context": build_context(matches),
            "conversation_context": bounded_history(history),
        },
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )


def summarize_logs(logs, as_of):
    """Summarize validated daily records with ISO dates and an explicit end date.

    Callers convert database date objects to ISO strings before calling. Missing
    days are not imputed; recorded zero protein/sleep values remain in means.
    """
    end = date.fromisoformat(as_of)
    start = end - timedelta(days=6)

    recent = sorted(
        [
            row for row in logs
            if start <= date.fromisoformat(row["date"]) <= end
        ],
        key=lambda row: row["date"],
    )

    logged_dates = [row["date"] for row in recent]

    if len(logged_dates) != len(set(logged_dates)):
        raise ValueError("Multiple entries for the same date.")

    for row in recent:
        for key in ("protein", "sleep_hours", "workout"):
            value = row[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Daily values must be finite and non-negative.")
        if row["sleep_hours"] > 24 or row["workout"] not in (0, 1):
            raise ValueError("Invalid sleep hours or workout flag.")
    weights = [row["weight"] for row in recent if row.get("weight") is not None]
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Recorded weights must be finite and positive.")

    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "logged_dates": logged_dates,
        "logged_days": len(recent),
        "window_days": 7,
        "missing_days": 7 - len(recent),
        "avg_protein_g": (
            mean(row["protein"] for row in recent) if recent else None
        ),
        "avg_sleep_hours": (
            mean(row["sleep_hours"] for row in recent) if recent else None
        ),
        "workout_days": (
            sum(row["workout"] for row in recent) if recent else None
        ),
        "avg_weight_kg": mean(weights) if weights else None,
        "weight_observations": len(weights),
        "average_basis": "Logged days only; missing days are unknown.",
    }


def build_log_header(log_summary):
    if log_summary is None:
        return ""

    dates = ", ".join(log_summary["logged_dates"]) or "None"

    return (
        "### Personal log coverage\n"
        f"- Reporting window: {log_summary['window_start']} "
        f"to {log_summary['window_end']} (inclusive)\n"
        f"- Logged dates: {dates}\n"
        f"- Coverage: {log_summary['logged_days']}/"
        f"{log_summary['window_days']} days logged; "
        f"{log_summary['missing_days']} {'day' if log_summary['missing_days'] == 1 else 'days'} missing\n"
        "- Averages use logged days only; missing days are unknown."
    )


def generate_answer(question, matches, log_summary=None, history=None, **model_options):
    """Notebook-compatible interface; single selection unless review is explicit."""
    return draft_and_review(
        question, matches, log_summary=log_summary, history=history, **model_options,
    )["answer"]


def generate_result(question, matches, log_summary=None, history=None, **model_options):
    reviewed = draft_and_review(
        question, matches, log_summary=log_summary, history=history, **model_options,
    )
    if reviewed["support_status"] == "deterministic_log_summary":
        # Knowledge retrieval is not evidence for personal arithmetic or dates.
        matches = []
    cited_ids = sorted({claim["source_id"] for claim in reviewed["claims"]})
    unknown_ids = reviewed["unknown_ids"]
    usage = reviewed.get("log_usage", {
        "summary_available": log_summary is not None,
        "fresh_summary_sent_to_model": None, "rendered_fields": None,
    })
    show_coverage = (usage["rendered_fields"] is None or bool(usage["rendered_fields"])
                     or usage["fresh_summary_sent_to_model"] is True)
    return {
        "answer": reviewed["answer"],
        "log_header": build_log_header(log_summary) if show_coverage else "",
        "log_usage": usage,
        "source_footer": build_source_footer(matches),
        "source_review_status": build_source_review_status(matches, cited_ids),
        "retrieved_ids": sorted({doc.metadata["id"] for doc, _ in matches}),
        "cited_ids": cited_ids,
        "unknown_citation_ids": unknown_ids,
        "citation_warning": (
            "Removed draft references to unavailable passages: " + ", ".join(unknown_ids) + "."
            if unknown_ids else ""
        ),
        "response_mode": "normal",
        "evidence_used": build_evidence_used(matches, cited_ids),
        "safety_source_ids": [],
        "grounding": {
            "citation_ids_valid": not unknown_ids,
            "provenance_issues": reviewed["mechanical_issues"],
            "support_review": reviewed["support_status"],
            "review_issues": reviewed["review_issues"],
            "model_issues": reviewed.get("model_issues", []),
            "model_issue_stage": reviewed.get("model_issue_stage"),
            "rendered_log_field_status": reviewed.get("rendered_log_field_status", {}),
            "selected_evidence": reviewed["claims"],
            "selection_contract": reviewed.get("selection_contract", "legacy_or_no_model"),
            "log_usage": usage,
        },
    }


def generate_checked_result(question, matches,
                            confirmed_unexplained_weight_loss=False,
                            log_summary=None, *, documents=None, history=None,
                            **model_options):
    """Explicit one-off confirmation only; never a keyword medical classifier."""
    if not isinstance(confirmed_unexplained_weight_loss, bool):
        raise ValueError("Safety confirmation must be a boolean.")
    if not confirmed_unexplained_weight_loss:
        return generate_result(
            question, matches, log_summary=log_summary, history=history, **model_options,
        )
    if documents is None:
        documents = prep_documents(load_passages())
    warning = next((doc for doc in documents if doc.metadata["id"] == "weight_04"), None)
    if warning is None:
        raise ValueError("Required safety passage weight_04 is missing.")
    guidance = warning.page_content.split("\n\n", 1)[-1].partition("Source review warning:")[0].strip()
    warning_matches = [(warning, None)]
    return {
        "answer": f"{guidance} [weight_04]",
        "log_header": "",
        "log_usage": {"summary_available": log_summary is not None,
                      "fresh_summary_sent_to_model": False, "rendered_fields": []},
        "source_footer": build_source_footer(warning_matches, heading="### Safety source"),
        "source_review_status": build_source_review_status(warning_matches, ["weight_04"]),
        "retrieved_ids": sorted({doc.metadata["id"] for doc, _ in matches}),
        "cited_ids": ["weight_04"],
        "evidence_used": build_evidence_used(warning_matches, ["weight_04"]),
        "unknown_citation_ids": [],
        "response_mode": "confirmed_safety",
        "citation_warning": "",
        "safety_source_ids": ["weight_04"],
        "grounding": {"citation_ids_valid": True, "provenance_issues": [],
                      "support_review": "curated_text", "review_issues": [], "selected_evidence": [],
                      "log_usage": {"summary_available": log_summary is not None,
                                    "fresh_summary_sent_to_model": False, "rendered_fields": []}},
    }
