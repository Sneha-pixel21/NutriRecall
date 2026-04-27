"""
NutriRecall AI pipeline
Handles: scoring, insights, RAG knowledge retrieval, ChromaDB memory, LLM calls
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

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
                model="llama-3.1-8b-instant",
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