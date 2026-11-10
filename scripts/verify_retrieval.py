"""Opt-in MiniLM/Chroma integration check using public passages only.

Seed may download the public embedding model. Reopen does not rebuild the index.
Run each phase in a separate container; use --network none and HF offline flags
for reopen to verify the mounted cache and persisted index. No LLM or health DB.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import rag

INDEX_DIR = rag.CHROMA_DIR.parent / "retrieval_verification"
COLLECTION = "nutrirecall_verification_v1"
QUERIES = [
    ("What foods provide protein without powder?", "protein_03"),
    ("What habits could help me sleep better?", "sleep_02"),
    ("How do I repair a spacecraft propulsion engine?", None),
]


def check(phase, index_dir=INDEX_DIR):
    index_dir = Path(index_dir)
    if index_dir.resolve() == rag.CHROMA_DIR.resolve():
        raise ValueError("Verification must not use the application's knowledge index.")
    marker = index_dir / "verification_manifest.json"
    if phase not in {"seed", "reopen"}:
        raise ValueError("Choose seed or reopen.")
    if phase == "seed" and marker.exists():
        raise ValueError("Verification index already seeded; use reopen. No files removed.")
    if phase == "reopen" and not marker.is_file():
        raise ValueError("No seed manifest found on this volume; run seed first.")

    passages = rag.load_passages()
    docs = rag.prep_documents(passages)
    ids = sorted(doc.metadata["id"] for doc in docs)
    if len(ids) != 16:
        raise ValueError("This verification expects the curated 16-passage knowledge base.")
    identity = {"protocol": "retrieval_verification_v1", "model": rag.EMBEDDING_MODEL,
                "kb_sha256": hashlib.sha256(rag.KB_PATH.read_bytes()).hexdigest(), "ids": ids}
    previous = None
    if phase == "reopen":
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Seed identity differs from this model/KB; no index was rebuilt.")

    embeddings = rag.get_embeddings_function(device="cpu")
    vector = embeddings.embed_query("Public nutrition knowledge verification")
    if len(vector) != 384 or not all(math.isfinite(v) for v in vector):
        raise ValueError("Expected 384 finite embedding dimensions.")
    if not math.isclose(sum(v * v for v in vector), 1.0, abs_tol=1e-4):
        raise ValueError("Expected normalized embeddings.")

    if phase == "seed":
        store = rag.build_vector_store(docs, embeddings, persist_directory=index_dir,
                                      collection_name=COLLECTION)
    else:
        store = rag.open_vector_store(embeddings, persist_directory=index_dir,
                                     collection_name=COLLECTION)
    saved = store.get(include=["documents", "metadatas"])
    if sorted(saved["ids"]) != ids:
        raise ValueError("Persisted passage IDs differ from the 16 expected IDs.")
    by_id = {doc.metadata["id"]: doc for doc in docs}
    for passage_id, text, metadata in zip(saved["ids"], saved["documents"], saved["metadatas"]):
        expected = by_id[passage_id]
        if text != expected.page_content or metadata != expected.metadata:
            raise ValueError("Persisted passage text or source metadata differs from the KB.")

    results = []
    for question, expected_id in QUERIES:
        matches = rag.retrieve_relevant_passages(question, store)
        retrieved = [doc.metadata["id"] for doc, _ in matches]
        if (expected_id is None and matches) or (expected_id and expected_id not in retrieved):
            raise ValueError("Retrieval smoke check failed for a public test question: " + question)
        results.append({"question": question, "ids": retrieved,
                        "distances": [float(distance) for _, distance in matches]})
    if previous is not None:
        for current, old in zip(results, previous["results"]):
            if current["ids"] != old["ids"] or len(current["distances"]) != len(old["distances"]):
                raise ValueError("Reopened index returned different retrieval IDs.")
            if any(not math.isclose(a, b, abs_tol=1e-5) for a, b in
                   zip(current["distances"], old["distances"])):
                raise ValueError("Reopened index returned different retrieval distances.")
    else:
        index_dir.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as file:
            json.dump({**identity, "results": results}, file, indent=2, allow_nan=False)

    return {"phase": phase, "device": "cpu", "vector_dimensions": len(vector),
            "passages": len(ids), "rebuilt_index": phase == "seed", "results": results,
            "scope": "Public retrieval smoke checks only; no health logs or LLM calls."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["seed", "reopen"])
    args = parser.parse_args()
    try:
        result = check(args.phase)
    except Exception as error:
        # Do not print raw provider bodies, local profiles or credential-bearing errors.
        print(f"Verification stopped ({type(error).__name__}). No automatic retry.", file=sys.stderr)
        if isinstance(error, ValueError):
            print(str(error), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
