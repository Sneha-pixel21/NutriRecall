"""Experimental source-backed retrieval extracted from RAG_Development.ipynb.

No model is loaded and no index is opened at import time. Callers explicitly
initialize these resources. Distance defaults are provisional development settings,
not a confidence score or a guarantee that a passage supports an answer.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB_PATH = ROOT / "knowledge_base" / "nutrition_v1.json"
CHROMA_DIR = ROOT / "chroma_db" / "knowledge_minilm_v1"
COLLECTION_NAME = "neutricall_knowledge_v1"  # Match the existing notebook index.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_K = 6
MAX_DISTANCE = 0.60


def load_passages(path=KB_PATH):
    """Load curated passages, rejecting missing metadata and duplicate IDs."""
    with Path(path).open(encoding="utf-8") as file:
        passages = json.load(file)
    if not isinstance(passages, list) or not passages:
        raise ValueError("Knowledge base must be a non-empty array.")

    passage_fields = {"id", "topic", "title", "text", "source"}
    source_fields = {"name", "url", "section", "checked_on"}
    seen = set()
    for passage in passages:
        if not isinstance(passage, dict) or set(passage) != passage_fields:
            raise ValueError("Unexpected passage fields.")
        source = passage["source"]
        if not isinstance(source, dict) or set(source) != source_fields:
            raise ValueError("Unexpected source fields.")
        values = [passage[key] for key in passage_fields - {"source"}]
        values.extend(source.values())
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("Passage and source fields must be non-empty strings.")
        if passage["id"] in seen:
            raise ValueError("Duplicate passage IDs.")
        seen.add(passage["id"])
    return passages


def get_embeddings_function(device="cpu"):
    """First use may download MiniLM; subsequent embedding runs are local."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def prep_documents(passages):
    from langchain_core.documents import Document

    documents = []
    for passage in passages:
        source = passage["source"]
        documents.append(Document(
            page_content=f"{passage['title']}\n\n{passage['text']}",
            metadata={
                "id": passage["id"],
                "topic": passage["topic"],
                "title": passage["title"],
                "source_name": source["name"],
                "source_url": source["url"],
                "source_section": source["section"],
                "checked_on": source["checked_on"],
            },
        ))
    return documents


def open_vector_store(embeddings, persist_directory=CHROMA_DIR,
                      collection_name=COLLECTION_NAME):
    """Open/create only the dedicated knowledge store, not legacy memory."""
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vector_store(documents, embeddings, persist_directory=CHROMA_DIR,
                       collection_name=COLLECTION_NAME):
    """Explicitly upsert passages by stable ID. Removed IDs are not pruned."""
    if not documents:
        raise ValueError("Cannot build an index without documents.")
    ids = [doc.metadata["id"] for doc in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs.")
    db = open_vector_store(embeddings, persist_directory, collection_name)
    db.add_documents(documents=documents, ids=ids)
    return db


def retrieve_passages(query, vector_store, k=RETRIEVAL_K):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")
    if not query.strip():
        return []
    return vector_store.similarity_search_with_score(query, k=k)


def retrieve_relevant_passages(query, vector_store, k=RETRIEVAL_K,
                               max_distance=MAX_DISTANCE):
    if not 0 <= max_distance <= 2:
        raise ValueError("Cosine distance cutoff must be between 0 and 2.")
    return [
        (doc, distance)
        for doc, distance in retrieve_passages(query, vector_store, k=k)
        if distance <= max_distance
    ]


def build_context(matches):
    if not matches:
        return "No sufficiently relevant knowledge passages retrieved."
    sections = []
    for doc, _ in matches:
        metadata = doc.metadata
        sections.append(
            f"[{metadata['id']}]\n{doc.page_content}\n"
            f"Source: {metadata['source_name']}\n"
            f"Section: {metadata['source_section']}\n"
            f"URL: {metadata['source_url']}\n"
            f"Checked on: {metadata['checked_on']}"
        )
    return "\n\n---\n\n".join(sections)


def evidence_packet(matches):
    """Separate source scope from advice without changing the curated file."""
    packet = {}
    for doc, _ in matches:
        text = doc.page_content.split("\n\n", 1)[-1]
        packet[doc.metadata["id"]] = {
            "text": text,
            "population": text.partition("Applies to:")[2].partition("Limitations:")[0].strip(),
            "limitations": text.partition("Limitations:")[2].partition("Source review warning:")[0].strip(),
            "source_review_warning": text.partition("Source review warning:")[2].strip(),
        }
    return packet


def build_source_footer(matches, heading="### Retrieved sources"):
    """List retrieved sources, without implying their claims were verified."""
    if not matches:
        return ""
    lines = [heading]
    seen_ids = set()
    for doc, _ in matches:
        metadata = doc.metadata
        passage_id = metadata["id"]
        if passage_id in seen_ids:
            continue
        seen_ids.add(passage_id)
        lines.append(
            f"- [{passage_id}] "
            f"[{metadata['source_name']}]({metadata['source_url']})"
            f" — {metadata['source_section']}"
            f" — checked {metadata['checked_on']}"
        )
        _, marker, warning = doc.page_content.partition("Source review warning:")
        if marker:
            lines.append(f"  - Source review warning: {warning.strip()}")
    return "\n".join(lines)


def build_evidence_used(matches, cited_ids):
    """Show cited passages verbatim; this is not an entailment check."""
    cited_ids = set(cited_ids)
    sections = []
    seen = set()
    for doc, _ in matches:
        passage_id = doc.metadata["id"]
        if passage_id in cited_ids and passage_id not in seen:
            seen.add(passage_id)
            sections.append(f"#### [{passage_id}]\n\n{doc.page_content}")
    return "\n\n".join(sections)


def build_citation_warning(unknown_ids):
    if not unknown_ids:
        return ""
    return (
        "Citation error: the answer references passages that were not "
        "supplied as evidence: " + ", ".join(unknown_ids)
        + ". Do not treat those references as verified sources."
    )


def build_source_review_status(matches, cited_ids):
    """Compact source facts, independent of whether a model selected a warning.

    Keep the full warning in the expandable footer; deduplicate identical review
    metadata so several passages from one page do not repeat a warning paragraph.
    """
    notes, seen = [], set()
    for doc, _ in matches:
        if doc.metadata["id"] not in cited_ids:
            continue
        warning = doc.page_content.partition("Source review warning:")[2].strip()
        if not warning or warning in seen:
            continue
        seen.add(warning)
        status = warning.partition("Recheck before release;")[0].strip() or warning
        notes.append(f"Source review: {status} [{doc.metadata['id']}]")
    return "\n".join(notes)
