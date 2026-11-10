"""Offline checks of integration-check safeguards; not proof of real retrieval."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import verify_retrieval as verification


class RetrievalVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = Path(self.temp.name) / "public_index"
        self.docs = verification.rag.prep_documents(verification.rag.load_passages())
        self.by_id = {doc.metadata["id"]: doc for doc in self.docs}
        saved = {"ids": [doc.metadata["id"] for doc in self.docs],
                 "documents": [doc.page_content for doc in self.docs],
                 "metadatas": [doc.metadata for doc in self.docs]}
        self.store = Mock()
        self.store.get.return_value = saved
        self.embedding = Mock()
        self.embedding.embed_query.return_value = [1.] + [0.] * 383
        self.patches = {}
        for name, kwargs in {
            "get_embeddings_function": {"return_value": self.embedding},
            "build_vector_store": {"return_value": self.store},
            "open_vector_store": {"return_value": self.store},
            "retrieve_relevant_passages": {"side_effect": self.retrieve},
        }.items():
            patcher = patch.object(verification.rag, name, **kwargs)
            self.patches[name] = patcher.start()
            self.addCleanup(patcher.stop)

    def retrieve(self, question, store):
        expected = next(pid for query, pid in verification.QUERIES if query == question)
        return [(self.by_id[expected], .2)] if expected else []

    def test_reopen_preserves_evidence_and_never_rebuilds(self):
        seed = verification.check("seed", self.index)
        self.patches["build_vector_store"].reset_mock()
        reopened = verification.check("reopen", self.index)
        self.assertEqual(seed["results"], reopened["results"])
        self.assertFalse(reopened["rebuilt_index"])
        self.patches["build_vector_store"].assert_not_called()
        self.patches["open_vector_store"].assert_called_once()

    def test_refuses_app_index_duplicate_seed_and_missing_manifest(self):
        with self.assertRaises(ValueError):
            verification.check("seed", verification.rag.CHROMA_DIR)
        with self.assertRaises(ValueError):
            verification.check("reopen", self.index)
        self.patches["get_embeddings_function"].assert_not_called()
        verification.check("seed", self.index)
        self.patches["get_embeddings_function"].reset_mock()
        with self.assertRaises(ValueError):
            verification.check("seed", self.index)
        self.patches["get_embeddings_function"].assert_not_called()

    def test_rejects_bad_embedding_and_does_not_build(self):
        self.embedding.embed_query.return_value = [1., 0.]
        with self.assertRaises(ValueError):
            verification.check("seed", self.index)
        self.patches["build_vector_store"].assert_not_called()

    def test_reopen_detects_changed_source_metadata(self):
        verification.check("seed", self.index)
        saved = copy.deepcopy(self.store.get.return_value)
        saved["metadatas"][0]["source_url"] = "https://example.invalid/changed"
        self.store.get.return_value = saved
        with self.assertRaises(ValueError):
            verification.check("reopen", self.index)


if __name__ == "__main__":
    unittest.main()
