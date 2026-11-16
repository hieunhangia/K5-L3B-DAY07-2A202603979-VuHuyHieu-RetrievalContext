"""Unit tests for the three retrieval methods: Dense, BM25, and Hybrid."""
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.retrieval import BM25Retriever, HybridRetriever, tokenize_vi


class MockDenseRetriever:
    """Mock dense retriever for fast unit testing."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, filters=None, top_k=3):
        # simple mock ranking
        res = []
        for c in self.chunks:
            if filters and filters.get("page") and c.get("page") != filters["page"]:
                continue
            item = dict(c)
            item["score"] = 0.9 if query.lower() in c["text"].lower() else 0.1
            res.append(item)
        res.sort(key=lambda x: -x["score"])
        return res[:top_k]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "c1",
                "text": "Sinh viên hoàn thành hai chương trình được cấp bằng kép nếu đủ điều kiện.",
                "source_id": "doc1",
                "source": "doc1.pdf",
                "page": 4,
                "metadata": {"section": "thuat_ngu"},
            },
            {
                "id": "c2",
                "text": "Ngôn ngữ giảng dạy và đánh giá chính thức tại trường là tiếng Anh.",
                "source_id": "doc1",
                "source": "doc1.pdf",
                "page": 7,
                "metadata": {"section": "chuong_1"},
            },
            {
                "id": "c3",
                "text": "Sinh viên rút hồ sơ tối đa 18 tín chỉ trong toàn bộ khóa học.",
                "source_id": "doc1",
                "source": "doc1.pdf",
                "page": 15,
                "metadata": {"section": "chuong_2"},
            },
        ]

    def test_tokenization_vietnamese(self):
        tokens = tokenize_vi("Trường Đại học VinUni! (Năm 2024)")
        self.assertIn("trường", tokens)
        self.assertIn("đại", tokens)
        self.assertIn("học", tokens)
        self.assertIn("vinuni", tokens)
        self.assertIn("2024", tokens)

    def test_bm25_retriever(self):
        bm25 = BM25Retriever(self.chunks)
        results = bm25.search("tiếng Anh giảng dạy", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "c2")
        self.assertIn("score", results[0])

    def test_bm25_metadata_filter(self):
        bm25 = BM25Retriever(self.chunks)
        results = bm25.search("sinh viên", filters={"section": "thuat_ngu"}, top_k=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "c1")

    def test_hybrid_retriever_rrf(self):
        dense_mock = MockDenseRetriever(self.chunks)
        bm25 = BM25Retriever(self.chunks)
        hybrid = HybridRetriever(dense_mock, bm25, rrf_k=60, pool_size=5)

        results = hybrid.search("bằng kép sinh viên", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "c1")
        self.assertGreater(results[0]["score"], 0.0)

    def test_empty_query_returns_empty(self):
        bm25 = BM25Retriever(self.chunks)
        self.assertEqual(bm25.search(""), [])
        self.assertEqual(bm25.search("   "), [])


if __name__ == "__main__":
    unittest.main()
