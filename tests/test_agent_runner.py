"""Unit tests for OpenAIController and Agentic RAG loop using mocks."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "starter"))

from src.agent import OpenAIController, run_agentic_rag


class MockRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query: str, *, filters=None, top_k=3):
        return self.chunks[:top_k]


class AgentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "chunk_p4_01",
                "text": "Sinh viên được cấp bằng kép nếu hoàn thành đầy đủ cả hai bằng.",
                "source_id": "doc",
                "source": "doc.pdf",
                "page": 4,
                "metadata": {},
            }
        ]
        self.retriever = MockRetriever(self.chunks)

    @patch("src.agent.get_openai_client")
    def test_controller_search_action(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock tool call response for search
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"query": "bằng kép sinh viên"}'
        mock_choice = MagicMock()
        mock_choice.message.tool_calls = [mock_tool_call]
        mock_choice.message.content = None
        mock_res = MagicMock()
        mock_res.choices = [mock_choice]
        mock_res.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = mock_res

        controller = OpenAIController()
        action = controller.decide("Làm sao để nhận bằng kép?", [])
        self.assertEqual(action["action"], "search")
        self.assertEqual(action["query"], "bằng kép sinh viên")

    @patch("src.agent.get_openai_client")
    def test_agent_full_loop_answer(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: Search call
        tc_search = MagicMock()
        tc_search.function.name = "search"
        tc_search.function.arguments = '{"query": "bằng kép"}'
        msg_search = MagicMock(tool_calls=[tc_search], content=None)
        res_search = MagicMock(choices=[MagicMock(message=msg_search)], usage=None)

        # Turn 2: Answer call
        tc_answer = MagicMock()
        tc_answer.function.name = "answer"
        tc_answer.function.arguments = '{"answer": "Sinh viên được cấp bằng kép nếu hoàn thành [chunk_p4_01].", "citations": ["chunk_p4_01"]}'
        msg_answer = MagicMock(tool_calls=[tc_answer], content=None)
        res_answer = MagicMock(choices=[MagicMock(message=msg_answer)], usage=None)

        mock_client.chat.completions.create.side_effect = [res_search, res_answer]

        controller = OpenAIController()
        result = run_agentic_rag(
            "Làm sao nhận bằng kép?",
            controller,
            self.retriever,
            query_id="q_test",
            max_searches=3,
            budget=800,
        )

        self.assertEqual(result["status"], "answered")
        self.assertIn("chunk_p4_01", result["citations"])
        self.assertEqual(result["query_id"], "q_test")

    @patch("src.agent.get_openai_client")
    def test_agent_abstain_on_unanswerable(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Turn 1: Search
        tc_search = MagicMock()
        tc_search.function.name = "search"
        tc_search.function.arguments = '{"query": "học bổng"}'
        res_search = MagicMock(choices=[MagicMock(message=MagicMock(tool_calls=[tc_search], content=None))], usage=None)

        # Turn 2: Abstain
        tc_ans = MagicMock()
        tc_ans.function.name = "answer"
        tc_ans.function.arguments = '{"answer": "Không đủ bằng chứng để trả lời.", "citations": []}'
        res_ans = MagicMock(choices=[MagicMock(message=MagicMock(tool_calls=[tc_ans], content=None))], usage=None)

        mock_client.chat.completions.create.side_effect = [res_search, res_ans]

        controller = OpenAIController()
        result = run_agentic_rag(
            "Mức học bổng là bao nhiêu?",
            controller,
            self.retriever,
            query_id="q_unans",
            max_searches=3,
            budget=800,
        )

        self.assertEqual(result["status"], "abstain")
        self.assertEqual(result["answer"], "Không đủ bằng chứng để trả lời.")
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
