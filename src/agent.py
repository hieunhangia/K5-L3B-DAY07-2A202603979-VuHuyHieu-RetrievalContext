"""Agentic RAG Controller and Runner for Day 7.
Connects real LLM (OpenAI Chat Completion) with real Retrievers (Dense, BM25, Hybrid).
Handles tool calling, context packing, citation validation, abstention, and full trace logging.
"""
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Protocol

# Ensure starter is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "starter"))

from baseline import pack_context
from src.config import OPENAI_MODEL, get_openai_client
from src.retrieval import Retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên nghiệp trả lời câu hỏi dựa trên Quy chế Đào tạo VinUniversity.
Nhiệm vụ của bạn là trả lời chính xác, trung thực và CHỈ DỰA TRÊN CÁC ĐOẠN VĂN BẢN ĐƯỢC CUNG CẤP TRONG CONTEXT.

HƯỚNG DẪN HOẠT ĐỘNG:
1. Nếu chưa có context hoặc context hiện tại chưa đủ thông tin để trả lời đầy đủ câu hỏi, bạn PHẢI gọi tool `search` với từ khóa phù hợp để tìm thêm bằng chứng.
2. Khi đã tìm thấy đủ bằng chứng trong context:
   - Bạn PHẢI gọi tool `answer` với câu trả lời hoàn chỉnh.
   - Trong câu trả lời, BẮT BUỘC phải trích dẫn định danh [chunk_id] ngay tại câu chứa thông tin đó. Ví dụ: "Thời gian đào tạo là 4 năm [quy_che_dao_tao_vinuni_2024_stratC_p11_c001]."
   - Liệt kê danh sách các chunk_id đã trích dẫn vào trường `citations`.
3. Nếu sau khi tìm kiếm (hoặc trong câu hỏi về những thông tin không có trong tài liệu quy chế) mà không có bằng chứng:
   - Bạn PHẢI gọi tool `answer` với câu trả lời chính xác: "Không đủ bằng chứng để trả lời." và để `citations: []`.
   - TUYỆT ĐỐI KHÔNG bịa đặt, suy đoán hay sử dụng thông tin ngoài context."""

RETRIEVAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Tìm kiếm các đoạn văn bản trong quy chế đào tạo dựa trên từ khóa truy vấn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Từ khóa tìm kiếm (ngắn gọn, tập trung vào thuật ngữ chính)",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Đưa ra câu trả lời cuối cùng kèm trích dẫn [chunk_id], hoặc từ chối khi không đủ bằng chứng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Nội dung câu trả lời kèm trích dẫn [chunk_id], hoặc 'Không đủ bằng chứng để trả lời.'",
                    },
                    "citations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Danh sách các chunk_id xuất hiện trong câu trả lời.",
                    },
                },
                "required": ["answer", "citations"],
            },
        },
    },
]


class OpenAIController:
    """Controller implementation using OpenAI Chat Completion API with Tool Calling."""

    def __init__(self, model_name: str | None = None, timeout: float = 60.0):
        self.model_name = model_name or OPENAI_MODEL
        self.client = get_openai_client()
        self.timeout = timeout
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Decide next action: search or answer."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Câu hỏi của sinh viên: {question}"},
        ]

        # Add history if available
        if history:
            latest = history[-1]
            queries_str = ", ".join(repr(q) for q in latest.get("queries", []))
            context_str = latest.get("context", "")
            context_ids = latest.get("context_ids", [])
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Kết quả tra cứu cho các truy vấn [{queries_str}]:\n\n"
                        f"CONTEXT HIỆN TẠI (Gồm các chunks: {context_ids}):\n"
                        f"{context_str}\n\n"
                        "Nếu context trên đã đủ bằng chứng, hãy gọi tool `answer`. "
                        "Nếu chưa đủ, hãy gọi tool `search` với từ khóa khác để tìm tiếp. "
                        "Nếu chắc chắn tài liệu không có thông tin, hãy gọi tool `answer` với 'Không đủ bằng chứng để trả lời.' và citations: []."
                    ),
                }
            )

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=RETRIEVAL_TOOLS,
                tool_choice="auto",
                timeout=self.timeout,
            )
            usage = res.usage
            if usage:
                self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0)
                self.total_completion_tokens += getattr(usage, "completion_tokens", 0)

            msg = res.choices[0].message
            # 1. Check Tool Calls
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                fn_name = tc.function.name
                fn_args_raw = tc.function.arguments
                try:
                    fn_args = json.loads(fn_args_raw)
                except Exception:
                    fn_args = {}

                if fn_name == "search":
                    query = fn_args.get("query", "").strip()
                    if query:
                        return {"action": "search", "query": query}
                elif fn_name == "answer":
                    ans = fn_args.get("answer", "").strip()
                    cits = fn_args.get("citations", [])
                    if isinstance(cits, list):
                        cits = [str(c) for c in cits]
                    else:
                        cits = []
                    return {"action": "answer", "answer": ans, "citations": cits}

            # 2. Check Text Content Fallback
            content = (msg.content or "").strip()
            if content:
                # Check for explicit abstention
                if "không đủ bằng chứng" in content.lower():
                    return {
                        "action": "answer",
                        "answer": "Không đủ bằng chứng để trả lời.",
                        "citations": [],
                    }
                # Extract citation IDs from text [chunk_id]
                found_cits = re.findall(r"\[([a-zA-Z0-9_\-:]+)\]", content)
                return {
                    "action": "answer",
                    "answer": content,
                    "citations": list(dict.fromkeys(found_cits)),
                }

            # Default fallback: search query
            return {"action": "search", "query": question}

        except Exception as e:
            logger.error(f"Error calling OpenAI API in Controller: {e}")
            raise e


def run_agentic_rag(
    question: str,
    controller: OpenAIController,
    retriever: Retriever,
    *,
    query_id: str = "q",
    filters: dict[str, Any] | None = None,
    max_searches: int = 4,
    top_k: int = 3,
    budget: int = 800,
) -> dict[str, Any]:
    """Execute Agentic RAG loop with real Controller, Retriever, and logging."""
    t0 = time.perf_counter()
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    for turn in range(max_searches + 1):
        try:
            action = controller.decide(question, history)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return {
                "query_id": query_id,
                "status": "error",
                "question": question,
                "answer": f"Lỗi hệ thống: {exc}",
                "citations": [],
                "events": events + [{"event": "controller_error", "error": str(exc)}],
                "elapsed_ms": round(elapsed_ms, 2),
            }

        if action.get("action") == "search":
            search_count = sum(e.get("event") == "search" for e in events)
            if search_count >= max_searches:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return {
                    "query_id": query_id,
                    "status": "abstain",
                    "question": question,
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "search_limit_reached"}],
                    "elapsed_ms": round(elapsed_ms, 2),
                }

            query = action.get("query", "").strip()
            if not query:
                query = question

            try:
                results = retriever.search(query, filters=filters, top_k=top_k)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                return {
                    "query_id": query_id,
                    "status": "error",
                    "question": question,
                    "answer": f"Lỗi retriever: {exc}",
                    "citations": [],
                    "events": events + [{"event": "retriever_error", "error": str(exc)}],
                    "elapsed_ms": round(elapsed_ms, 2),
                }

            candidates.extend(results)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c["id"]: c for c in candidates if c["id"] in ids}

            events.append(
                {
                    "event": "search",
                    "turn": turn + 1,
                    "query": query,
                    "filters": filters,
                    "retrieved_count": len(results),
                    "context_ids": ids,
                }
            )

            history = [
                {
                    "tool": "search",
                    "queries": [e["query"] for e in events if e["event"] == "search"],
                    "context": context,
                    "context_ids": ids,
                }
            ]

        elif action.get("action") == "answer":
            answer = action.get("answer", "")
            citations = action.get("citations", [])
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Validate citation IDs
            valid_citations = [cid for cid in citations if cid in chunks and f"[{cid}]" in answer]

            # If answer is explicit abstention
            if "không đủ bằng chứng" in answer.lower():
                return {
                    "query_id": query_id,
                    "status": "abstain",
                    "question": question,
                    "answer": "Không đủ bằng chứng để trả lời.",
                    "citations": [],
                    "events": events + [{"event": "abstain"}],
                    "elapsed_ms": round(elapsed_ms, 2),
                }

            # Valid answer with supported citations
            if answer.strip() and valid_citations:
                return {
                    "query_id": query_id,
                    "status": "answered",
                    "question": question,
                    "answer": answer,
                    "citations": valid_citations,
                    "events": events + [{"event": "answered"}],
                    "elapsed_ms": round(elapsed_ms, 2),
                }

            # Fallback abstention if citations are invalid or context is empty
            return {
                "query_id": query_id,
                "status": "abstain",
                "question": question,
                "answer": "Không đủ bằng chứng để trả lời.",
                "citations": [],
                "events": events + [{"event": "invalid_or_missing_citations"}],
                "elapsed_ms": round(elapsed_ms, 2),
            }

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return {
        "query_id": query_id,
        "status": "abstain",
        "question": question,
        "answer": "Không đủ bằng chứng để trả lời.",
        "citations": [],
        "events": events + [{"event": "max_turns_exceeded"}],
        "elapsed_ms": round(elapsed_ms, 2),
    }
