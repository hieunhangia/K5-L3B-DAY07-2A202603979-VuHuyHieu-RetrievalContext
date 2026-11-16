"""Run Agent evaluation on Dev or Test questions and save traces for agent_metrics.py."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "starter") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "starter"))

from src.agent import OpenAIController, run_agentic_rag
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluation on questions")
    parser.add_argument("--chunks", default="data/chunks.json")
    parser.add_argument("--questions", default="data/all_questions.json")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--retriever", default="hybrid", choices=("dense", "bm25", "hybrid", "weighted_hybrid"))
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=800)
    parser.add_argument("--max_searches", type=int, default=3)
    parser.add_argument("--output", default="results/agent_traces.json")
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    selected_qs = [q for q in questions if q.get("split") == args.split]

    logger.info(f"Loaded {len(chunks)} chunks and {len(selected_qs)} questions for split='{args.split}'.")

    # Select retriever
    if args.retriever == "dense":
        ret = DenseRetriever(chunks)
    elif args.retriever == "bm25":
        ret = BM25Retriever(chunks)
    elif args.retriever == "weighted_hybrid":
        dense_ret = DenseRetriever(chunks)
        bm25_ret = BM25Retriever(chunks)
        ret = HybridRetriever(
            dense_ret, bm25_ret, rrf_k=20, pool_size=20, dense_weight=0.30, bm25_weight=0.70
        )
    else:
        dense_ret = DenseRetriever(chunks)
        bm25_ret = BM25Retriever(chunks)
        ret = HybridRetriever(dense_ret, bm25_ret, rrf_k=60, pool_size=20)

    controller = OpenAIController()
    traces = []

    logger.info(f"Starting agent evaluation using {args.retriever} retriever...")
    for idx, q in enumerate(selected_qs, 1):
        qid = q["id"]
        q_text = q["question"]
        filters = q.get("filter")

        logger.info(f"[{idx}/{len(selected_qs)}] Running QID: {qid} | {q_text[:60]}...")
        res = run_agentic_rag(
            q_text,
            controller,
            ret,
            query_id=qid,
            filters=filters,
            max_searches=args.max_searches,
            top_k=args.top_k,
            budget=args.budget,
        )
        logger.info(
            f"  -> Result: status={res['status']}, citations={res['citations']}, elapsed={res['elapsed_ms']}ms"
        )
        traces.append(res)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(traces)} agent traces to {out_path}.")


if __name__ == "__main__":
    main()
