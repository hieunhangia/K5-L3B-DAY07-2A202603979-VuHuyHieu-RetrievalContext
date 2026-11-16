"""Benchmark and compare the 3 retrieval methods on Dev and Test splits.
Outputs Hit@1, Evidence Hit@3, Context Evidence Rate, Latency, and Traces.
"""
import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure root and starter are in sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "starter"))

from baseline import pack_context
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_gold_hit(chunk: dict[str, Any], gold_list: list[dict[str, Any]]) -> bool:
    """Check if chunk matches any gold evidence in source_id, page, and exact quote substring."""
    for g in gold_list:
        if (
            chunk.get("source_id") == g.get("source_id")
            and chunk.get("page") == g.get("page")
            and g.get("evidence", "") in chunk.get("text", "")
        ):
            return True
    return False


def evaluate_method(
    retriever: Retriever,
    questions: list[dict[str, Any]],
    method_name: str,
    split: str,
    top_k: int = 3,
    budget: int = 800,
) -> dict[str, Any]:
    """Evaluate a retrieval method on a list of questions."""
    selected_qs = [q for q in questions if q.get("split") == split and q.get("answerable", True)]
    n = len(selected_qs)
    if n == 0:
        return {"method": method_name, "split": split, "n": 0}

    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0
    latencies: list[float] = []
    traces: list[dict[str, Any]] = []

    for q in selected_qs:
        qid = q["id"]
        question_text = q["question"]
        gold = q.get("gold", [])
        filters = q.get("filter")

        t0 = time.perf_counter()
        results = retriever.search(question_text, filters=filters, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        hit1 = is_gold_hit(results[0], gold) if len(results) >= 1 else False
        hit3 = any(is_gold_hit(r, gold) for r in results[:3])

        context, context_ids = pack_context(results, budget=budget)
        context_hit = any(g.get("evidence", "") in context for g in gold)

        if hit1:
            hit1_count += 1
        if hit3:
            hit3_count += 1
        if context_hit:
            context_hit_count += 1

        traces.append(
            {
                "id": qid,
                "question": question_text,
                "split": split,
                "filters": filters,
                "hit1": hit1,
                "hit3": hit3,
                "context_hit": context_hit,
                "latency_ms": round(elapsed_ms, 2),
                "top_results": [
                    {
                        "rank": idx + 1,
                        "id": r["id"],
                        "score": round(r.get("score", 0.0), 4),
                        "page": r.get("page"),
                        "text_snippet": r.get("text", "")[:120],
                    }
                    for idx, r in enumerate(results)
                ],
                "context_ids": context_ids,
            }
        )

    summary = {
        "method": method_name,
        "split": split,
        "n": n,
        "hit1": hit1_count,
        "hit1_rate": round(hit1_count / n, 4),
        "hit3": hit3_count,
        "hit3_rate": round(hit3_count / n, 4),
        "context_evidence_rate": round(context_hit_count / n, 4),
        "context_hits": context_hit_count,
        "median_latency_ms": round(statistics.median(latencies), 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "traces": traces,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3 retrieval methods on Day 7 benchmark")
    parser.add_argument("--chunks", default="data/chunks.json")
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing retrievers...")
    dense = DenseRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25, rrf_k=60, pool_size=20)

    methods = [
        ("Dense Retrieval (Google Gemini)", dense),
        ("Lexical Retrieval (BM25)", bm25),
        ("Hybrid Retrieval (Dense + BM25 via RRF)", hybrid),
    ]

    all_summaries = []

    for name, ret in methods:
        logger.info(f"Evaluating {name} on DEV split...")
        dev_res = evaluate_method(ret, questions, name, split="dev", top_k=3, budget=800)
        logger.info(
            f"  [DEV] Hit@1: {dev_res['hit1']}/{dev_res['n']} ({dev_res['hit1_rate']*100:.1f}%), "
            f"Hit@3: {dev_res['hit3']}/{dev_res['n']} ({dev_res['hit3_rate']*100:.1f}%), "
            f"Context Hit: {dev_res['context_hits']}/{dev_res['n']} ({dev_res['context_evidence_rate']*100:.1f}%)"
        )

        logger.info(f"Evaluating {name} on TEST split...")
        test_res = evaluate_method(ret, questions, name, split="test", top_k=3, budget=800)
        logger.info(
            f"  [TEST] Hit@1: {test_res['hit1']}/{test_res['n']} ({test_res['hit1_rate']*100:.1f}%), "
            f"Hit@3: {test_res['hit3']}/{test_res['n']} ({test_res['hit3_rate']*100:.1f}%), "
            f"Context Hit: {test_res['context_hits']}/{test_res['n']} ({test_res['context_evidence_rate']*100:.1f}%)"
        )

        slug = name.split()[0].lower()
        method_output = output_dir / f"retrieval_{slug}.json"
        method_output.write_text(
            json.dumps({"dev": dev_res, "test": test_res}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_summaries.append(
            {
                "method": name,
                "dev_hit1": f"{dev_res['hit1']}/{dev_res['n']} ({dev_res['hit1_rate']*100:.1f}%)",
                "dev_hit3": f"{dev_res['hit3']}/{dev_res['n']} ({dev_res['hit3_rate']*100:.1f}%)",
                "dev_context_rate": f"{dev_res['context_hits']}/{dev_res['n']} ({dev_res['context_evidence_rate']*100:.1f}%)",
                "dev_median_ms": dev_res["median_latency_ms"],
                "test_hit1": f"{test_res['hit1']}/{test_res['n']} ({test_res['hit1_rate']*100:.1f}%),",
                "test_hit3": f"{test_res['hit3']}/{test_res['n']} ({test_res['hit3_rate']*100:.1f}%)",
                "test_context_rate": f"{test_res['context_hits']}/{test_res['n']} ({test_res['context_evidence_rate']*100:.1f}%)",
                "test_median_ms": test_res["median_latency_ms"],
            }
        )

    comparison_path = output_dir / "retrieval_comparison.json"
    comparison_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Done! Saved comparison to {comparison_path}")


if __name__ == "__main__":
    main()
