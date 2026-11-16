"""Stage 5: Independent Evaluation on Test Split.

Runs:
1. Baseline Configuration:
   - Retriever: DenseRetriever (Google Gemini Embedding)
   - Context Budget: 800 chars
   - Agent Controller: OpenAI Controller (real LLM)
2. Selected Configuration:
   - Retriever: WeightedHybridRetriever (k=20, w_bm25=0.70, w_dense=0.30, pool_size=20)
   - Context Budget: 2000 chars
   - Agent Controller: OpenAI Controller (real LLM)

Evaluates on all 7 test cases:
- 5 answerable questions: q16, q17, q18, q19, q20
- 2 unanswerable questions: q_unans_test_01, q_unans_test_02

Saves outputs to:
- results/test_traces_baseline.json
- results/test_traces_selected.json
- results/stage5_test_retrieval_comparison.json
"""
import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "starter"))

from baseline import pack_context
from src.agent import OpenAIController, run_agentic_rag
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_retrieval_test(
    retriever,
    questions: list[dict[str, Any]],
    budget: int,
    config_name: str,
    top_k: int = 3,
) -> dict[str, Any]:
    ans_qs = [q for q in questions if q.get("split") == "test" and q.get("answerable", True)]
    n = len(ans_qs)
    h1_count = 0
    h3_count = 0
    ctx_count = 0
    latencies: list[float] = []

    for q in ans_qs:
        gold = q.get("gold", [])
        filters = q.get("filter")
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        h1 = any(
            g["source_id"] == results[0]["source_id"]
            and g["page"] == results[0]["page"]
            and g["evidence"] in results[0]["text"]
            for g in gold
        ) if results else False

        h3 = any(
            g["source_id"] == r["source_id"]
            and g["page"] == r["page"]
            and g["evidence"] in r["text"]
            for r in results[:3]
            for g in gold
        )

        context, _ = pack_context(results, budget=budget)
        ctx = any(g["evidence"] in context for g in gold)

        if h1:
            h1_count += 1
        if h3:
            h3_count += 1
        if ctx:
            ctx_count += 1

    return {
        "config_name": config_name,
        "n": n,
        "budget": budget,
        "hit1": f"{h1_count}/{n} ({h1_count/n*100:.1f}%)",
        "hit1_rate": round(h1_count / n, 4),
        "hit3": f"{h3_count}/{n} ({h3_count/n*100:.1f}%)",
        "hit3_rate": round(h3_count / n, 4),
        "context_evidence_rate": f"{ctx_count}/{n} ({ctx_count/n*100:.1f}%)",
        "context_rate": round(ctx_count / n, 4),
        "median_latency_ms": round(statistics.median(latencies), 2),
    }


def run_agent_test(
    controller: OpenAIController,
    retriever,
    test_cases: list[dict[str, Any]],
    budget: int,
    config_name: str,
    top_k: int = 3,
    max_searches: int = 3,
) -> list[dict[str, Any]]:
    logger.info(f"--- Running Agent on Test Set: {config_name} (budget={budget}) ---")
    traces = []
    for idx, case in enumerate(test_cases, 1):
        qid = case["id"]
        q_text = case["question"]
        is_ans = case.get("answerable", True)
        filters = case.get("filter")

        logger.info(f"[{idx}/{len(test_cases)}] QID: {qid} (answerable={is_ans}) | {q_text[:60]}...")
        t0 = time.perf_counter()
        res = run_agentic_rag(
            q_text,
            controller,
            retriever,
            query_id=qid,
            filters=filters,
            max_searches=max_searches,
            top_k=top_k,
            budget=budget,
        )
        elapsed_sec = (time.perf_counter() - t0)
        logger.info(
            f"  -> Result: status={res['status']}, citations={res['citations']}, elapsed={elapsed_sec:.2f}s"
        )
        logger.info(f"  -> Answer: {res['answer'][:100]}...")
        traces.append(res)
    return traces


def main():
    chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
    questions = json.loads(Path("data/all_questions.json").read_text(encoding="utf-8"))
    test_cases = [q for q in questions if q.get("split") == "test"]
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing retrievers...")
    dense = DenseRetriever(chunks)
    bm25 = BM25Retriever(chunks)
    weighted_hybrid = HybridRetriever(
        dense, bm25, rrf_k=20, pool_size=20, dense_weight=0.30, bm25_weight=0.70
    )

    # 1. Retrieval Metrics Evaluation on Test set
    retrieval_comparison = [
        evaluate_retrieval_test(dense, questions, budget=800, config_name="Baseline Dense (budget=800)"),
        evaluate_retrieval_test(bm25, questions, budget=800, config_name="Baseline BM25 (budget=800)"),
        evaluate_retrieval_test(
            weighted_hybrid, questions, budget=2000, config_name="Selected Config: Weighted Hybrid (budget=2000)"
        ),
    ]

    retrieval_file = output_dir / "stage5_test_retrieval_comparison.json"
    retrieval_file.write_text(json.dumps(retrieval_comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved Test retrieval comparison to {retrieval_file}")

    print("\n" + "=" * 90)
    print("STAGE 5: TEST SET RETRIEVAL METRICS (5 Answerable Questions)")
    print("=" * 90)
    print(f"{'Configuration':<45} | {'Hit@1':<12} | {'Hit@3':<12} | {'Context Rate':<14} | {'Median Latency'}")
    print("-" * 90)
    for r in retrieval_comparison:
        print(f"{r['config_name']:<45} | {r['hit1']:<12} | {r['hit3']:<12} | {r['context_evidence_rate']:<14} | {r['median_latency_ms']} ms")
    print("=" * 90)

    # 2. Agent Execution on Test Set
    controller = OpenAIController()

    # 2.1 Baseline Agent
    logger.info("Executing Baseline Agent on Test set...")
    baseline_traces = run_agent_test(
        controller, dense, test_cases, budget=800, config_name="Baseline Dense (budget=800)"
    )
    baseline_path = output_dir / "test_traces_baseline.json"
    baseline_path.write_text(json.dumps(baseline_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved Baseline traces to {baseline_path}")

    # 2.2 Selected Config Agent
    logger.info("Executing Selected Config Agent on Test set...")
    selected_traces = run_agent_test(
        controller, weighted_hybrid, test_cases, budget=2000, config_name="Selected Config (budget=2000)"
    )
    selected_path = output_dir / "test_traces_selected.json"
    selected_path.write_text(json.dumps(selected_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved Selected Config traces to {selected_path}")

    print("\n" + "=" * 90)
    print("STAGE 5: AGENT TEST RUN COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
