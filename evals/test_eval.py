"""
RAG Evaluation Suite — Measures retrieval, faithfulness, citation, e2e, latency, and context precision.

Run:
    pytest evals/ -v -s --tb=short

Requires real API keys: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
"""
import json
import logging
import time
from pathlib import Path

import pytest

from graph.rag_graph import build_graph
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from evals.eval_metrics import evaluate_single, compute_aggregate_scores

BENCHMARK_PATH = Path(__file__).parent / "benchmark.jsonl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ── Fixtures (session-scoped — built once, reused across all tests) ──

@pytest.fixture(scope="session")
def rag_pipeline():
    """Build the real RAG pipeline (requires API keys)."""
    retriever = Retriever()
    model_loader = ModelLoader()
    graph = build_graph(retriever, model_loader, max_retries=2)
    llm = model_loader.load_llm()
    return graph, llm


@pytest.fixture(scope="session")
def benchmark_data():
    """Load benchmark entries from JSONL."""
    entries = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _run_pipeline(graph, question: str) -> tuple:
    """Invoke the full RAG pipeline for a question. Returns (result, start_time, end_time)."""
    start = time.time()
    result = graph.invoke({
        "question": question,
        "rewritten_query": "",
        "documents": [],
        "sources": [],
        "grade": "",
        "answer": "",
        "retries": 0,
        "chat_history": "No previous conversation.",
    })
    end = time.time()
    return result, start, end


# ── Load benchmark IDs at collection time for parametrize ──

def _load_benchmark_ids():
    entries = []
    with open(BENCHMARK_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                entries.append(entry["id"])
    return entries


# ── Parametrized per-entry evaluation ──

@pytest.mark.parametrize("eval_id", _load_benchmark_ids())
def test_eval_benchmark(eval_id, rag_pipeline, benchmark_data):
    """Evaluate a single benchmark entry through the full RAG pipeline."""
    graph, llm = rag_pipeline
    entry = next(e for e in benchmark_data if e["id"] == eval_id)

    result, start, end = _run_pipeline(graph, entry["question"])
    scores = evaluate_single(
        question=entry["question"],
        benchmark_entry=entry,
        pipeline_result=result,
        llm=llm,
        latency_start=start,
        latency_end=end,
    )

    # Log results
    logging.info(f"\n{'='*60}")
    logging.info(f"EVAL: {eval_id} — {entry['question'][:60]}")
    logging.info(f"  Retrieval Relevance:  {scores['retrieval_relevance']:.2f}")
    logging.info(f"  Context Precision:    {scores['context_precision']:.2f} ({scores['relevant_docs']}/{scores['total_docs']} docs)")
    logging.info(f"  Faithfulness:         {scores['faithfulness']:.2f}")
    logging.info(f"  Citation Validity:    {scores['citation_validity']:.2f}")
    logging.info(f"  Citation Relevance:   {scores['citation_relevance']:.2f}")
    logging.info(f"  E2E Quality:          {scores['e2e_quality']:.2f}")
    logging.info(f"  Latency:              {scores['latency_seconds']:.1f}s ({scores['latency_rating']})")
    logging.info(f"  Has Citations:        {scores['has_citations']}")
    if scores.get("unsupported_claims"):
        logging.info(f"  Unsupported Claims:   {scores['unsupported_claims']}")
    if scores.get("missing_points"):
        logging.info(f"  Missing Points:       {scores['missing_points']}")

    # Soft assertions — fail only on catastrophically bad results
    difficulty = entry.get("difficulty", "easy")
    if difficulty != "hard":
        assert scores["retrieval_relevance"] >= 0.3, (
            f"Retrieval too poor: {scores['retrieval_relevance']:.2f}"
        )
        assert scores["faithfulness"] >= 0.3, (
            f"Faithfulness too low: {scores['faithfulness']:.2f}"
        )


# ── Aggregate evaluation ──

def test_eval_aggregate(rag_pipeline, benchmark_data):
    """Run all benchmarks and report aggregate scores."""
    graph, llm = rag_pipeline
    all_scores = []

    for entry in benchmark_data:
        result, start, end = _run_pipeline(graph, entry["question"])
        scores = evaluate_single(
            question=entry["question"],
            benchmark_entry=entry,
            pipeline_result=result,
            llm=llm,
            latency_start=start,
            latency_end=end,
        )
        all_scores.append(scores)

    aggregates = compute_aggregate_scores(all_scores)

    logging.info(f"\n{'='*60}")
    logging.info("AGGREGATE EVALUATION RESULTS")
    logging.info(f"  Entries evaluated:        {aggregates['total_entries']}")
    logging.info(f"  Mean Retrieval Relevance: {aggregates['mean_retrieval_relevance']:.2f}")
    logging.info(f"  Mean Context Precision:   {aggregates['mean_context_precision']:.2f}")
    logging.info(f"  Mean Faithfulness:        {aggregates['mean_faithfulness']:.2f}")
    logging.info(f"  Mean Citation Validity:   {aggregates['mean_citation_validity']:.2f}")
    logging.info(f"  Mean Citation Relevance:  {aggregates['mean_citation_relevance']:.2f}")
    logging.info(f"  Mean E2E Quality:         {aggregates['mean_e2e_quality']:.2f}")
    logging.info(f"  Citation Rate:            {aggregates['citation_rate']:.0%}")
    logging.info(f"  Mean Latency:             {aggregates['mean_latency_seconds']:.1f}s")
    logging.info(f"{'='*60}")

    # Aggregate thresholds
    assert aggregates["mean_retrieval_relevance"] >= 0.5, (
        f"Mean retrieval too low: {aggregates['mean_retrieval_relevance']:.2f}"
    )
    assert aggregates["mean_faithfulness"] >= 0.5, (
        f"Mean faithfulness too low: {aggregates['mean_faithfulness']:.2f}"
    )
    assert aggregates["citation_rate"] >= 0.7, (
        f"Citation rate too low: {aggregates['citation_rate']:.0%}"
    )
