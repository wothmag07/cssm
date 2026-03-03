"""
RAG Evaluation Metrics — 6 metrics for retrieval, faithfulness, citation, e2e, latency, and context precision.

Metric 1, 3, 5: Deterministic (no LLM calls, fast and free)
Metric 2, 4, 6: LLM-as-judge (uses gpt-4o-mini)
"""
import json
import logging
import re
import time
from typing import List

from langchain_core.documents import Document

# ── Minimal stopwords (no NLTK dependency) ──
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could and but or nor for yet so at by from in into "
    "of on to with as it its this that these those i me my we our you your he she "
    "they them their what which who whom how when where why all each every both few "
    "more most other some such no not only very".split()
)


def _significant_words(text: str, top_n: int = 5) -> set:
    """Extract top significant words from text (no stopwords, no short words)."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    filtered = [w for w in words if w not in _STOPWORDS]
    # Return first top_n unique significant words
    seen = []
    for w in filtered:
        if w not in seen:
            seen.append(w)
        if len(seen) >= top_n:
            break
    return set(seen)


# ── Metric 1: Retrieval Relevance (deterministic) ──

def score_retrieval_relevance(
    documents: List[Document],
    expected_keywords: List[str],
    expected_categories: List[str],
) -> dict:
    """Score retrieval relevance based on keyword and category hit rates."""
    if not expected_keywords and not expected_categories:
        return {
            "retrieval_relevance": 1.0,
            "keyword_hits": 0,
            "keyword_total": 0,
            "category_hits": 0,
            "category_total": 0,
        }

    all_content = " ".join(doc.page_content.lower() for doc in documents)
    all_categories = " ".join(
        str(doc.metadata.get("category", "")).lower() for doc in documents
    )

    keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in all_content)
    keyword_total = len(expected_keywords)
    keyword_rate = keyword_hits / keyword_total if keyword_total > 0 else 1.0

    category_hits = sum(1 for cat in expected_categories if cat.lower() in all_categories)
    category_total = len(expected_categories)
    category_rate = category_hits / category_total if category_total > 0 else 1.0

    score = 0.6 * keyword_rate + 0.4 * category_rate

    return {
        "retrieval_relevance": round(score, 3),
        "keyword_hits": keyword_hits,
        "keyword_total": keyword_total,
        "category_hits": category_hits,
        "category_total": category_total,
    }


# ── Metric 2: Answer Faithfulness (LLM-judge) ──

_FAITHFULNESS_PROMPT = """You are an evaluation judge. Given an ANSWER and the SOURCE DOCUMENTS it was generated from, determine what fraction of claims in the answer are supported by the source documents.

SOURCE DOCUMENTS:
{context}

ANSWER:
{answer}

Score from 0.0 to 1.0 where:
- 1.0 = every claim is directly supported by the sources
- 0.5 = about half the claims are supported
- 0.0 = the answer is entirely fabricated

Respond with ONLY a JSON object: {{"score": <float>, "unsupported_claims": ["claim1", ...]}}"""


def score_faithfulness(
    answer: str,
    documents: List[Document],
    llm,
) -> dict:
    """Score answer faithfulness using LLM-as-judge."""
    context = "\n\n".join(
        f"[{i+1}] {doc.page_content[:500]}" for i, doc in enumerate(documents)
    )

    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json_response(raw)
        return {
            "faithfulness": round(float(parsed.get("score", 0.0)), 3),
            "unsupported_claims": parsed.get("unsupported_claims", []),
        }
    except Exception as e:
        logging.warning(f"Faithfulness eval failed: {e}")
        return {"faithfulness": 0.0, "unsupported_claims": [f"eval_error: {e}"]}


# ── Metric 3: Citation Accuracy (deterministic) ──

def score_citation_accuracy(
    answer: str,
    documents: List[Document],
) -> dict:
    """Score citation accuracy using regex parsing."""
    citations = re.findall(r"\[(\d+)\]", answer)

    if not citations:
        return {
            "citation_validity": 0.0,
            "citation_relevance": 0.0,
            "has_citations": False,
            "total_citations": 0,
            "invalid_citations": [],
        }

    total = len(citations)
    num_docs = len(documents)
    invalid = []
    valid_count = 0
    relevant_count = 0

    # Split answer into sentences for per-citation context
    sentences = re.split(r"[.!?]+", answer)

    for cite_str in citations:
        cite_num = int(cite_str)
        if cite_num < 1 or cite_num > num_docs:
            invalid.append(cite_num)
            continue
        valid_count += 1

        # Find the sentence containing this citation
        doc = documents[cite_num - 1]
        doc_keywords = _significant_words(doc.page_content, top_n=5)

        for sent in sentences:
            if f"[{cite_str}]" in sent:
                sent_words = set(re.findall(r"[a-z]{3,}", sent.lower()))
                if doc_keywords & sent_words:
                    relevant_count += 1
                break

    validity_rate = valid_count / total if total > 0 else 0.0
    relevance_rate = relevant_count / valid_count if valid_count > 0 else 0.0

    return {
        "citation_validity": round(validity_rate, 3),
        "citation_relevance": round(relevance_rate, 3),
        "has_citations": True,
        "total_citations": total,
        "invalid_citations": invalid,
    }


# ── Metric 4: End-to-End Quality (LLM-judge) ──

_E2E_PROMPT = """You are an evaluation judge. Compare a GENERATED answer with a REFERENCE answer for the same question about electronics products.

REFERENCE (ideal) ANSWER:
{reference}

GENERATED ANSWER:
{answer}

Score from 0.0 to 1.0 where:
- 1.0 = the generated answer covers all key points from the reference and is well-structured
- 0.5 = partially covers the reference answer
- 0.0 = completely misses the point

Respond with ONLY a JSON object: {{"score": <float>, "missing_points": ["point1", ...], "extra_points": ["point1", ...]}}"""


def score_end_to_end(
    answer: str,
    reference_answer: str,
    llm,
) -> dict:
    """Score end-to-end quality by comparing to reference answer."""
    prompt = _E2E_PROMPT.format(reference=reference_answer, answer=answer)

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json_response(raw)
        return {
            "e2e_quality": round(float(parsed.get("score", 0.0)), 3),
            "missing_points": parsed.get("missing_points", []),
            "extra_points": parsed.get("extra_points", []),
        }
    except Exception as e:
        logging.warning(f"E2E eval failed: {e}")
        return {"e2e_quality": 0.0, "missing_points": [f"eval_error: {e}"], "extra_points": []}


# ── Metric 5: Latency (deterministic) ──

def score_latency(start_time: float, end_time: float) -> dict:
    """Measure pipeline latency in seconds."""
    duration = round(end_time - start_time, 3)
    # Thresholds: <5s good, 5-15s acceptable, >15s slow
    if duration < 5.0:
        rating = "good"
    elif duration < 15.0:
        rating = "acceptable"
    else:
        rating = "slow"

    return {
        "latency_seconds": duration,
        "latency_rating": rating,
    }


# ── Metric 6: Context Precision (LLM-judge) ──

_CONTEXT_PRECISION_PROMPT = """You are an evaluation judge. Given a QUESTION and RETRIEVED DOCUMENTS, determine how precisely the retrieved documents match the question's intent. Focus on whether the documents are specifically about what the user asked — not just vaguely related.

QUESTION: {question}

RETRIEVED DOCUMENTS:
{context}

For each document, score:
- 1 = directly relevant and useful for answering the question
- 0 = irrelevant, off-topic, or only tangentially related

Respond with ONLY a JSON object: {{"scores": [1, 0, 1, ...], "reasoning": "brief explanation"}}"""


def score_context_precision(
    question: str,
    documents: List[Document],
    llm,
) -> dict:
    """Score context precision — what fraction of retrieved docs are truly relevant."""
    if not documents:
        return {"context_precision": 0.0, "relevant_docs": 0, "total_docs": 0, "precision_reasoning": "no documents"}

    context = "\n\n".join(
        f"[{i+1}] (Product: {doc.metadata.get('product_name', 'Unknown')}) "
        f"{doc.page_content[:400]}"
        for i, doc in enumerate(documents)
    )

    prompt = _CONTEXT_PRECISION_PROMPT.format(question=question, context=context)

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = _parse_json_response(raw)
        scores = parsed.get("scores", [])
        if not scores:
            return {"context_precision": 0.0, "relevant_docs": 0, "total_docs": len(documents), "precision_reasoning": "parse error"}
        relevant = sum(1 for s in scores if s == 1)
        total = len(scores)
        return {
            "context_precision": round(relevant / total, 3) if total > 0 else 0.0,
            "relevant_docs": relevant,
            "total_docs": total,
            "precision_reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        logging.warning(f"Context precision eval failed: {e}")
        return {"context_precision": 0.0, "relevant_docs": 0, "total_docs": len(documents), "precision_reasoning": f"eval_error: {e}"}


# ── Helpers ──

def _parse_json_response(raw: str) -> dict:
    """Parse JSON from LLM response, with fallback for malformed output."""
    raw = raw.strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try extracting JSON from markdown code block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: extract a float
    float_match = re.search(r"(\d+\.?\d*)", raw)
    if float_match:
        return {"score": float(float_match.group(1))}
    return {"score": 0.0}


def evaluate_single(
    question: str,
    benchmark_entry: dict,
    pipeline_result: dict,
    llm,
    latency_start: float = 0.0,
    latency_end: float = 0.0,
) -> dict:
    """Run all 6 metrics for a single benchmark entry."""
    documents = pipeline_result.get("documents", [])
    answer = pipeline_result.get("answer", "")

    retrieval = score_retrieval_relevance(
        documents=documents,
        expected_keywords=benchmark_entry.get("expected_keywords", []),
        expected_categories=benchmark_entry.get("expected_categories", []),
    )

    faithfulness = score_faithfulness(
        answer=answer,
        documents=documents,
        llm=llm,
    )

    citation = score_citation_accuracy(
        answer=answer,
        documents=documents,
    )

    e2e = score_end_to_end(
        answer=answer,
        reference_answer=benchmark_entry.get("reference_answer", ""),
        llm=llm,
    )

    latency = score_latency(latency_start, latency_end)

    context_prec = score_context_precision(
        question=question,
        documents=documents,
        llm=llm,
    )

    return {
        "id": benchmark_entry["id"],
        "question": question,
        **retrieval,
        **faithfulness,
        **citation,
        **e2e,
        **latency,
        **context_prec,
    }


def compute_aggregate_scores(all_scores: list) -> dict:
    """Compute mean scores across all benchmark entries."""
    n = len(all_scores)
    if n == 0:
        return {}

    return {
        "total_entries": n,
        "mean_retrieval_relevance": round(
            sum(s["retrieval_relevance"] for s in all_scores) / n, 3
        ),
        "mean_faithfulness": round(
            sum(s["faithfulness"] for s in all_scores) / n, 3
        ),
        "mean_citation_validity": round(
            sum(s["citation_validity"] for s in all_scores) / n, 3
        ),
        "mean_citation_relevance": round(
            sum(s["citation_relevance"] for s in all_scores) / n, 3
        ),
        "mean_e2e_quality": round(
            sum(s["e2e_quality"] for s in all_scores) / n, 3
        ),
        "citation_rate": round(
            sum(1 for s in all_scores if s["has_citations"]) / n, 3
        ),
        "mean_latency_seconds": round(
            sum(s["latency_seconds"] for s in all_scores) / n, 3
        ),
        "mean_context_precision": round(
            sum(s["context_precision"] for s in all_scores) / n, 3
        ),
    }
