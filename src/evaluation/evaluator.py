"""
evaluator.py
------------
RAG evaluation suite — three metrics:

  1. Faithfulness       — Is the answer grounded in the retrieved context?
                          Uses the LLM to check each sentence of the answer
                          against the context. Score = % supported sentences.

  2. Answer Relevance   — Does the answer address the actual question?
                          Uses embedding similarity between the answer and
                          LLM-reconstructed questions from the answer.

  3. Context Precision  — Were the retrieved chunks actually useful?
                          Uses the LLM to judge each chunk's relevance to
                          the query. Score = % relevant chunks in top-K.

All scores are logged to MLflow.
"""

import logging
import os
import re
import numpy as np
import mlflow
from typing import List, Dict, Any

from src.generation.llm_client   import complete
from src.retrieval.embeddings     import embed_texts

log = logging.getLogger(__name__)


# 1. FAITHFULNESS

FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is faithful to its source context.

Context:
{context}

Answer:
{answer}

For each sentence in the answer, determine if it is directly supported by the context above.
Respond with ONLY a JSON list of booleans, one per sentence (true=supported, false=not supported).
Example: [true, true, false, true]"""


def score_faithfulness(answer: str, context_chunks: List[Dict]) -> float:
    """
    Returns 0.0–1.0: proportion of answer sentences supported by context.
    """
    context = "\n\n".join(c["text"] for c in context_chunks)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if len(s.strip()) > 10]

    if not sentences:
        return 1.0

    try:
        response = complete(
            messages=[{"role": "user", "content":
                FAITHFULNESS_PROMPT.format(context=context[:3000], answer=answer)}],
            temperature=0.0
        )
        bools = re.findall(r'(true|false)', response.lower())
    except Exception as exc:
        log.warning("faithfulness check failed: %s", exc)
        return float("nan")
    if not bools:
        log.warning("faithfulness check returned no verdicts")
        return float("nan")
    return round(sum(b == "true" for b in bools) / len(bools), 3)


# 2. ANSWER RELEVANCE

RELEVANCE_PROMPT = """Given the following answer, generate {n} questions that this answer could be responding to.

Answer: {answer}

Respond with ONLY {n} questions, one per line."""


def score_answer_relevance(query: str, answer: str, n_questions: int = 3) -> float:
    """
    Generates n questions from the answer and measures cosine similarity
    between the original query and each generated question.
    Score = mean similarity → how well the answer addresses the actual question.
    """
    try:
        response = complete(
            messages=[{"role": "user", "content":
                RELEVANCE_PROMPT.format(answer=answer[:1000], n=n_questions)}],
            temperature=0.3
        )
        questions = [q.strip() for q in response.strip().split("\n") if q.strip()][:n_questions]
    except Exception as exc:
        log.warning("answer relevance check failed: %s", exc)
        return float("nan")

    if not questions:
        return float("nan")

    all_texts   = [query] + questions
    embeddings  = embed_texts(all_texts)
    q_emb       = embeddings[0]
    gen_embs    = embeddings[1:]

    # Cosine similarities
    sims = []
    for g in gen_embs:
        cos = float(np.dot(q_emb, g) / (np.linalg.norm(q_emb) * np.linalg.norm(g) + 1e-9))
        sims.append(cos)

    return round(float(np.mean(sims)), 3)


# 3. CONTEXT PRECISION

PRECISION_PROMPT = """Does the following text contain information that is directly relevant to answering the query?

Query: {query}

Text: {text}

Respond with ONLY 'yes' or 'no'."""


def score_context_precision(query: str, chunks: List[Dict]) -> float:
    """
    Proportion of retrieved chunks that are actually relevant to the query.
    """
    if not chunks:
        return 0.0

    verdicts = []
    for chunk in chunks:
        try:
            response = complete(
                messages=[{"role": "user", "content":
                    PRECISION_PROMPT.format(query=query, text=chunk["text"][:500])}],
                temperature=0.0
            )
            verdicts.append(response.strip().lower().startswith("yes"))
        except Exception as exc:
            log.warning("context precision check failed for one chunk: %s", exc)

    if not verdicts:
        return float("nan")
    return round(sum(verdicts) / len(verdicts), 3)


# MASTER EVALUATION RUN

def evaluate(
    query:    str,
    answer:   str,
    chunks:   List[Dict],
    log_mlflow: bool = True,
) -> Dict[str, float]:
    """Run all three metrics and return a scores dict."""
    print("\nRunning RAG evaluation...")

    faithfulness     = score_faithfulness(answer, chunks)
    answer_relevance = score_answer_relevance(query, answer)
    context_precision = score_context_precision(query, chunks)

    scores = {
        "faithfulness":      faithfulness,
        "answer_relevance":  answer_relevance,
        "context_precision": context_precision,
        "mean_score":        round(float(np.nanmean([faithfulness, answer_relevance, context_precision])), 3),
    }

    print(f"  Faithfulness:      {faithfulness:.3f}")
    print(f"  Answer Relevance:  {answer_relevance:.3f}")
    print(f"  Context Precision: {context_precision:.3f}")
    print(f"  Mean Score:        {scores['mean_score']:.3f}")

    if log_mlflow:
        try:
            mlflow.set_experiment("doc-intel-evaluation")
            with mlflow.start_run(run_name="rag_eval"):
                mlflow.log_param("query",    query[:100])
                mlflow.log_param("n_chunks", len(chunks))
                mlflow.log_metrics(scores)
        except Exception:
            pass

    return scores
