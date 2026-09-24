"""Post-generation tests that decide whether an LLM answer is acceptable."""

from __future__ import annotations

import re

from app.evaluator import grounding, tokens


_SECRET_OUTPUT = re.compile(r"\b(?:github_pat_|sgp_|sk-)[A-Za-z0-9_-]+\b", re.IGNORECASE)
_PROMPT_LEAK = re.compile(r"\b(?:system prompt|developer message|ignore previous instructions)\b", re.IGNORECASE)
_PROMPT_ECHO = re.compile(r"(?:^|\n)\s*(?:live facts|rag evidence|code evidence|final output rule)\s*:", re.IGNORECASE)


def _question_terms(question):
    ignored = {"answer", "does", "explain", "how", "is", "the", "this", "what", "which", "with"}
    return {term for term in tokens(question) if term not in ignored}


def validate_output(question: str, answer: str, category: str, context: dict, *, mode: str | None = None) -> dict:
    """Apply deterministic relevance, support, format, and safety tests.

    The original answer is retained as diagnostic data in comparisons, but an
    answer with a blocking failure is never decision-eligible or returned by
    Ask Nexus as an accepted answer.
    """
    answer = (answer or "").strip()
    failures, warnings = [], []
    if not answer:
        failures.append("empty_answer")
    if len(answer) > 4_000:
        failures.append("output_too_long")
    if _SECRET_OUTPUT.search(answer):
        failures.append("credential_leak")
    if _PROMPT_LEAK.search(answer):
        failures.append("prompt_leak")
    if _PROMPT_ECHO.search(answer):
        failures.append("prompt_echo")

    answer_terms = tokens(answer)
    question_terms = _question_terms(question)
    relevance = round(100 * len(answer_terms & question_terms) / len(question_terms), 1) if question_terms else 100.0
    if question_terms and not (answer_terms & question_terms):
        failures.append("not_relevant_to_question")

    grounded = grounding(answer, context)
    evidence_categories = {"Code Retrieval", "Dependency Understanding", "Bug Analysis", "RAG-based Question", "Refactoring"}
    if category in evidence_categories and (context.get("rag") or context.get("sourcegraph")) and (grounded is None or grounded < 18):
        failures.append("insufficient_evidence_support")

    if mode == "fast" and answer and not re.search(r"[.!?][\"'\])}]*$", answer):
        failures.append("fast_format_incomplete")
    if mode == "fast" and len(re.findall(r"[.!?](?:\s|$)", answer)) > max(1, len(context.get("question_parts") or [])):
        warnings.append("fast_answer_has_extra_sentences")

    return {
        "accepted": not failures,
        "failures": failures,
        "warnings": warnings,
        "relevance_percent": relevance,
        "grounding_percent": grounded,
        "checks": {
            "non_empty": bool(answer),
            "relevant": "not_relevant_to_question" not in failures,
            "supported": "insufficient_evidence_support" not in failures,
            "safe": not any(item in failures for item in ("credential_leak", "prompt_leak", "prompt_echo")),
            "format": "fast_format_incomplete" not in failures,
        },
    }
