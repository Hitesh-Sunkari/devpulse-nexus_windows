"""Deterministic, evidence-aware scoring for live model comparisons.

For benchmark questions the metrics use matching reference concepts. For live
questions they measure the actual answer against the telemetry, RAG documents,
and Sourcegraph results collected in that exact comparison job.
"""

import json
import re
from pathlib import Path


REFERENCE_PATH = Path("evaluation/reference_answers.json")
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "is",
    "are", "was", "were", "does", "do", "what", "how", "why", "where",
    "which", "for", "with", "from", "that", "this", "it", "be", "can",
    "should", "as", "by", "into", "about", "function", "purpose",
}
CATEGORIES = [
    "Correctness", "Relevance", "Grounding", "Coverage", "Completeness",
    "Clarity", "Hallucination",
]


def tokens(text):
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_.:/-]*", (text or "").lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def load_references():
    if not REFERENCE_PATH.exists():
        return []
    try:
        data = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return next((data[key] for key in ("questions", "references", "data")
                         if isinstance(data.get(key), list)), [])
    except (OSError, json.JSONDecodeError):
        pass
    return []


def find_reference(question):
    normalized = question.strip().lower()
    return next((item for item in load_references()
                 if str(item.get("question", "")).strip().lower() == normalized), None)


def concept_list(reference):
    concepts = reference.get("expected_concepts", [])
    if isinstance(concepts, dict):
        concepts = list(concepts)
    if not isinstance(concepts, list):
        concepts = [concepts]
    return [str(item).strip() for item in concepts if str(item).strip()]


def concept_coverage(answer, reference):
    concepts = concept_list(reference)
    answer_lower = (answer or "").lower()
    answer_tokens = tokens(answer)
    if not concepts:
        reference_tokens = tokens(reference.get("reference_answer", ""))
        return round(100 * len(reference_tokens & answer_tokens) / len(reference_tokens), 1) if reference_tokens else None
    matched = sum(
        concept.lower() in answer_lower
        or bool(tokens(concept) and tokens(concept).issubset(answer_tokens))
        for concept in concepts
    )
    return round(100 * matched / len(concepts), 1)


def _context_text(context):
    values = []
    values.extend(str(item) for item in context.get("rag", []))
    values.extend(str(source.get(field, "")) for source in context.get("sourcegraph", [])
                  for field in ("repository", "path", "preview", "code"))
    if context.get("telemetry"):
        values.append(json.dumps(context["telemetry"], sort_keys=True))
    if context.get("diagnosis"):
        values.append(json.dumps(context["diagnosis"], sort_keys=True))
    return "\n".join(values)


def relevance(question, answer):
    question_tokens, answer_tokens = tokens(question), tokens(answer)
    if not question_tokens or not answer_tokens:
        return 0.0
    return round(100 * len(question_tokens & answer_tokens) / len(question_tokens), 1)


def grounding(answer, context):
    answer_tokens, evidence_tokens = tokens(answer), tokens(_context_text(context))
    if not answer_tokens or not evidence_tokens:
        return None
    supported = len(answer_tokens & evidence_tokens)
    claim_tokens = max(1, min(len(answer_tokens), len(evidence_tokens)))
    return round(min(100.0, 100 * supported / claim_tokens), 1)


def hallucination_safety(answer, context):
    known_paths = {str(item.get("path")) for item in context.get("sourcegraph", []) if item.get("path")}
    referenced_paths = set(re.findall(r"(?:[\w.-]+/)*[\w.-]+\.py", answer or ""))
    if not referenced_paths:
        return 100.0
    if not known_paths:
        return None
    unsupported = [path for path in referenced_paths if path not in known_paths]
    return round(100 * (1 - len(unsupported) / len(referenced_paths)), 1)


def clarity(answer):
    words = (answer or "").split()
    if not words:
        return 0.0
    count = len(words)
    lines = [line.strip().lower() for line in (answer or "").splitlines() if line.strip()]
    repetition = len(lines) - len(set(lines))
    score = 100.0 if 20 <= count <= 220 else 75.0 if 8 <= count <= 320 else 45.0
    return round(max(0.0, score - repetition * 12), 1)


def category_scores(question, answer, context, reference=None):
    rel = relevance(question, answer)
    ground = grounding(answer, context)
    safety = hallucination_safety(answer, context)
    if reference:
        coverage = concept_coverage(answer, reference)
        correctness = coverage
    else:
        # Live correctness is evidence alignment, not fabricated ground truth.
        coverage = rel
        correctness = round((rel + ground) / 2, 1) if ground is not None else rel
    evidence_factor = ground if ground is not None else rel
    clarity_score = clarity(answer)
    completeness = round((coverage + evidence_factor + clarity_score) / 3, 1)
    return {
        "Correctness": correctness,
        "Relevance": rel,
        "Grounding": ground,
        "Coverage": coverage,
        "Completeness": completeness,
        "Clarity": clarity_score,
        "Hallucination": safety,
    }


def overall_score(scores):
    measured = [value for value in scores.values() if value is not None]
    return round(sum(measured) / len(measured), 1) if measured else None


def evaluate_response(question, category, response, context):
    answer = response.get("answer", "")
    reference = find_reference(question)
    scores = category_scores(question, answer, context, reference)
    result = dict(response)
    result.update({
        "category": category,
        "categories": scores,
        "relevance": scores["Relevance"],
        "grounding": scores["Grounding"],
        "hallucination": None if scores["Hallucination"] is None else round(100 - scores["Hallucination"], 1),
        "coverage": scores["Coverage"],
        "correctness": scores["Correctness"],
        "score": overall_score(scores),
        "evaluation_basis": (
            "Reference-concept evaluation" if reference else
            "Live evidence alignment across telemetry, RAG, and repository context"
        ),
    })
    return result


def evaluate_all(question, category, responses, context):
    return {model: evaluate_response(question, category, response, context)
            for model, response in responses.items()}
