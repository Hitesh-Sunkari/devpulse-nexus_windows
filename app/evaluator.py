"""Deterministic, evidence-aware evaluation for live model comparisons.

These are transparent evidence-alignment indicators, not a claim of semantic
ground truth.  They deliberately reject an otherwise fluent answer that omits
an explicit part of the user's request.
"""

import json
import re
from pathlib import Path

from app.question_analysis import analyse_question, important_terms


REFERENCE_PATH = Path("evaluation/reference_answers.json")
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "is",
    "are", "was", "were", "does", "do", "what", "how", "why", "where",
    "which", "for", "with", "from", "that", "this", "it", "be", "can",
    "should", "as", "by", "into", "about", "function", "purpose",
}

# Merely repeating the named component in a question is not an explanation.
# These signals represent the minimum factual content needed for the recurring
# DevPulse concepts to count as addressed.
CONCEPT_EXPLANATION_SIGNALS = {
    "digital twin": {"telemetry", "snapshot", "environment", "host", "container", "docker", "combine", "collect"},
    "docker": {"container", "telemetry", "diagnosis", "contribut", "sole", "budget", "host", "memory"},
    "sourcegraph": {"repository", "search", "evidence", "code"},
    "rag": {"retrieve", "knowledge", "chunk", "context", "chromadb"},
    "ollama": {"model", "generate", "inference", "local"},
}
DANGLING_ENDINGS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in",
    "is", "of", "or", "that", "the", "to", "what", "with",
}
EXPLANATORY_PREDICATE = re.compile(
    r"\b(?:is|are|can|may|does|combines?|collects?|creates?|returns?|"
    r"provides?|uses?|produces?|builds?|captures?|represents?|contributes?)\b",
    flags=re.IGNORECASE,
)

# These verbs describe the form of a request, not its subject.  An answer to
# “is Docker responsible for memory usage?” cannot qualify merely because it
# repeats “Docker responsible”; it must also deal with “memory”.
GENERIC_REQUEST_TERMS = {
    "answer", "compare", "describe", "explain", "function", "purpose",
    "question", "responsible", "usage", "working",
}


def _answer_content(answer):
    """Remove numbered question echoes before measuring an actual answer."""
    useful = []
    question_heading = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:\d+[.)]\s*)?"
        r"(?:what|how|why|where|which|who|is|are|does|do|can|could)\b",
        flags=re.IGNORECASE,
    )
    for line in (answer or "").splitlines():
        cleaned = line.strip()
        if not cleaned or question_heading.match(cleaned):
            continue
        useful.append(cleaned)
    return "\n".join(useful)


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
            return next((data[key] for key in ("questions", "references", "data") if isinstance(data.get(key), list)), [])
    except (OSError, json.JSONDecodeError):
        pass
    return []


def find_reference(question):
    normalized = question.strip().lower()
    return next((item for item in load_references() if str(item.get("question", "")).strip().lower() == normalized), None)


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
    matched = sum(concept.lower() in answer_lower or bool(tokens(concept) and tokens(concept).issubset(answer_tokens)) for concept in concepts)
    return round(100 * matched / len(concepts), 1)


def _context_text(context):
    values = []
    values.extend(str(item) for item in context.get("rag", []))
    values.extend(
        str(source.get(field, ""))
        for source in context.get("sourcegraph", [])
        for field in ("repository", "path", "preview", "code")
    )
    if context.get("telemetry"):
        values.append(json.dumps(context["telemetry"], sort_keys=True))
    if context.get("diagnosis"):
        values.append(json.dumps(context["diagnosis"], sort_keys=True))
    return "\n".join(values)


def question_completion(question, answer, context):
    """Measure whether each explicit user request is materially addressed."""
    meaningful_answer = _answer_content(answer)
    answer_lower = meaningful_answer.lower()
    answer_terms = tokens(meaningful_answer)
    parts = context.get("question_parts") or analyse_question(question)
    details = []

    for part in parts:
        terms = set(part.get("terms") or important_terms(part.get("question", "")))
        anchors = set(part.get("anchors") or [])
        matched_terms = sorted(term for term in terms if term in answer_terms)
        matched_anchors = sorted(anchor for anchor in anchors if anchor in answer_lower)
        anchor_words = {
            word
            for anchor in anchors
            for word in re.findall(r"[a-z]+", anchor)
        }
        subject_terms = terms - anchor_words - GENERIC_REQUEST_TERMS
        matched_subject_terms = sorted(
            term for term in subject_terms if term in answer_terms
        )
        expected_signals = set().union(
            *(CONCEPT_EXPLANATION_SIGNALS.get(anchor, set()) for anchor in anchors)
        ) if anchors else set()
        matched_signals = sorted(
            signal for signal in expected_signals
            if signal in answer_lower
        )
        term_score = 100 * len(matched_terms) / len(terms) if terms else 100.0
        anchor_score = 100 * len(matched_anchors) / len(anchors) if anchors else 100.0
        signal_score = 100 * len(matched_signals) / len(expected_signals) if expected_signals else 100.0
        score = round(term_score * 0.45 + anchor_score * 0.35 + signal_score * 0.20, 1)
        # A named DevPulse concept must be explicitly named, while ordinary
        # questions also need enough distinctive terms to avoid keyword-only
        # answers being marked complete.
        addressed = bool(matched_anchors) if anchors else term_score >= 45
        if anchors and term_score < 25:
            addressed = False
        # A concrete subject in the user's request (memory, network traffic,
        # a repository name, etc.) must be mentioned.  This prevents a model
        # answering a nearby but different Docker question from winning.
        if subject_terms and not matched_subject_terms:
            addressed = False
        if expected_signals and not matched_signals:
            addressed = False
        if anchors and not EXPLANATORY_PREDICATE.search(meaningful_answer):
            addressed = False
        details.append({
            "index": part.get("index"),
            "question": part.get("question"),
            "score": score,
            "addressed": addressed,
            "matched_terms": matched_terms,
            "matched_anchors": matched_anchors,
            "subject_terms": sorted(subject_terms),
            "matched_subject_terms": matched_subject_terms,
            "matched_signals": matched_signals,
        })

    coverage = round(sum(item["score"] for item in details) / len(details), 1) if details else 0.0
    final_word = re.findall(r"[A-Za-z]+", meaningful_answer.lower())
    appears_truncated = bool(final_word and final_word[-1] in DANGLING_ENDINGS)
    return {
        "coverage": coverage,
        "is_complete": bool(details) and all(item["addressed"] for item in details) and not appears_truncated,
        "appears_truncated": appears_truncated,
        "parts": details,
    }


def _citation_score(answer, context):
    known_paths = {str(item.get("path")) for item in context.get("sourcegraph", []) if item.get("path")}
    cited_paths = set(re.findall(r"\[([^\]\n]+?\.py):\d+\]", answer or ""))
    if not known_paths:
        return 100.0, []
    if not cited_paths:
        return 15.0, []
    supported = [path for path in cited_paths if path in known_paths]
    return round(100 * len(supported) / len(cited_paths), 1), sorted(cited_paths - set(supported))


def _numeric_claim_score(answer, context):
    evidence = re.sub(r"\s+", "", _context_text(context).lower())
    claims = re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|gib|mib|gb|mb|cores?)\b", answer or "", flags=re.IGNORECASE)
    if not claims:
        return 100.0, []
    unsupported = [claim for claim in claims if re.sub(r"\s+", "", claim.lower()) not in evidence]
    return round(100 * (1 - len(unsupported) / len(claims)), 1), unsupported


def _contradictions(answer, context):
    answer_lower = (answer or "").lower()
    diagnosis = context.get("diagnosis") or {}
    assessment = diagnosis.get("assessment")
    problems = []
    if assessment == "potentially_significant" and re.search(r"docker (?:is )?(?:not responsible|cannot be responsible|isn't responsible)", answer_lower):
        problems.append("contradicts the deterministic Docker contribution assessment")
    if assessment in {"unlikely", "not_responsible"} and re.search(r"docker (?:is )?(?:the |sole )?(?:cause|responsible)", answer_lower):
        problems.append("overstates Docker responsibility against the deterministic assessment")
    return problems


def grounding(answer, context):
    answer_tokens, evidence_tokens = tokens(answer), tokens(_context_text(context))
    if not answer_tokens or not evidence_tokens:
        return None
    overlap = 100 * len(answer_tokens & evidence_tokens) / len(answer_tokens)
    citation_score, _ = _citation_score(answer, context)
    numeric_score, _ = _numeric_claim_score(answer, context)
    return round(overlap * 0.55 + citation_score * 0.30 + numeric_score * 0.15, 1)


def evidence_safety(answer, context):
    citation_score, unsupported_paths = _citation_score(answer, context)
    numeric_score, unsupported_numbers = _numeric_claim_score(answer, context)
    contradictions = _contradictions(answer, context)
    penalty = len(unsupported_paths) * 25 + len(unsupported_numbers) * 12 + len(contradictions) * 35
    # Citation and numeric support are evidence signals. Contradictions and
    # invented paths/numbers are explicit safety failures.
    score = max(0.0, min(100.0, (citation_score * 0.45 + numeric_score * 0.55) - penalty))
    return round(score, 1), {
        "unsupported_paths": unsupported_paths,
        "unsupported_numbers": unsupported_numbers,
        "contradictions": contradictions,
    }


def clarity(answer):
    words = (answer or "").split()
    if not words:
        return 0.0
    count = len(words)
    lines = [line.strip().lower() for line in (answer or "").splitlines() if line.strip()]
    repetition = len(lines) - len(set(lines))
    score = 100.0 if 18 <= count <= 135 else 75.0 if 8 <= count <= 180 else 45.0
    return round(max(0.0, score - repetition * 15), 1)


def category_scores(question, answer, context, reference=None):
    completion = question_completion(question, answer, context)
    ground = grounding(answer, context)
    safety, safety_details = evidence_safety(answer, context)
    coverage = completion["coverage"]
    relevance = coverage
    if reference:
        reference_score = concept_coverage(answer, reference)
        correctness = round(reference_score * 0.70 + (ground or 0) * 0.20 + safety * 0.10, 1)
    else:
        correctness = round(coverage * 0.45 + (ground or 0) * 0.45 + safety * 0.10, 1)
    if not completion["is_complete"]:
        correctness = min(correctness, 45.0)
    clarity_score = clarity(answer)
    completeness = round(coverage * 0.75 + (ground or 0) * 0.15 + clarity_score * 0.10, 1)
    return {
        "Correctness": correctness,
        "Relevance": relevance,
        "Grounding": ground,
        "Coverage": coverage,
        "Completeness": completeness,
        "Clarity": clarity_score,
        "Evidence safety": safety,
    }, completion, safety_details


def overall_score(scores, completion):
    weights = {
        "Correctness": 0.25,
        "Relevance": 0.15,
        "Grounding": 0.20,
        "Coverage": 0.20,
        "Completeness": 0.10,
        "Clarity": 0.05,
        "Evidence safety": 0.05,
    }
    measured_weight = sum(weights[name] for name, value in scores.items() if value is not None)
    if not measured_weight:
        return None
    score = sum(weights[name] * value for name, value in scores.items() if value is not None) / measured_weight
    # Missing a user-requested part makes an answer ineligible to win and
    # visibly caps its headline score rather than hiding the failure.
    if not completion["is_complete"]:
        score = min(score, 59.0)
    return round(score, 1)


def evaluate_response(question, category, response, context):
    answer = response.get("answer", "")
    reference = find_reference(question)
    scores, completion, safety_details = category_scores(question, answer, context, reference)
    warnings = []
    missed = [part["question"] for part in completion["parts"] if not part["addressed"]]
    if missed:
        warnings.append("Did not address: " + "; ".join(missed))
    if completion["appears_truncated"]:
        warnings.append("Answer appears to end mid-sentence")
    if safety_details["contradictions"]:
        warnings.extend(safety_details["contradictions"])
    if safety_details["unsupported_numbers"]:
        warnings.append("Contains unsupported numeric claim(s): " + ", ".join(safety_details["unsupported_numbers"]))
    result = dict(response)
    result.update({
        "category": category,
        "categories": scores,
        "relevance": scores["Relevance"],
        "grounding": scores["Grounding"],
        "hallucination": round(100 - scores["Evidence safety"], 1),
        "coverage": scores["Coverage"],
        "correctness": scores["Correctness"],
        "score": overall_score(scores, completion),
        "completion": completion,
        "decision_eligible": completion["is_complete"] and not response.get("error"),
        "evaluation_warnings": warnings,
        "evaluation_basis": "Reference-concept evaluation" if reference else "Deterministic completion and evidence-alignment indicators",
    })
    return result


def evaluate_all(question, category, responses, context):
    return {model: evaluate_response(question, category, response, context) for model, response in responses.items()}
