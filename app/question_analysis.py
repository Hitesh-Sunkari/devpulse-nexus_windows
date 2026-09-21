"""Small, deterministic helpers for analysing a user request.

The local models are intentionally lightweight.  Splitting a clearly compound
question before retrieval gives every model an explicit checklist and prevents
one fluent answer from silently ignoring half of the request.
"""

import re


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "how", "i", "in", "is", "it", "of",
    "on", "or", "should", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you",
    "your", "about", "explain", "tell", "me", "please",
}

_QUESTION_START = (
    "what", "how", "why", "where", "which", "who", "is", "are", "do",
    "does", "can", "could", "should", "will", "explain", "describe",
    "tell", "give", "compare",
)


def _normalise(value):
    return re.sub(r"\s+", " ", (value or "").strip())


def split_question(question, maximum_parts=3):
    """Split only explicit compound questions, never arbitrary prose.

    For example, ``is Docker responsible ... and what is Digital Twin?`` is
    two independently answerable requests.  A phrase such as ``Docker and
    Ollama`` is deliberately left intact.
    """
    cleaned = _normalise(question)
    if not cleaned:
        return []

    boundary = (
        r"\s+(?:and|also|then)\s+(?=(?:" + "|".join(_QUESTION_START) + r")\b)"
    )
    parts = re.split(boundary, cleaned, flags=re.IGNORECASE)

    # Multiple explicit question marks are also a reliable separator.
    if len(parts) == 1 and cleaned.count("?") > 1:
        parts = [part.strip() for part in cleaned.split("?") if part.strip()]

    result = []
    for part in parts:
        part = _normalise(part).strip(" ;")
        if part and part not in result:
            result.append(part)
        if len(result) >= maximum_parts:
            break
    return result or [cleaned]


def important_terms(question):
    """Return stable content terms used for completion and retrieval checks."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_-]*", (question or "").lower())
    return {
        word for word in words
        if word not in STOPWORDS and len(word) >= 3
    }


def required_anchors(question):
    """Concepts that must be addressed for common DevPulse questions."""
    lowered = (question or "").lower()
    anchors = set()
    if "digital twin" in lowered:
        anchors.add("digital twin")
    if "sourcegraph" in lowered:
        anchors.add("sourcegraph")
    if "docker" in lowered:
        anchors.add("docker")
    if "rag" in lowered or "knowledge base" in lowered:
        anchors.add("rag")
    if "ollama" in lowered or "model" in lowered:
        anchors.add("ollama")
    return anchors


def analyse_question(question):
    """Return answerable parts with explicit anchors and useful terms."""
    return [
        {
            "index": index,
            "question": part,
            "anchors": sorted(required_anchors(part)),
            "terms": sorted(important_terms(part)),
        }
        for index, part in enumerate(split_question(question), start=1)
    ]
