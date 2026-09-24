"""Deterministic request guardrails for DevPulse Nexus.

DevPulse is a repository-aware developer-environment assistant.  These rules
run before retrieval or inference, so a blocked request never reaches Ollama,
Sourcegraph, or the RAG store.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


MAX_INPUT_CHARACTERS = 1_200
MAX_INPUT_WORDS = 220

_SCOPE_TERMS = {
    "app", "bug", "chroma", "chromadb", "code", "codebase", "compose",
    "container", "database", "dependency", "devpulse", "digital", "docker",
    "endpoint", "evaluation", "fastapi", "git", "github", "kubernetes",
    "llm", "memory", "model", "ollama", "phi3", "programming", "python",
    "qwen", "rag", "refactor", "repository", "software", "source", "sourcegraph",
    "telemetry", "tinyllama", "twin", "vector", "workflow",
}

_INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?\b",
    r"\b(?:reveal|show|print|repeat)\s+(?:the\s+)?(?:system|developer)\s+prompt\b",
    r"\bjailbreak\b",
    r"\byou\s+are\s+now\b",
)

_SECRET_PATTERNS = (
    r"\bgithub_pat_[A-Za-z0-9_]+\b",
    r"\bsgp_[A-Za-z0-9]+\b",
    r"\bsk-[A-Za-z0-9_-]{12,}\b",
    r"\b(?:reveal|show|print|extract|dump)\b.{0,60}\b(?:token|secret|password|api[ _-]?key|\.env|environment variables?)\b",
)

_UNSAFE_PATTERNS = (
    r"\brm\s+-rf\b",
    r"\bdrop\s+(?:database|table)\b",
    r"\bformat\s+(?:disk|drive)\b",
    r"\b(?:create|write|build)\b.{0,50}\b(?:malware|ransomware|keylogger)\b",
    r"\b(?:credential stuffing|ddos|denial of service)\b",
)


@dataclass(frozen=True)
class GuardrailDecision:
    """A serialisable decision explaining why inference may or may not run."""

    allowed: bool
    code: str
    message: str
    checks: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def _blocked(code, message, check):
    return GuardrailDecision(False, code, message, (check,))


def _words(text):
    return re.findall(r"[A-Za-z][A-Za-z0-9_-]*", text.lower())


def evaluate_input(question: str, *, enforce: bool = True) -> GuardrailDecision:
    """Validate a request before any external call or local model inference.

    ``enforce=False`` is used only in the guardrail demonstration runner to
    show the old, unprotected routing decision.  Application paths always use
    the default enforced mode.
    """
    text = (question or "").strip()
    if not text:
        return _blocked("empty_input", "Please provide a developer-environment question.", "non_empty")

    violations = []
    if len(text) > MAX_INPUT_CHARACTERS:
        violations.append(("input_too_long", f"Keep requests under {MAX_INPUT_CHARACTERS} characters.", "max_characters"))
    elif len(_words(text)) > MAX_INPUT_WORDS:
        violations.append(("input_too_long", f"Keep requests under {MAX_INPUT_WORDS} words.", "max_words"))

    lowered = text.lower()
    if any(re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL) for pattern in _INJECTION_PATTERNS):
        violations.append(("prompt_injection", "I can help with DevPulse evidence and code, but I cannot follow prompt-override requests.", "prompt_injection"))
    if any(re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL) for pattern in _SECRET_PATTERNS):
        violations.append(("credential_request", "I cannot reveal, process, or reproduce credentials. Rotate exposed credentials and use environment variables locally.", "credential_protection"))
    if any(re.search(pattern, lowered, flags=re.IGNORECASE | re.DOTALL) for pattern in _UNSAFE_PATTERNS):
        violations.append(("unsafe_operation", "I cannot provide destructive or harmful operational instructions. Ask for a safe diagnostic, backup, or recovery procedure instead.", "safe_operations"))

    terms = set(_words(text))
    if not terms & _SCOPE_TERMS:
        violations.append(("outside_scope", "DevPulse Nexus only answers repository, software-engineering, Docker, telemetry, RAG, and local-model questions.", "application_scope"))

    if violations and enforce:
        code, message, check = violations[0]
        return _blocked(code, message, check)

    return GuardrailDecision(
        True,
        "allowed" if enforce else "unprotected_would_route",
        "Request is within the DevPulse Nexus developer-environment scope.",
        ("non_empty", "bounded_input", "safe_operations", "credential_protection", "application_scope"),
    )


def require_evidence(category: str, context: dict, *, docker_available: bool | None = None) -> GuardrailDecision:
    """Refuse factual answers when their required evidence is unavailable."""
    sourcegraph = context.get("sourcegraph") or []
    rag = context.get("rag") or []
    category = category or "Explanation"

    if category in {"Code Retrieval", "Dependency Understanding", "Bug Analysis", "Refactoring"} and not sourcegraph:
        return _blocked(
            "insufficient_repository_evidence",
            "I cannot establish that repository fact because no matching Sourcegraph evidence was retrieved. Try a more specific file, function, or module name.",
            "repository_evidence",
        )
    if category == "RAG-based Question" and not rag:
        return _blocked(
            "insufficient_rag_evidence",
            "I cannot answer that knowledge-base question because no relevant RAG evidence was retrieved.",
            "rag_evidence",
        )
    runtime_words = {"current", "running", "memory", "cpu", "usage", "container"}
    asks_for_runtime_state = any(
        "docker" in str(part.get("question", "")).lower()
        and bool(runtime_words & set(_words(str(part.get("question", "")))))
        for part in context.get("question_parts", [])
    )
    if docker_available is False and asks_for_runtime_state:
        return _blocked(
            "telemetry_unavailable",
            "Docker telemetry is unavailable, so I cannot make a supported claim about the current containers.",
            "docker_telemetry",
        )
    return GuardrailDecision(True, "evidence_available", "Required evidence is available for this request.", ("evidence_available",))


def controlled_response(decision: GuardrailDecision) -> dict:
    """Return the standard no-inference response shape for a blocked request."""
    return {
        "answer": decision.message,
        "guardrail": {**decision.to_dict(), "blocked": True},
        "ai": {"model": None, "latency_seconds": None, "error": decision.code},
    }
