"""Category-wise, repeatable benchmark for DevPulse's seven task types.

Every selected model receives every selected question under identical retrieved
context.  Results are reported by category; no single cross-category accuracy
number is used to select a general-purpose winner.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.docker_monitor import get_docker_metrics
from app.model_runner import run_model
from app.output_validation import validate_output


CATEGORIES = (
    "Explanation",
    "Code Retrieval",
    "Dependency Understanding",
    "Bug Analysis",
    "Code Generation",
    "Refactoring",
    "RAG-based Question",
)

EVALUATION_DIR = Path("evaluation")
QUESTIONS_PATH = EVALUATION_DIR / "questions.json"
RUBRIC_PATH = EVALUATION_DIR / "rubric.json"
REFERENCE_PATH = EVALUATION_DIR / "reference_answers.json"
LIVE_REPORT_PATH = EVALUATION_DIR / "live_category_benchmark.json"
LIVE_CSV_PATH = EVALUATION_DIR / "live_category_benchmark.csv"
LIVE_MD_PATH = EVALUATION_DIR / "live_category_benchmark.md"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "how", "in", "is", "it", "of", "on",
    "or", "should", "the", "this", "to", "was", "what", "when", "where",
    "which", "who", "why", "with", "write", "according",
}


def _read_json(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def _tokens(value):
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_.:/-]*", str(value or ""))
        if token.lower() not in STOPWORDS and len(token) > 1
    }


def _concept_match_score(answer, concepts):
    concepts = [str(item).strip() for item in (concepts or []) if str(item).strip()]
    if not concepts:
        return None
    answer_lower, answer_terms = str(answer or "").lower(), _tokens(answer)
    hits = 0
    for concept in concepts:
        concept_terms = _tokens(concept)
        if concept.lower() in answer_lower or (
            concept_terms and len(concept_terms & answer_terms) >= max(1, math.ceil(len(concept_terms) / 2))
        ):
            hits += 1
    return round(100 * hits / len(concepts), 1)


def _question_relevance(question, answer):
    terms = _tokens(question)
    if not terms:
        return 100.0
    return round(100 * len(terms & _tokens(answer)) / len(terms), 1)


def _extract_code(answer):
    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", str(answer or ""), flags=re.IGNORECASE | re.DOTALL)
    return "\n".join(matches).strip() if matches else str(answer or "").strip()


def _static_code_validation(answer):
    """Static-only check: generated code is parsed but never executed."""
    code = _extract_code(answer)
    if not code or not re.search(r"\b(?:def|class|assert|@app\.)\b", code):
        return False, "no_python_construct"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax_error_line_{exc.lineno}"
    return True, "syntax_valid"


def _ollama_memory_mib():
    try:
        docker = get_docker_metrics()
        for container in docker.get("containers", []):
            if container.get("name") == "devpulse-ollama":
                match = re.search(r"([0-9.]+)\s*(?:MiB|MB)", str(container.get("memory_usage", "")), re.IGNORECASE)
                return round(float(match.group(1)), 2) if match else None
    except Exception:
        pass
    return None


def _question_maps():
    questions = _read_json(QUESTIONS_PATH, [])
    rubric = {item.get("id"): item for item in _read_json(RUBRIC_PATH, []) if isinstance(item, dict)}
    references = {item.get("id"): item for item in _read_json(REFERENCE_PATH, []) if isinstance(item, dict)}
    return questions, rubric, references


def benchmark_tasks(categories=None):
    requested = set(categories or CATEGORIES)
    invalid = requested - set(CATEGORIES)
    if invalid:
        raise ValueError("Unknown benchmark categories: " + ", ".join(sorted(invalid)))
    questions, rubric, references = _question_maps()
    tasks = []
    for task in questions:
        if task.get("category") not in requested:
            continue
        item = dict(task)
        item["rubric"] = rubric.get(task.get("id"), {})
        item["reference"] = references.get(task.get("id"), {})
        tasks.append(item)
    return tasks


def _benchmark_prompt(task, context):
    category, question = task["category"], task["question"]
    format_rule = {
        "Code Retrieval": "Name the exact file and function when the evidence provides them.",
        "Dependency Understanding": "Describe the components and their order; do not invent links.",
        "Bug Analysis": "State the observed behavior, supported cause, and safe next step.",
        "Code Generation": "Return one fenced Python code block and a short explanation. Do not claim it was executed.",
        "Refactoring": "Give a concrete, evidence-based refactoring proposal and its expected benefit.",
        "RAG-based Question": "Answer only from the retrieved knowledge-base evidence.",
    }.get(category, "Give a concise, evidence-based explanation.")
    evidence_parts = []
    for item in (context.get("rag_evidence") or [])[:3]:
        evidence_parts.append("RAG: " + str(item.get("content", ""))[:700])
    for item in (context.get("sourcegraph") or [])[:4]:
        evidence_parts.append(
            "CODE [" + str(item.get("path", "unknown")) + ":" + str(item.get("line", "?")) + "]: "
            + str(item.get("code") or item.get("preview") or "")[:700]
        )
    evidence = "\n\n".join(evidence_parts) or "No relevant evidence was retrieved."
    return f"""You are being evaluated on the DevPulse Nexus category: {category}.

Task: {question}

Use only the evidence below. If it does not establish the answer, explicitly say
that the evidence is insufficient. {format_rule}

EVIDENCE:
{evidence}

ANSWER:
""".strip()


def _retrieval_support(task, context):
    rubric = task.get("rubric") or {}
    targets = rubric.get("required_concepts") or rubric.get("expected_concepts") or []
    evidence = "\n".join(
        [str(item) for item in context.get("rag", [])]
        + [str(item.get("path", "")) + "\n" + str(item.get("code") or item.get("preview") or "") for item in context.get("sourcegraph", [])]
    )
    return _concept_match_score(evidence, targets)


def _task_row(task, model, response, context, ollama_memory_mib):
    rubric = task.get("rubric") or {}
    answer = response.get("answer", "")
    validation = validate_output(task["question"], answer, task["category"], context, mode="benchmark")
    required = rubric.get("required_concepts") or rubric.get("expected_concepts") or []
    expected = rubric.get("expected_concepts") or required
    hallucination_checks = [str(item).lower() for item in rubric.get("hallucination_checks", [])]
    named_hallucinations = sum(check in answer.lower() for check in hallucination_checks)
    hallucination_rate = round(
        100 * named_hallucinations / len(hallucination_checks), 1
    ) if hallucination_checks else 0.0
    if not validation["accepted"]:
        hallucination_rate = max(hallucination_rate, 100.0 if any(
            item in validation["failures"] for item in ("credential_leak", "prompt_leak", "prompt_echo", "insufficient_evidence_support")
        ) else 25.0)
    code_valid, code_reason = (None, None)
    if task["category"] == "Code Generation":
        code_valid, code_reason = _static_code_validation(answer)
    return {
        "question_id": task["id"],
        "category": task["category"],
        "question": task["question"],
        "model": model,
        "answer": answer,
        "error": response.get("error"),
        "latency_seconds": response.get("latency_seconds"),
        "response_tokens": response.get("response_tokens"),
        "ollama_memory_mib": ollama_memory_mib,
        "correctness_percent": _concept_match_score(answer, required),
        "coverage_percent": _concept_match_score(answer, expected),
        "relevance_percent": _question_relevance(task["question"], answer),
        "grounding_percent": validation.get("grounding_percent"),
        "hallucination_rate_percent": hallucination_rate,
        "retrieval_support_percent": _retrieval_support(task, context),
        "output_guard_pass": validation["accepted"],
        "output_validation": validation,
        "static_code_validation_pass": code_valid,
        "static_code_validation_reason": code_reason,
        "executable_test_pass": None,
    }


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(statistics.fmean(values), 2) if values else None


def _p95(rows, key):
    values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
    if not values:
        return None
    return round(values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)], 2)


def _quality_score(category, metrics):
    # Category-specific quality weights.  Latency and token/resource use are
    # reported quantitatively but do not outweigh correctness in winner choice.
    weights = {
        "Explanation": {"correctness_percent": .45, "coverage_percent": .25, "relevance_percent": .15, "grounding_percent": .10, "safety": .05},
        "Code Retrieval": {"correctness_percent": .35, "retrieval_support_percent": .25, "grounding_percent": .20, "relevance_percent": .10, "safety": .10},
        "Dependency Understanding": {"correctness_percent": .40, "coverage_percent": .20, "grounding_percent": .20, "relevance_percent": .10, "safety": .10},
        "Bug Analysis": {"correctness_percent": .40, "coverage_percent": .20, "grounding_percent": .20, "relevance_percent": .10, "safety": .10},
        "Code Generation": {"correctness_percent": .30, "coverage_percent": .15, "static_code_validation_pass_rate": .30, "grounding_percent": .10, "relevance_percent": .05, "safety": .10},
        "Refactoring": {"correctness_percent": .35, "coverage_percent": .25, "grounding_percent": .20, "relevance_percent": .10, "safety": .10},
        "RAG-based Question": {"correctness_percent": .35, "retrieval_support_percent": .30, "grounding_percent": .15, "coverage_percent": .10, "safety": .10},
    }[category]
    weighted, available = 0.0, 0.0
    for key, weight in weights.items():
        value = 100 - metrics["hallucination_rate_percent"] if key == "safety" else metrics.get(key)
        if value is not None:
            weighted += float(value) * weight
            available += weight
    return round(weighted / available, 2) if available else None


def _aggregate_category(category, rows, models):
    model_metrics = {}
    for model in models:
        items = [row for row in rows if row["model"] == model]
        if not items:
            continue
        metrics = {
            "questions": len(items),
            "successful_responses_percent": round(100 * sum(not row["error"] and bool(row["answer"]) for row in items) / len(items), 2),
            "correctness_percent": _mean(items, "correctness_percent"),
            "coverage_percent": _mean(items, "coverage_percent"),
            "relevance_percent": _mean(items, "relevance_percent"),
            "grounding_percent": _mean(items, "grounding_percent"),
            "hallucination_rate_percent": _mean(items, "hallucination_rate_percent"),
            "retrieval_support_percent": _mean(items, "retrieval_support_percent"),
            "output_guard_pass_rate": round(100 * sum(row["output_guard_pass"] for row in items) / len(items), 2),
            "static_code_validation_pass_rate": (round(100 * sum(row["static_code_validation_pass"] is True for row in items) / len(items), 2) if category == "Code Generation" else None),
            "executable_test_pass_rate": None,
            "average_latency_seconds": _mean(items, "latency_seconds"),
            "p95_latency_seconds": _p95(items, "latency_seconds"),
            "average_response_tokens": _mean(items, "response_tokens"),
            "total_response_tokens": sum(row["response_tokens"] or 0 for row in items),
            "average_ollama_memory_mib": _mean(items, "ollama_memory_mib"),
        }
        metrics["quality_score_percent"] = _quality_score(category, metrics)
        model_metrics[model] = metrics
    eligible = [
        (model, data) for model, data in model_metrics.items()
        if data["successful_responses_percent"] > 0 and data["output_guard_pass_rate"] > 0
    ]
    winner = None
    if eligible:
        winner = sorted(
            eligible,
            key=lambda item: (-float(item[1]["quality_score_percent"] or 0), float(item[1]["average_latency_seconds"] or float("inf"))),
        )[0][0]
    return {"models": model_metrics, "winner": winner, "winner_metric": "category-specific quality score; latency breaks exact ties"}


def _write_report(report):
    EVALUATION_DIR.mkdir(exist_ok=True)
    LIVE_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["category", "model", "questions", "quality_score_percent", "correctness_percent", "coverage_percent", "relevance_percent", "grounding_percent", "hallucination_rate_percent", "retrieval_support_percent", "static_code_validation_pass_rate", "output_guard_pass_rate", "average_latency_seconds", "p95_latency_seconds", "average_response_tokens", "total_response_tokens", "average_ollama_memory_mib", "winner"]
    with LIVE_CSV_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for category, data in report["categories"].items():
            for model, metrics in data["models"].items():
                row = {"category": category, "model": model, "winner": data["winner"], **metrics}
                # Keep CSV stable even when the JSON report gains diagnostic
                # metrics that are not represented as a column yet.
                writer.writerow({field: row.get(field) for field in fields})
    lines = ["# DevPulse Nexus — Live Category-Wise Benchmark", "", "## Methodology", "", "- Every selected model receives the identical question set within each category.", "- Winners are selected independently for each category using category-specific quality metrics.", "- Output-guard failures are excluded from winner selection.", "- Code is parsed statically only; executable generated-code test pass rate is reported as unavailable rather than invented.", "", "## Results", ""]
    for category, data in report["categories"].items():
        lines.extend([f"### {category}", "", f"Winner: **{data['winner'] or 'No eligible model'}**", "", "| Model | Quality | Correctness | Coverage | Guard pass | Latency | Tokens |", "|---|---:|---:|---:|---:|---:|---:|"])
        for model, item in data["models"].items():
            lines.append(f"| {model} | {item['quality_score_percent']} | {item['correctness_percent']} | {item['coverage_percent']} | {item['output_guard_pass_rate']}% | {item['average_latency_seconds']}s | {item['average_response_tokens']} |")
        lines.append("")
    LIVE_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_category_benchmark(models, categories=None, on_progress=None):
    """Run all shared questions in selected categories against every model."""
    tasks = benchmark_tasks(categories)
    if not tasks:
        raise ValueError("No benchmark tasks were found.")
    if not models:
        raise ValueError("No selected models are installed.")
    rows = []
    total = len(tasks) * len(models)
    index = 0
    # Importing RAG lazily keeps report-format and guardrail tests independent
    # of a running vector-store volume.
    from app.context import build_context
    for task in tasks:
        context = build_context(task["question"])
        prompt = _benchmark_prompt(task, context)
        for model in models:
            index += 1
            if on_progress:
                on_progress(index, total, task, model, "running")
            response = run_model(model, prompt, max_tokens=180, mode="benchmark", question_parts=1)
            rows.append(_task_row(task, model, response, context, _ollama_memory_mib()))
            if on_progress:
                on_progress(index, total, task, model, "complete")
    categories_in_run = [category for category in CATEGORIES if any(row["category"] == category for row in rows)]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "same_questions_for_all_models": True,
            "models": list(models),
            "categories": categories_in_run,
            "question_count": len(tasks),
            "evaluation_count": len(rows),
            "winner_policy": "Independent category-specific quality score; no overall model winner is reported.",
            "output_testing": "Relevance, evidence support, prompt/credential leakage, and format checks run before an answer is eligible.",
            "code_execution": "Generated code is parsed statically but never executed; executable test-pass rate is unavailable.",
        },
        "categories": {category: _aggregate_category(category, [row for row in rows if row["category"] == category], models) for category in categories_in_run},
        "rows": rows,
    }
    _write_report(report)
    return report


def load_benchmark_report():
    """Return the latest live run, or the existing Week 4 benchmark clearly labelled historical."""
    live = _read_json(LIVE_REPORT_PATH, None)
    if live:
        return {"available": True, "kind": "live", "report": live}
    historical = _read_json(EVALUATION_DIR / "week4_final_category_metrics.json", None)
    if historical:
        return {"available": True, "kind": "historical_week4", "report": historical}
    return {"available": False, "kind": None, "report": None}
