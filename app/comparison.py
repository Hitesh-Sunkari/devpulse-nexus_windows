
import json
from pathlib import Path


HISTORY_PATH = Path(
    "evaluation/week4_final_category_metrics.json"
)

CATEGORIES = [
    "Explanation",
    "Code Retrieval",
    "Dependency Understanding",
    "Bug Analysis",
    "Code Generation",
    "Refactoring",
    "RAG-based Question",
]

MODELS = [
    "qwen2.5:1.5b",
    "phi3:mini",
    "tinyllama",
]


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_historical():
    if not HISTORY_PATH.exists():
        return {
            "available": False,
            "categories": {},
        }

    try:
        data = json.loads(
            HISTORY_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "categories": {},
        }

    raw_categories = data.get(
        "categories",
        {},
    )

    output = {}

    for category in CATEGORIES:
        info = raw_categories.get(
            category,
            {},
        )

        models = (
            info.get("models", {})
            if isinstance(info, dict)
            else {}
        )

        # Support the existing metric file even if
        # model records are directly inside the category.
        if not models and isinstance(info, dict):
            for model in MODELS:
                if model in info:
                    models[model] = info[model]

        normalized = {}

        for model in MODELS:
            metric = models.get(model, {})

            if not isinstance(metric, dict):
                metric = {}

            normalized[model] = {
                "correctness": _number(
                    metric.get("correctness_percent",
                              metric.get("correct")),
                ),
                "relevance": _number(
                    metric.get("relevance_percent",
                              metric.get("relevance")),
                ),
                "coverage": _number(
                    metric.get("coverage_percent",
                              metric.get("coverage")),
                ),
                "hallucination": _number(
                    metric.get("hallucination_percent",
                              metric.get("hallucination")),
                ),
                "latency": _number(
                    metric.get("latency_seconds",
                              metric.get("latency")),
                ),
            }

        output[category] = {
            "models": normalized,
            "winner": info.get("winner"),
        }

    return {
        "available": True,
        "categories": output,
        "source": str(HISTORY_PATH),
    }


def compare_models(evaluated, category):
    completed = {
        model: info
        for model, info in evaluated.items()
        if info.get("error") is None
        and info.get("answer")
        and info.get("score") is not None
    }

    if not completed:
        return {
            "best_model": None,
            "reason": "No model completed successfully.",
            "category": category,
            "confidence": "none",
        }

    valid = {
        model: info for model, info in completed.items()
        if info.get("decision_eligible")
    }

    if not valid:
        return {
            "best_model": None,
            "reason": (
                "No reliable winner: every completed answer missed at least "
                "one explicit part of the request."
            ),
            "category": category,
            "confidence": "low",
            "scores": {model: info["score"] for model, info in completed.items()},
        }

    ordered = sorted(
        valid.items(),
        key=lambda x: x[1]["score"],
        reverse=True,
    )

    best_model, best = ordered[0]

    return {
        "best_model": best_model,
        "reason": (
            f"Highest eligible evidence-alignment score for "
            f"the {category} request: "
            f"{best['score']:.1f}/100."
        ),
        "category": category,
        "confidence": "evidence-aligned",
        "scores": {
            model: info["score"]
            for model, info in valid.items()
        },
    }
