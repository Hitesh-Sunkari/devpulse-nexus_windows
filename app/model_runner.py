
import os
import time
import urllib.request
import urllib.error
import json

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_URL",
    "http://ollama:11434",
).rstrip("/")

if OLLAMA_BASE_URL.endswith("/api/generate"):
    OLLAMA_GENERATE_URL = OLLAMA_BASE_URL
else:
    OLLAMA_GENERATE_URL = OLLAMA_BASE_URL + "/api/generate"


# Models used by the live comparison.
# Run sequentially so only one model is actively loaded at a time.
OLLAMA_MODELS = [
    "qwen2.5:1.5b",
    "phi3:mini",
    "tinyllama",
]


def _matches_requested_model(requested, installed):
    """Treat Ollama's implicit :latest tag as the requested base name."""
    return installed == requested or installed.split(":", 1)[0] == requested


def list_available_models():
    """Return the models that are actually installed in the Ollama service."""
    tags_url = OLLAMA_BASE_URL + "/api/tags"

    try:
        with urllib.request.urlopen(tags_url, timeout=10) as response:
            data = json.loads(response.read().decode())

        names = [
            item.get("name")
            for item in data.get("models", [])
            if item.get("name")
        ]

        # Keep the comparison order stable while still exposing any additional
        # local Ollama models in the Ask Nexus picker.
        # Ollama reports TinyLlama as ``tinyllama:latest`` but accepts
        # ``tinyllama`` for generation. Keep the canonical name so it is not
        # silently omitted from live comparison.
        ordered = [
            model for model in OLLAMA_MODELS
            if any(_matches_requested_model(model, name) for name in names)
        ]
        ordered.extend(
            model for model in names
            if not any(_matches_requested_model(item, model) for item in ordered)
        )
        return ordered

    except Exception:
        # The fixed set is useful during a transient Ollama restart, but callers
        # still receive a truthful ``available`` flag from their status endpoint.
        return []


def run_model(model, prompt, timeout=None):
    start = time.perf_counter()

    if timeout is None:
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        # A comparison runs models sequentially. Unload after each response so
        # Phi-3 is never competing with Qwen or TinyLlama for the WSL memory
        # budget on a local Docker Desktop installation.
        "keep_alive": "0",
        "options": {
            # Enough for a useful, structured answer without needlessly
            # extending CPU-only inference time for Phi-3 Mini.
            "num_predict": 128,
            "temperature": 0,
        },
    }).encode()

    request = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            data = json.loads(
                response.read().decode()
            )

        latency = time.perf_counter() - start

        return {
            "model": model,
            "answer": data.get("response", ""),
            "latency_seconds": round(latency, 2),
            "total_duration_ns": data.get("total_duration"),
            "load_duration_ns": data.get("load_duration"),
            "eval_duration_ns": data.get("eval_duration"),
            "response_tokens": data.get("eval_count"),
            "error": None,
        }

    except Exception as exc:
        latency = time.perf_counter() - start

        return {
            "model": model,
            "answer": "",
            "latency_seconds": round(latency, 2),
            "total_duration_ns": None,
            "load_duration_ns": None,
            "eval_duration_ns": None,
            "response_tokens": None,
            "error": str(exc),
        }


def run_all_models(prompt, models=None, on_progress=None):
    results = {}

    models = models or OLLAMA_MODELS

    for model in models:
        if on_progress:
            on_progress(model, "running", None)
        results[model] = run_model(model, prompt)
        if on_progress:
            on_progress(model, "complete", results[model])

    return results
