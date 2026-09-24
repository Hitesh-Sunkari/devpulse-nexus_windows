
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

# Fast comparison is the default for CPU-bound local models. It asks for a
# concise, structured answer rather than allowing a small model to spend
# minutes producing generic background material. Thorough mode remains useful
# for deliberate deep dives.
FAST_COMPARISON_TOKENS = int(os.getenv("COMPARISON_FAST_NUM_PREDICT", "56"))
THOROUGH_COMPARISON_TOKENS = int(os.getenv("COMPARISON_THOROUGH_NUM_PREDICT", "192"))
COMPARISON_CONTEXT_TOKENS = int(os.getenv("COMPARISON_NUM_CTX", "1536"))
# Docker Desktop exposes twelve CPUs here. Reserve headroom for Sourcegraph,
# FastAPI, and the host while avoiding Ollama's conservative one-thread path.
OLLAMA_NUM_THREADS = int(os.getenv("OLLAMA_NUM_THREADS", "8"))
COMPARISON_KEEP_ALIVE = os.getenv("OLLAMA_COMPARISON_KEEP_ALIVE", "5m")


def comparison_token_budget(mode, question_parts=1):
    """Choose a response budget that is fast without cutting off compound answers."""
    if mode == "thorough":
        return THOROUGH_COMPARISON_TOKENS
    # A single direct question remains very fast. Each explicit additional
    # request receives a small allocation so a model can close its first
    # section and actually explain the next one.
    return min(112, max(FAST_COMPARISON_TOKENS, 44 * max(1, question_parts)))


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


def run_model(
    model,
    prompt,
    timeout=None,
    max_tokens=None,
    mode="fast",
    question_parts=1,
):
    start = time.perf_counter()

    if timeout is None:
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "300"))

    requested_tokens = max_tokens or FAST_COMPARISON_TOKENS
    options = {
        "num_predict": requested_tokens,
        "num_ctx": COMPARISON_CONTEXT_TOKENS,
        "num_thread": OLLAMA_NUM_THREADS,
        "temperature": 0,
    }
    # One-part Fast responses must stop before a small model starts an
    # unnecessary second numbered item or echoes a prompt section.
    if mode == "fast" and question_parts == 1:
        options["stop"] = ["\n2.", "\n3.", "\n\n", "\n#"]

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        # The models run sequentially, but retaining them briefly avoids a
        # costly reload during the user's next comparison.  The three chosen
        # models fit comfortably within the configured Docker/WSL budget.
        "keep_alive": COMPARISON_KEEP_ALIVE,
        "options": options,
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
            "done_reason": data.get("done_reason"),
            # Ollama marks length-limited output explicitly on supported
            # versions.  The token-count fallback protects older versions.
            "truncated": (
                data.get("done_reason") in {"length", "max_tokens"}
                or int(data.get("eval_count") or 0) >= requested_tokens
            ),
            "mode": mode,
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
            "done_reason": None,
            "truncated": False,
            "mode": mode,
            "error": str(exc),
        }


def run_all_models(
    prompt,
    models=None,
    on_progress=None,
    max_tokens=None,
    mode="fast",
    question_parts=1,
):
    results = {}

    models = models or OLLAMA_MODELS

    for model in models:
        if on_progress:
            on_progress(model, "running", None)
        results[model] = run_model(
            model,
            prompt,
            max_tokens=max_tokens,
            mode=mode,
            question_parts=question_parts,
        )
        if on_progress:
            on_progress(model, "complete", results[model])

    return results
