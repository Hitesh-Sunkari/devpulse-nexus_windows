"""Focused Sourcegraph retrieval with compact, attributable code evidence."""

import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache


SOURCEGRAPH_URL = os.getenv("SOURCEGRAPH_URL", "http://host.docker.internal:7080").rstrip("/")
SOURCEGRAPH_TOKEN = os.getenv("SG_TOKEN", "")
SOURCEGRAPH_PUBLIC_URL = os.getenv("SOURCEGRAPH_PUBLIC_URL", "http://localhost").rstrip("/")
SOURCEGRAPH_REPOSITORY = os.getenv(
    "SOURCEGRAPH_REPOSITORY",
    "github.com/Hitesh-Sunkari/devpulse-nexus_windows",
).strip()
SOURCEGRAPH_TIMEOUT = int(os.getenv("SOURCEGRAPH_TIMEOUT", "8"))

_SEARCH_QUERY = """
query Search($query: String!) {
  search(query: $query) {
    results {
      matchCount
      limitHit
      results {
        __typename
        ... on FileMatch {
          repository { name }
          file { path }
          lineMatches { preview lineNumber }
        }
      }
    }
  }
}
"""

_FILE_CONTEXT_QUERY = """
query FileContext($repository: String!, $path: String!, $start: Int!, $end: Int!) {
  repository(name: $repository) {
    commit(rev: "HEAD") {
      blob(path: $path) { content(startLine: $start, endLine: $end) }
    }
  }
}
"""


def _graphql_request(query, variables):
    body = json.dumps({"query": query, "variables": variables}).encode()
    headers = {"Content-Type": "application/json"}
    if SOURCEGRAPH_TOKEN:
        headers["Authorization"] = f"token {SOURCEGRAPH_TOKEN}"
    request = urllib.request.Request(
        f"{SOURCEGRAPH_URL}/.api/graphql",
        data=body,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=SOURCEGRAPH_TIMEOUT) as response:
        data = json.loads(response.read().decode())
    if data.get("errors"):
        raise RuntimeError(str(data["errors"]))
    return data


def _request(query):
    return _graphql_request(_SEARCH_QUERY, {"query": query})


def sourcegraph_status():
    """Expose connection state without leaking tokens or response details."""
    try:
        _request("type:file count:1")
        return {
            "available": True,
            "authenticated": True,
            "message": "Sourcegraph GraphQL search is connected.",
        }
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            return {
                "available": True,
                "authenticated": False,
                "message": "Sourcegraph is running but DevPulse needs a Sourcegraph access token in SG_TOKEN to retrieve repository evidence.",
            }
        return {"available": False, "authenticated": bool(SOURCEGRAPH_TOKEN), "message": f"Sourcegraph returned HTTP {exc.code}."}
    except Exception as exc:
        return {"available": False, "authenticated": bool(SOURCEGRAPH_TOKEN), "message": f"Sourcegraph is unavailable: {exc}"}


def _focused_query(question):
    lowered = (question or "").lower()
    if "digital twin" in lowered:
        # Prefer the implementation module over wrapper calls in main.py.
        terms = "file:app/digital_twin.py get_digital_twin"
    elif "docker" in lowered or "memory" in lowered:
        # Retrieve the returned diagnostic evidence, not just the function
        # docstring or its API wrapper. `build_diagnosis` is present in the
        # indexed repository's stable implementation, whereas a particular
        # response-field name can change as the schema evolves.
        terms = "file:main.py build_diagnosis"
    elif "rag" in lowered or "knowledge" in lowered:
        terms = "file:app/rag.py retrieve_evidence"
    elif "sourcegraph" in lowered:
        terms = "search_sourcegraph OR SOURCEGRAPH_REPOSITORY"
    else:
        identifiers = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", question or "")
        terms = " OR ".join(identifiers[:4]) or "type:file"
    scope = f"repo:{SOURCEGRAPH_REPOSITORY}" if SOURCEGRAPH_REPOSITORY else ""
    return " ".join(part for part in (scope, terms, "type:file count:20") if part)


def _source_score(item, question):
    path = (item.get("path") or "").lower()
    preview = (item.get("preview") or "").lower()
    lowered = (question or "").lower()
    score = 0
    if "digital twin" in lowered:
        score += 8 if "digital_twin" in path else 0
        score += 5 if "get_digital_twin" in preview else 0
    if "docker" in lowered or "memory" in lowered:
        score += 8 if path.endswith("main.py") else 0
        score += 6 if "build_diagnosis" in preview else 0
    if "rag" in lowered or "knowledge" in lowered:
        score += 8 if any(name in path for name in ("rag", "vector_store", "context")) else 0
    terms = set(re.findall(r"[a-z_]{4,}", lowered))
    score += sum(term in path or term in preview for term in terms)
    return score


@lru_cache(maxsize=128)
def _file_context(repository, path, line_number):
    """Fetch a short Sourcegraph code window after search has selected a line."""
    start = max(0, int(line_number) - 3)
    end = int(line_number) + 6
    try:
        data = _graphql_request(
            _FILE_CONTEXT_QUERY,
            {"repository": repository, "path": path, "start": start, "end": end},
        )
        blob = (((data.get("data") or {}).get("repository") or {}).get("commit") or {}).get("blob") or {}
        content = str(blob.get("content") or "").strip()
        if not content:
            return ""
        return "\n".join(
            f"{start + index + 1}: {value}"
            for index, value in enumerate(content.splitlines())
        )
    except Exception:
        # A preview remains useful if the optional code-window request is not
        # supported by an older Sourcegraph server or a file is unavailable.
        return ""


def search_sourcegraph(question, limit=6):
    """Return ranked, deduplicated evidence for one focused question."""
    data = _request(_focused_query(question))
    results = (((data.get("data") or {}).get("search") or {}).get("results") or {}).get("results") or []
    candidates, seen = [], set()

    for item in results:
        if item.get("__typename") != "FileMatch":
            continue
        repository = (item.get("repository") or {}).get("name")
        path = (item.get("file") or {}).get("path")
        if not repository or not path:
            continue
        for match in item.get("lineMatches") or []:
            line = match.get("lineNumber")
            preview = match.get("preview") or ""
            key = (repository, path, line)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "repository": repository,
                "path": path,
                "line": line,
                "preview": preview,
                "relevance_score": _source_score({"path": path, "preview": preview}, question),
                # Sourcegraph is reached from the app container through
                # host.docker.internal, but the browser must receive the
                # externally reachable site URL (localhost for this setup).
                "url": f"{SOURCEGRAPH_PUBLIC_URL}/{repository}/-/blob/{path}#L{int(line or 0) + 1}",
            })

    candidates.sort(key=lambda item: item["relevance_score"], reverse=True)
    # One strongest match from each file prevents a function definition and a
    # __main__ sample from crowding out separate useful evidence.
    selected, selected_files = [], set()
    for item in candidates:
        if item["path"] in selected_files:
            continue
        selected.append(item)
        selected_files.add(item["path"])
        if len(selected) >= limit:
            break
    for item in selected:
        item["code"] = _file_context(item["repository"], item["path"], item["line"] or 0)
    return selected
