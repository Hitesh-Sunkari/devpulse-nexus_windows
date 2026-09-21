"""Evidence retrieval and prompt context construction for DevPulse."""

from app.question_analysis import analyse_question
from app.rag import retrieve_evidence
from app.sourcegraph import search_sourcegraph


def _deduplicate(items, key, limit):
    output, seen = [], set()
    for item in items:
        value = key(item)
        if value in seen:
            continue
        seen.add(value)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def build_context(question):
    """Retrieve diverse, attributable evidence for every explicit question part."""
    parts = analyse_question(question)
    rag_evidence, sourcegraph_evidence, retrieval_parts = [], [], []

    for part in parts:
        text = part["question"]
        try:
            part_rag = retrieve_evidence(text, number_of_results=2)
        except Exception as exc:
            part_rag = []
            rag_error = str(exc)
        else:
            rag_error = None

        try:
            # A focused query per sub-question produces less duplicate and
            # more relevant Sourcegraph evidence than one broad natural query.
            part_sourcegraph = search_sourcegraph(text, limit=3)
        except Exception as exc:
            part_sourcegraph = []
            sourcegraph_error = str(exc)
        else:
            sourcegraph_error = None

        for item in part_rag:
            rag_evidence.append({**item, "question_part": part["index"]})
        for item in part_sourcegraph:
            sourcegraph_evidence.append({**item, "question_part": part["index"]})
        retrieval_parts.append({
            **part,
            "rag_count": len(part_rag),
            "sourcegraph_count": len(part_sourcegraph),
            "rag_error": rag_error,
            "sourcegraph_error": sourcegraph_error,
        })

    rag_evidence = _deduplicate(
        rag_evidence,
        lambda item: (item.get("source"), item.get("chunk"), item.get("content")),
        limit=4,
    )
    sourcegraph_evidence = _deduplicate(
        sourcegraph_evidence,
        lambda item: (item.get("repository"), item.get("path"), item.get("line")),
        limit=6,
    )

    return {
        "question_parts": parts,
        "retrieval_parts": retrieval_parts,
        "rag": [item["content"] for item in rag_evidence],
        "rag_evidence": rag_evidence,
        "sourcegraph": sourcegraph_evidence,
    }


def format_context(context, max_rag_characters=420, max_code_characters=520):
    """Format a compact evidence packet for CPU-bound local models."""
    parts = []
    for item in context.get("rag_evidence", [])[:3]:
        parts.append(
            f"RAG [{item.get('source', 'knowledge base')}]: "
            f"{item.get('content', '')[:max_rag_characters]}"
        )

    for item in context.get("sourcegraph", [])[:4]:
        location = f"{item.get('path', 'unknown')}:{item.get('line', '?')}"
        excerpt = item.get("code") or item.get("preview") or ""
        parts.append(f"CODE [{location}]: {excerpt[:max_code_characters]}")

    return "\n".join(parts) if parts else "No relevant retrieval evidence was available."


def build_combined_prompt(question, context):
    """Compact prompt used by the single-model Ask Nexus endpoint."""
    parts = context.get("question_parts") or analyse_question(question)
    checklist = "\n".join(
        f"{item['index']}. {item['question']}" for item in parts
    )
    return f"""You are DevPulse Nexus. Answer every numbered request below.

Rules:
- Use only the supplied evidence for factual code, telemetry, and architecture claims.
- If evidence is insufficient, say so; do not fill gaps with generic advice.
- Use one short numbered section per request and keep the answer under 140 words.
- Cite supplied code as [path:line] when making a repository claim.

REQUESTS:
{checklist}

EVIDENCE:
{format_context(context)}
""".strip()
