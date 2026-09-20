
from app.rag import retrieve_context
from app.sourcegraph import search_sourcegraph


def build_context(question):
    try:
        rag_documents = retrieve_context(
            question,
            number_of_results=2,
        )
    except Exception:
        rag_documents = []

    try:
        sourcegraph_results = search_sourcegraph(
            question,
            limit=8,
        )
    except Exception:
        sourcegraph_results = []

    return {
        "rag": rag_documents or [],
        "sourcegraph": sourcegraph_results or [],
    }


def format_context(context):
    parts = []

    rag = context.get("rag", [])
    if rag:
        parts.append("KNOWLEDGE BASE:\n")
        for i, document in enumerate(rag, 1):
            parts.append(
                f"[Knowledge {i}]\n{document}\n"
            )

    sourcegraph = context.get("sourcegraph", [])
    if sourcegraph:
        parts.append("\nREPOSITORY SOURCES:\n")
        for i, source in enumerate(sourcegraph, 1):
            parts.append(
                f"[Repository Source {i}]\n"
                f"Repository: {source.get('repository')}\n"
                f"File: {source.get('path')}\n"
                f"Line: {source.get('line')}\n"
                f"Match: {source.get('preview')}\n"
            )

    if not parts:
        return "No retrieved context was available."

    return "\n".join(parts)


def build_combined_prompt(question, context):
    return f"""
You are DevPulse Nexus Repository Intelligence.

Answer the user's question using ONLY the supplied repository
sources and knowledge-base context when those sources are relevant.

Do not invent files, functions, dependencies, APIs, or architecture.

If the supplied context does not establish something, say that
the available context does not establish it.

Give a concise technical answer and mention relevant file paths
and functions when the repository sources support them.

USER QUESTION:
{question}

{format_context(context)}
""".strip()
