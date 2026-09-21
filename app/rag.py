import os
from functools import lru_cache

import chromadb

from sentence_transformers import SentenceTransformer


DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
MODEL_NAME = "all-MiniLM-L6-v2"

# The local knowledge base currently contains Docker troubleshooting material.
# A semantic nearest neighbour is still returned for unrelated questions, so a
# small lexical guard prevents that material from being presented as evidence
# for an architecture question such as "what is a digital twin?".
_QUERY_STOPWORDS = {
    "about", "after", "against", "answer", "are", "can", "could", "does",
    "explain", "for", "from", "function", "have", "how", "into", "is",
    "its", "may", "please", "should", "that", "the", "this", "use", "what",
    "when", "which", "with", "would", "you", "your",
}

MIN_LEXICAL_OVERLAP = 2


def _terms(value):
    import re

    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", value or "")
        if token.lower() not in _QUERY_STOPWORDS
    }


def filter_relevant_documents(question, documents):
    """Keep retrieval results only when they share a meaningful query term."""
    query_terms = _terms(question)
    if not query_terms:
        return list(documents or [])

    return [
        document for document in (documents or [])
        if query_terms & _terms(document)
    ]


client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection(
    name="devpulse_knowledge"
)

model = SentenceTransformer(MODEL_NAME)


def _normalise_distance(distance):
    """Map Chroma distances to a bounded, display-friendly relevance score."""
    try:
        return round(100 / (1 + max(0.0, float(distance))), 1)
    except (TypeError, ValueError):
        return 0.0


def _lexical_score(question, document):
    query_terms = _terms(question)
    document_terms = _terms(document)
    if not query_terms:
        return 0.0
    return round(100 * len(query_terms & document_terms) / len(query_terms), 1)


@lru_cache(maxsize=96)
def _retrieve_cached(question, number_of_results):
    """Retrieve, filter, and rerank documents for one normalised question.

    The cache avoids recomputing the sentence embedding when each comparison
    reuses the same sub-question.  It contains only public project knowledge,
    never live telemetry or credentials.
    """
    question_embedding = model.encode([question])[0].tolist()
    requested = max(number_of_results * 3, number_of_results)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=requested,
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0] or []
    metadatas = (results.get("metadatas") or [[]])[0] or []
    distances = (results.get("distances") or [[]])[0] or []
    query_terms = _terms(question)
    ranked = []

    for index, document in enumerate(documents):
        document = str(document or "").strip()
        if not document:
            continue
        overlap = len(query_terms & _terms(document))
        # A single word such as “Docker” is not enough to call a document
        # evidence for a question about Docker memory.  Short two-word concept
        # queries still require both meaningful words where they exist.
        minimum_overlap = min(MIN_LEXICAL_OVERLAP, len(query_terms))
        if query_terms and overlap < minimum_overlap:
            continue
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        lexical = _lexical_score(question, document)
        semantic = _normalise_distance(distance)
        # Semantic ranking finds paraphrases; lexical overlap prevents an
        # unrelated nearest neighbour from being passed to a model as evidence.
        score = round(semantic * 0.65 + lexical * 0.35, 1)
        ranked.append({
            "content": document,
            "source": (metadata or {}).get("source", "knowledge base"),
            "chunk": (metadata or {}).get("chunk"),
            "semantic_score": semantic,
            "lexical_score": lexical,
            "relevance_score": score,
        })

    ranked.sort(key=lambda item: item["relevance_score"], reverse=True)
    return tuple(ranked[:number_of_results])


def retrieve_evidence(question, number_of_results=2):
    """Return attributable RAG evidence, including its relevance signals."""
    normalised = " ".join((question or "").split())
    if not normalised:
        return []
    return [dict(item) for item in _retrieve_cached(normalised, number_of_results)]


def retrieve_context(question, number_of_results=2):
    """Backward-compatible text-only retrieval for existing API consumers."""
    return [
        item["content"]
        for item in retrieve_evidence(question, number_of_results)
    ]


if __name__ == "__main__":

    question = "How can I investigate high Docker memory usage?"

    documents = retrieve_context(question)

    print("\nQuestion:")
    print(question)

    print("\nRetrieved knowledge:")

    for i, document in enumerate(documents, start=1):

        print("\n" + "=" * 60)
        print(f"RESULT {i}")
        print("=" * 60)
        print(document)
