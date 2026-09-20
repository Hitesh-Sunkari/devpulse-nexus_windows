import os

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


def retrieve_context(question, number_of_results=2):

    question_embedding = model.encode(
        [question]
    )[0].tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results["documents"][0]

    return filter_relevant_documents(question, documents)


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
