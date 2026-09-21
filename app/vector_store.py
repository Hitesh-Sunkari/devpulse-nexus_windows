import os
from pathlib import Path

import chromadb

from app.chunking import load_document, chunk_text
from app.embeddings import create_embeddings


DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")


def build_vector_store():

    client = chromadb.PersistentClient(
        path=DB_PATH
    )

    collection = client.get_or_create_collection(
        name="devpulse_knowledge"
    )

    knowledge_files = sorted(Path("knowledge").glob("*.md"))
    if not knowledge_files:
        raise RuntimeError("No Markdown knowledge files were found in knowledge/.")

    chunks = []
    metadatas = []
    ids = []
    for file_path in knowledge_files:
        document_chunks = chunk_text(load_document(file_path))
        for index, chunk in enumerate(document_chunks):
            chunks.append(chunk)
            ids.append(f"{file_path.stem}_chunk_{index}")
            metadatas.append({"source": str(file_path).replace("\\", "/"), "chunk": index})

    # This collection is owned by DevPulse. Rebuilding removes chunks from an
    # older knowledge document that is no longer part of the curated corpus.
    existing = collection.get(include=[]).get("ids", [])
    if existing:
        collection.delete(ids=existing)

    embeddings = create_embeddings(chunks)

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print("Vector database created successfully!")
    print(f"Collection: {collection.name}")
    print(f"Documents stored: {collection.count()}")


if __name__ == "__main__":
    build_vector_store()
