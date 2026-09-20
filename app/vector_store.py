import os

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

    file_path = "knowledge/docker.md"

    text = load_document(file_path)
    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    ids = [
        f"docker_chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "source": file_path,
                "chunk": i
            }
            for i in range(len(chunks))
        ]
    )

    print("Vector database created successfully!")
    print(f"Collection: {collection.name}")
    print(f"Documents stored: {collection.count()}")


if __name__ == "__main__":
    build_vector_store()
