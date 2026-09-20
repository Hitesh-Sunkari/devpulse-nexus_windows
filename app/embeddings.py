from sentence_transformers import SentenceTransformer

from app.chunking import load_document, chunk_text


MODEL_NAME = "all-MiniLM-L6-v2"


def create_embeddings(chunks):
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )

    return embeddings


if __name__ == "__main__":

    file_path = "knowledge/docker.md"

    text = load_document(file_path)

    chunks = chunk_text(text)

    embeddings = create_embeddings(chunks)

    print("\nEmbedding complete!")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")

    for i, embedding in enumerate(embeddings, start=1):
        print(
            f"Chunk {i}: "
            f"{len(embedding)} dimensions"
        )
