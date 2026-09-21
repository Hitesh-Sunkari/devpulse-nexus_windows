from pathlib import Path


def load_document(file_path):
    return Path(file_path).read_text(encoding="utf-8")


def chunk_text(text, chunk_size=700, overlap=120):
    """Create Markdown-aware chunks without splitting sentences mid-word."""
    paragraphs = [
        paragraph.strip()
        for paragraph in text.replace("\r\n", "\n").split("\n\n")
        if paragraph.strip()
    ]
    chunks, current = [], ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        # Retain a small amount of preceding context while keeping headings
        # and paragraph boundaries readable to both the embedding model and LLM.
        prefix = current[-overlap:].strip() if current else ""
        current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph

        while len(current) > chunk_size:
            split_at = current.rfind(" ", 0, chunk_size)
            split_at = split_at if split_at > chunk_size // 2 else chunk_size
            chunks.append(current[:split_at].strip())
            current = current[max(0, split_at - overlap):].strip()

    if current:
        chunks.append(current)
    return chunks


if __name__ == "__main__":
    file_path = "knowledge/docker.md"

    text = load_document(file_path)

    chunks = chunk_text(text)

    print(f"Document length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks, start=1):
        print("\n" + "=" * 60)
        print(f"CHUNK {i}")
        print("=" * 60)
        print(chunk)
