import ollama

EMBEDDING_MODEL = "nomic-embed-text"


def embed_text(text: str) -> list[float]:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.embeddings[0]


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    return [embed_text(chunk) for chunk in chunks]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks