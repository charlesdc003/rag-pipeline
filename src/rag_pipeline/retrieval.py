import json
import psycopg2
from .database import get_connection
from .embeddings import embed_text, chunk_text


def ingest_document(content: str, metadata: dict = {}) -> int:
    chunks = chunk_text(content)
    conn = get_connection()
    cur = conn.cursor()

    inserted = 0
    for chunk in chunks:
        embedding = embed_text(chunk)
        cur.execute(
            """
            INSERT INTO documents (content, embedding, metadata)
            VALUES (%s, %s, %s)
            RETURNING id;
            """,
            (chunk, embedding, json.dumps(metadata))
        )
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()
    return inserted


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    query_embedding = embed_text(query)
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, content, metadata,
               1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_embedding, query_embedding, top_k)
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "id": row[0],
            "content": row[1],
            "metadata": row[2],
            "similarity": round(float(row[3]), 4)
        }
        for row in rows
    ]