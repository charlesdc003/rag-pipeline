import json
import ollama
import weave
from pydantic import BaseModel, Field
from .retrieval import retrieve

GENERATE_MODEL = "llama3.2"

weave.init("rag-pipeline")


class RAGResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_used: int
    action: str


@weave.op()
def retrieve_context(query: str) -> list[dict]:
    return retrieve(query, top_k=3)


@weave.op()
def call_llm(prompt: str) -> str:
    response = ollama.chat(
        model=GENERATE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json"
    )
    return response.message.content


@weave.op()
def generate(query: str) -> RAGResponse:
    results = retrieve_context(query)

    if not results:
        return RAGResponse(
            answer="No relevant context found.",
            confidence=0.0,
            sources_used=0,
            action="escalate"
        )

    avg_similarity = sum(r["similarity"] for r in results) / len(results)
    context = "\n\n".join(r["content"] for r in results)

    prompt = f"""You are a support assistant. Use the context below to answer the query.
Respond in this exact JSON format with no extra text:
{{
  "answer": "your answer here",
  "confidence": 0.0 to 1.0 based on how well context supports your answer,
  "action": "auto_reply" if confidence > 0.7, "escalate" if confidence < 0.4, otherwise "review"
}}

Context:
{context}

Query: {query}

JSON response:"""

    raw = json.loads(call_llm(prompt))

    return RAGResponse(
        answer=raw.get("answer", ""),
        confidence=float(raw.get("confidence", avg_similarity)),
        sources_used=len(results),
        action=raw.get("action", "review")
    )