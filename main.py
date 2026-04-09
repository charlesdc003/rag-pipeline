from fastapi import FastAPI
from src.rag_pipeline.generator import generate, RAGResponse

app = FastAPI(title="RAG Pipeline")


@app.post("/query", response_model=RAGResponse)
def query(request: dict) -> RAGResponse:
    return generate(request["query"])