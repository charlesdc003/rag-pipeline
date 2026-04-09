from unittest.mock import patch, MagicMock
from src.rag_pipeline.generator import RAGResponse


def make_mock_results():
    return [
        {"id": 1, "content": "Enterprise auth failures should be escalated immediately.", "metadata": {}, "similarity": 0.85},
        {"id": 2, "content": "Do not resolve auth outages through standard channels.", "metadata": {}, "similarity": 0.75},
    ]


def test_rag_response_schema():
    r = RAGResponse(
        answer="Escalate immediately.",
        confidence=0.9,
        sources_used=2,
        action="escalate"
    )
    assert r.confidence == 0.9
    assert r.action == "escalate"


def test_rag_response_rejects_bad_confidence():
    import pytest
    with pytest.raises(Exception):
        RAGResponse(
            answer="test",
            confidence=1.5,
            sources_used=1,
            action="auto_reply"
        )


@patch("src.rag_pipeline.generator.retrieve_context")
@patch("src.rag_pipeline.generator.call_llm")
def test_generate_returns_escalate(mock_llm, mock_retrieve):
    import json
    mock_retrieve.return_value = make_mock_results()
    mock_llm.return_value = json.dumps({
        "answer": "Escalate to on-call team.",
        "confidence": 0.9,
        "action": "escalate"
    })

    from src.rag_pipeline.generator import generate
    result = generate("enterprise auth outage")
    assert result.action == "escalate"
    assert result.sources_used == 2