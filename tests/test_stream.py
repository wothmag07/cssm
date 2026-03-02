"""Tests for the /stream SSE endpoint."""
import json
from unittest.mock import patch


def test_stream_returns_sse(client):
    """Stream endpoint should return text/event-stream content type."""
    with patch("main.run_pre_generate", return_value={
        "question": "best laptops",
        "documents": [],
        "sources": [{"content": "review...", "metadata": {"product_name": "Dell XPS"}, "similarity": 0.85}],
        "chat_history": "No previous conversation.",
    }), patch("main.generate_stream", return_value=iter(["The ", "Dell ", "XPS ", "is great."])):
        resp = client.post("/stream", data={"msg": "best laptops"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_stream_event_format(client):
    """Stream should emit sources, tokens, and done events."""
    with patch("main.run_pre_generate", return_value={
        "question": "best laptops",
        "documents": [],
        "sources": [{"content": "review...", "metadata": {}, "similarity": 0.85}],
        "chat_history": "No previous conversation.",
    }), patch("main.generate_stream", return_value=iter(["Hello ", "world"])):
        resp = client.post("/stream", data={"msg": "best laptops"})

    events = []
    for line in resp.text.strip().split("\n\n"):
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    # First event should be sources
    assert events[0]["type"] == "sources"
    assert isinstance(events[0]["sources"], list)

    # Middle events should be tokens
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 2
    assert token_events[0]["token"] == "Hello "
    assert token_events[1]["token"] == "world"

    # Last event should be done
    assert events[-1]["type"] == "done"


def test_stream_empty_query(client):
    """Empty string is rejected by FastAPI's Form(...) as missing field."""
    resp = client.post("/stream", data={"msg": ""})
    assert resp.status_code == 422


def test_stream_too_long_query(client):
    resp = client.post("/stream", data={"msg": "x" * 1001})
    assert resp.status_code == 400
