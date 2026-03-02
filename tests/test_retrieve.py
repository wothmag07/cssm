"""Tests for the /retrieve endpoint."""


def test_retrieve_returns_answer_and_sources(client):
    resp = client.post("/retrieve", data={"msg": "best laptops for students"})
    assert resp.status_code == 200

    data = resp.json()
    assert "response" in data
    assert "sources" in data
    assert len(data["response"]) > 0
    assert len(data["sources"]) > 0
    assert data["sources"][0]["metadata"]["product_name"] == "Dell XPS 15"


def test_retrieve_empty_query(client):
    """Empty string is rejected by FastAPI's Form(...) as missing field."""
    resp = client.post("/retrieve", data={"msg": ""})
    assert resp.status_code == 422


def test_retrieve_whitespace_query(client):
    resp = client.post("/retrieve", data={"msg": "   "})
    assert resp.status_code == 400


def test_retrieve_too_long_query(client):
    resp = client.post("/retrieve", data={"msg": "x" * 1001})
    assert resp.status_code == 400
    assert "too long" in resp.json()["error"].lower()


def test_retrieve_max_length_query(client):
    """Exactly 1000 chars should be accepted."""
    resp = client.post("/retrieve", data={"msg": "x" * 1000})
    assert resp.status_code == 200


def test_retrieve_with_chat_history(client):
    resp = client.post(
        "/retrieve",
        data={"msg": "what about cheaper ones?", "chat_history": "User: best laptops\nAssistant: Dell XPS 15..."},
    )
    assert resp.status_code == 200


def test_retrieve_missing_msg_field(client):
    resp = client.post("/retrieve", data={})
    assert resp.status_code == 422  # FastAPI validation error
