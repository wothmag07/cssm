"""Tests for API key authentication."""


def test_no_auth_in_dev_mode(client):
    """When API_KEY is not set, requests should pass without a key."""
    resp = client.post("/retrieve", data={"msg": "best laptops"})
    assert resp.status_code == 200


def test_auth_rejects_missing_key(auth_client):
    """When API_KEY is set, requests without X-API-Key should get 401."""
    resp = auth_client.post("/retrieve", data={"msg": "best laptops"})
    assert resp.status_code == 401


def test_auth_rejects_wrong_key(auth_client):
    """Wrong API key should get 401."""
    resp = auth_client.post(
        "/retrieve",
        data={"msg": "best laptops"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_auth_accepts_correct_key(auth_client):
    """Correct API key should pass."""
    resp = auth_client.post(
        "/retrieve",
        data={"msg": "best laptops"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 200


def test_stream_auth_rejects_missing_key(auth_client):
    """Stream endpoint should also require auth."""
    resp = auth_client.post("/stream", data={"msg": "best laptops"})
    assert resp.status_code == 401


def test_stream_auth_accepts_correct_key(auth_client):
    """Stream endpoint with correct key should work."""
    resp = auth_client.post(
        "/stream",
        data={"msg": "best laptops"},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 200
