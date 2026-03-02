"""Tests for the /health endpoint."""


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_no_auth_required(auth_client):
    """Health endpoint should work even when API key auth is enabled."""
    resp = auth_client.get("/health")
    assert resp.status_code == 200
