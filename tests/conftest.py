"""Shared fixtures for CSSM API tests."""
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _clear_api_key(monkeypatch):
    """Ensure API_KEY is unset by default (auth disabled)."""
    monkeypatch.delenv("API_KEY", raising=False)


@pytest.fixture()
def mock_rag_graph():
    """Create a mock RAG graph that returns a canned response."""
    graph = MagicMock()
    graph.invoke.return_value = {
        "answer": "The XPS 15 is highly rated for students [1].",
        "sources": [
            {
                "content": "Great laptop for college...",
                "metadata": {"product_name": "Dell XPS 15", "rating": 4.5, "category": "Laptops"},
                "similarity": 0.87,
            }
        ],
    }
    return graph


@pytest.fixture()
def client(mock_rag_graph):
    """FastAPI TestClient with mocked RAG graph (auth disabled)."""
    # Patch heavy dependencies so the app can import without real credentials
    with (
        patch("main.Retriever"),
        patch("main.ModelLoader"),
        patch("main.build_graph", return_value=mock_rag_graph),
    ):
        # Re-import after patching so lifespan uses the mock
        import main

        main.rag_graph = mock_rag_graph
        yield TestClient(main.app)


@pytest.fixture()
def auth_client(mock_rag_graph, monkeypatch):
    """FastAPI TestClient with API key auth enabled."""
    monkeypatch.setenv("API_KEY", "test-secret-key")

    with (
        patch("main.Retriever"),
        patch("main.ModelLoader"),
        patch("main.build_graph", return_value=mock_rag_graph),
    ):
        import main

        # Reload the API key from env
        main._API_KEY = "test-secret-key"
        main.rag_graph = mock_rag_graph
        yield TestClient(main.app)

    # Reset after test
    main._API_KEY = ""
