import importlib
import sys
import types

import pytest
from fastapi.testclient import TestClient

from faq_rag_assistant.agent.exceptions import LLMProviderError


@pytest.fixture
def app_module(monkeypatch):
    # Импорт app по цепочке тянет модули с побочными эффектами при импорте:
    # vector_store создаёт QdrantClient (сразу делает HTTP-запрос к Qdrant)
    # и грузит SentenceTransformer. Подменяем его заглушкой до импорта.
    fake_vector_store = types.ModuleType("faq_rag_assistant.rag.vector_store")
    fake_vector_store.semantic_search = lambda *args, **kwargs: []
    monkeypatch.setitem(
        sys.modules, "faq_rag_assistant.rag.vector_store", fake_vector_store
    )
    # OpenAI() при импорте core требует ключ; фиктивный ключ не даёт
    # load_dotenv подставить настоящий из .env.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    return importlib.import_module("faq_rag_assistant.api.app")


def test_chat_returns_502_when_llm_provider_unavailable(app_module, monkeypatch):
    def fake_run_agent(messages):
        raise LLMProviderError("test")

    def fail_on_redis_call(*args, **kwargs):
        raise AssertionError("session storage must not be called")

    # Патчим имена в модуле app, где они используются, а не где определены.
    monkeypatch.setattr("faq_rag_assistant.api.app.run_agent", fake_run_agent)
    monkeypatch.setattr("faq_rag_assistant.api.app.get_session", fail_on_redis_call)
    monkeypatch.setattr("faq_rag_assistant.api.app.save_session", fail_on_redis_call)

    client = TestClient(app_module.app)
    response = client.post("/chat", json={"session_id": None, "message": "Привет"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM provider is unavailable"}
