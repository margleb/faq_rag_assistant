import pytest
from fastapi.testclient import TestClient

from faq_rag_assistant.api.app import app


@pytest.fixture
def client() -> TestClient:
    """Клиенты OpenAI/Qdrant/Redis создаются лениво, поэтому импорт app безопасен."""
    return TestClient(app)


@pytest.fixture
def session_store(monkeypatch) -> dict[str, list[dict]]:
    """Redis подменяется словарём: тесты не зависят от внешнего сервиса."""
    storage: dict[str, list[dict]] = {}

    monkeypatch.setattr(
        "faq_rag_assistant.api.app.save_session",
        lambda session_id, messages: storage.__setitem__(session_id, messages),
    )
    monkeypatch.setattr(
        "faq_rag_assistant.api.app.get_session",
        storage.get,
    )
    return storage
