import pytest

from faq_rag_assistant.agent.exceptions import LLMProviderError


@pytest.fixture
def echo_agent(monkeypatch):
    """Агент без сети: возвращает последний вопрос и накопленную историю."""

    def fake_run_agent(messages):
        return f"ответ на: {messages[-1]['content']}", [
            *messages,
            {"role": "assistant", "content": "ответ"},
        ]

    monkeypatch.setattr("faq_rag_assistant.api.app.run_agent", fake_run_agent)


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_without_session_id_starts_new_session(client, session_store, echo_agent):
    response = client.post("/chat", json={"message": "Сколько длится курс?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "ответ на: Сколько длится курс?"
    assert body["session_id"] in session_store

    # Системный промпт и вопрос пользователя должны попасть в сохранённую историю.
    saved = session_store[body["session_id"]]
    assert saved[0]["role"] == "system"
    assert {"role": "user", "content": "Сколько длится курс?"} in saved


def test_chat_continues_existing_session(client, session_store, echo_agent):
    first = client.post("/chat", json={"message": "Сколько длится курс?"}).json()
    session_id = first["session_id"]

    second = client.post(
        "/chat",
        json={"message": "А сколько стоит?", "session_id": session_id},
    )

    assert second.status_code == 200
    assert second.json()["session_id"] == session_id
    # История растёт, а не перезаписывается новым диалогом.
    contents = [message.get("content") for message in session_store[session_id]]
    assert "Сколько длится курс?" in contents
    assert "А сколько стоит?" in contents


def test_chat_returns_404_for_unknown_session(client, session_store, echo_agent):
    response = client.post(
        "/chat",
        json={"message": "Привет", "session_id": "не-существует"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_chat_rejects_empty_message(client, session_store):
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422


def test_chat_returns_502_when_llm_provider_unavailable(client, session_store, monkeypatch):
    def fake_run_agent(messages):
        raise LLMProviderError("test")

    monkeypatch.setattr("faq_rag_assistant.api.app.run_agent", fake_run_agent)

    response = client.post("/chat", json={"message": "Привет"})

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM provider is unavailable"}
    # Сломанный запрос не должен оставлять сессию в хранилище.
    assert session_store == {}
