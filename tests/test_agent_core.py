from types import SimpleNamespace

import pytest
from openai import APIConnectionError

from faq_rag_assistant.agent import core
from faq_rag_assistant.agent.exceptions import LLMProviderError


def make_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none: bool = False) -> dict:
        return {"role": "assistant", "content": self.content}


@pytest.fixture
def fake_llm(monkeypatch):
    """Подменяет клиента OpenAI очередью заранее заданных ответов модели."""

    def _install(messages_to_return):
        queue = list(messages_to_return)

        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=queue.pop(0))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(core, "get_client", lambda: client)

    return _install


def test_run_agent_returns_answer_without_tool_calls(fake_llm):
    fake_llm([FakeMessage(content="Курс длится 8 недель.")])

    answer, messages = core.run_agent(core.create_conversation())

    assert answer == "Курс длится 8 недель."
    assert messages[-1]["content"] == "Курс длится 8 недель."


def test_run_agent_feeds_tool_result_back_to_model(fake_llm, monkeypatch):
    fake_llm(
        [
            FakeMessage(tool_calls=[make_tool_call("call-1", "search", '{"query": "цена"}')]),
            FakeMessage(content="Курс стоит 30000 рублей."),
        ]
    )
    monkeypatch.setitem(core.TOOL_FUNCTIONS, "search", lambda query, top_k=None: "30000 рублей")

    answer, messages = core.run_agent(core.create_conversation())

    assert answer == "Курс стоит 30000 рублей."
    tool_messages = [message for message in messages if message["role"] == "tool"]
    assert tool_messages == [{"role": "tool", "tool_call_id": "call-1", "content": "30000 рублей"}]


def test_run_agent_passes_tool_failure_to_model_instead_of_raising(fake_llm, monkeypatch):
    def broken_search(query, top_k=None):
        raise RuntimeError("Qdrant недоступен")

    fake_llm(
        [
            FakeMessage(tool_calls=[make_tool_call("call-1", "search", '{"query": "цена"}')]),
            FakeMessage(content="Сейчас не могу проверить базу."),
        ]
    )
    monkeypatch.setitem(core.TOOL_FUNCTIONS, "search", broken_search)

    answer, messages = core.run_agent(core.create_conversation())

    assert answer == "Сейчас не могу проверить базу."
    assert "Qdrant недоступен" in messages[-2]["content"]


def test_run_agent_stops_after_max_steps(fake_llm, monkeypatch):
    monkeypatch.setattr(core.get_settings(), "max_steps", 2)
    fake_llm(
        [
            FakeMessage(tool_calls=[make_tool_call("call-1", "search", '{"query": "x"}')]),
            FakeMessage(tool_calls=[make_tool_call("call-2", "search", '{"query": "x"}')]),
        ]
    )
    monkeypatch.setitem(core.TOOL_FUNCTIONS, "search", lambda query, top_k=None: "результат")

    answer, _ = core.run_agent(core.create_conversation())

    assert answer == core.NO_ANSWER_REPLY


def test_run_agent_wraps_provider_errors(monkeypatch):
    def create(**kwargs):
        raise APIConnectionError(request=None)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(core, "get_client", lambda: client)

    with pytest.raises(LLMProviderError):
        core.run_agent(core.create_conversation())
