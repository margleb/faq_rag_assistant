from types import SimpleNamespace

import pytest

from faq_rag_assistant.agent import tools


def make_point(score: float, question: str, answer: str) -> SimpleNamespace:
    return SimpleNamespace(score=score, payload={"question": question, "answer": answer})


@pytest.fixture
def fake_search(monkeypatch):
    def _install(points):
        monkeypatch.setattr(tools, "semantic_search", lambda query, top_k=None: points)

    return _install


def test_search_formats_points_for_the_model(fake_search):
    fake_search([make_point(0.87654, "Сколько длится курс?", "8 недель.")])

    result = tools.search("длительность")

    assert result == "score: 0.877\nВопрос: Сколько длится курс?\nОтвет: 8 недель."


def test_search_separates_points_with_blank_line(fake_search):
    fake_search([make_point(0.9, "Q1", "A1"), make_point(0.8, "Q2", "A2")])

    assert tools.search("что-нибудь").count("\n\n") == 1


def test_search_reports_empty_result_explicitly(fake_search):
    fake_search([])

    assert tools.search("нет такого") == tools.NOTHING_FOUND
