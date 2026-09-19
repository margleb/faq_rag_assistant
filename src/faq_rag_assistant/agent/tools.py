from faq_rag_assistant.rag.vector_store import semantic_search

NOTHING_FOUND = "В базе FAQ ничего не найдено."


def search(query: str, top_k: int | None = None) -> str:
    points = semantic_search(query, top_k=top_k)

    if not points:
        # Пустая строка выглядела бы как сбой инструмента; говорим явно.
        return NOTHING_FOUND

    return "\n\n".join(
        f"score: {point.score:.3f}\n"
        f"Вопрос: {point.payload['question']}\n"
        f"Ответ: {point.payload['answer']}"
        for point in points
    )


TOOL_FUNCTIONS = {
    "search": search,
}
