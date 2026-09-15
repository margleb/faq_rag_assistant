from tool_calling.vector_store import semantic_search


def search(query: str, top_k: int = 3) -> str:
    points = semantic_search(query, top_k=top_k)
    chunks = []

    for point in points:
        chunks.append(
            f"score: {point.score}\n"
            f"Вопрос: {point.payload['question']}\n"
            f"Ответ: {point.payload['answer']}"
        )

    return "\n\n".join(chunks)


TOOL_FUNCTIONS = {
    "search": search,
}
