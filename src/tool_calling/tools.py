import json
from pathlib import Path

FAQ_PATH = Path(__file__).parent / "data" / "faq.json"

# читаем один раз, чтобы не вызывать несколько раз
with open(FAQ_PATH, "r", encoding="utf-8") as file:
    faq = json.load(file)


def search(query: str, top_k: int = 3) -> str:

    query_words = [
        word
        for word in query.lower().split()
        if len(word) > 3  # берем слова которые больше 3eх букв
    ]

    scored = []  # RAG должен хотя бы нескольк а не одно
    for item in faq:
        text = (item["question"] + " " + item["answer"]).lower()
        score = sum(1 for word in query_words if word in text)

        if score:  # берем по лучшим совпадениям
            scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)  # лучшие сверху
    top = scored[:top_k]  # до трех вариантов

    if not top:
        return "В базе FAQ ничего подходящего не найдено"

    return "\n\n".join(
        f"Вопрос: {item['question']}\nОтвет: {item['answer']}" for _, item in top
    )


TOOL_FUNCTIONS = {
    "search": search,
}
