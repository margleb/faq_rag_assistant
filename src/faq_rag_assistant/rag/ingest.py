import json
from pathlib import Path

from qdrant_client.models import PointStruct

from faq_rag_assistant.rag.embeddings import encode
from faq_rag_assistant.rag.vector_store import reset_collection, upsert_points

FAQ_PATH = Path(__file__).resolve().parents[1] / "data" / "faq.json"

# Исходный JSON нужен только для индексации.
with open(FAQ_PATH, "r", encoding="utf-8") as file:
    faq = json.load(file)

reset_collection()

# Индексируем вопрос вместе с ответом, сохраняя исходные поля в payload.
points = []

for idx, item in enumerate(faq, start=1):
    text = f"Вопрос: {item['question']}\nОтвет: {item['answer']}"

    points.append(
        PointStruct(
            id=idx,
            vector=encode(text),
            payload={
                "question": item["question"],
                "answer": item["answer"],
            },
        )
    )

upsert_points(points)
