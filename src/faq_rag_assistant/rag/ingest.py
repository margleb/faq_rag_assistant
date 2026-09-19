import json
import logging
from pathlib import Path

from qdrant_client.models import PointStruct

from faq_rag_assistant.rag.embeddings import encode
from faq_rag_assistant.rag.vector_store import reset_collection, upsert_points

logger = logging.getLogger(__name__)

FAQ_PATH = Path(__file__).resolve().parents[1] / "data" / "faq.json"


def load_faq(path: Path = FAQ_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def build_points(faq: list[dict]) -> list[PointStruct]:
    """ID назначаются по порядку записей в JSON — на них опирается evaluation."""
    return [
        PointStruct(
            id=idx,
            # Индексируем вопрос вместе с ответом: так находятся и перефразировки.
            vector=encode(f"Вопрос: {item['question']}\nОтвет: {item['answer']}"),
            payload={
                "question": item["question"],
                "answer": item["answer"],
            },
        )
        for idx, item in enumerate(faq, start=1)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    faq = load_faq()
    reset_collection()
    upsert_points(build_points(faq))

    logger.info("Проиндексировано записей: %s", len(faq))


if __name__ == "__main__":
    main()
