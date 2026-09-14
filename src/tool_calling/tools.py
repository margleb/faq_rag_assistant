import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

FAQ_PATH = Path(__file__).parent / "data" / "faq.json"

# читаем один раз, чтобы не вызывать несколько раз
with open(FAQ_PATH, "r", encoding="utf-8") as file:
    faq = json.load(file)


client = QdrantClient(url="http://localhost:6333")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def search(query: str, top_k: int = 3) -> str:

    result = client.query_points(
        collection_name="faq",
        query=model.encode(query).tolist(),
        limit=top_k,
        with_payload=True,
    )

    chunks = []

    for point in result.points:
        chunks.append(
            f"score: {point.score}\n"
            f"Вопрос: {point.payload['question']}\n"
            f"Ответ: {point.payload['answer']}"
        )

    return "\n\n".join(chunks)


TOOL_FUNCTIONS = {
    "search": search,
}
