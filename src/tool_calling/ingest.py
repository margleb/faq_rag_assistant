import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

FAQ_PATH = Path(__file__).parent / "data" / "faq.json"
COLLECTION_NAME = "faq"

# Читаем данные по вопросам
with open(FAQ_PATH, "r", encoding="utf-8") as file:
    faq = json.load(file)

# Подключаемся к БД Qdreant
client = QdrantClient(url="http://localhost:6333")

# Используем модель для embadding
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# vector = model.encode("тестовая строка")
# print(vector.shape)

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# делаем поинты (энкод)
points = []

for idx, item in enumerate(faq, start=1):
    text = f"Вопрос: {item['question']}\nОтвет: {item['answer']}"

    vector = model.encode(text).tolist()

    points.append(
        PointStruct(
            id=idx,
            vector=vector,
            payload={
                "question": item["question"],
                "answer": item["answer"],
            },
        )
    )


# вставляем в базу
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)
