from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from tool_calling.config import COLLECTION_NAME, QDRANT_URL, VECTOR_SIZE
from tool_calling.embeddings import encode

client = QdrantClient(url=QDRANT_URL)


def reset_collection() -> None:
    """Пересоздаём коллекцию перед полной индексацией FAQ."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def upsert_points(points: list[PointStruct]) -> None:
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def semantic_search(query: str, top_k: int = 3) -> list[ScoredPoint]:
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=encode(query),
        limit=top_k,
        with_payload=True,
    )
    return result.points
