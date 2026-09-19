from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

from faq_rag_assistant.config import get_settings
from faq_rag_assistant.rag.embeddings import encode


@lru_cache
def get_client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def reset_collection() -> None:
    """Пересоздаём коллекцию перед полной индексацией FAQ."""
    settings = get_settings()
    client = get_client()

    if client.collection_exists(settings.collection_name):
        client.delete_collection(settings.collection_name)

    client.create_collection(
        settings.collection_name,
        vectors_config=VectorParams(
            size=settings.vector_size,
            distance=Distance.COSINE,
        ),
    )


def upsert_points(points: list[PointStruct]) -> None:
    get_client().upsert(
        collection_name=get_settings().collection_name,
        points=points,
    )


def semantic_search(query: str, top_k: int | None = None) -> list[ScoredPoint]:
    settings = get_settings()

    result = get_client().query_points(
        collection_name=settings.collection_name,
        query=encode(query),
        limit=top_k or settings.top_k,
        with_payload=True,
    )
    return result.points
