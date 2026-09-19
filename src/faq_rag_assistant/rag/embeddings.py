from functools import lru_cache

from numpy import ndarray
from sentence_transformers import SentenceTransformer

from faq_rag_assistant.config import get_settings


@lru_cache
def get_model() -> SentenceTransformer:
    """Одна модель на процесс: загрузка весов стоит дорого."""
    return SentenceTransformer(get_settings().embedding_model)


def encode(text: str) -> ndarray:
    return get_model().encode(text)
