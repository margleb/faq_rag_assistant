from numpy import ndarray
from sentence_transformers import SentenceTransformer

from faq_rag_assistant.config import EMBEDDING_MODEL

# Одна модель на модуль для индексации и поиска.
_model = SentenceTransformer(EMBEDDING_MODEL)


def encode(text: str) -> ndarray:
    return _model.encode(text)
