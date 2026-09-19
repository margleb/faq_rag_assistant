from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения: значения берутся из окружения или .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"

    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "faq"

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    vector_size: int = 384
    top_k: int = Field(default=3, ge=1)

    redis_url: str = "redis://localhost:6379"
    # Сессии не живут вечно: без TTL Redis копил бы диалоги бесконечно.
    session_ttl_seconds: int = Field(default=24 * 60 * 60, ge=1)

    # Предохранитель от бесконечного цикла «модель зовёт tool -> tool отвечает».
    max_steps: int = Field(default=5, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
