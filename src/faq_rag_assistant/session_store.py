import json
from functools import lru_cache

import redis

from faq_rag_assistant.config import get_settings


@lru_cache
def get_client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def save_session(session_id: str, messages: list[dict]) -> None:
    get_client().set(
        session_id,
        json.dumps(messages, ensure_ascii=False),
        ex=get_settings().session_ttl_seconds,
    )


def get_session(session_id: str) -> list[dict] | None:
    data = get_client().get(session_id)
    if data is None:
        return None
    return json.loads(data)
