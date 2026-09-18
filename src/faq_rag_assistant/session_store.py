import json

import redis

from faq_rag_assistant.config import REDIS_URL

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
)


def save_session(session_id: str, messages: list[dict]) -> None:
    messages = json.dumps(messages, ensure_ascii=False)
    redis_client.set(session_id, messages)


def get_session(session_id: str) -> list[dict] | None:
    data = redis_client.get(session_id)
    if not data:
        return None
    return json.loads(data)
