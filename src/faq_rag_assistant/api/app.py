import logging
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from faq_rag_assistant.agent.core import create_conversation, run_agent
from faq_rag_assistant.agent.exceptions import LLMProviderError
from faq_rag_assistant.session_store import get_session, save_session

logger = logging.getLogger(__name__)

app = FastAPI(
    title="FAQ RAG Assistant",
    description="FAQ-помощник с tool calling и semantic search по Qdrant.",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    # Пусто или отсутствует — начинаем новый диалог.
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness-проба: отвечает только когда приложение поднялось целиком."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id

    if session_id is None:
        session_id = str(uuid.uuid4())
        messages = create_conversation()
        logger.info("Начата сессия %s", session_id)
    else:
        messages = get_session(session_id)

        if messages is None:
            # Истёкший TTL неотличим от неизвестного ID — оба дают 404.
            raise HTTPException(status_code=404, detail="Session not found")

    messages.append({"role": "user", "content": request.message})

    try:
        answer, messages = run_agent(messages)
    except LLMProviderError as error:
        logger.exception("LLM-провайдер недоступен для сессии %s", session_id)
        raise HTTPException(
            status_code=502,
            detail="LLM provider is unavailable",
        ) from error

    save_session(session_id, messages)

    return ChatResponse(session_id=session_id, answer=answer)
