import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from faq_rag_assistant.agent.core import run_agent

app = FastAPI(title="FAQ RAG Assistant")

SESSIONS = {}


class ChatRequest(BaseModel):
    session_id: str | None
    message: str


class ChatResponse(BaseModel):
    session_id: str | None
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if request.session_id is None:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = []
        messages = SESSIONS[session_id]
    else:
        session_id = request.session_id
        messages = SESSIONS[session_id]

    messages.append(
        {
            "role": "user",
            "content": request.message,
        }
    )

    answer, messages = run_agent(messages)

    SESSIONS[session_id] = messages

    return ChatResponse(session_id=session_id, answer=answer)
