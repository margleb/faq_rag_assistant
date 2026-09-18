import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from faq_rag_assistant.agent.core import create_conversation, run_agent
from faq_rag_assistant.session_store import get_session, save_session

app = FastAPI(title="FAQ RAG Assistant")


class ChatRequest(BaseModel):
    session_id: str | None
    message: str


class ChatResponse(BaseModel):
    session_id: str | None
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    session_id = request.session_id

    if session_id is None:
        session_id = str(uuid.uuid4())
        messages = create_conversation()

    else:
        messages = get_session(session_id)

        if messages is None:
            raise HTTPException(
                status_code=404,
                detail="Session not found",
            )

    messages.append(
        {
            "role": "user",
            "content": request.message,
        }
    )

    answer, messages = run_agent(messages)

    save_session(session_id, messages)

    return ChatResponse(session_id=session_id, answer=answer)
