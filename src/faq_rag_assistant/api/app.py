from fastapi import FastAPI
from pydantic import BaseModel

from faq_rag_assistant.agent.core import run_agent

app = FastAPI(title="FAQ RAG Assistant")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [
        {
            "role": "user",
            "content": request.message,
        }
    ]

    answer, messages = run_agent(messages)

    return ChatResponse(answer=answer)
