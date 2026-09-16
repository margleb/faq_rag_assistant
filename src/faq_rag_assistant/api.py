from fastapi import FastAPI
from pydantic import BaseModel

from faq_rag_assistant.agent import run_agent

app = FastAPI(title="FAQ RAG Assistant")


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(request: ChatRequest):
    messages = [
        {
            "role": "user",
            "content": request.message,
        }
    ]

    answer, messages = run_agent(messages)
    return answer
