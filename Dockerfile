FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src

RUN uv sync --frozen --no-dev

CMD ["/app/.venv/bin/uvicorn", "faq_rag_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]