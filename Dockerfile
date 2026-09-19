FROM python:3.11-slim

# Меньше мусора в слоях и предсказуемый вывод логов.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN pip install --no-cache-dir uv

# Зависимости ставим отдельным слоем: он переиспользуется, пока не изменился uv.lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# Приложению не нужен root. Каталог кеша создаём заранее, чтобы
# смонтированный named volume унаследовал права пользователя app.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app
USER app

ENV HF_HOME=/home/app/.cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "faq_rag_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
