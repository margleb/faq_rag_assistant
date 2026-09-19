# FAQ RAG Assistant

[![CI](https://github.com/margleb/faq_rag_assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/margleb/faq_rag_assistant/actions/workflows/ci.yml)

FAQ-помощник с LLM tool calling и semantic search. Модель сама решает, нужен ли поиск,
вызывает инструмент `search`, получает подходящие записи из Qdrant и отвечает только по ним.

Доступен как HTTP API (`POST /chat` с историей диалога в Redis) и как CLI. Используется
`gpt-4o-mini` через OpenAI SDK, локальная SentenceTransformer и Qdrant. База знаний —
10 вопросов и ответов о курсе в [faq.json](src/faq_rag_assistant/data/faq.json).

## Как работает

При индексации каждая пара вопроса и ответа кодируется в embedding и сохраняется как point
с полями `id`, `vector` и `payload`:

```text
faq.json -> ingest.py -> SentenceTransformer -> Qdrant
```

Во время запроса LLM получает описание `search` и решает, нужен ли поиск. Python выполняет
запрошенную функцию, кодирует query и получает top-3 результата по cosine similarity.
Найденные вопросы, ответы и score возвращаются модели в сообщении `role="tool"`.

```text
Вопрос -> LLM -> tool call: search -> embedding запроса -> Qdrant
            ^                                               |
            +---------------- tool result <-----------------+
            |
            v
      Финальный ответ
```

Если модель не запрашивает tool, ответ возвращается сразу. Цикл ограничен пятью обращениями
к LLM (`max_steps`). Ошибка инструмента не роняет запрос: её текст передаётся модели, и она
может ответить, что проверить базу не удалось.

История диалога живёт в Redis под ключом `session_id` с TTL в сутки. Первый запрос без
`session_id` заводит новую сессию и возвращает её идентификатор; последующие запросы с этим
идентификатором продолжают диалог.

## Структура

```text
.
├── .github/workflows/ci.yml          # Lint, format и тесты
├── .agents/skills/explain-diff/      # Skill для объяснения изменений
├── Dockerfile
├── docker-compose.yml                # api + qdrant + redis, профиль tools для индексации
├── pyproject.toml                    # Зависимости, ruff и pytest
├── uv.lock                           # Зафиксированные зависимости
├── tests/                            # Тесты API, агента, инструмента и метрик
└── src/faq_rag_assistant/
    ├── main.py                       # Интерактивный CLI
    ├── config.py                     # Настройки через pydantic-settings
    ├── session_store.py              # История диалога в Redis
    ├── api/app.py                    # FastAPI: POST /chat, GET /health
    ├── agent/
    │   ├── core.py                   # LLM loop и выполнение tools
    │   ├── tools.py                  # search и форматирование результата
    │   ├── tool_schemas.py           # Schema инструмента search
    │   └── exceptions.py             # LLMProviderError
    ├── rag/
    │   ├── ingest.py                 # Загрузка FAQ в Qdrant
    │   ├── embeddings.py             # Embedding-модель и encode
    │   └── vector_store.py           # Collection, upsert и semantic_search
    ├── evaluation/retrieval.py       # Hit Rate и MRR
    └── data/faq.json                 # Исходные вопросы и ответы
```

Клиенты OpenAI, Qdrant, Redis и embedding-модель создаются лениво (`@lru_cache`), а не при
импорте. Поэтому приложение стартует за секунды, тесты обходятся без внешних сервисов,
а модель грузится при первом реальном поиске.

## Запуск в Docker

Нужны Docker и API-ключ с доступом к `gpt-4o-mini`.

```bash
cp .env.example .env    # и вписать OPENAI_API_KEY
docker compose up -d --build
docker compose run --rm ingest    # разовая индексация FAQ
```

Проверка:

```bash
curl http://localhost:8000/health
```

Первый вопрос — без `session_id`, в ответе придёт идентификатор сессии:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Можно потом посмотреть запись урока?"}'
```

```json
{"session_id": "17d77a12-...", "answer": "Да, записи занятий доступны в вашем личном кабинете."}
```

Продолжение диалога — с тем же `session_id`:

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "А сколько это стоит?", "session_id": "17d77a12-..."}'
```

Swagger UI — на [localhost:8000/docs](http://localhost:8000/docs), дашборд Qdrant — на
[localhost:6333/dashboard](http://localhost:6333/dashboard).

| Код | Когда |
| --- | --- |
| `200` | Ответ получен |
| `404` | `session_id` неизвестен или её TTL истёк |
| `422` | Пустое сообщение или сообщение длиннее 2000 символов |
| `502` | LLM-провайдер недоступен |

## Локальный запуск

Нужны Python 3.11 и `uv`. Qdrant и Redis удобнее поднять тем же compose:

```bash
uv sync --locked --group dev
docker compose up -d qdrant redis
uv run --locked python -m faq_rag_assistant.rag.ingest
```

**Индексация удаляет существующую collection `faq` и создаёт её заново.** Для текущих данных
в collection должно быть 10 points: `curl http://localhost:6333/collections/faq`.

API:

```bash
uv run --locked uvicorn faq_rag_assistant.api.app:app --reload
```

Интерактивный CLI (та же логика, что и `/chat`, но диалог живёт в памяти процесса):

```bash
uv run --locked faq-rag-assistant
```

Поиск можно проверить отдельно, без OpenAI:

```bash
uv run --locked python -c 'from faq_rag_assistant.agent.tools import search; print(search("Можно потом посмотреть запись урока?"))'
```

## Настройки

Все параметры — в [config.py](src/faq_rag_assistant/config.py), читаются из окружения или `.env`
(см. [.env.example](.env.example)). Обязателен только `OPENAI_API_KEY`.

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | Ключ OpenAI |
| `LLM_MODEL` | `gpt-4o-mini` | Модель для агента |
| `QDRANT_URL` | `http://localhost:6333` | Адрес Qdrant |
| `COLLECTION_NAME` | `faq` | Имя collection |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Модель эмбеддингов |
| `VECTOR_SIZE` | `384` | Размерность векторов |
| `TOP_K` | `3` | Сколько записей возвращает `search` |
| `REDIS_URL` | `redis://localhost:6379` | Адрес Redis |
| `SESSION_TTL_SECONDS` | `86400` | Время жизни сессии |
| `MAX_STEPS` | `5` | Лимит обращений к LLM за запрос |

`Distance.COSINE` задаётся в `vector_store.py`. В compose `QDRANT_URL` и `REDIS_URL`
переопределяются на имена сервисов.

## Evaluation

`evaluation/retrieval.py` проверяет retrieval на 10 размеченных парах
«перефразированный запрос + `expected_id`». Он использует существующую collection и не
вызывает LLM.

```bash
uv run --locked python -m faq_rag_assistant.evaluation.retrieval
```

Baseline от 19.09.2026, на 10 FAQ-документах без score threshold:

```text
Hit Rate@3: 0.90
MRR@3: 0.90
```

Hit Rate@3 — доля запросов с ожидаемым документом в top-3. MRR@3 — среднее обратной позиции
этого документа; отсутствие в top-3 даёт 0. Девять запросов нашли нужный документ на первом
месте. Единственный промах — `«Если пропущу урок, смогу посмотреть его потом?»`: вернулось
`[1, 7, 2]` вместо ожидаемого ID 3, потому что в тексте запроса нет слова «запись», а модель
эмбеддингов цепляется за «пропущу» и «урок». Это оценка поиска на маленькой выборке,
а не точность ответов LLM.

Baseline получен с Qdrant 1.19.1, qdrant-client 1.19.0 и sentence-transformers 6.0.1. Версии
Python-зависимостей зафиксированы в `uv.lock`, image Qdrant — в `docker-compose.yml`; revision
embedding-модели не закреплён.

## Проверка кода

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

Тесты не ходят в сеть: LLM, Qdrant и Redis подменяются заглушками, поэтому весь набор
проходит за доли секунды. Покрыты сценарии `/chat` (новая сессия, продолжение диалога, 404,
422, 502), LLM loop (ответ без tool call, передача результата инструмента обратно в модель,
ошибка инструмента, исчерпание лимита шагов, ошибка провайдера), форматирование `search`
и подсчёт метрик retrieval. Те же три команды выполняет CI.

## Ограничения

- ID назначаются по порядку записей в JSON. Перестановка FAQ требует сверки `expected_id`
  в evaluation.
- Score threshold не используется: поиск всегда возвращает top-3, даже если они нерелевантны.
  Отсечение мусора возложено на системный промпт.
- Нет reranker, hybrid search и оценки качества финальных ответов — только retrieval.
- Сессии не привязаны к пользователю: знание `session_id` даёт доступ к диалогу.
  Аутентификации и rate limiting нет.
- Embedding-модель загружается в каждый процесс приложения; горизонтальное масштабирование
  потребует вынести её в отдельный сервис.

## Объяснение изменений

В проект добавлен [skill explain-diff](.agents/skills/explain-diff/SKILL.md) из соседнего
`event-bot`. Он объясняет diff, коммит или диапазон коммитов в режимах `summary`, `teach-me`
и `expert`. Флаг `--html` дополнительно сохраняет самостоятельную HTML-страницу в `tmp/`.

```text
$explain-diff summary HEAD
$explain-diff teach-me 560777e
$explain-diff teach-me --html 560777e
```
