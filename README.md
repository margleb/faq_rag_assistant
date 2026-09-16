# FAQ RAG Assistant

Учебный FAQ-помощник с LLM tool calling и semantic search. Модель может вызвать инструмент `search`, получить подходящие записи из Qdrant и использовать их при формировании ответа.

Проект использует `gpt-4o-mini` через OpenAI SDK, локальную SentenceTransformer и Qdrant. База знаний — 10 вопросов и ответов о курсе в [faq.json](src/faq_rag_assistant/data/faq.json).

## Как работает

При индексации каждая пара вопроса и ответа кодируется в embedding и сохраняется как point с полями `id`, `vector` и `payload`:

```text
faq.json -> ingest.py -> SentenceTransformer -> Qdrant
```

Во время запроса LLM получает описание `search` и решает, нужен ли поиск. Python выполняет запрошенную функцию, кодирует query и получает top-3 результатов по cosine similarity. Найденные вопросы, ответы и score возвращаются модели в сообщении `role="tool"`.

```text
Вопрос -> LLM -> tool call: search -> embedding запроса -> Qdrant
            ^                                               |
            +---------------- tool result <-----------------+
            |
            v
      Финальный ответ
```

Если модель не запрашивает tool, ответ печатается сразу. Цикл ограничен пятью обращениями к LLM.

## Структура

```text
.
├── .agents/skills/explain-diff/  # Skill для объяснения изменений
├── pyproject.toml              # Зависимости и настройки сборки
├── uv.lock                     # Зафиксированные зависимости
└── src/faq_rag_assistant/
    ├── __init__.py             # Шаблонный console entry point
    ├── main.py                 # LLM loop и выполнение tools
    ├── tool_schemas.py         # Schema инструмента search
    ├── tools.py                # search и форматирование результата
    ├── vector_store.py         # Collection, upsert и semantic_search
    ├── embeddings.py           # Общая embedding-модель и encode
    ├── config.py               # Настройки retrieval
    ├── ingest.py               # Загрузка FAQ в Qdrant
    ├── evaluate.py             # Оценка retrieval
    └── data/faq.json           # Исходные вопросы и ответы
```

## Запуск

Команды выполняются из корня проекта. Нужны Python 3.11, `uv` и Docker. Для LLM-сценария также нужен API-ключ с доступом к `gpt-4o-mini`.

### 1. Установить зависимости

```bash
uv sync --locked --group dev
```

В `pyproject.toml` для PyTorch настроен CPU-индекс. При первом запуске SentenceTransformer загружает веса с Hugging Face, затем использует локальный кеш.

### 2. Запустить Qdrant

```bash
docker pull qdrant/qdrant
docker run --name qdrant -d \
  -p 127.0.0.1:6333:6333 \
  -v "$PWD/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Для уже созданного контейнера достаточно `docker start qdrant`. Файлы базы сохраняются в `qdrant_storage/`; эта папка исключена из Git. Docker Compose в проекте нет.

Проверка: `curl http://localhost:6333/`. Веб-интерфейс доступен по адресу [localhost:6333/dashboard](http://localhost:6333/dashboard).

### 3. Загрузить FAQ

**Индексация удаляет существующую collection `faq` и создаёт её заново.**

```bash
uv run --locked python -m faq_rag_assistant.rag.ingest
```

После изменения `faq.json` нужно повторить индексацию. Для текущих данных в collection должно быть 10 points:

```bash
curl http://localhost:6333/collections/faq
```

### 4. Запустить LLM-сценарий

Создайте `.env` в корне проекта со своим ключом:

```dotenv
OPENAI_API_KEY=ваш_ключ
```

`.env` исключён из Git. Затем выполните:

```bash
uv run --locked python -m faq_rag_assistant.main
```

В VS Code этот модуль запускает конфигурация `Python: FAQ RAG Assistant`.

Вопрос задаётся в `messages` внутри `main.py`; сейчас это `«Какая погода в Москве?»`. Чтобы проверить FAQ-сценарий, замените его, например, на `«Можно потом посмотреть запись урока?»`. Интерактивного ввода и инструмента погоды нет.

Команда `uv run faq-rag-assistant` пока вызывает шаблонную функцию из `__init__.py` и печатает `Hello from FAQ RAG Assistant!`. Для LLM loop используйте запуск модуля выше.

Поиск можно проверить отдельно, без OpenAI и изменения исходников:

```bash
uv run --locked python -c 'from faq_rag_assistant.agent.tools import search; print(search("Можно потом посмотреть запись урока?"))'
```

## Настройки

Основные параметры находятся в [config.py](src/faq_rag_assistant/config.py):

| Параметр | Значение |
| --- | --- |
| `QDRANT_URL` | `http://localhost:6333` |
| `COLLECTION_NAME` | `faq` |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `VECTOR_SIZE` | `384` |

`Distance.COSINE` задаётся в `vector_store.py`. `top_k` по умолчанию равен 3; строка `score_threshold=0.4` закомментирована. Модель `gpt-4o-mini` и лимит пяти итераций заданы в `main.py`.

## Evaluation

`evaluate.py` проверяет retrieval на трёх размеченных парах `query + expected_id`. Он использует существующую collection и не вызывает LLM.

```bash
uv run --locked python -m faq_rag_assistant.evaluation.retrieval
```

Последний проверенный baseline от 16.09.2026, на 10 FAQ-документах без score threshold:

```text
Hit Rate@3: 0.67
MRR@3: 0.67
```

Hit Rate@3 показывает долю запросов с ожидаемым документом в top-3. MRR@3 — среднее обратной позиции этого документа; отсутствие в top-3 даёт 0.

Два запроса нашли правильный документ на первом месте. Запрос `«Если пропущу урок, смогу посмотреть его потом?»` вернул `[1, 7, 2]` вместо ожидаемого ID 3. Это оценка поиска на маленькой выборке, а не точность ответов LLM.

Baseline получен с Qdrant server 1.19.1, qdrant-client 1.19.0 и sentence-transformers 6.0.1. Версии Python-зависимостей хранятся в `uv.lock`; Docker image и revision embedding-модели в проекте не закреплены.

## Проверка кода

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Применить форматирование:

```bash
uv run --locked ruff format .
```

## Ограничения

- ID назначаются по порядку записей в JSON. Перестановка FAQ требует сверки `expected_id` в evaluation.
- Без score threshold поиск может вернуть нерелевантные документы. Пустой результат превращается в пустую строку; отдельного fallback нет.
- Нет reranker, hybrid search и оценки качества финальных ответов.
- При исчерпании пяти итераций скрипт может завершиться без финального ответа. Ошибки выполнения tools обрабатываются, ошибки JSON-аргументов и LLM API — нет.
- Модель и клиенты создаются при импорте; ingestion и evaluation также запускаются на уровне модуля.

## Объяснение изменений

В проект добавлен [skill explain-diff](.agents/skills/explain-diff/SKILL.md) из соседнего `event-bot`. Он объясняет diff, коммит или диапазон коммитов в режимах `summary`, `teach-me` и `expert`. Флаг `--html` дополнительно сохраняет самостоятельную HTML-страницу в `tmp/`.

Примеры запросов в Codex:

```text
$explain-diff summary HEAD
$explain-diff teach-me 560777e
$explain-diff teach-me --html 560777e
```

`560777e` — коммит с расширением FAQ, добавлением evaluation и документации. Для подготовки рассказа можно дописать к запросу: «Объясни по-русски и помоги составить краткий рассказ о том, что добавлено, как работает и какие есть ограничения».
