# tool_calling

Учебный проект по LLM Engineering: FAQ-помощник с tool calling, локальными embeddings и semantic retrieval в Qdrant. README описывает текущий код и помогает объяснить его на техническом собеседовании.

Состояние проверено **16 сентября 2026 года**: 10 FAQ-записей, 3 evaluation-запроса, `Hit Rate@3 = 0.67` и `MRR@3 = 0.67`. Это метрики retrieval на маленькой выборке, а не точность ответов всей системы.

## Оглавление

1. [Что это за проект](#section-1)
2. [Что я изучаю этим проектом](#section-2)
3. [Архитектура](#section-3)
4. [Tool calling — подробно](#section-4)
5. [Semantic search и embeddings](#section-5)
6. [Qdrant](#section-6)
7. [Индексация](#section-7)
8. [Retrieval](#section-8)
9. [Evaluation](#section-9)
10. [Почему архитектура разделена именно так](#section-10)
11. [Как запустить проект](#section-11)
12. [Что происходит при запуске](#section-12)
13. [Ключевые инженерные решения](#section-13)
14. [Ограничения и учебные упрощения](#section-14)
15. [Что можно улучшить](#section-15)
16. [Вопросы на собеседовании](#section-16)
17. [Как рассказать проект за 2 минуты](#section-17)
18. [Как рассказать проект за 30 секунд](#section-18)
19. [Что я должен точно понимать перед собеседованием](#section-19)

<a id="section-1"></a>

# 1. Что это за проект

Проект показывает, как подключить LLM к небольшой базе знаний о курсе. Данные находятся в [faq.json](src/tool_calling/data/faq.json): сроки обучения, записи занятий, оплата, сертификат и другие вопросы.

Сценарий для вопроса о курсе:

1. Пользовательский текст попадает в `messages`.
2. LLM `gpt-4o-mini` получает сообщения и описание доступного tool `search`.
3. Модель может запросить `search` и сформировать аргумент `query`.
4. Python исполняет `search()`, который вызывает `semantic_search()`.
5. Локальная SentenceTransformer превращает поисковый текст в embedding.
6. Qdrant сравнивает его с заранее сохранёнными векторами FAQ и возвращает top-k кандидатов.
7. Приложение собирает из их payload текст и добавляет сообщение `role="tool"`.
8. Следующий запрос к LLM содержит найденные сведения; модель может сформировать финальный ответ.

Это небольшой **RAG pipeline**: Retrieval-Augmented Generation означает генерацию с привлечением найденного контекста. Здесь retrieval выполняется через tool calling, поэтому решение о поиске принимает LLM. Обучения или fine-tuning модели на FAQ нет: знания передаются в контекст во время запроса.

**Фактический интерфейс пока учебный.** В [main.py](src/tool_calling/main.py) вопрос записан прямо в коде: `«Какая погода в Москве?»`. Интерактивного ввода, HTTP API и инструмента погоды нет. По описанию `search` предназначен для курса; вызов поиска по любому вопросу не обязателен. Возможен короткий путь `User → LLM → final answer`.

<a id="section-2"></a>

# 2. Что я изучаю этим проектом

| Термин | Что означает и где используется |
| --- | --- |
| LLM tool/function calling | Модель выдаёт структурированный запрос на вызов функции. В `main.py` это `message.tool_calls`; реальный вызов выполняет Python. |
| Tool schemas | Машиночитаемое описание имени, назначения и аргументов tool. В `tool_schemas.py` список `TOOLS` описывает `search(query: string)`. |
| Agent/tool loop | Цикл «LLM → выполнение tools → результаты → LLM». В `main.py` он ограничен пятью итерациями. Это минимальная orchestration, без агентного фреймворка. |
| Embeddings | Числовое представление текста. `embeddings.encode()` возвращает вектор длиной 384 для документа или запроса. |
| Semantic search | Поиск по близости таких представлений, позволяющий сопоставлять разные формулировки. Реализован в `vector_store.semantic_search()`. |
| Vector database | Хранилище векторов и связанных данных с операциями поиска. Приложение передаёт числовые vectors отдельному серверу. |
| Qdrant | Конкретная vector database проекта. `QdrantClient` подключается к `http://localhost:6333`, collection называется `faq`. |
| Cosine similarity | Сходство направлений векторов. Задаётся через `Distance.COSINE` в `reset_collection()`. |
| Retrieval | Получение кандидатов до генерации ответа. `semantic_search()` возвращает `list[ScoredPoint]`, не пользовательский ответ. |
| Payload | Данные при point: здесь исходные `question` и `answer`. `tools.search()` превращает их в контекст для LLM. |
| Indexing / ingestion | Предварительная подготовка базы: `ingest.py` читает FAQ, считает embeddings и загружает points. Это не обучение модели. |
| Retrieval evaluation | Проверка поиска по размеченным примерам независимо от LLM. `evaluate.py` сравнивает найденные ID с `expected_id`. |
| Hit Rate@k | Доля запросов, для которых ожидаемый документ найден среди первых k. В `evaluate.py` бинарные значения собираются в `hits`. |
| Reciprocal Rank, RR | Обратная позиция правильного документа: `1 / rank`; при отсутствии в top-k — 0. Считается отдельно для каждого запроса. |
| MRR@k | Среднее RR по evaluation-запросам с обрезкой выдачи на k. В проекте `k=3` и список `reciprocal_ranks`. |

<a id="section-3"></a>

# 3. Архитектура

## Дерево проекта

Ниже все исходные файлы и конфигурация текущего проекта. Локальные данные и кеши вынесены под дерево отдельно.

```text
tool_calling/
├── .gitignore
├── .python-version
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── README.md
├── pyproject.toml
├── uv.lock
└── src/
    └── tool_calling/
        ├── __init__.py
        ├── config.py
        ├── embeddings.py
        ├── evaluate.py
        ├── ingest.py
        ├── main.py
        ├── tool_schemas.py
        ├── tools.py
        ├── vector_store.py
        └── data/
            └── faq.json
```

В рабочей среде также есть `.env`, `.venv/`, `qdrant_storage/`, `.ruff_cache/` и Python-кеши. Это не исходный код. `.env`, `.venv`, `qdrant_storage/` и `__pycache__/` исключены правилами корневого `.gitignore`.

| Файл | Ответственность |
| --- | --- |
| [main.py](src/tool_calling/main.py) | Загружает `.env`, создаёт `OpenAI()`, ведёт `messages`, вызывает LLM, исполняет tools и печатает финальный ответ. |
| [tool_schemas.py](src/tool_calling/tool_schemas.py) | Содержит `TOOLS` — описание интерфейса `search` для модели. |
| [tools.py](src/tool_calling/tools.py) | `search()` форматирует найденные FAQ; `TOOL_FUNCTIONS` связывает имя из tool call с Python-функцией. |
| [vector_store.py](src/tool_calling/vector_store.py) | Содержит Qdrant-клиент и функции `reset_collection()`, `upsert_points()`, `semantic_search()`. |
| [embeddings.py](src/tool_calling/embeddings.py) | Создаёт `_model` и предоставляет единый `encode(text)` для ingestion и поиска. |
| [ingest.py](src/tool_calling/ingest.py) | Читает JSON, пересоздаёт collection, кодирует FAQ и загружает points. |
| [evaluate.py](src/tool_calling/evaluate.py) | Содержит `GROUND_TRUTH`, запускает retrieval и выводит Hit Rate@3 / MRR@3. |
| [config.py](src/tool_calling/config.py) | URL Qdrant, имя collection, embedding-модель и размерность. |
| [data/faq.json](src/tool_calling/data/faq.json) | 10 пар `question + answer`; собственных полей ID в JSON нет. |
| [__init__.py](src/tool_calling/__init__.py) | Шаблонная `main()`, печатающая `Hello from tool-calling!`. Именно на неё пока указывает console script. |
| [pyproject.toml](pyproject.toml) | Метаданные пакета, Python ≥3.11, зависимости, dev-группа Ruff, сборка через `uv_build`, CPU-источник PyTorch. |
| [uv.lock](uv.lock) | Зафиксированные версии и источники зависимостей для воспроизводимой установки. |
| [.python-version](.python-version) | Выбран Python 3.11. |
| [.gitignore](.gitignore) | Исключает окружение, секреты, генерируемые Python-файлы и хранилище Qdrant. |
| [.vscode/settings.json](.vscode/settings.json) | Локальный интерпретатор, `.env`, Ruff как formatter и действия при сохранении. |
| [.vscode/launch.json](.vscode/launch.json) | Конфигурация отладки; `program` пока указывает на отсутствующий `src/tool_calling/tool_calling.py`. |
| [README.md](README.md) | Документация, команды воспроизведения и подготовка к собеседованию. |

## Runtime flow

Ветка, в которой LLM выбрала поиск:

```text
User
  |
  v
LLM: gpt-4o-mini
  |
  v
tool_call: search(query=...)
  |
  v
main.py: TOOL_FUNCTIONS["search"](**tool_arguments)
  |
  v
tools.py: search()
  |
  v
vector_store.py: semantic_search()
  |
  v
embeddings.py: encode() -> embedding
  |
  v
Qdrant: query_points(collection="faq")
  |
  v
top-k ScoredPoint: id, score, payload
  |
  v
tools.py: текст из score + question + answer
  |
  v
main.py: role="tool", tool_call_id, content
  |
  v
LLM
  |
  v
final answer -> print()
```

Модель может запросить несколько tools в одном сообщении или ещё один поиск на следующей итерации.

## Ingestion flow

```text
data/faq.json
  |
  v
ingest.py: json.load()
  |
  v
reset_collection(): удалить старую faq -> создать новую faq
  |
  v
"Вопрос: ...\nОтвет: ..."
  |
  v
embeddings.encode() -> embedding
  |
  v
PointStruct(id, vector, payload)
  |
  v
список points -> upsert_points()
  |
  v
Qdrant collection "faq"
```

## Evaluation flow

```text
GROUND_TRUTH: query + expected_id
  |
  v
semantic_search(query, top_k=3)
  |
  v
top-k IDs
  |
  v
сравнение с expected_id -> hit и reciprocal rank
  |
  v
усреднение -> Hit Rate@3 + MRR@3
```

Evaluation использует тот же retrieval, что и tool, но не обращается к OpenAI и не оценивает генерацию.

<a id="section-4"></a>

# 4. Tool calling — подробно

## Tool schema и Python-функция — разные части

В [tool_schemas.py](src/tool_calling/tool_schemas.py) находится фактическая схема:

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Поиск по базе FAQ курса. Используй, когда вопрос касается курса.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]
```

`description` объясняет модели назначение поиска. `parameters` описывает JSON-объект с обязательной строкой `query`. `additionalProperties: False` запрещает дополнительные поля в рамках схемы.

Схема не содержит исполняемый Python-код, содержимое FAQ или доступ к базе. Модель получает только описание интерфейса. `top_k` существует в Python-сигнатуре `search(query: str, top_k: int = 3)`, но не объявлен в schema, поэтому корректный tool call передаёт только `query`, а функция использует значение 3.

`strict=True` в проекте не установлен. Схема также не заменяет валидацию аргументов на стороне приложения: отдельной проверки JSON по schema сейчас нет.

## Фактический цикл main.py

```python
for _ in range(5):  # максимум 5 итераций
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )

    message = resp.choices[0].message
    # Сообщение модели с tool_calls должно идти перед результатами инструментов.
    messages.append(message)

    # Если вызовов инструментов нет, печатаем финальный ответ.
    if not message.tool_calls:
        print(message.content)
        break

    # Выполняем все инструменты, которые запросила модель.
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments)

        try:
            result = TOOL_FUNCTIONS[tool_name](**tool_arguments)
        except Exception as e:  # noqa: BLE001
            result = f"Ошибка при вызове {tool_name}: {e}"

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)

        messages.append(
            {  # Передаём результат функции с идентификатором её вызова.
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )
```

Пошагово:

1. **`messages`.** История начинается с одного сообщения `role="user"` с вопросом о погоде. System prompt и интерактивный диалог не реализованы. `load_dotenv()` загружает окружение, затем `OpenAI()` создаёт API-клиент.
2. **`tools=TOOLS`.** Список схем передаётся модели в каждом запросе. Это описание доступных действий.
3. **Первый запрос к LLM.** `client.chat.completions.create()` получает `model="gpt-4o-mini"`, историю и tools. `tool_choice` не указан: при наличии tools стандартное поведение — `auto`, модель может вызвать tool или ответить текстом.
4. **`message.tool_calls`.** Берётся `resp.choices[0].message`. Список `tool_calls` содержит запросы модели на действия; сам по себе он ещё ничего не исполняет.
5. **`function.name`.** Имя запрошенного действия, для текущей schema — `"search"`.
6. **`function.arguments`.** JSON, представленный строкой. Например, строка `{"query":"Можно потом посмотреть запись урока?"}` — иллюстрация возможных аргументов, а не сохранённый ответ API.
7. **`json.loads`.** Превращает JSON-строку в Python-объект, ожидаемо `dict`. Это разбор данных, а не выполнение кода.
8. **`TOOL_FUNCTIONS`.** Реестр `{"search": search}` выбирает доступную Python-функцию по имени. Здесь нет `eval` или исполнения присланного моделью кода.
9. **Вызов Python-функции.** `TOOL_FUNCTIONS[tool_name](**tool_arguments)` распаковывает словарь в именованные аргументы и запускает поиск. Именно здесь начинается реальная работа с embedding-моделью и Qdrant.
10. **Возврат assistant message в историю.** `messages.append(message)` происходит до обработки tools и сохраняет исходные tool calls с их ID. Следующий запрос содержит информацию о том, что сама модель попросила сделать.
11. **Сообщение `role="tool"`.** Приложение добавляет результат функции как отдельное сообщение. У `search()` это строка; другие нестроковые результаты общий код сериализовал бы через `json.dumps(..., ensure_ascii=False)`.
12. **`tool_call_id`.** Связывает результат с конкретным вызовом из assistant message. Это особенно важно при нескольких вызовах. Это не имя функции и не ID документа в Qdrant.
13. **Повторный вызов LLM.** На следующей итерации модель получает расширенную историю и результаты поиска. Она может ответить или запросить ещё один tool.
14. **Завершение.** Если `message.tool_calls` отсутствует или пуст, печатается `message.content` и выполняется `break`. Иначе цикл продолжится, но максимум до пяти обращений к `chat.completions.create()`.

При достижении лимита нет отдельного сообщения об остановке. Если пятая итерация снова вернула tool calls, Python выполнит их и добавит результаты, но шестого запроса, который передаст эти результаты модели, уже не будет. Финальный ответ в таком случае может не появиться. Пять итераций — не ограничение в пять Python-вызовов: на одной итерации их может быть несколько.

## Три события, которые нельзя путать

| Событие | Кто его выполняет | Что уже произошло |
| --- | --- | --- |
| «Нужно вызвать `search`» | LLM | Сформированы имя и аргументы. Поиск ещё не запущен. |
| `search(query=...)` | Python-приложение | Посчитан embedding, выполнен запрос к Qdrant, подготовлен результат. |
| Результат доступен модели | Приложение отправляет следующий запрос LLM | `role="tool"` уже включён в `messages`, модель может использовать данные. |

Просто добавление результата в локальный список не отправляет его по сети. Передача происходит при следующем `client.chat.completions.create()`. Такой lifecycle соответствует [официальному описанию function calling](https://developers.openai.com/api/docs/guides/function-calling).

## Границы обработки ошибок

`try/except` охватывает выбор функции из реестра и её выполнение. Неизвестное имя или ошибка retrieval превратятся в строку `Ошибка при вызове ...` и попадут модели.

Но `json.loads` находится **до `try`**: повреждённый JSON остановит скрипт. Вызов OpenAI, загрузка embedding-модели и сериализация нестрокового результата также не защищены этим блоком. Собственной стратегии повторов и восстановления в проекте нет; это не утверждение об отсутствии внутренних retries в SDK.

<a id="section-5"></a>

# 5. Semantic search и embeddings

```text
текст
  -> SentenceTransformer
  -> embedding: numpy.ndarray, shape=(384,), dtype=float32
  -> Qdrant
```

В [config.py](src/tool_calling/config.py) зафиксированы настройки:

```python
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "faq"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VECTOR_SIZE = 384
```

А [embeddings.py](src/tool_calling/embeddings.py) содержит общую точку кодирования:

```python
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from tool_calling.config import EMBEDDING_MODEL

# Одна модель на модуль для индексации и поиска.
_model = SentenceTransformer(EMBEDDING_MODEL)


def encode(text: str) -> ndarray:
    return _model.encode(text)
```

## Что представляет собой embedding

Embedding — плотный числовой вектор, в котором обученная модель представляет текст. Близкие по смыслу тексты могут получить близкие направления векторов, даже если используют разные слова.

Это не словарь «слово → координата» и не гарантия правильного понимания любого перефразирования. В текущем evaluation запрос про пропущенный урок как раз демонстрирует ошибку такого сопоставления.

Модель `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` преобразует текст в 384-мерное пространство; размерность подтверждается [model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) и локальным запуском `encode()`.

**`VECTOR_SIZE = 384` не заставляет модель выдать 384 числа.** Размер выхода задаёт сама модель. Константа сообщает Qdrant, сколько координат ожидать. Эти величины должны совпадать.

## Почему кодируем query и документы одной моделью

Документ кодируется как `Вопрос: ...\nОтвет: ...`, а query — как поисковая строка. Оба проходят через один `encode()`, поэтому их векторы находятся в согласованном пространстве.

Одинаковая длина векторов разных моделей ещё не делает их сопоставимыми: координаты могут иметь другой смысл. При замене embedding-модели нужно заново закодировать документы; при изменении размерности также изменить схему collection. Простая смена модели только для запросов нарушит retrieval.

`_model` создаётся при импорте `embeddings.py` и повторно используется внутри процесса. Запуски ingestion, evaluation и main — отдельные процессы: каждый загружает собственный экземпляр модели.

## Encoding — не decoding

`encode(text)` вычисляет представление текста. Здесь нет обратного преобразования embedding в исходный вопрос или ответ.

Текст для LLM берётся из `payload`. Вектор помогает найти point, а ответ хранится рядом с ним в обычном текстовом виде. Embedding нельзя рассматривать как обратимый архив документа.

## numpy.ndarray и list

| Тип | В контексте проекта |
| --- | --- |
| `numpy.ndarray` | Числовой массив с размерностью `shape` и типом элементов `dtype`. `encode()` при проверке вернул одномерный `float32`-массив длиной 384. Удобен для векторных вычислений. |
| `list` | Обычный Python-список. Может содержать объекты разных типов; список чисел удобен для сериализации в API. |
| `vector.tolist()` | Конвертация массива в Python-список. Не пересчитывает embedding и не меняет его смысл. |

В текущем коде нет ручного `.tolist()`. Это проверено с установленными `qdrant-client 1.19.0` и `numpy 2.4.6`:

- `query_points(query=...)` поддерживает `numpy.ndarray`; внутренний `_resolve_query()` клиента преобразует его через `query.tolist()`.
- `PointStruct(vector=encode(text), ...)` также успешно создаётся; его модель данных приводит полученный массив к списку. При проверке `type(point.vector).__name__` был `list`.

Таким образом, Python-клиент принимает массив напрямую на входе, но это не означает, что объект NumPy передаётся по HTTP без преобразования. Поддержка конкретного типа зависит от используемого API и версии, поэтому не стоит переносить это утверждение на любой клиент vector database.

В проекте `numpy` импортируется напрямую, но в `pyproject.toml` отдельно не объявлен: он установлен как транзитивная зависимость.

<a id="section-6"></a>

# 6. Qdrant

## Основные сущности

| Термин | Реализация в проекте |
| --- | --- |
| Collection | Набор points с заданной схемой векторов. Здесь одна collection `faq`. |
| Point | Одна FAQ-пара вместе с ID, embedding и payload. Создаётся как `PointStruct`. |
| `id` | Целое число от 1 до 10, полученное при перечислении текущего JSON. Используется для адресации point и evaluation. |
| `vector` | 384 координаты embedding текста вопроса и ответа. Используется для векторного поиска. |
| `payload` | Объект с полями `question` и `answer`. Возвращается приложению и превращается в текст для LLM. |
| `VectorParams` | Настройки vector space collection: `size=VECTOR_SIZE`, `distance=Distance.COSINE`. |
| `upsert` | Записывает points: вставляет новый ID или обновляет существующий. В текущем ingestion выполняется после пересоздания collection. |
| `query_points` | API поиска. Здесь получает dense query vector, лимит и `with_payload=True`. |
| `top_k` | Параметр Python-функций; передаётся Qdrant как `limit`. По умолчанию 3, то есть не более трёх результатов. |
| `score` | Оценка векторной близости найденного point. При Cosine большее значение означает большее сходство. Это не вероятность правильного ответа. |
| `score_threshold` | Порог допустимого score. Сейчас строка `score_threshold=0.4` закомментирована, фильтрация по порогу выключена. |

Пример point для третьей записи FAQ; координаты намеренно не выдуманы:

```text
id: 3
vector: encode("Вопрос: Есть ли записи занятий?\nОтвет: Записи занятий доступны в личном кабинете.")
        -> 384 числа
payload:
  question: "Есть ли записи занятий?"
  answer: "Записи занятий доступны в личном кабинете."
```

`vector` отвечает на вопрос «насколько эта запись близка запросу?». `payload` отвечает на вопрос «какой текст передать приложению?». В текущем `query_points` нет payload-фильтров: сами поля `question` и `answer` не участвуют в отдельном keyword-поиске.

## Cosine similarity

Для ненулевых векторов `q` и `d`:

```text
cosine_similarity(q, d) = (q · d) / (||q|| * ||d||)
```

Числитель — скалярное произведение, знаменатель — произведение длин векторов. Сравниваются направления: изменение длины вектора без изменения направления не меняет cosine similarity.

Математический диапазон — от −1 до 1. В Qdrant для Cosine векторы нормализуются при загрузке, а сходство вычисляется через скалярное произведение нормализованных векторов. В нашем `encode()` явная нормализация не включена, но метрику обеспечивает Qdrant. См. [описание collections и метрик](https://qdrant.tech/documentation/manage-data/collections/).

## Почему «лучшие» не обязательно релевантные

Nearest-neighbor search ранжирует имеющиеся документы. Даже если все они не относятся к вопросу, среди них всё равно найдутся ближайшие.

Проверенный прямой вызов `semantic_search("Какая погода в Москве?", top_k=3)` вернул:

| ID | FAQ | Score, округлён |
| --- | --- | --- |
| 7 | Когда проходят занятия? | 0.183643 |
| 9 | Есть ли домашние задания? | 0.092014 |
| 2 | Сколько длится курс? | 0.088561 |

Эта проверка вызывала retrieval напрямую; она не доказывает, что LLM выберет `search` для вопроса о погоде.

Если включить порог 0.4, эти три результата будут отсеяны. Но **0.4 не является измеренным оптимальным порогом**. Его нужно подбирать на релевантных и нерелевантных запросах: высокий порог снижает количество шума, но может удалить полезные документы. В провальном evaluation-примере score нужного документа около 0.299634, поэтому такой порог удалил бы и его.

Для Cosine `score_threshold` задаёт нижнюю границу score; после фильтрации результатов может быть меньше k или ни одного. Семантика зависит от выбранной метрики; см. [Qdrant: search и threshold](https://qdrant.tech/documentation/search/search/). Сейчас при пустой выдаче `search()` возвращает пустую строку, отдельная логика «ответа в базе нет» не реализована.

<a id="section-7"></a>

# 7. Индексация

[ingest.py](src/tool_calling/ingest.py) выполняет весь pipeline на уровне модуля:

```python
import json
from pathlib import Path

from qdrant_client.models import PointStruct

from tool_calling.embeddings import encode
from tool_calling.vector_store import reset_collection, upsert_points

FAQ_PATH = Path(__file__).parent / "data" / "faq.json"

# Исходный JSON нужен только для индексации.
with open(FAQ_PATH, "r", encoding="utf-8") as file:
    faq = json.load(file)

reset_collection()

# Индексируем вопрос вместе с ответом, сохраняя исходные поля в payload.
points = []

for idx, item in enumerate(faq, start=1):
    text = f"Вопрос: {item['question']}\nОтвет: {item['answer']}"

    points.append(
        PointStruct(
            id=idx,
            vector=encode(text),
            payload={
                "question": item["question"],
                "answer": item["answer"],
            },
        )
    )

upsert_points(points)
```

Порядок действий:

1. `FAQ_PATH` строится относительно `__file__`. Это путь к `src/tool_calling/data/faq.json`, а не к папке `data` в корне.
2. JSON читается в UTF-8 и превращается в список словарей.
3. `reset_collection()` удаляет существующую `faq`, если она есть, и создаёт пустую collection с размерностью 384 и Cosine.
4. Для каждой записи формируется единый текст `Вопрос: {question}\nОтвет: {answer}`.
5. `encode(text)` вычисляет embedding этого текста.
6. `PointStruct` получает числовой ID, vector и исходные поля payload.
7. Все points собираются в список и передаются `upsert_points(points)` одним вызовом.

Одна FAQ-пара здесь равна одному документу и одному point. Chunking длинных документов не реализован. Переменная `chunks` в `tools.py` означает фрагменты форматируемого результата, а не отдельный алгоритм разбиения текста.

## Зачем считать embeddings заранее

Документы меняются реже, чем приходят пользовательские запросы. Их embeddings можно вычислить один раз при индексации, сохранить и использовать многократно.

Во время поиска вычисляется только embedding запроса. Повторное кодирование всей базы на каждый запрос добавило бы работу пропорционально числу документов и лишнюю задержку. Предварительная индексация переносит эти затраты на этап подготовки или обновления данных.

Изменение `faq.json` само по себе не меняет Qdrant: необходим новый ingestion.

## ID и upsert

При неизменном порядке JSON `enumerate(faq, start=1)` повторно создаёт те же ID. Текущее соответствие:

| ID | Вопрос |
| --- | --- |
| 1 | Можно ли записаться после начала курса? |
| 2 | Сколько длится курс? |
| 3 | Есть ли записи занятий? |
| 4 | Сколько стоит обучение? |
| 5 | Можно ли оплатить курс частями? |
| 6 | Выдается ли сертификат? |
| 7 | Когда проходят занятия? |
| 8 | Можно ли вернуть деньги за курс? |
| 9 | Есть ли домашние задания? |
| 10 | Как связаться с преподавателем? |

**Это позиционные, а не независимо стабильные ID.** Вставка записи в начало или перестановка FAQ изменит соответствия и может сделать `expected_id` неверным. Для устойчивой идентификации понадобились бы явные ID в данных; сейчас их нет.

Концептуально `upsert` объединяет insert и update: point с новым ID создаётся, с существующим — обновляется. Это позволяет адресовать тот же документ при повторной загрузке, а не постоянно добавлять новые записи.

Но текущий pipeline **сначала удаляет collection**, поэтому на практике каждый ingestion делает полную загрузку в пустое хранилище. Называть его инкрементальным обновлением нельзя. Если после удаления возникнет ошибка кодирования или загрузки, прежней collection уже не будет. Атомарного переключения индексов нет.

## Почему qdrant_storage не хранится в Git

`qdrant_storage/` — файлы состояния сервера: данные points, служебные структуры и журналы. Это изменяемые, в том числе бинарные, артефакты, а не удобный формат для review исходных данных.

Для этого учебного проекта база восстанавливается из `faq.json` и ingestion-кода. Поэтому directory исключена из Git, а версии исходных данных и зависимостей хранятся в репозитории. В реальной системе резервное копирование базы было бы отдельной задачей.

<a id="section-8"></a>

# 8. Retrieval

Фактическая функция из [vector_store.py](src/tool_calling/vector_store.py):

```python
def semantic_search(query: str, top_k: int = 3) -> list[ScoredPoint]:
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=encode(query),
        limit=top_k,
        # score_threshold=0.4,  # отсекаем мусор
        with_payload=True,
    )
    return result.points
```

```text
query: str
  -> encode(query)
  -> query_points(query=embedding, limit=top_k, with_payload=True)
  -> QueryResponse.points
  -> list[ScoredPoint]
```

`query_points` возвращает объект ответа; `semantic_search()` извлекает из него только `result.points`. Поиск идёт по уже загруженной collection, JSON в этом пути не читается.

## Что такое ScoredPoint

`ScoredPoint` — модель результата поиска из `qdrant_client.models`. Для проекта важны три поля:

| Поле | Для чего используется |
| --- | --- |
| `id` | `evaluate.py` строит `found_ids` и сравнивает их с `expected_id`. |
| `score` | `tools.py` включает значение в tool result. Порядок выдачи отражает ранжирование Qdrant. |
| `payload` | Из него `tools.py` достаёт `question` и `answer`. |

У `ScoredPoint` есть и другие поля, например `version` и необязательный `vector`. Наш код их не использует. Векторы в ответе не запрашиваются; в установленном клиенте `with_vectors` по умолчанию выключен.

`with_payload=True` нужен потому, что для ответа необходим исходный текст. Код предполагает наличие обоих полей payload и обращается к ним напрямую.

## Как retrieval становится tool result

Полная реализация [tools.py](src/tool_calling/tools.py):

```python
from tool_calling.vector_store import semantic_search


def search(query: str, top_k: int = 3) -> str:
    points = semantic_search(query, top_k=top_k)
    chunks = []

    for point in points:
        chunks.append(
            f"score: {point.score}\n"
            f"Вопрос: {point.payload['question']}\n"
            f"Ответ: {point.payload['answer']}"
        )

    return "\n\n".join(chunks)


TOOL_FUNCTIONS = {
    "search": search,
}
```

Результат `search()` — одна строка, где кандидаты разделены пустыми строками. Она содержит score, вопрос и ответ каждого кандидата. ID документа в этот текст не включён.

`search()` не выбирает окончательный ответ и не вызывает LLM. Он подготавливает контекст. **Retrieval** находит документы, **generation** составляет текст с учётом истории и полученных сведений.

Поэтому успешный retrieval не гарантирует правильной генерации, а внешне правильный ответ LLM не доказывает, что retrieval сработал: модель могла ответить из собственных знаний или догадаться. Эти этапы нужно проверять отдельно.

<a id="section-9"></a>

# 9. Evaluation

## Что проверяет evaluate.py

[evaluate.py](src/tool_calling/evaluate.py) напрямую вызывает `semantic_search`. Он измеряет, попал ли ожидаемый FAQ-документ в top-3 и на каком месте.

Не проверяются решение LLM вызвать tool, сформулированный ею `query`, корректность tool calling, содержание финального ответа, latency и стоимость API. Ключ OpenAI для этого скрипта не нужен.

## GROUND_TRUTH

В коде три размеченных примера:

```python
GROUND_TRUTH = [
    {
        "query": "Я опоздал к началу обучения, ещё можно присоединиться?",
        "expected_id": 1,
    },
    {
        "query": "Сколько времени займёт вся программа?",
        "expected_id": 2,
    },
    {
        "query": "Если пропущу урок, смогу посмотреть его потом?",
        "expected_id": 3,
    },
    # дальше добавим остальные
]
```

Каждый пример содержит:

- `query` — пользовательскую формулировку, которую отправляем в retrieval;
- `expected_id` — ID FAQ-документа, который человек считает правильным для этого запроса.

Evaluation query должен проверять разные способы выразить намерение. Если использовать только исходный FAQ-вопрос, можно получить слишком оптимистичную оценку: поиск будет сопоставлять почти одинаковые тексты. Перефразирование показывает, связывает ли модель, например, «пропущу урок» с «записи занятий».

Это не абсолютный запрет на точные вопросы в тестах: они полезны как простая проверка. Но ими нельзя ограничивать оценку semantic search.

Разметка опирается на текущий порядок `faq.json`. Она не извлекается автоматически из ответа LLM и должна соответствовать данным в Qdrant.

## Что происходит в цикле

Ключевой фрагмент существующего кода:

```python
hits = []
reciprocal_ranks = []

for item in GROUND_TRUTH:
    results = semantic_search(item["query"], top_k=3)

    expected_id = item["expected_id"]
    found_ids = [point.id for point in results]

    hit = int(expected_id in found_ids)  # считаем hit Rate

    # считаем rr
    if hit:
        rank = found_ids.index(expected_id) + 1
        rr = 1 / rank
    else:
        rr = 0.0

    hits.append(hit)
    reciprocal_ranks.append(rr)
```

Для каждого query:

1. Вызывается тот же поиск, что использует приложение, с `top_k=3`.
2. Из `ScoredPoint` извлекаются ID в порядке выдачи.
3. Считается бинарный hit.
4. При попадании определяется позиция с единицы: `index(...) + 1`.
5. Считается RR и сохраняется для последующего усреднения.
6. Скрипт печатает query, expected ID, найденные ID, hit и RR.

## Hit Rate@k

Hit одного запроса равен 1, если ожидаемый документ присутствует среди первых k, иначе 0.

Учебный пример, не результат текущего запуска:

```text
expected_id = 3
found_ids = [7, 3, 5]

Hit@3 = 1
```

Документ найден на втором месте, но для Hit Rate позиция внутри top-k не важна.

```text
Hit Rate@k =
    число запросов, где правильный документ попал в top-k
    / число всех запросов

Hit Rate@k = (1/N) * sum(hit_i)
```

При одном ожидаемом документе на query эта метрика также совпадает с Recall@k для такой разметки. Для нескольких релевантных документов их смысл уже различается: Hit Rate проверяет хотя бы одно попадание, Recall учитывает долю найденных релевантных документов.

## Reciprocal Rank и MRR@k

Reciprocal Rank учитывает позицию правильного документа:

| Позиция в выдаче | RR |
| --- | --- |
| 1 | `1 / 1 = 1` |
| 2 | `1 / 2 = 0.5` |
| 3 | `1 / 3 ≈ 0.3333` |
| Не найден в top-k | `0` |

Для примера `found_ids = [7, 3, 5]` при `expected_id = 3` получаем `RR@3 = 0.5`.

```text
RR@k(query) =
    1 / rank, если expected_id найден на позиции rank <= k
    0,        если не найден среди первых k

MRR@k = (RR_1@k + RR_2@k + ... + RR_N@k) / N
```

В коде усреднение выглядит так:

```python
hit_rate = sum(hits) / len(hits)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)


print(f"Hit Rate@3: {hit_rate:.2f}")
print(f"MRR@3: {mrr:.2f}")
```

Обычно RR определяется по первому релевантному результату; здесь каждому query соответствует ровно один `expected_id`. Обозначение `@3` важно: позиция за пределами трёх не участвует, такой запрос получает RR=0.

**Hit Rate отвечает «нашли ли вообще в top-k?». MRR отвечает «насколько высоко поставили правильный документ?».**

## Воспроизведённый baseline

На 16.09.2026 фактический запуск текущего `evaluate.py` дал:

| Query | Expected ID | Found IDs, top-3 | Hit | RR |
| --- | --- | --- | --- | --- |
| Я опоздал к началу обучения, ещё можно присоединиться? | 1 | `[1, 7, 2]` | 1 | 1.0 |
| Сколько времени займёт вся программа? | 2 | `[2, 1, 7]` | 1 | 1.0 |
| Если пропущу урок, смогу посмотреть его потом? | 3 | `[1, 7, 2]` | 0 | 0.0 |

Конечные строки вывода:

```text
Hit Rate@3: 0.67
MRR@3: 0.67
```

Условия проверки:

- 10 документов из текущего `faq.json`; payload существующих points совпал с JSON;
- сохранённые vectors совпали с заново посчитанными embeddings `question + answer` после нормализации с максимальной разницей координат около `2 × 10⁻⁸`;
- `top_k=3`, Cosine, без `score_threshold`;
- `qdrant-client 1.19.0`, `sentence-transformers 6.0.1`, `numpy 2.4.6`, `torch 2.14.0+cpu`;
- локальный Qdrant server `1.19.1`;
- кешированная revision embedding-модели: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`.

Проверка использовала существующую collection без её перезаписи. `evaluate.py` был выполнен с локальной моделью в offline-режиме. Это реальный retrieval-результат; LLM в проверке не участвовала.

Python-зависимости закреплены в `uv.lock`, но server image tag и revision модели в коде не закреплены. Поэтому при изменении окружения, FAQ, модели или коллекции baseline нужно измерить заново.

## Почему обе метрики равны 0.67

`hits = [1, 1, 0]` и `reciprocal_ranks = [1.0, 1.0, 0.0]`. Каждый успешный поиск поставил правильный документ сразу на первое место:

```text
Hit Rate@3 = (1 + 1 + 0) / 3 = 2/3 ≈ 0.67
MRR@3      = (1 + 1 + 0) / 3 = 2/3 ≈ 0.67
```

Если бы второй успешный запрос нашёл документ на втором месте, Hit Rate остался бы `2/3`, а MRR стал бы `(1 + 0.5 + 0) / 3 = 0.50`.

Для этой разметки всегда `MRR@k ≤ Hit Rate@k`. Без округления равенство означает, что все попадания были на первом месте, либо попаданий не было. Одинаковые значения после округления до двух знаков сами по себе ещё не доказывают равенство; в нашем запуске это видно из RR каждого запроса.

## Провальный запрос и границы reranker

Запрос `«Если пропущу урок, смогу посмотреть его потом?»` должен найти ID 3 — `«Есть ли записи занятий?»`. Но top-3 состоит из ID 1, 7 и 2: запись после начала курса, расписание, длительность.

Дополнительная диагностическая проверка с `top_k=10` показала:

```text
found_ids = [1, 7, 2, 6, 10, 5, 4, 3, 8, 9]
                                 ^
                          expected_id=3, rank=8
```

У ID 3 score примерно `0.299634`; у первого кандидата — `0.507143`. Это наблюдаемая ошибка ранжирования относительно человеческой разметки. Возможное объяснение — выбранная модель и представление `question + answer` недостаточно хорошо связывают неявную формулировку с записью занятия. Одного примера недостаточно, чтобы доказать точную причину.

**Reranker только переупорядочивает полученных кандидатов.** Если передать ему `[1, 7, 2]`, он не сможет вернуть отсутствующий ID 3. Нужен более широкий или качественный candidate retrieval: другой k, представление документов, embedding-модель или дополнительный способ поиска.

Для этих десяти документов `k=10` включает нужный FAQ, но это ещё не улучшенная production-система и не измеренная работа reranker. В проекте reranker отсутствует. Увеличение k повышает объём шума и контекста; эффект нужно проверять.

## Что baseline позволяет утверждать

Он подтверждает два успешных и один неуспешный retrieval на данной выборке. Он не доказывает качество на всех вопросах о курсе.

Три запроса — слишком мало для устойчивой оценки. Нет отдельного validation/test split, отрицательных примеров без ответа в FAQ, нескольких правильных документов на query и оценки генерации. Настройка модели или порога на этих же трёх примерах может привести к подгонке под них.

<a id="section-10"></a>

# 10. Почему архитектура разделена именно так

```text
main.py         -> orchestration / LLM loop
tool_schemas.py -> описание интерфейса tool для LLM
tools.py        -> Python tool interface и формат результата
vector_store.py -> доступ к Qdrant
embeddings.py   -> embedding-модель
ingest.py       -> indexing
evaluate.py     -> retrieval evaluation
config.py       -> общие настройки retrieval
```

Это separation of concerns: каждый модуль отвечает за свой уровень. Например, формат tool result можно менять в `tools.py`, не переписывая запрос к Qdrant, а retrieval проверять через `evaluate.py` без вызова OpenAI.

История Git подтверждает рефакторинг в коммите `f8ddc9e` «Рефакторинг модулей embeddings, Qdrant и инструментов»:

| До рефакторинга | В текущем коде |
| --- | --- |
| `SentenceTransformer` создавался отдельно в `ingest.py` и `tools.py` | Один модуль `embeddings.py` с `_model` и `encode()`. |
| `QdrantClient` создавался в ingestion и tools | Клиент в `vector_store.py`. |
| Имя embedding-модели, URL, collection и размерность были разбросаны по коду | Настройки собраны в `config.py`. |
| `tools.py` непосредственно вызывал `query_points` | Tool делегирует retrieval в `semantic_search()`. |
| `tools.py` читал FAQ при импорте в runtime, хотя использовал payload результатов | JSON читает только `ingest.py`; runtime получает текст из Qdrant. |

Раньше JSON читался один раз при импорте `tools.py`, а не на каждый поисковый запрос. Рефакторинг убрал именно это лишнее runtime-чтение.

Разделение остаётся небольшим: функции, модули и один словарь диспетчеризации. Нет необходимости вводить сложный framework или иерархию классов для одного tool.

Есть и ограничения: модель и клиент — глобальные объекты модулей, dependency injection отсутствует, их создание связано с импортами. Не все настройки вынесены в `config.py`: `gpt-4o-mini` и лимит цикла находятся в `main.py`, `top_k` — в сигнатурах и evaluation.

<a id="section-11"></a>

# 11. Как запустить проект

Команды выполняются из корня `tool_calling`. Нужны установленный `uv`, Python 3.11 и работающий Docker. `pyproject.toml` допускает Python ≥3.11, а `.python-version` выбирает 3.11.

## Зависимости

```bash
uv sync --locked --group dev
```

Команда устанавливает проект и зависимости в `.venv`. `--locked` требует согласованности с `uv.lock` без изменения lock-файла; `--group dev` явно включает Ruff.

| Зависимость | В pyproject.toml | Версия при проверке | Назначение |
| --- | --- | --- | --- |
| `openai` | `>=3.13.0` | 3.13.0 | Клиент Chat Completions API. |
| `python-dotenv` | `>=1.2.3` | 1.2.3 | Загрузка `.env` в `main.py`. |
| `qdrant-client` | `>=1.19.0` | 1.19.0 | Работа с сервером Qdrant. |
| `sentence-transformers` | `>=6.0.1` | 6.0.1 | Локальный расчёт embeddings. |
| `torch` | Без ограничения версии | 2.14.0+cpu | Вычислительный backend модели в проверенной Linux-среде. |
| `ruff` | `>=0.16.7`, dev | 0.16.7 | Lint и форматирование Python. |

В `tool.uv.sources` для `torch` выбран индекс `https://download.pytorch.org/whl/cpu`. CUDA для проверенного запуска не нужна. NumPy приходит транзитивно; версия в проверенной Python 3.11-среде — 2.4.6, а lock-файл содержит и другие варианты для других окружений.

При первом создании `SentenceTransformer` потребуются загрузка весов модели и доступ к Hugging Face. Далее модель может использовать локальный кеш. OpenAI нужен только для `main.py`; ingestion и evaluation работают без API-ключа OpenAI.

## Qdrant через Docker

В репозитории нет Dockerfile или Compose-файла. Следующая команда запускает стандартный Qdrant image, использует HTTP-порт из `config.py` и локальную папку, исключённую в `.gitignore`. Основа команды — [официальный Qdrant quickstart](https://qdrant.tech/documentation/quickstart/).

Для нового контейнера:

```bash
docker pull qdrant/qdrant
docker run --name qdrant -d \
  -p 127.0.0.1:6333:6333 \
  -v "$PWD/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant
```

Если контейнер с именем `qdrant` уже существует, повторно создавать его не нужно:

```bash
docker start qdrant
```

Проверка доступности:

```bash
curl http://localhost:6333/
```

Dashboard: [localhost:6333/dashboard](http://localhost:6333/dashboard). Проект использует HTTP; порт gRPC 6334 для этих вызовов не требуется.

В проверенной среде image назывался `qdrant/qdrant`, сервер сообщил версию 1.19.1. Тег версии в репозитории не закреплён, поэтому `docker pull` в другое время может получить другую версию.

## Индексация

**Команда удаляет существующую collection `faq` и создаёт её заново.** Использовать для полной переиндексации учебной базы.

```bash
uv run --locked python -m tool_calling.ingest
```

Проверить состояние collection:

```bash
curl http://localhost:6333/collections/faq
```

После загрузки текущего JSON ожидается `points_count: 10`, размерность 384 и distance `Cosine`. Сам `ingest.py` не печатает отдельное сообщение об успешной загрузке.

## Настройка и запуск LLM

В корневом `.env` укажите свой API-ключ; ниже шаблон значения, а не рабочий ключ:

```dotenv
OPENAI_API_KEY=ваш_ключ
```

`main.py` вызывает `load_dotenv()`, а `OpenAI()` читает ключ из окружения. `.env` исключён из Git; `.env.example` в проекте нет. Для запуска нужны доступ к API и к указанной в коде модели `gpt-4o-mini`.

```bash
uv run --locked python -m tool_calling.main
```

Сейчас выполняется один встроенный вопрос `«Какая погода в Москве?»`. Аргумента `--query` и `input()` нет. Перед демонстрацией другого запроса в самом `main.py` пришлось бы изменить содержимое его `messages`; приведённый далее walkthrough объясняет этот сценарий, а не наличие готового CLI.

**Не путать с console script:**

```bash
uv run --locked tool-calling
```

Он пока вызывает `tool_calling:main` из `__init__.py` и печатает только:

```text
Hello from tool-calling!
```

Команда `python -m tool_calling` также не запускает агентный цикл: файла `__main__.py` нет. Для цикла нужна именно `python -m tool_calling.main`.

## Evaluation и отдельная проверка поиска

```bash
uv run --locked python -m tool_calling.evaluate
```

Команда использует существующую collection, ничего в неё не загружает и не вызывает LLM.

Посмотреть tool result для собственного вопроса без изменения исходников:

```bash
uv run --locked python -c 'from tool_calling.tools import search; print(search("Можно потом посмотреть запись урока?"))'
```

Если модель уже полностью закеширована, evaluation можно выполнить с запретом обращений к Hugging Face:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run --no-sync python -B -m tool_calling.evaluate
```

`--no-sync` предполагает уже подготовленную `.venv`. Offline-режим модели не отключает соединение с локальным Qdrant.

## Ruff

Проверка Python-кода:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Применение форматирования Python, когда это необходимо:

```bash
uv run --locked ruff format .
```

Последняя команда изменяет Python-файлы. При подготовке этого README запускались только проверки: `ruff check --no-cache .` и `ruff format --check --no-cache .`. Они прошли; проверенные файлы уже отформатированы. Ruff не проверяет смысл Markdown-документа.

## Частые причины проблем

| Симптом | Что проверить |
| --- | --- |
| Ошибка соединения с Qdrant | Работает ли контейнер и доступен ли `http://localhost:6333`. |
| Collection `faq` отсутствует | Выполнен ли ingestion после запуска сервера. |
| Поиск не отражает изменения JSON | После редактирования FAQ нужна переиндексация. |
| Ошибка credentials в `OpenAI()` | Есть ли `OPENAI_API_KEY` в окружении или корневом `.env`. |
| Не загружается SentenceTransformer | Доступ к модели при первой загрузке или полноту локального кеша. |
| Несовпадение размерности | Соответствуют ли embedding-модель и схема collection значению `VECTOR_SIZE`. |
| Отладка VS Code не запускает файл | `launch.json` содержит устаревший путь к `tool_calling.py`; используйте показанную команду запуска модуля. |

<a id="section-12"></a>

# 12. Что происходит при запуске

Разберём вопрос `«Можно потом посмотреть запись урока?»`. Это пример для объяснения пути исполнения; текущий литерал в `main.py` остаётся вопросом о погоде.

**Данные retrieval ниже измерены напрямую. Выбор tool, его аргументы и финальный ответ LLM в этом walkthrough иллюстративные: end-to-end вызов OpenAI при подготовке README не выполнялся.**

1. **Импорты.** `main.py` импортирует `TOOL_FUNCTIONS` из `tools.py`, тот импортирует `semantic_search` из `vector_store.py`, а он — `encode` из `embeddings.py`. Загружается `SentenceTransformer`, создаётся Qdrant-клиент. Поэтому модель embeddings загружается даже если позже LLM не выберет поиск.
2. **Окружение и история.** Загружается `.env`, создаётся `OpenAI()`. Для рассматриваемого сценария пользовательское сообщение содержит вопрос о записи урока.
3. **Первый запрос.** `gpt-4o-mini` получает сообщение и schema `search`. Содержимого Qdrant и Python-кода функции в этом запросе нет.
4. **Решение модели.** Допустим, LLM выдаёт tool call `search` с `query="Можно потом посмотреть запись урока?"`. Модель может сформулировать query иначе; тогда результаты поиска могут отличаться от приведённых.
5. **Диспетчеризация.** `main.py` сохраняет assistant message, разбирает arguments через `json.loads`, получает функцию из `TOOL_FUNCTIONS` и реально вызывает `search(query=...)`.
6. **Tool interface.** `search()` вызывает `semantic_search(query, top_k=3)`.
7. **Embedding.** `encode(query)` возвращает массив из 384 чисел. Веса модели уже в памяти. Документы при этом заново не кодируются.
8. **Qdrant.** `query_points` ищет по `faq`, ограничивает выдачу тремя points и прикладывает payload. Порог score выключен.
9. **Candidates.** Для точно этой поисковой строки проверенная выдача такова:

| Rank | ID | Score, округлён | Question | Answer |
| --- | --- | --- | --- | --- |
| 1 | 1 | 0.645229 | Можно ли записаться после начала курса? | Да, запись доступна в течение первой недели. |
| 2 | 3 | 0.595237 | Есть ли записи занятий? | Записи занятий доступны в личном кабинете. |
| 3 | 7 | 0.546067 | Когда проходят занятия? | Занятия проходят по вторникам и четвергам вечером. |

Нужный документ есть, но на втором месте. Возможная причина первого кандидата — неоднозначность слова «запись» между записью на курс и видеозаписью. Это гипотеза по результатам, а не установленная причина внутри модели.

10. **Tool result.** `search()` превращает candidates в одну строку. Ниже значения score округлены для чтения; код выводит их без такого округления:

```text
score: 0.645229
Вопрос: Можно ли записаться после начала курса?
Ответ: Да, запись доступна в течение первой недели.

score: 0.595237
Вопрос: Есть ли записи занятий?
Ответ: Записи занятий доступны в личном кабинете.

score: 0.546067
Вопрос: Когда проходят занятия?
Ответ: Занятия проходят по вторникам и четвергам вечером.
```

11. **Возврат в messages.** `main.py` добавляет эту строку как `role="tool"` с ID исходного tool call. В истории теперь есть пользовательский вопрос, запрос действия от assistant и результат.
12. **Второй запрос к LLM.** Модель видит все три фрагмента и может использовать ответ про личный кабинет. Наличие более высокого score у другого документа не обязывает её копировать первый фрагмент.
13. **Финальный ответ.** Возможный ответ: «Да, записи занятий доступны в личном кабинете». Это пример ответа, который поддерживается FAQ, а не зафиксированный output LLM. Если `tool_calls` отсутствуют, `main.py` печатает текст и завершает цикл; иначе обрабатывает новые вызовы.

Для этой отдельной строки retrieval дал бы `Hit@3=1` и `RR@3=0.5` при `expected_id=3`. Она **не входит** в `GROUND_TRUTH` и не меняет описанный baseline.

На доске важно показать две границы: LLM передаёт приложению структурированное намерение, а приложение возвращает LLM найденный текст. Embedding и Qdrant работают между этими двумя API-вызовами.

<a id="section-13"></a>

# 13. Ключевые инженерные решения

Таблица объясняет практический смысл текущей реализации. Это анализ её свойств, а не утверждение, что автор измерил превосходство над всеми альтернативами.

| Решение | Почему сделано так | Альтернатива | Trade-off |
| --- | --- | --- | --- |
| Qdrant вместо ручного сравнения всех vectors | Даёт хранение points, payload и API retrieval; позволяет изучить vector database. | Матрица NumPy и cosine similarity со всеми документами. | Для 10 FAQ ручной поиск проще; Qdrant добавляет сервер и настройку. Выигрыш скорости здесь не измерялся. |
| Embeddings вместо keyword matching | Позволяют искать перефразирования без точного совпадения слов. | Поиск подстроки, полнотекстовый поиск, BM25. | Могут путать смыслы и пропускать релевантные документы; вычисление vectors требует модели. |
| Top-k retrieval, k=3 | Ограничивает объём контекста и даёт несколько кандидатов. | Top-1, большее k, адаптивный лимит. | Малое k теряет документы; большое приносит шум и больше текста в LLM. Тройка не доказана как оптимальная. |
| Cosine similarity | Сравнивает направления embeddings, схема явно задана в collection. | Dot product или Euclidean distance. | Подходящую метрику нужно согласовывать с embedding-моделью; сравнение альтернатив не проводилось. |
| Отдельный ingestion | Документы кодируются заранее, runtime считает только query. | Пересчёт базы на каждом запросе. | Нужна явная переиндексация при изменении данных; текущее пересоздание collection разрушает предыдущее состояние. |
| Tool calling | LLM выбирает поиск по назначению tool. | Жёсткий вызов retrieval для каждого вопроса. | Гибкость добавляет зависимость от решения модели и разбор аргументов; успешный сценарий обычно требует минимум двух LLM-запросов. |
| Отдельный evaluation | Можно измерять retrieval без генерации и API-расходов OpenAI. | Оценивать только финальные ответы end-to-end. | Легче локализовать ошибку, но качество генерации остаётся вне измерения. |
| Hit Rate + MRR | Видно и наличие нужного документа, и его позицию. | Только одна метрика; для других задач — Recall или nDCG. | Эти две метрики с одним expected ID не описывают полноту сложной выдачи или качество текста ответа. |
| Локальная SentenceTransformer | Один encoder для query и документов, локальный расчёт после загрузки весов. | Внешний embedding API. | Нет API-вызова для каждого embedding, но требуются память, CPU и время загрузки; качество нужно измерять. |
| Разделение модулей | Orchestration, formatting, storage и encoding можно понимать и менять по отдельности. | Один скрипт; сложный framework с интерфейсами. | Несколько файлов вместо одного, при этом глобальные объекты ещё затрудняют изоляцию компонентов. |
| Question + answer как текст документа | Вектор учитывает и вопрос, и содержание ответа. | Только question, несколько перефразирований, отдельные chunks. | Больше смысловых сигналов, но текст ответа может смещать сходство; превосходство этого представления не проверялось. |
| Строковый tool result | Простая передача score и исходного FAQ в контекст. | Структурированный JSON с ID и полями. | Легко читать, но нет формального контракта результата и явных ID источников для LLM. |

Наличие Qdrant не означает, что на этом объёме обязательно используется HNSW. При проверке collection содержала 10 points и `indexed_vectors_count=0`. Это не отсутствие загруженных vectors; нельзя приписывать текущему примеру измеренное ускорение ANN-поиском.

<a id="section-14"></a>

# 14. Ограничения и учебные упрощения

| Ограничение текущего кода | Практическое следствие |
| --- | --- |
| 10 документов и 3 evaluation query | Метрики полезны для изучения механики, но не дают надёжной оценки на широком наборе вопросов. |
| Встроенный вопрос в `main.py` | Это демонстрационный скрипт, не готовый чат или HTTP-сервис. |
| Один tool `search` | Погода и любые внешние актуальные данные недоступны. |
| Нет system prompt с правилами ответа по FAQ | Модель не обязана строго опираться только на найденный контекст; groundedness не контролируется отдельной логикой. |
| `score_threshold` закомментирован | Даже нерелевантный query получает ближайшие FAQ-кандидаты. |
| Пустая выдача превращается в `""` | Нет явного формата «ничего не найдено» и гарантированного корректного отказа от ответа. |
| Dense retrieval без reranker и hybrid search | Ошибки semantic matching ничем дополнительно не исправляются. |
| ID зависят от порядка FAQ | Перестановка записей требует сверки evaluation-разметки. |
| Ingestion пересоздаёт collection | Это полная замена базы, без incremental updates и сохранения прежнего состояния при сбое. |
| Загрузка модели и создание клиента при импорте | Старт main включает инициализацию retrieval даже при ответе без tool; компоненты трудно подменять изолированно. |
| `main.py`, `ingest.py`, `evaluate.py` исполняются на уровне модуля | Их импорт запускает соответствующий сценарий. В частности, импорт ingestion может пересоздать collection. Защиты `if __name__ == "__main__"` нет. |
| Частичная обработка ошибок | Ошибки функции возвращаются модели, но некорректный JSON аргументов и ошибки LLM API могут прервать процесс. |
| Лимит пяти итераций без явного fallback | Возможен выход без финального ответа. |
| Несколько tool calls исполняются последовательно | Асинхронная обработка и конкурентное выполнение не реализованы. |
| Версии модели и сервера не полностью закреплены | Один `uv.lock` не фиксирует весь эксперимент. |
| Нет автоматических тестов и CI-конфигурации | Evaluation — диагностический скрипт с печатью метрик, без assertions и проверки минимального качества. |
| Нет измерений latency, throughput и стоимости | Нельзя заявлять конкретную производительность или экономию. |
| Console script и VS Code launch остались от шаблона/старой структуры | Рабочая команда агентного сценария — запуск `tool_calling.main` как модуля. |

Эти ограничения не отменяют полезность учебного проекта: он позволяет пройти lifecycle от tool schema до retrieval-метрик на небольшом коде. Но их нужно назвать при обсуждении готовности к реальному использованию.

<a id="section-15"></a>

# 15. Что можно улучшить

Ниже направления дальнейшей работы. **Они не реализованы в текущем проекте.** Их стоит рассматривать как проверяемые гипотезы, а не обещание автоматического роста качества.

| Шаг | Что изменить | Как проверить пользу |
| --- | --- | --- |
| Расширить разметку | Добавить перефразирования всех FAQ, неоднозначные и нерелевантные вопросы; выделить отложенную выборку. | Считать метрики на ней отдельно от примеров, использованных для настройки. |
| Сделать ID устойчивыми | Хранить явный ID у каждого FAQ вместо позиции в массиве. | После перестановки записей соответствие `query → expected_id` остаётся верным. |
| Проверить разные k | Сравнить несколько размеров candidate pool. | Смотреть Hit Rate/MRR вместе с количеством кандидатов и объёмом контекста; найденный ID 3 на позиции 8 показывает смысл эксперимента. |
| Сравнить представления документов | Question-only против текущего question+answer; при необходимости добавить полезные перефразирования. | Полная переиндексация для каждого варианта и одинаковый evaluation dataset. |
| Сравнить embedding-модели | Выбрать encoder по качеству на русских FAQ, а не только по названию. | Переиндексировать всю базу, измерить retrieval и ресурсы. Не смешивать vector spaces. |
| Исследовать hybrid retrieval | Дополнить semantic search лексическим сигналом, например BM25. | Проверить, появляются ли ранее пропущенные документы среди candidates. |
| Добавить reranker | Переоценивать достаточно широкий набор найденных кандидатов с учётом пары query/document. | Отдельно измерять попадание до reranking и позиции после него; отсутствующий candidate переупорядочить нельзя. |
| Подобрать порог и fallback | Калибровать `score_threshold` и явно возвращать отсутствие подходящих данных. | Использовать положительные и отрицательные запросы, считать ошибочные ответы и пропуски. |
| Усилить tool loop | Валидировать arguments, обрабатывать ошибки разбора, сообщать об исчерпании итераций. | Проверить некорректный JSON, неизвестный tool, сбой Qdrant и повторные tool calls. |
| Сделать запуск удобным | Добавить ввод запроса, исправить entry point и конфигурацию отладки, убрать побочные эффекты импорта. | Проверить команды запуска и безопасный импорт модулей. |
| Улучшить ingestion | Перейти к обновлению изменившихся записей либо загрузке новой collection с последующим переключением. | Проверить поведение при изменении/удалении FAQ и прерывании загрузки. |
| Измерять всю систему | Оценивать выбор tool, query от LLM, верность ответа источнику, задержку и расход токенов. | Сопоставлять ошибки orchestration, retrieval и generation по отдельным этапам. |
| Фиксировать эксперимент | Закреплять revision модели, версию сервера, данные и параметры поиска. | Повторный запуск в том же окружении даёт сопоставимые результаты. |

На десяти документах полезно начать с расширения evaluation и разбора конкретных промахов. Усложнение архитектуры само по себе не является улучшением.

<a id="section-16"></a>

# 16. Вопросы на собеседовании

**1. Чем tool calling отличается от обычного текстового ответа?**

В ответе появляются структурированные `tool_calls` с именем и JSON-аргументами. Приложение разбирает их, выполняет разрешённую функцию и отправляет результат модели. Текст «я сейчас поищу» сам по себе функцию не запускает.

**2. Выполняет ли LLM ваш Python-код?**

Нет. LLM запрашивает действие. `main.py` находит функцию в `TOOL_FUNCTIONS` и вызывает её в Python-процессе. Модель видит результат только в следующем API-запросе.

**3. Зачем сохранять assistant message перед tool result?**

Оно содержит исходный запрос действия и его ID. `role="tool"` с `tool_call_id` отвечает на этот конкретный запрос. Одного текста результата без истории вызова недостаточно для корректного tool lifecycle.

**4. Чем tool schema отличается от реализации tool?**

Schema в `tool_schemas.py` описывает контракт для модели. `tools.search()` — реализация. Реестр `TOOL_FUNCTIONS` связывает их по имени `search`. Изменения интерфейса нужно согласовывать в обоих местах.

**5. Почему это RAG, если нет LangChain или LlamaIndex?**

RAG определяется потоком данных: поиск внешнего контекста и генерация с этим контекстом. В проекте он реализован обычными функциями, Qdrant-клиентом и Chat Completions API.

**6. Где здесь две разные модели?**

`gpt-4o-mini` через API выбирает tool и генерирует ответ. Локальная `paraphrase-multilingual-MiniLM-L12-v2` кодирует тексты в embeddings. Она не составляет пользовательский ответ.

**7. Что означает размерность 384?**

Это число координат одного embedding, задаваемое encoder. Оно не связано с числом FAQ или длиной ответа. `VECTOR_SIZE` должен соответствовать этому выходу при создании collection.

**8. Можно ли заменить encoder другим с той же размерностью?**

Можно как эксперимент, но придётся перекодировать документы. Совпадение размерности не означает совпадения vector space.

**9. Откуда берётся текст ответа, если в базе vectors?**

У point также есть payload с вопросом и ответом. Qdrant возвращает payload, `search()` форматирует его, а LLM получает строку. Decoding embedding здесь отсутствует.

**10. Почему выбирается Cosine?**

Текущая схема сравнивает направления vectors и не использует их длину как отдельный сигнал. Это понятный baseline для таких embeddings; превосходство над другими метриками в проекте не измерено.

**11. Может ли высокий score означать неправильный FAQ?**

Да. Score отражает близость представлений, а не истинность ответа. В примере про запись урока вопрос о записи на курс ранжируется выше вопроса о видеозаписях.

**12. Почему Hit Rate и MRR нужны вместе?**

Hit Rate фиксирует попадание в top-k, MRR штрафует низкую позицию. Для `expected_id=3` и `[7, 3, 5]` hit равен 1, а RR — 0.5.

**13. Что показывает ваш baseline 0.67 / 0.67?**

Два из трёх запросов нашли правильный документ на первом месте, третий не нашёл его в top-3. Это не «67% точности чат-бота», а retrieval-метрики на трёх примерах.

**14. Поможет ли reranker провальному запросу?**

Для текущих трёх кандидатов — нет: ID 3 отсутствует. Сначала нужно получить его в candidate pool. При диагностическом `top_k=10` он оказался восьмым; оценка reranker была бы следующим отдельным экспериментом.

**15. Почему Qdrant для десяти записей?**

Чтобы изучить vector database, ingestion, payload и retrieval API. Для такого объёма обычная матрица NumPy была бы достаточной альтернативой. Доказанного ускорения или требования масштабирования здесь нет.

**16. Что бы вы улучшили первым?**

Расширил бы evaluation и закрепил ID, затем сравнил бы k и представление документов на одинаковой выборке. Это даст основание выбирать дальнейшие изменения, включая модель, hybrid search или reranker.

<a id="section-17"></a>

# 17. Как рассказать проект за 2 минуты

> Я сделал учебный FAQ-помощник, чтобы разобраться в полном цикле tool calling и в измерении retrieval. В базе десять вопросов и ответов о курсе. Проект состоит из Python-модулей, локальной embedding-модели, Qdrant и вызовов gpt-4o-mini через OpenAI SDK.
>
> Сначала отдельный ingestion читает JSON. Для каждой FAQ-пары он объединяет вопрос с ответом, считает 384-мерный embedding через SentenceTransformer и сохраняет в Qdrant point с ID, vector и текстовым payload. Сейчас ingestion полностью пересоздаёт коллекцию, это учебное упрощение.
>
> Во время обработки вопроса LLM получает описание tool search и может решить его вызвать. Важно, что модель не выполняет Python: она возвращает имя функции и JSON-аргументы. Мой код разбирает их, выбирает функцию из реестра и запускает её.
>
> Поиск кодирует только query и запрашивает top-3 в Qdrant по cosine similarity. Tool превращает найденные payload в текст. Я сохраняю assistant message с вызовом, добавляю отдельное сообщение role tool с тем же tool_call_id и снова обращаюсь к модели. Теперь она может сформировать ответ с найденным контекстом.
>
> Retrieval я проверяю отдельно от генерации. В evaluation есть три перефразированных запроса с ожидаемыми ID. Hit Rate показывает попадание в top-3, MRR учитывает позицию. При проверке обе метрики получились 0.67: два документа были первыми, третий не попал в выдачу.
>
> Промах связан с вопросом про пропущенный урок. Нужный FAQ о записях оказался восьмым при расширенном поиске. Поэтому reranker только первых трёх результатов проблему не решит: сначала нужно улучшить набор кандидатов.
>
> Ограничения я понимаю: выборка маленькая, порог score выключен, ID зависят от порядка JSON, нет оценки финальных ответов и полноценного пользовательского интерфейса. Следующим шагом я бы расширил разметку и сравнил параметры retrieval на отложенных примерах. Разделение модулей позволяет делать это без изменения LLM-цикла.

<a id="section-18"></a>

# 18. Как рассказать проект за 30 секунд

> Это учебный RAG-помощник по FAQ с tool calling. LLM решает вызвать search, Python исполняет функцию, локальная SentenceTransformer кодирует запрос, а Qdrant возвращает top-3 документов. Их текст передаётся модели как tool result для ответа. Поиск оценивается отдельно: на трёх запросах Hit Rate@3 и MRR@3 равны 0.67. Один промах показал, что reranker не исправит отсутствие нужного документа среди кандидатов. Следующий шаг — расширить evaluation и улучшить retrieval; качество генерации пока не измерялось.

<a id="section-19"></a>

# 19. Что я должен точно понимать перед собеседованием

- [ ] Могу объяснить задачу проекта и показать фактический вопрос в `main.py`.
- [ ] Могу нарисовать runtime flow от user message до final answer.
- [ ] Могу отдельно нарисовать ingestion flow и evaluation flow.
- [ ] Понимаю разницу между решением LLM вызвать tool, выполнением Python-функции и передачей результата модели.
- [ ] Могу показать в коде `TOOLS`, `message.tool_calls`, `function.name` и `function.arguments`.
- [ ] Могу объяснить, зачем нужны `json.loads`, `TOOL_FUNCTIONS` и распаковка `**tool_arguments`.
- [ ] Могу объяснить, почему assistant message добавляется до результатов tools.
- [ ] Понимаю назначение `role="tool"` и отличие `tool_call_id` от ID документа Qdrant.
- [ ] Понимаю, когда цикл заканчивается и что произойдёт при tool calls на пятой итерации.
- [ ] Могу показать, какие ошибки ловит `try/except` и почему ошибочный JSON находится вне него.
- [ ] Понимаю разницу между генеративной LLM и embedding-моделью проекта.
- [ ] Могу назвать фактическую embedding-модель и объяснить размерность 384.
- [ ] Понимаю, почему query и документы должны кодироваться согласованной моделью.
- [ ] Могу объяснить, почему embedding — не decoding и почему текст берётся из payload.
- [ ] Понимаю разницу между `numpy.ndarray` и `list` и где клиент преобразует массив.
- [ ] Могу объяснить collection, point, ID, vector и payload на примере FAQ с ID 3.
- [ ] Могу объяснить формулу cosine similarity и смысл score без трактовки его как вероятности.
- [ ] Могу показать `query_points`, `limit=top_k`, `with_payload=True` и выключенный `score_threshold`.
- [ ] Понимаю, почему nearest-neighbor search может вернуть нерелевантные документы.
- [ ] Могу объяснить, зачем ingestion вынесен отдельно и какие embeddings считаются при запросе.
- [ ] Понимаю отличие upsert от текущего полного пересоздания collection.
- [ ] Понимаю зависимость ID от порядка FAQ и связь с `GROUND_TRUTH`.
- [ ] Могу показать поля `ScoredPoint`, которые используют tools и evaluation.
- [ ] Понимаю разницу между retrieval и generation и границы текущей оценки качества.
- [ ] Могу вручную вычислить Hit Rate@k, RR и MRR@k для небольшого примера.
- [ ] Могу объяснить, почему текущие 0.67 / 0.67 совпали и почему они не обязаны совпадать.
- [ ] Могу разобрать провальный query про пропущенный урок и объяснить ограничение reranker.
- [ ] Могу запустить установку, Qdrant, ingestion, main, evaluation и Ruff по командам README.
- [ ] Знаю, почему `tool-calling` сейчас печатает приветствие, и могу назвать правильный запуск main.
- [ ] Могу объяснить фактический рефакторинг и его пользу без приписывания проекту лишних абстракций.
- [ ] Могу назвать ограничения: размер выборки, отсутствие fallback, неустойчивые ID, полная переиндексация, отсутствие оценки генерации.
- [ ] Могу предложить следующий эксперимент и заранее сказать, какой результат будет считаться улучшением.
