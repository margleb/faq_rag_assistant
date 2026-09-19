import json
import logging
from functools import lru_cache

from openai import OpenAI, OpenAIError

from faq_rag_assistant.agent.exceptions import LLMProviderError
from faq_rag_assistant.agent.tool_schemas import TOOLS
from faq_rag_assistant.agent.tools import TOOL_FUNCTIONS
from faq_rag_assistant.config import get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты ассистент службы поддержки курса. "
    "Отвечай только на основе результатов инструмента search. "
    "Если в базе нет ответа — так и скажи, не придумывай."
)

NO_ANSWER_REPLY = "Не удалось сформировать ответ за допустимое число шагов."


@lru_cache
def get_client() -> OpenAI:
    settings = get_settings()
    # Пустой ключ передавать бессмысленно: OpenAI() сам возьмёт OPENAI_API_KEY.
    if settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key)
    return OpenAI()


def create_conversation() -> list[dict]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]


def _call_tool(name: str, arguments: str) -> str:
    """Любая ошибка инструмента возвращается модели текстом, а не роняет запрос."""
    try:
        parsed = json.loads(arguments)
        result = TOOL_FUNCTIONS[name](**parsed)
    except Exception as error:  # noqa: BLE001
        logger.warning("Вызов инструмента %s завершился ошибкой: %s", name, error)
        return f"Ошибка при вызове {name}: {error}"

    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False)


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    settings = get_settings()

    for step in range(1, settings.max_steps + 1):
        try:
            response = get_client().chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=TOOLS,
            )
        except OpenAIError as error:
            raise LLMProviderError("Ошибка при обращении к LLM-провайдеру") from error

        message = response.choices[0].message
        # Сообщение модели с tool_calls должно идти перед результатами инструментов.
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content, messages

        logger.info(
            "Шаг %s/%s: модель запросила инструменты %s",
            step,
            settings.max_steps,
            [call.function.name for call in message.tool_calls],
        )

        for tool_call in message.tool_calls:
            messages.append(
                {  # Результат функции передаётся с идентификатором её вызова.
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _call_tool(
                        tool_call.function.name,
                        tool_call.function.arguments,
                    ),
                }
            )

    logger.warning("Лимит в %s шагов исчерпан без финального ответа", settings.max_steps)
    return NO_ANSWER_REPLY, messages
