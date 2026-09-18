import json

from dotenv.main import load_dotenv
from openai import OpenAI

from faq_rag_assistant.agent.tool_schemas import TOOLS
from faq_rag_assistant.agent.tools import TOOL_FUNCTIONS
from faq_rag_assistant.config import MAX_STEPS

load_dotenv()

client = OpenAI()


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    for _ in range(MAX_STEPS):  # максимум 5 итераций
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
        )

        message = resp.choices[0].message
        # Сообщение модели с tool_calls должно идти перед результатами инструментов.
        messages.append(message.model_dump(exclude_none=True))

        # Если вызовов инструментов нет, возвращаем финальный ответ.
        if not message.tool_calls:
            # print(message.content)
            return message.content, messages

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
    return "Не удалось сформировать ответ за допустимое число шагов.", messages


SYSTEM_PROMPT = (
    "Ты ассистент службы поддержки курса. "
    "Отвечай только на основе результатов инструмента search. "
    "Если в базе нет ответа — так и скажи, не придумывай."
)


def create_conversation() -> list[dict]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]
