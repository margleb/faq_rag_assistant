import json

from dotenv import load_dotenv
from openai import OpenAI

from faq_rag_assistant.tool_schemas import TOOLS
from faq_rag_assistant.tools import TOOL_FUNCTIONS

load_dotenv()

client = OpenAI()

messages = [
    {
        "role": "user",
        "content": "Какая погода в Москве?",
    }
]

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
