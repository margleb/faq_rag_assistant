import json
from dotenv import load_dotenv
from openai import OpenAI
from tool_calling.tool_schemas import TOOLS
from tool_calling.tools import TOOL_FUNCTIONS

load_dotenv()

client = OpenAI()

messages = [{"role": "user", "content": "Можно ли записаться на курс после его начала?"}]

for _ in range(5): # максимум 5 итераций
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )

    message = resp.choices[0].message

    # если модель приняла решение о том что нужно вызывать инструмент
    if message.tool_calls: 
        
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments)

        tool_function = TOOL_FUNCTIONS[tool_name]
        result = tool_function(**tool_arguments)

        messages.append(message) # добавляем сам ответ модели

        messages.append({ # а затем то что вернула функция
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # иначе отвечаем то как модель ответила сама
    else: 
        print(message.content)
        break