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
    messages.append(message) # добавляем сам ответ модели

    # если модель приняла решение о том что отвечать больше не следует
    if not message.tool_calls:             
        print(message.content)
        break

    # если модель приняла решение о том что нужно вызывать инструмент
    for tool_call in message.tool_calls: 
        
        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments)

        try:
            result = TOOL_FUNCTIONS[tool_name](**tool_arguments)
        except Exception as e:
            result = f"Ошибка при вызове {tool_name}: {e}" 

        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)     

        messages.append({ # а затем то что вернула функция
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })