from faq_rag_assistant.agent import run_agent

messages = [
    {
        "role": "user",
        "content": "Какая погода в Москве?",
    }
]

answer, messages = run_agent(messages)

print(answer)
