"""Интерактивный CLI поверх агента: то же, что делает POST /chat, но в терминале."""

import logging

from faq_rag_assistant.agent.core import create_conversation, run_agent
from faq_rag_assistant.agent.exceptions import LLMProviderError


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    messages = create_conversation()
    print("FAQ RAG Assistant. Пустая строка или Ctrl+C — выход.\n")

    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not question:
            return

        messages.append({"role": "user", "content": question})

        try:
            answer, messages = run_agent(messages)
        except LLMProviderError as error:
            print(f"Ассистент недоступен: {error}\n")
            return

        print(f"Ассистент: {answer}\n")


if __name__ == "__main__":
    main()
