import json
from pathlib import Path

FAQ_PATH = Path(__file__).parent / "data" / "faq.json"

# читаем один раз, чтобы не вызывать несколько раз
with open(FAQ_PATH, "r", encoding="utf-8") as file:
    faq = json.load(file)

def search(query: str) -> str:

        query_words = [
            word
            for word in query.lower().split()
            if len(word) > 3 # берем слова которые больше 3eх букв
        ] 

        for item in faq:
            question = item["question"].lower()

            if any(word in question for word in query_words):
                return item["answer"]

        return "В базе FAQ ничего подходящего не найдено"


TOOL_FUNCTIONS = {
    "search": search,
}
