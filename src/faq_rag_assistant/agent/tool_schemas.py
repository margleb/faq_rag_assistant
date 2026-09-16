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
