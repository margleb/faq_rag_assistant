def search(query: str) -> str:
    return (
        f"По запросу '{query}' найдено: "
        "записаться на курс после начала можно."
    )


TOOL_FUNCTIONS = {
    "search": search,
}