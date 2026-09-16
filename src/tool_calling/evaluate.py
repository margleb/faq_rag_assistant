from tool_calling.vector_store import semantic_search

GROUND_TRUTH = [
    {
        "query": "Я опоздал к началу обучения, ещё можно присоединиться?",
        "expected_id": 1,
    },
    {
        "query": "Сколько времени займёт вся программа?",
        "expected_id": 2,
    },
    {
        "query": "Если пропущу урок, смогу посмотреть его потом?",
        "expected_id": 3,
    },
    # дальше добавим остальные
]


# Hit Rate

hits = []
reciprocal_ranks = []

for item in GROUND_TRUTH:
    results = semantic_search(item["query"], top_k=3)

    expected_id = item["expected_id"]
    found_ids = [point.id for point in results]

    hit = int(expected_id in found_ids)  # считаем hit Rate

    # считаем rr
    if hit:
        rank = found_ids.index(expected_id) + 1
        rr = 1 / rank
    else:
        rr = 0.0

    hits.append(hit)
    reciprocal_ranks.append(rr)

    print(
        f"query={item['query']}\n"
        f"expected={expected_id}\n"
        f"found={found_ids}\n"
        f"hit={hit}\n"
        f"rr={rr}\n"
    )


hit_rate = sum(hits) / len(hits)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)


print(f"Hit Rate@3: {hit_rate:.2f}")
print(f"MRR@3: {mrr:.2f}")
