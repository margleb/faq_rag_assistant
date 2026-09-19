"""Оценка retrieval на размеченных парах «перефразированный запрос -> ожидаемый ID».

Использует уже существующую collection и не вызывает LLM.
"""

from dataclasses import dataclass

from faq_rag_assistant.rag.vector_store import semantic_search

# expected_id совпадает с порядковым номером записи в faq.json (нумерация с 1).
GROUND_TRUTH = [
    {"query": "Я опоздал к началу обучения, ещё можно присоединиться?", "expected_id": 1},
    {"query": "Сколько времени займёт вся программа?", "expected_id": 2},
    {"query": "Если пропущу урок, смогу посмотреть его потом?", "expected_id": 3},
    {"query": "Какая цена обучения?", "expected_id": 4},
    {"query": "Есть ли рассрочка платежа?", "expected_id": 5},
    {"query": "Получу ли я документ об окончании?", "expected_id": 6},
    {"query": "В какие дни недели расписание занятий?", "expected_id": 7},
    {"query": "Как оформить возврат средств?", "expected_id": 8},
    {"query": "Нужно ли делать практику после модулей?", "expected_id": 9},
    {"query": "Куда написать вопрос ментору?", "expected_id": 10},
]


@dataclass
class RetrievalMetrics:
    """Hit Rate@k — доля запросов с нужным документом в top-k.

    MRR@k — среднее обратной позиции этого документа (нет в top-k -> 0).
    """

    hit_rate: float
    mrr: float


def reciprocal_rank(found_ids: list[int], expected_id: int) -> float:
    if expected_id not in found_ids:
        return 0.0
    return 1 / (found_ids.index(expected_id) + 1)


def compute_metrics(ranked_ids: list[list[int]], expected_ids: list[int]) -> RetrievalMetrics:
    ranks = [
        reciprocal_rank(found, expected)
        for found, expected in zip(ranked_ids, expected_ids, strict=True)
    ]
    total = len(ranks)

    return RetrievalMetrics(
        hit_rate=sum(rank > 0 for rank in ranks) / total,
        mrr=sum(ranks) / total,
    )


def main(top_k: int = 3) -> RetrievalMetrics:
    ranked_ids = []

    for item in GROUND_TRUTH:
        found_ids = [point.id for point in semantic_search(item["query"], top_k=top_k)]
        ranked_ids.append(found_ids)

        rank = reciprocal_rank(found_ids, item["expected_id"])
        print(
            f"query={item['query']}\n"
            f"expected={item['expected_id']}\n"
            f"found={found_ids}\n"
            f"hit={int(rank > 0)}\n"
            f"rr={rank:.2f}\n"
        )

    metrics = compute_metrics(ranked_ids, [item["expected_id"] for item in GROUND_TRUTH])

    print(f"Hit Rate@{top_k}: {metrics.hit_rate:.2f}")
    print(f"MRR@{top_k}: {metrics.mrr:.2f}")

    return metrics


if __name__ == "__main__":
    main()
