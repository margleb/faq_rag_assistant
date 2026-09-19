import pytest

from faq_rag_assistant.evaluation.retrieval import compute_metrics, reciprocal_rank


@pytest.mark.parametrize(
    ("found_ids", "expected_id", "expected_rr"),
    [
        ([3, 1, 2], 3, 1.0),
        ([1, 3, 2], 3, 0.5),
        ([1, 2, 3], 3, pytest.approx(1 / 3)),
        ([1, 2, 4], 3, 0.0),
        ([], 3, 0.0),
    ],
)
def test_reciprocal_rank_depends_on_position(found_ids, expected_id, expected_rr):
    assert reciprocal_rank(found_ids, expected_id) == expected_rr


def test_compute_metrics_averages_hits_and_ranks():
    metrics = compute_metrics(
        ranked_ids=[[1, 2, 3], [9, 8, 7], [5, 2, 1]],
        expected_ids=[1, 3, 2],
    )

    # Попадания в 2 запросах из 3; ранги 1 и 2 дают (1 + 0.5 + 0) / 3.
    assert metrics.hit_rate == pytest.approx(2 / 3)
    assert metrics.mrr == pytest.approx(0.5)


def test_compute_metrics_requires_matching_lengths():
    with pytest.raises(ValueError):
        compute_metrics(ranked_ids=[[1]], expected_ids=[1, 2])
