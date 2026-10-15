"""Contract tests for OpenAI-compatible embedding providers."""

import pytest

from src.embeddings import _parse_embedding_response


def test_parse_embedding_response_orders_by_index_and_preserves_values():
    payload = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]
    }

    assert _parse_embedding_response(payload, 2) == [[1.0, 0.0], [0.0, 1.0]]


def test_parse_embedding_response_rejects_missing_items():
    with pytest.raises(ValueError, match="count mismatch"):
        _parse_embedding_response({"data": []}, 1)


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"data": [{"index": 0, "embedding": [1.0, float("nan")]}]}, "non-finite"),
        ({"data": [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 2.0]}]}, "dimensions"),
        ({"data": [{"index": 0, "embedding": []}]}, "non-empty"),
        ({"data": [{"index": 0, "embedding": [True]}]}, "non-finite"),
    ],
)
def test_parse_embedding_response_rejects_invalid_vectors(payload, message):
    with pytest.raises(ValueError, match=message):
        _parse_embedding_response(payload, 2 if len(payload["data"]) == 2 else 1)


def test_parse_embedding_response_rejects_duplicate_or_out_of_range_indexes():
    with pytest.raises(ValueError, match="duplicate"):
        _parse_embedding_response(
            {"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [1.0]}]},
            2,
        )
    with pytest.raises(ValueError, match="out of range"):
        _parse_embedding_response({"data": [{"index": 2, "embedding": [1.0]}]}, 1)
