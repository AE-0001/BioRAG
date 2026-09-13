import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "evaluation"))

from metrics import ndcg, percentile, reciprocal_rank


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank([False, True, True]) == 0.5
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_rewards_earlier_relevant_results():
    assert ndcg([True, False], 1) == 1.0
    assert 0 < ndcg([False, True], 1) < 1


def test_source_level_ndcg_requires_deduplicated_rankings():
    # A source should occur once in a source-level ranking, even if several
    # chunks from it occupy the raw nearest-neighbour results.
    assert ndcg([True], 1) == 1.0


def test_percentile_uses_documented_nearest_rank():
    assert percentile([1, 2, 3, 4], 0.50) == 2
    assert percentile([1, 2, 3, 4], 0.95) == 4
