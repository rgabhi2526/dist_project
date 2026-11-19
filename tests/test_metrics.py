"""Unit tests for flsim.metrics: Jain's index, participation counts, eval CSV."""

import pytest

from flsim.metrics import EvalLogger, Traffic, jain_index, participation_counts


def test_jain_equal_is_one_and_one_hot_is_inverse_n() -> None:
    """Equal counts give 1, a single participant gives 1/n, all-zero is defined as 1."""
    assert jain_index([3, 3, 3, 3]) == pytest.approx(1.0)
    assert jain_index([5, 0, 0, 0, 0]) == pytest.approx(1 / 5)
    assert jain_index([0, 0, 0]) == 1.0


def test_participation_counts_from_agg_events() -> None:
    """Only agg events count, a repeated cid (Libra multi-upload) counts twice."""
    events = [
        {"kind": "select", "cids": [0, 1, 2]},
        {"kind": "agg", "cids": [0, 1, 1]},
        {"kind": "reject", "cids": [2]},
        {"kind": "agg", "cids": [3]},
    ]
    assert participation_counts(events, 4) == [1, 2, 0, 1]


def test_eval_logger_flushes_each_row(tmp_path) -> None:
    """A row is readable from disk before the logger is closed (crash safety, D3)."""
    path = tmp_path / "evals.csv"
    log = EvalLogger(path)
    log.log(1.5, 2, 0.8, 0.5, Traffic(n_up=3, bytes_up=10))
    lines = path.read_text().splitlines()
    assert lines[0].startswith("t,version,acc,loss") and len(lines) == 2
    log.close()
