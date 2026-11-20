"""Tests for the pure helpers of ``plots.py`` on tiny synthetic arrays."""

from __future__ import annotations

import numpy as np

from plots import (
    colour_map,
    config_differences,
    fairness_stats,
    fluctuation,
    fmt_reasons,
    inclusion_counts,
    interp_to_grid,
    mean_std,
    reason_counts,
    target_row,
    time_to_target,
)


def test_time_to_target_exact() -> None:
    """The first eval at or above the target is returned, including equality."""
    t = np.array([0.0, 5.0, 10.0, 15.0])
    acc = np.array([0.1, 0.5, 0.8, 0.9])
    assert time_to_target(t, acc, 0.8) == 10.0
    assert time_to_target(t, acc, 0.05) == 0.0


def test_time_to_target_never_reached() -> None:
    """None signals a run that never hits the target."""
    assert time_to_target(np.array([0.0, 5.0]), np.array([0.1, 0.2]), 0.5) is None


def test_interp_to_grid_and_mean_std() -> None:
    """Interpolation is linear, NaN beyond the curve, and seeds average correctly."""
    a = interp_to_grid(np.array([0.0, 10.0]), np.array([0.0, 1.0]), np.array([0.0, 5.0, 20.0]))
    b = interp_to_grid(np.array([0.0, 20.0]), np.array([1.0, 1.0]), np.array([0.0, 5.0, 20.0]))
    assert a[1] == 0.5 and np.isnan(a[2])
    mean, std = mean_std([a, b])
    assert np.allclose(mean, [0.5, 0.75, 1.0])  # NaN ignored at the last point
    assert np.allclose(std, [0.5, 0.25, 0.0])
    _, single = mean_std([b])
    assert np.all(single == 0)


def test_fluctuation_uses_last_five() -> None:
    """Only the last five evals count; constant tail means zero fluctuation."""
    acc = np.array([0.0, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert fluctuation(acc) == 0.0
    assert np.isclose(fluctuation(np.array([0.0, 1.0])), 0.5)


def test_colour_map_families_are_distinct() -> None:
    """Different families never share a colour; same name always maps the same way."""
    c = colour_map(["fedavg", "seafl", "seafl_scaled", "libra"])
    assert len({c["fedavg"], c["seafl"], c["libra"]}) == 3
    assert c["seafl"] != c["seafl_scaled"]


def test_inclusion_counts_repeated_cids_count_once() -> None:
    """A client listed twice in one agg event counts once; non-agg events are ignored."""
    events = [
        {"kind": "agg", "cids": [0, 0, 1]},
        {"kind": "agg", "cids": [0]},
        {"kind": "reject", "cid": 2, "reason": "theta"},
    ]
    assert inclusion_counts(events, 3) == [2, 1, 0]


def test_fairness_stats() -> None:
    """Variance, Jain and % never on a known vector; all-zero follows the Jain convention."""
    var, jain, never = fairness_stats([0, 0, 1, 1])
    assert (var, jain, never) == (0.25, 0.5, 50.0)
    assert fairness_stats([0, 0])[1] == 1.0


def test_reason_counts_and_format() -> None:
    """Reasons are counted per kind and rendered most frequent first."""
    ev = [{"kind": "halt", "reason": "a"}] * 2 + [{"kind": "halt", "reason": "b"}] * 3
    ev.append({"kind": "reject", "reason": "a"})
    assert reason_counts(ev, "halt") == {"a": 2, "b": 3}
    assert fmt_reasons(reason_counts(ev, "halt")) == "b:3 a:2"
    assert fmt_reasons({}) == ""


def test_target_row_and_config_differences() -> None:
    """First row reaching the target; seed/out are ignored when comparing configs."""
    assert target_row(np.array([0.0, 5.0]), np.array([0.1, 0.6]), 0.5) == 1
    assert target_row(np.array([0.0]), np.array([0.1]), 0.5) is None
    cfgs = [{"seed": 0, "out": "a", "lr": 0.1, "K": 2}, {"seed": 1, "out": "b", "lr": 0.2, "K": 2}]
    assert config_differences(cfgs) == ["lr"]
