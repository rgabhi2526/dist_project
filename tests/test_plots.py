"""Tests for the pure helpers of ``plots.py`` on tiny synthetic arrays."""

from __future__ import annotations

import numpy as np

from plots import colour_map, fluctuation, interp_to_grid, mean_std, time_to_target


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
