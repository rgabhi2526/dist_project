"""Unit tests for flsim.strategies.seafl: pure math and the strategy state machine."""

import math

import numpy as np
import pytest
import torch

from flsim.strategies.seafl import (
    SEAFLStrategy,
    aggregation_weights,
    cosine,
    importance_factor,
    mix,
    staleness_factor,
)


def test_gamma_endpoints() -> None:
    """γ(0) = α, γ(β) = α/2, γ with β = inf is α."""
    assert staleness_factor(0, 3.0, 10) == pytest.approx(3.0)
    assert staleness_factor(10, 3.0, 10) == pytest.approx(1.5)
    assert staleness_factor(7, 3.0, math.inf) == 3.0


def test_weights_sum_to_one_and_uniform() -> None:
    """p_k sums to 1; uniform=True gives equal weights."""
    p = aggregation_weights([10, 30, 60], [3.0, 2.0, 1.5], [0.5, 0.9, 0.1], uniform=False)
    assert p.sum() == pytest.approx(1.0)
    u = aggregation_weights([10, 30, 60], [3.0, 2.0, 1.5], [0.5, 0.9, 0.1], uniform=True)
    assert np.allclose(u, 1 / 3)


def test_importance_range() -> None:
    """s lies in [0, μ] for parallel, opposite and orthogonal vectors."""
    g = torch.tensor([1.0, 0.0])
    assert importance_factor(g, g, 2.0) == pytest.approx(2.0)
    assert importance_factor(-g, g, 2.0) == pytest.approx(0.0)
    assert importance_factor(torch.tensor([0.0, 1.0]), g, 2.0) == pytest.approx(1.0)
    assert cosine(torch.zeros(2), g) == 0.0


def test_mix_extremes() -> None:
    """ϑ = 1 returns w_new, ϑ = 0 returns w_old."""
    old, new = {"w": torch.zeros(3)}, {"w": torch.ones(3)}
    assert torch.equal(mix(old, new, 1.0)["w"], new["w"])
    assert torch.equal(mix(old, new, 0.0)["w"], old["w"])


def _make(K: int = 2, beta: float = 10, partial: bool = False) -> SEAFLStrategy:
    torch.manual_seed(0)
    return SEAFLStrategy(
        torch.nn.Linear(2, 2),
        {i: 10 for i in range(4)},
        K=K,
        stale_bound=beta,
        alpha=3.0,
        mu=1.0,
        theta=0.8,
        n_active=4,
        epochs=1,
        partial=partial,
    )


def _update(strat: SEAFLStrategy, cid: int, base: int) -> dict:
    """UPDATE whose weights are the base snapshot plus a small constant step."""
    w = {k: v + 0.1 for k, v in strat.snapshots[base].items()}
    return {
        "type": "UPDATE",
        "cid": cid,
        "base_version": base,
        "weights": w,
        "n_samples": 10,
        "loss": 0.0,
        "epochs_done": 1,
        "upload_idx": 0,
    }


def test_start_and_aggregation() -> None:
    """start sends 4 TRAIN; K=2 updates aggregate and re-train only the reporters."""
    s = _make()
    out = s.start(0.0)
    assert len(out) == 4 and all(m["type"] == "TRAIN" and m["version"] == 0 for _, m in out)
    assert s.on_update(_update(s, 0, 0), 1.0) == []
    out = s.on_update(_update(s, 1, 0), 2.0)
    assert s.version == 1
    assert sorted(c for c, _ in out) == [0, 1]
    assert all(m["version"] == 1 for _, m in out)
    assert s.events[-1]["kind"] == "agg" and s.events[-1]["cids"] == [0, 1]


def test_staleness_bound_waits() -> None:
    """With β=1, a client still on base 0 blocks the aggregation that would make it stale 2."""
    s = _make(beta=1)
    s.start(0.0)
    s.on_update(_update(s, 0, 0), 1.0)
    s.on_update(_update(s, 1, 0), 2.0)  # version 1; clients 2, 3 still at base 0
    s.on_update(_update(s, 0, 1), 3.0)
    assert s.on_update(_update(s, 1, 1), 4.0) == []  # blocked: 2 would hit staleness 2 > β
    assert s.version == 1
    s.on_update(_update(s, 2, 0), 5.0)
    assert s.version == 1  # client 3 still blocks
    s.on_update(_update(s, 3, 0), 6.0)
    assert s.version == 2 and len(s.events[-1]["cids"]) == 4


def test_partial_notifies_once() -> None:
    """SEAFL²: blocking clients get exactly one NOTIFY each."""
    s = _make(beta=1, partial=True)
    s.start(0.0)
    s.on_update(_update(s, 0, 0), 1.0)
    s.on_update(_update(s, 1, 0), 2.0)
    s.on_update(_update(s, 0, 1), 3.0)
    first = s.on_update(_update(s, 1, 1), 4.0)
    assert sorted(c for c, _ in first) == [2, 3]
    assert all(m == {"type": "NOTIFY"} for _, m in first)
    # Client 2 reports; client 3 still blocks but was already notified -> no second NOTIFY.
    assert s.on_update(_update(s, 2, 0), 5.0) == []
    assert [e["cid"] for e in s.events if e["kind"] == "notify"] == [2, 3]
