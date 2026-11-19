"""Unit tests for flsim.strategies.fedavg: the synchronous barrier and sample-weighted average."""

import pytest
import torch

from flsim.strategies.fedavg import FedAvgStrategy


def _update(cid: int, base_version: int, value: float, n: int) -> dict:
    """UPDATE whose Linear(1, 1) weight and bias are filled with ``value``."""
    w = {"weight": torch.full((1, 1), value), "bias": torch.full((1,), value)}
    return {"type": "UPDATE", "cid": cid, "base_version": base_version, "weights": w,
            "n_samples": n, "loss": 0.1, "epochs_done": 1, "upload_idx": 1}  # fmt: skip


def test_two_rounds_and_sample_weighting() -> None:
    """4 clients, 2 per round: barrier waits for both, then version bumps with a new TRAIN."""
    s = FedAvgStrategy(torch.nn.Linear(1, 1), {0: 10, 1: 10, 2: 10, 3: 10}, 2, epochs=1, seed=0)
    out = s.start(0.0)
    assert len(out) == 2 and all(m["type"] == "TRAIN" and m["version"] == 0 for _, m in out)
    a, b = (cid for cid, _ in out)

    assert s.on_update(_update(a, 0, 1.0, 10), 1.0) == []  # barrier: still waiting for b
    out = s.on_update(_update(b, 0, 4.0, 30), 2.0)
    assert s.version == 1
    assert len(out) == 2 and all(m["version"] == 1 for _, m in out)
    # (10*1 + 30*4) / 40 = 3.25: weighted by n_k, not a plain mean (2.5).
    assert s.global_model.weight.item() == pytest.approx(3.25)
    assert out[0][1]["weights"]["weight"].item() == pytest.approx(3.25)

    c, d = (cid for cid, _ in out)
    s.on_update(_update(c, 1, 2.0, 10), 3.0)
    s.on_update(_update(d, 1, 2.0, 10), 4.0)
    assert s.version == 2
    assert [e["kind"] for e in s.events] == ["select", "agg", "select", "agg", "select"]
    assert s.events[1]["n_updates"] == 2 and s.events[1]["cids"] == sorted([a, b])


def test_rejects_unexpected_update_and_bad_m() -> None:
    """An update from outside S_t is a bug; m must lie in [1, N]."""
    s = FedAvgStrategy(torch.nn.Linear(1, 1), {0: 5, 1: 5, 2: 5}, 1, epochs=1)
    ((cid, _),) = s.start(0.0)
    with pytest.raises(KeyError):
        s.on_update(_update((cid + 1) % 3, 0, 1.0, 5), 1.0)
    with pytest.raises(ValueError):
        FedAvgStrategy(torch.nn.Linear(1, 1), {0: 5}, 2, epochs=1)
