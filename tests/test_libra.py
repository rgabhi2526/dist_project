import math

import torch
from torch import nn

from flsim.strategies.libra import (
    LibraStrategy,
    assign_admission,
    biased_selection,
    early_drop,
    good_resource_set,
    output_distribution,
    participation_threshold,
    sinkhorn_distance,
    zero_one_cost,
)

COST = zero_one_cost(10)


def one_hot(i: int) -> torch.Tensor:
    v = torch.zeros(10, dtype=torch.float64)
    v[i] = 1.0
    return v


def test_sinkhorn_identical_is_zero():
    p = torch.softmax(torch.randn(10, dtype=torch.float64), dim=0)
    assert sinkhorn_distance(p, p, COST, 0.1, 100) < 1e-2


def test_sinkhorn_different_one_hots_is_about_one():
    d = sinkhorn_distance(one_hot(0), one_hot(1), COST, 0.1, 100)
    assert 0.95 < d < 1.05


def test_assign_admission_splits_half_half():
    part = {cid: cid for cid in range(10)}  # client 9 participates most
    theta = assign_admission(part, 0.5)
    assert sum(theta.values()) == 5
    assert all(theta[c] == 0 for c in (5, 6, 7, 8, 9))
    assert all(theta[c] == 1 for c in (0, 1, 2, 3, 4))


def test_assign_admission_ties_break_by_cid():
    theta = assign_admission({c: 0 for c in range(4)}, 0.5)
    assert theta == {0: 0, 1: 0, 2: 1, 3: 1}


def test_participation_threshold_boundary():
    part = {cid: cid for cid in range(10)}
    assert participation_threshold(part, 0.5) == 4  # sorted desc: 9..5 rejected, 4 is boundary


def test_early_drop_truth_table():
    assert not early_drop(5, 1, 2, 2)  # 4 < 0 false
    assert early_drop(5, 1, 6, 1)  # 4 < 5
    assert not early_drop(5, 1, 5, 1)  # 4 < 4 false
    assert early_drop(5, 5, 0, 0)  # v_i == tau_max always drops
    assert early_drop(5, 7, 0, 3)  # beyond tau_max
    assert not early_drop(5, 4, 0, 3)  # 1 < -3 false


def test_biased_selection_respects_good_set_and_loss_order():
    losses = {0: 0.5, 1: 2.0, 2: 1.0, 3: 9.0}  # 3 is high-loss but not good
    assert biased_selection(losses, {0, 1, 2}, 2) == [1, 2]
    assert biased_selection(losses, {0, 1, 2}, 10) == [1, 2, 0]


def test_biased_selection_unknown_loss_first():
    assert biased_selection({0: 5.0, 1: 3.0}, {0, 1, 2}, 2) == [2, 0]


def test_good_resource_set_picks_fastest():
    delays = {0: 5.0, 1: 1.0, 2: 3.0, 3: 2.0}
    assert good_resource_set(delays, 0.5) == {1, 3}


# ---- strategy -------------------------------------------------------------------------
def make(**kw) -> LibraStrategy:
    torch.manual_seed(0)
    args = dict(
        model=nn.Sequential(nn.Linear(4, 10)),
        n_samples={c: 100 for c in range(5)},
        delays={c: 1.0 + c for c in range(5)},
        x_val=torch.randn(8, 4),
        epochs=1,
        C=0.6,
        tau_max=5,
        sigma_max=5,
        M=0.5,
        good_frac=0.8,
        deadline_s=10.0,
        sinkhorn_L=None,
        sinkhorn_eps=0.1,
        sinkhorn_iters=100,
        select_every=10,
    )
    args.update(kw)
    return LibraStrategy(**args)


def update(s: LibraStrategy, cid: int, idx: int = 1, base: int | None = None, shift: float = 0.0):
    weights = {k: v.clone() + shift for k, v in s.global_model.state_dict().items()}
    return {
        "type": "UPDATE",
        "cid": cid,
        "base_version": s.version if base is None else base,
        "weights": weights,
        "n_samples": 100,
        "loss": 1.0,
        "epochs_done": 1,
        "upload_idx": idx,
    }


def test_start_selects_round_nc_clients():
    s = make()
    out = s.start(0.0)
    assert len(out) == round(5 * 0.6) == 3
    assert all(m["type"] == "TRAIN" and m["version"] == 0 for _, m in out)
    assert out[0][1]["policy"]["multi_upload"] is True
    assert s.events[0]["kind"] == "select"


def test_fresh_uploads_accepted_and_tick_aggregates():
    s = make()
    chosen = [cid for cid, _ in s.start(0.0)]
    for cid in chosen:
        assert s.on_update(update(s, cid, shift=1.0), 1.0) == []
    assert all(s.participation[c] == 1 for c in chosen)
    before = {k: v.clone() for k, v in s.global_model.state_dict().items()}
    assert s.on_tick(5.0) == []  # before deadline: nothing happens
    assert s.version == 0
    out = s.on_tick(10.0)
    assert s.version == 1
    assert {cid for cid, m in out if m["type"] == "TRAIN"} <= set(range(5))
    for k, v in s.global_model.state_dict().items():
        assert torch.allclose(v, before[k] + 1.0)  # every upload shifted by +1
    agg = [e for e in s.events if e["kind"] == "agg"][-1]
    assert agg["n_updates"] == 3 and agg["n_stale"] == 0


def test_stale_update_from_high_p_client_rejected():
    s = make()
    chosen = [cid for cid, _ in s.start(0.0)]
    s.on_tick(10.0)  # nobody uploaded: empty round, everyone becomes stale
    assert s.version == 1
    cid = chosen[0]
    s.theta[cid] = 0
    out = s.on_update(update(s, cid, base=0), 11.0)
    rejects = [e for e in s.events if e["kind"] == "reject"]
    assert len(rejects) == 1 and rejects[0]["reason"] == "theta"
    assert s.participation[cid] == 0
    assert out[0][1]["type"] == "HALT"


def test_stale_update_admitted_when_theta_one():
    s = make()
    chosen = [cid for cid, _ in s.start(0.0)]
    s.on_tick(10.0)
    cid = chosen[0]
    s.theta[cid] = 1
    s.on_update(update(s, cid, base=0), 11.0)
    assert s.participation[cid] == 1
    assert not [e for e in s.events if e["kind"] == "reject"]


def test_sigma_max_halts():
    s = make(sigma_max=3)
    cid = s.start(0.0)[0][0]
    out = s.on_update(update(s, cid, idx=3), 1.0)
    assert out == [(cid, {"type": "HALT"})]
    assert s.events[-1]["reason"] == "sigma_max"


def test_sinkhorn_threshold_halts_and_logs_distance():
    s = make(sinkhorn_L=0.0)
    cid = s.start(0.0)[0][0]
    assert s.on_update(update(s, cid, idx=2), 1.0) == []  # uploads 1-2 are free
    out = s.on_update(update(s, cid, idx=3, shift=0.5), 2.0)
    assert out == [(cid, {"type": "HALT"})]
    sk = [e for e in s.events if e["kind"] == "sinkhorn"]
    assert len(sk) == 1 and math.isfinite(sk[0]["dist"])


def test_no_limit_never_halts():
    s = make(no_limit=True, sinkhorn_L=0.0)
    cid = s.start(0.0)[0][0]
    assert s.on_update(update(s, cid, idx=5), 1.0) == []
    assert not [e for e in s.events if e["kind"] in ("sinkhorn", "halt")]


def test_output_distribution_is_probability_vector():
    p = output_distribution(nn.Sequential(nn.Linear(4, 10)), torch.randn(8, 4))
    assert p.shape == (10,) and abs(float(p.sum()) - 1.0) < 1e-5
