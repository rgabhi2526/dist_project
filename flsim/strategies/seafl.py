"""SEAFL server strategy (semi-asynchronous, staleness- and importance-weighted FL).

Sits in the server process of the flsim system: ``server.py`` owns the sockets and the single
main loop and feeds every received message to this class; the class is a pure state machine
(no I/O, no sleeping, no clock of its own) that returns the ``(cid, message)`` pairs to send.
FedBuff and FedAsync are not separate classes, they are flag settings of this one (see
:class:`SEAFLStrategy`).

Paper -> code map (SEAFL, Islam et al., IPDPS 2025, arXiv:2503.05755)
=====================================================================
Alg. 1 (SEAFL server)
    - sample N_active clients, broadcast w_0 ...... SEAFLStrategy.start()
    - buffer uploads, aggregate when |buffer| >= K . SEAFLStrategy.on_update()
    - wait for clients that would exceed S_k > beta  SEAFLStrategy._blocking_clients()
    - S_k = t - t_k (staleness) ................... SEAFLStrategy._aggregate()
    - Eq. gamma = alpha*beta/(S_k+beta) ........... staleness_factor()
    - Eq. s = mu*(cos(., w_g)+1)/2 ................ importance_factor(), cosine()
    - Eq. p_k = |D_k|/|D_buf| * (gamma+s), norm. .. aggregation_weights()
    - w_new = sum_k p_k w_k ....................... weighted_average()
    - w_{t+1} = (1-theta) w_t + theta w_new ....... mix()
    - send w_{t+1} only to the K reporters ........ SEAFLStrategy._aggregate()
Alg. 2 (SEAFL^2, partial training)
    - notify clients about to exceed beta ......... SEAFLStrategy.on_update() (NOTIFY)
      (the client-side "finish the epoch, upload now" lives in client.py)

Interpretation choices (details in decisions.md): D4 (FedBuff/FedAsync as flags),
D7 (cosine on the delta Delta_k = w_k - w_{t_k}; both cosines are logged), D13 (events).
"""

from __future__ import annotations

import math

import numpy as np
import torch

StateDict = dict[str, torch.Tensor]
Message = dict
Outbox = list[tuple[int, Message]]


# --------------------------------------------------------------------------------------------
# Pure functions (unit-tested without processes)
# --------------------------------------------------------------------------------------------
def staleness_factor(S: int, alpha: float, beta: float) -> float:
    """Staleness factor γ = α·β / (S + β)  (SEAFL staleness weighting).

    γ(0) = α and γ(β) = α/2, so older updates are smoothly down-weighted.

    Args:
        S: Staleness S_k = t - t_k of the update, in aggregations.
        alpha: α, scale of the staleness term.
        beta: β, staleness bound. ``math.inf`` means unbounded (FedBuff).

    Returns:
        γ as a float.
    """
    if math.isinf(beta):
        # lim_{β→∞} αβ/(S+β) = α: staleness is ignored when there is no bound.
        return float(alpha)
    return float(alpha * beta / (S + beta))


def flatten(state_dict: StateDict) -> torch.Tensor:
    """Concatenate all floating-point entries of a state dict into one 1-D vector.

    Key order is the dict's own (``nn.Module.state_dict`` order is fixed), so two vectors from
    the same architecture are aligned. Integer buffers (e.g. BatchNorm's counter) are skipped
    because they carry no direction for the cosine of the importance factor.

    Args:
        state_dict: Model weights.

    Returns:
        1-D float tensor.
    """
    parts = [v.detach().reshape(-1).float() for v in state_dict.values() if v.is_floating_point()]
    return torch.cat(parts)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two flat vectors, 0.0 if either has zero norm.

    Args:
        a: First vector (e.g. Δ_k).
        b: Second vector (e.g. the global model w_g).

    Returns:
        cos(a, b) in [-1, 1].
    """
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return 0.0  # a zero update has no direction; neutral value avoids NaN
    return float(torch.dot(a, b) / denom)


def importance_factor(delta: torch.Tensor, w_global: torch.Tensor, mu: float) -> float:
    """Importance factor s = μ·(cos(Δ_k, w_g) + 1) / 2  (delta version, decisions.md D7).

    Args:
        delta: Δ_k = w_k - w_{t_k}, flat update of client k.
        w_global: w_g, flat current global model.
        mu: μ, upper bound of the importance factor.

    Returns:
        s in [0, μ].
    """
    # (cos+1)/2 maps [-1, 1] -> [0, 1] so s is never negative.
    return float(mu * (cosine(delta, w_global) + 1.0) / 2.0)


def aggregation_weights(
    n_samples: list[int], gammas: list[float], importances: list[float], uniform: bool
) -> np.ndarray:
    """Aggregation weights p_k = (|D_k| / Σ|D|)·(γ_k + s_k), normalised to sum 1.

    Args:
        n_samples: |D_k| of every buffered client.
        gammas: γ_k, staleness factors.
        importances: s_k, importance factors.
        uniform: If True return 1/len (FedBuff, decisions.md D4).

    Returns:
        Array of p_k summing to 1.
    """
    n = len(n_samples)
    if uniform:
        return np.full(n, 1.0 / n)
    sizes = np.asarray(n_samples, dtype=np.float64)
    raw = (sizes / sizes.sum()) * (np.asarray(gammas) + np.asarray(importances))
    total = raw.sum()
    if total <= 0.0:
        return np.full(n, 1.0 / n)  # degenerate (all γ+s = 0): fall back to uniform
    return raw / total


def weighted_average(state_dicts: list[StateDict], weights: np.ndarray) -> StateDict:
    """w_new = Σ_k p_k · w_k over floating-point tensors.

    Non-float entries (integer counters such as BatchNorm ``num_batches_tracked``) cannot be
    meaningfully averaged, so they are copied from the first state dict.

    Args:
        state_dicts: Client weights w_k.
        weights: p_k, same length, summing to 1.

    Returns:
        New state dict w_new.
    """
    out: StateDict = {}
    for key, first in state_dicts[0].items():
        if first.is_floating_point():
            out[key] = sum(float(p) * sd[key] for p, sd in zip(weights, state_dicts))
        else:
            out[key] = first.clone()  # ponytail: copy not average, ceiling: BN counters only
    return out


def mix(w_old: StateDict, w_new: StateDict, theta: float) -> StateDict:
    """Server mixing w_{t+1} = (1 - ϑ)·w_t + ϑ·w_new.

    Args:
        w_old: w_t, current global weights.
        w_new: Weighted average of the buffer.
        theta: ϑ in [0, 1]; ϑ = 1 returns w_new (FedBuff).

    Returns:
        w_{t+1}. Non-float buffers are taken from ``w_new`` (the freshest value).
    """
    out: StateDict = {}
    for key, new in w_new.items():
        if new.is_floating_point():
            out[key] = (1.0 - theta) * w_old[key] + theta * new
        else:
            out[key] = new.clone()
    return out


def _cpu_snapshot(model: torch.nn.Module) -> StateDict:
    """Detached CPU copy of a model's state dict (weights are CPU tensors on the wire)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------------------------
class SEAFLStrategy:
    """SEAFL / SEAFL² server state machine (Alg. 1 and Alg. 2).

    Flag recipes (decisions.md D4), no extra code:
        * FedBuff:  ``uniform_weights=True, stale_bound=math.inf, theta=1.0``.
        * FedAsync: FedBuff with ``K=1``.
        * SEAFL²:   ``partial=True``.

    Attributes:
        global_model: The nn.Module w_t the server evaluates.
        version: t, number of aggregations done so far.
        events: Event dicts (``agg`` / ``notify``) drained by the server into CSV.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: dict[int, int],
        K: int,
        stale_bound: float,
        alpha: float,
        mu: float,
        theta: float,
        n_active: int,
        epochs: int,
        uniform_weights: bool = False,
        partial: bool = False,
        seed: int = 0,
    ) -> None:
        """Create the strategy.

        Args:
            model: Initial global model w_0 (kept as ``global_model``).
            n_samples: |D_k| per client id; its keys define the client population N.
            K: Buffer size that triggers aggregation.
            stale_bound: β, max tolerated staleness (``math.inf`` disables the bound).
            alpha: α, staleness-factor scale.
            mu: μ, importance-factor scale.
            theta: ϑ, server mixing rate.
            n_active: N_active, clients sampled at start (all if equal to N).
            epochs: E, local epochs requested in every TRAIN.
            uniform_weights: p_k = 1/len (FedBuff).
            partial: SEAFL² - NOTIFY clients that block aggregation.
            seed: Seed of the client-sampling RNG.

        Raises:
            ValueError: If K > n_active (the buffer could never fill).
        """
        if K > n_active:
            raise ValueError(f"K={K} > n_active={n_active}: buffer can never fill")
        self.global_model = model
        self.version = 0  # t
        self.events: list[dict] = []
        self.n_samples = n_samples  # |D_k|
        self.K = K  # buffer size K
        self.beta = stale_bound  # β
        self.alpha = alpha  # α
        self.mu = mu  # μ
        self.theta = theta  # ϑ
        self.n_active = n_active  # N_active
        self.epochs = epochs
        self.uniform = uniform_weights
        self.partial = partial
        self.rng = np.random.default_rng(seed)
        # cid -> base version t_k the client is currently training from.
        self.in_flight: dict[int, int] = {}
        self.buffer: list[Message] = []
        self.notified: set[int] = set()  # clients already sent NOTIFY (once each)
        # version -> CPU state dict of w_t, needed for Δ_k = w_k - w_{t_k}.
        # Memory ceiling: pruned to versions still referenced by in-flight or buffered
        # updates, so at most ~(beta + 1) snapshots of a ~400 KB model when beta is finite.
        self.snapshots: dict[int, StateDict] = {}

    def start(self, now: float) -> Outbox:
        """Alg. 1 init: sample N_active clients and broadcast w_0.

        Args:
            now: Server clock (unused; kept for the strategy interface).

        Returns:
            One TRAIN (version 0) per sampled client.
        """
        ids = sorted(self.n_samples)
        chosen = (
            ids
            if self.n_active >= len(ids)
            else sorted(int(c) for c in self.rng.choice(ids, size=self.n_active, replace=False))
        )
        self.snapshots[0] = _cpu_snapshot(self.global_model)
        return [(cid, self._train_msg(cid)) for cid in chosen]

    def on_update(self, msg: Message, now: float) -> Outbox:
        """Handle an UPDATE: buffer it and aggregate if allowed.

        Args:
            msg: UPDATE dict (``cid, base_version, weights, n_samples, ...``).
            now: Server clock, stored in events.

        Returns:
            TRAIN messages to the reporters after an aggregation, NOTIFY messages (SEAFL²)
            while blocked, else an empty list.

        Raises:
            KeyError: If the client is not in flight (trust boundary is the transport).
        """
        cid = msg["cid"]
        if cid not in self.in_flight:
            raise KeyError(f"UPDATE from client {cid} that is not in flight")
        del self.in_flight[cid]
        self.notified.discard(cid)
        self.buffer.append(msg)
        if len(self.buffer) < self.K:
            return []
        blockers = self._blocking_clients()
        if not blockers:
            return self._aggregate(now)
        out: Outbox = []
        if self.partial:
            for b in blockers:
                if b not in self.notified:
                    self.notified.add(b)
                    out.append((b, {"type": "NOTIFY"}))
                    self.events.append({"kind": "notify", "t": now, "cid": b})
        return out

    def on_tick(self, now: float) -> Outbox:
        """No timers: SEAFL is purely update-driven.

        The β wait ends when the blocking client's UPDATE arrives (a message), so no
        deadline is needed; the server's eval timer is handled outside the strategy.

        Args:
            now: Server clock (unused).

        Returns:
            Always an empty list.
        """
        return []

    def _blocking_clients(self) -> list[int]:
        """Clients that would exceed β if we aggregated now (Alg. 1 wait rule).

        After aggregating the version becomes t+1, so an in-flight base t_k has staleness
        (t+1) - t_k at its next report; it blocks if that is > β.

        Returns:
            Client ids sorted ascending.
        """
        return sorted(c for c, b in self.in_flight.items() if (self.version + 1 - b) > self.beta)

    def _train_msg(self, cid: int) -> Message:
        """Build a TRAIN for the current global model and mark ``cid`` in flight."""
        self.in_flight[cid] = self.version
        return {
            "type": "TRAIN",
            "version": self.version,
            "weights": self.snapshots[self.version],  # shared read-only snapshot
            "epochs": self.epochs,
            "policy": {},
        }

    def _aggregate(self, now: float) -> Outbox:
        """Alg. 1 aggregation of the whole buffer; returns TRAIN for the reporters."""
        buf, self.buffer = self.buffer, []
        w_g = flatten(self.snapshots[self.version])  # w_g, current global model
        staleness, gammas, importances, cos_d, cos_w, states = [], [], [], [], [], []
        for m in buf:
            S = self.version - m["base_version"]  # S_k = t - t_k
            staleness.append(S)
            gammas.append(staleness_factor(S, self.alpha, self.beta))  # γ_k
            w_k = flatten(m["weights"])
            delta = w_k - flatten(self.snapshots[m["base_version"]])  # Δ_k = w_k - w_{t_k}
            importances.append(importance_factor(delta, w_g, self.mu))  # s_k (D7: delta)
            cos_d.append(cosine(delta, w_g))  # logged: cos(Δ_k, w_g), the one s is built on
            cos_w.append(cosine(w_k, w_g))  # logged only, to justify D7
            states.append(m["weights"])
        sizes = [self.n_samples[m["cid"]] for m in buf]  # |D_k|
        p = aggregation_weights(sizes, gammas, importances, self.uniform)  # p_k
        w_new = weighted_average(states, p)
        w_next = mix(self.snapshots[self.version], w_new, self.theta)  # w_{t+1}
        self.global_model.load_state_dict(w_next)
        self.version += 1
        self.snapshots[self.version] = _cpu_snapshot(self.global_model)
        self.events.append(
            {
                "kind": "agg",
                "t": now,
                "version": self.version,
                "cids": [m["cid"] for m in buf],
                "staleness": staleness,
                "weights": [float(x) for x in p],
                "cos_delta": cos_d,
                "cos_weights": cos_w,
            }
        )
        out = [(m["cid"], self._train_msg(m["cid"])) for m in buf]
        self._prune()
        return out

    def _prune(self) -> None:
        """Drop snapshots older than the oldest base still needed (bounds memory)."""
        keep = self.version
        needed = list(self.in_flight.values()) + [m["base_version"] for m in self.buffer]
        oldest = min(needed + [keep])
        self.snapshots = {v: w for v, w in self.snapshots.items() if v >= oldest}
