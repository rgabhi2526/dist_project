"""Libra: semi-asynchronous FL that gates *who* participates (Wang et al., ICDCS 2024).

Sits in the server process. The server main loop (the only thread touching strategy state)
feeds ``UPDATE`` messages to :meth:`LibraStrategy.on_update` and calls
:meth:`LibraStrategy.on_tick` on a timer; the strategy returns ``(cid, message)`` pairs to send
and never does I/O itself (CLAUDE.md "Contracts", decisions.md D12). All Libra math lives in
module-level pure functions so pytest can check it without processes.

Paper -> code map (Libra, Wang et al., ICDCS 2024):
    Round structure (broadcast, collect until T, aggregate) -> LibraStrategy.start / on_tick
    (a) Limit fast devices (uploads 1-2 free, Sinkhorn test for 3..sigma_max, hard stop at
        sigma_max)                                          -> sinkhorn_distance(),
                                                               output_distribution(),
                                                               zero_one_cost(),
                                                               LibraStrategy._on_fresh_update()
    (b) Participation-based stale-model admission:
        theta (admit flag) from sort by P, top N(1-M) get 0 -> assign_admission()
        P^M, the quantile participation count               -> participation_threshold()
        rho (early drop), (tau_max - v_i) < (P_i - P^M)     -> early_drop()
        stale-upload gate (theta, rho, tau_max)             -> LibraStrategy._on_stale_update()
    (c) Biased client selection (good-resource set A, sort by loss, take min(N*C, |A|))
                                                            -> good_resource_set(),
                                                               biased_selection(),
                                                               LibraStrategy._select()
    Sample-weighted aggregation (FedAvg-style, requirements 3.4) -> _aggregate()
    Ablations: no_limit = Libra-NoL, random_select = Libra-RS, staleness_select = Libra-SS.

Quantities the paper does not give are ours (decisions.md D6 and D13) and are flagged inline.
"""

from __future__ import annotations

import copy
import math
import random

import torch
from torch import Tensor, nn

Message = dict
Outbox = list[tuple[int, dict]]

# Per-client lifecycle (see LibraStrategy docstring).
IDLE = "idle"  # not training anything
FRESH = "fresh"  # training the current round's global model
STALE = "stale"  # training an older round's model, nothing uploaded for it yet


# --------------------------------------------------------------------------------------
# (a) Limit fast devices: Sinkhorn distance between output distributions
# --------------------------------------------------------------------------------------
def zero_one_cost(n_classes: int) -> Tensor:
    """Build the ground cost between class labels, ``1 - I``.

    Args:
        n_classes: Number of output classes (10 for FMNIST / CIFAR-10).

    Returns:
        An ``(n_classes, n_classes)`` float64 matrix, 0 on the diagonal and 1 elsewhere.
    """
    # Not given by the paper -> decisions.md D6: 0/1 label cost (ponytail: approximates total
    # variation; a semantic class-distance cost only if L cannot be tuned, upgrade = cost matrix
    # from class-embedding distances).
    return 1.0 - torch.eye(n_classes, dtype=torch.float64)


def sinkhorn_distance(p: Tensor, q: Tensor, cost: Tensor, eps: float, iters: int) -> float:
    """Entropic optimal-transport cost between two output distributions (Libra mechanism (a)).

    Runs Sinkhorn scaling in the log domain, so it stays stable for small eps (0.1) and for
    distributions with exact zeros.

    Args:
        p: Probability vector, e.g. the round's global-model output distribution.
        q: Probability vector, e.g. the uploaded local-model output distribution.
        cost: Ground cost matrix ``C[i, j]`` (see :func:`zero_one_cost`).
        eps: Entropic regularisation strength (decisions.md D6: 0.1).
        iters: Number of Sinkhorn iterations (decisions.md D6: 100).

    Returns:
        The transport cost ``<plan, cost>`` as a float; ~0 when ``p == q`` and ~1 for two
        distinct one-hots under the 0/1 cost.
    """
    tiny = 1e-12  # log(0) guard; a clamped one-hot still behaves like a one-hot
    p = p.detach().cpu().double().clamp_min(tiny)
    q = q.detach().cpu().double().clamp_min(tiny)
    cost = cost.detach().cpu().double()
    p, q = p / p.sum(), q / q.sum()
    log_k = -cost / eps  # log of the Gibbs kernel K = exp(-C / eps)
    log_u = torch.zeros_like(p)
    log_v = torch.zeros_like(q)
    for _ in range(iters):
        log_u = p.log() - torch.logsumexp(log_k + log_v[None, :], dim=1)
        log_v = q.log() - torch.logsumexp(log_k + log_u[:, None], dim=0)
    plan = torch.exp(log_u[:, None] + log_k + log_v[None, :])  # entropic transport plan
    return float((plan * cost).sum())


@torch.no_grad()
def output_distribution(model: nn.Module, x_val: Tensor) -> Tensor:
    """Mean softmax output of a model over the validation set D_val.

    Args:
        model: Any classifier returning logits.
        x_val: Inputs of D_val, shape ``(n, ...)``.

    Returns:
        A probability vector of length ``n_classes``. Collapsing D_val to one vector means
        Sinkhorn solves one 10x10 problem per upload instead of one per sample (decisions.md D6,
        our choice; the paper does not specify the input).
    """
    was_training = model.training
    model.eval()  # BN/dropout off so the distance reflects weights only
    device = next(model.parameters()).device
    probs = torch.softmax(model(x_val.to(device)), dim=1).mean(dim=0)
    model.train(was_training)
    return probs


# --------------------------------------------------------------------------------------
# (b) Participation-based stale-model admission
# --------------------------------------------------------------------------------------
def _sorted_by_participation(participation: dict[int, int]) -> list[int]:
    """Client ids sorted by P descending; ties broken by smaller cid first (deterministic)."""
    return sorted(participation, key=lambda cid: (-participation[cid], cid))


def assign_admission(participation: dict[int, int], M: float) -> dict[int, int]:
    """Assign the stale-admission flag ϑ_i from participation counts P_i.

    Sorts clients by P descending; the top ``N*(1-M)`` (heaviest participants) get ϑ=0, so their
    stale uploads are rejected; the bottom ``N*M`` get ϑ=1. Ties in P are broken by smaller cid
    first (our choice: the paper is silent, and determinism keeps runs reproducible; at round 0
    all P are 0 so the lowest cids start with ϑ=0).

    Args:
        participation: ``cid -> P_i`` (number of accepted contributions).
        M: Fraction of clients whose stale models are admitted (paper: 0.5).

    Returns:
        ``cid -> ϑ_i`` with ϑ_i in ``{0, 1}``.
    """
    order = _sorted_by_participation(participation)
    n_reject = round(len(order) * (1.0 - M))  # N(1-M) clients with ϑ = 0
    return {cid: (0 if rank < n_reject else 1) for rank, cid in enumerate(order)}


def participation_threshold(participation: dict[int, int], M: float) -> int:
    """Participation count P^M at the M-quantile boundary of the sorted list.

    Args:
        participation: ``cid -> P_i``.
        M: Admission fraction (paper: 0.5).

    Returns:
        P^M: the count of the first client (in P-descending order) that gets ϑ=1. The paper does
        not define P^M precisely; this boundary reading is decisions.md D13.
    """
    if not participation:
        return 0
    order = _sorted_by_participation(participation)
    boundary = min(round(len(order) * (1.0 - M)), len(order) - 1)
    return participation[order[boundary]]


def early_drop(tau_max: int, v_i: int, P_i: int, P_M: int) -> bool:
    """Early-drop flag ρ_i for a stale in-flight client.

    ρ=1 when the staleness budget left (τ_max - v_i) is smaller than the client's participation
    excess over the boundary (P_i - P^M): a heavy participant gets less slack. Also drops when
    v_i has reached τ_max, since such an update would be rejected anyway.

    Args:
        tau_max: Maximum tolerated staleness τ_max (paper: 5).
        v_i: Staleness of client i in rounds (current version - base version).
        P_i: Participation count of client i.
        P_M: Boundary participation count P^M.

    Returns:
        True if ρ_i = 1 (drop the client's training early).
    """
    return v_i >= tau_max or (tau_max - v_i) < (P_i - P_M)


# --------------------------------------------------------------------------------------
# (c) Biased client selection
# --------------------------------------------------------------------------------------
def good_resource_set(delays: dict[int, float], good_frac: float) -> set[int]:
    """Good-resource client set A: the fastest ``good_frac`` fraction.

    Args:
        delays: ``cid -> injected delay`` (the seeded draws from ``config.client_delays``).
        good_frac: Fraction of clients kept (decisions.md D13: 0.7).

    Returns:
        Set of cids in A. Substitution: the paper picks A from ContextFL's resource signal;
        we have no device-context model, so delay rank is the stand-in (decisions.md D13).
    """
    n_keep = max(1, round(len(delays) * good_frac))
    ranked = sorted(delays, key=lambda cid: (delays[cid], cid))  # fastest first, tie by cid
    return set(ranked[:n_keep])


def biased_selection(losses: dict[int, float], good: set[int], n_select: int) -> list[int]:
    """Biased client selection: highest-loss clients inside the good-resource set A.

    Args:
        losses: Last reported local loss per cid; a cid missing from the dict is unseen.
        good: The good-resource set A.
        n_select: Target count ``N*C``.

    Returns:
        ``min(n_select, |A|)`` cids sorted by loss descending (ties by cid). An unseen client
        has loss +inf so it is picked first: it has had no chance to report, and starving it
        would bias the sample (our choice; the paper does not cover cold start).
    """
    candidates = sorted(good, key=lambda cid: (-losses.get(cid, math.inf), cid))
    return candidates[: min(n_select, len(candidates))]


# --------------------------------------------------------------------------------------
# The strategy (pure state machine)
# --------------------------------------------------------------------------------------
class LibraStrategy:
    """Libra server logic: deadline rounds, fast-device limit, stale gating, biased selection.

    Per-client state is one of ``idle`` / ``fresh`` (training the current round's model) /
    ``stale`` (training an older round's model with zero uploads yet). A round ends when
    ``now >= round_start + deadline_s`` (the paper's deadline T).

    Attributes:
        global_model: Current global model, evaluated by the server.
        version: Round index; also the base version stamped on every ``TRAIN``.
        events: Event dicts (``kind`` in agg/reject/halt/select/sinkhorn) drained by the server.
        participation: ``cid -> P_i``, accepted contributions so far.
        losses: ``cid -> last reported local loss``.
        theta: ``cid -> ϑ_i`` stale-admission flag.
        state: ``cid -> idle|fresh|stale``.
        uploads: ``cid -> σ``, uploads received in the current round.
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: dict[int, int],
        delays: dict[int, float],
        x_val: Tensor,
        epochs: int,
        C: float,
        tau_max: int,
        sigma_max: int,
        M: float,
        good_frac: float,
        deadline_s: float,
        sinkhorn_L: float | None,
        sinkhorn_eps: float,
        sinkhorn_iters: int,
        select_every: int,
        no_limit: bool = False,
        random_select: bool = False,
        staleness_select: bool = False,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        """Store hyperparameters and initialise empty per-client state.

        Args:
            model: Initial global model (moved to ``device``; the strategy owns it afterwards).
            n_samples: ``cid -> local dataset size`` (defines N and the client id set).
            delays: ``cid -> injected delay``, used only to build the good-resource set A.
            x_val: D_val inputs for the Sinkhorn test (decisions.md D6: 500 samples).
            epochs: Local epochs E sent in every ``TRAIN``.
            C: Fraction of clients selected per round (paper: 0.6).
            tau_max: Max staleness τ_max (paper: 5).
            sigma_max: Hard cap σ_max on uploads per client per round (paper: 5).
            M: Stale-admission fraction (paper: 0.5).
            good_frac: Size of A as a fraction of N (decisions.md D13: 0.7).
            deadline_s: Round deadline T in seconds (decisions.md D6: median expected round time).
            sinkhorn_L: Distance threshold L; None disables the threshold but still logs
                distances (used by the pilot that picks L, decisions.md D6).
            sinkhorn_eps: Sinkhorn ε (decisions.md D6: 0.1).
            sinkhorn_iters: Sinkhorn iterations (decisions.md D6: 100).
            select_every: Re-run selection + ϑ assignment every this many rounds (paper: 10).
            no_limit: Libra-NoL, disable both the Sinkhorn limit and σ_max.
            random_select: Libra-RS, pick clients uniformly at random instead of by loss.
            staleness_select: Libra-SS, admit stale uploads by staleness instead of by P.
            seed: Seed of the strategy's own RNG (random selection).
            device: Torch device for the global model and Sinkhorn probes.
        """
        self.global_model = model.to(device)
        self._probe = copy.deepcopy(self.global_model)  # scratch copy to load local weights into
        self.x_val = x_val.to(device)
        self.n_samples = dict(n_samples)
        self.epochs = epochs
        self.C = C
        self.tau_max = tau_max
        self.sigma_max = sigma_max
        self.M = M
        self.deadline_s = deadline_s
        self.L = sinkhorn_L
        self.eps = sinkhorn_eps
        self.iters = sinkhorn_iters
        self.select_every = select_every
        self.no_limit = no_limit
        self.random_select = random_select
        self.staleness_select = staleness_select
        self._rng = random.Random(seed)

        self.cids = sorted(n_samples)
        self.good = good_resource_set(delays, good_frac)
        self.n_select = round(len(self.cids) * C)  # N*C clients per round
        self.cost = zero_one_cost(self._n_classes())

        self.version = 0
        self.round_start = 0.0
        self.events: list[dict] = []
        self.participation: dict[int, int] = {cid: 0 for cid in self.cids}  # P_i
        self.losses: dict[int, float] = {}
        self.theta: dict[int, int] = {cid: 1 for cid in self.cids}  # ϑ_i
        self.state: dict[int, str] = {cid: IDLE for cid in self.cids}
        self.uploads: dict[int, int] = {cid: 0 for cid in self.cids}  # σ per cid this round
        self.train_version: dict[int, int] = {}  # base version each client is training on
        self.selected: list[int] = []
        self._buffer: list[tuple[int, dict, int, bool]] = []  # (cid, weights, n, is_stale)
        self._p_global = output_distribution(self.global_model, self.x_val)

    # ---- public strategy interface (CLAUDE.md Contracts) --------------------------------
    def start(self, now: float) -> Outbox:
        """Round 0: select clients, assign ϑ, send the first ``TRAIN`` broadcast.

        Args:
            now: Server monotonic time.

        Returns:
            ``(cid, TRAIN)`` pairs for the selected clients.
        """
        self.round_start = now
        self._select(now, initial=True)
        return self._train_msgs(self.selected)

    def on_update(self, msg: Message, now: float) -> Outbox:
        """Handle one ``UPDATE``: record loss, then admit or reject it.

        Args:
            msg: ``UPDATE`` dict (``cid, base_version, weights, n_samples, loss, epochs_done,
                upload_idx``).
            now: Server monotonic time.

        Returns:
            Zero or one control message (``HALT``) for the sender.
        """
        cid = msg["cid"]
        self.losses[cid] = msg["loss"]  # feeds biased selection (c)
        if msg["base_version"] == self.version:
            return self._on_fresh_update(msg, now)
        return self._on_stale_update(msg, now)

    def on_tick(self, now: float) -> Outbox:
        """Timer hook: close the round at the deadline T, aggregate, start the next round.

        Args:
            now: Server monotonic time.

        Returns:
            ``TRAIN`` / ``HALT`` messages for the new round (empty before the deadline).
        """
        if now < self.round_start + self.deadline_s:
            return []
        n_fresh, n_stale = self._aggregate(now)
        self.version += 1
        self.round_start = now
        self._p_global = output_distribution(self.global_model, self.x_val)
        self.events.append(
            {
                "kind": "agg",
                "t": now,
                "version": self.version,
                "cids": [c for c, *_ in self._buffer],
                "n_updates": n_fresh + n_stale,
                "n_stale": n_stale,
            }
        )
        self._buffer = []

        # Clients with zero uploads are still training the old model -> they become stale.
        uploaded = {cid for cid in self.cids if self.uploads[cid] > 0}
        for cid in self.cids:
            if self.state[cid] == FRESH and cid not in uploaded:
                self.state[cid] = STALE
        # Selection refresh happens before TRAIN so new rounds use the new set and ϑ.
        if self.version % self.select_every == 0:
            self._select(now, initial=False)
        selected = set(self.selected)

        out: Outbox = []
        # Server-side early drop ρ (decisions.md D13): halt stale clients that can no longer help.
        p_m = participation_threshold(self.participation, self.M)
        for cid in self.cids:
            if self.state[cid] != STALE:
                continue
            v_i = self.version - self.train_version[cid]
            if early_drop(self.tau_max, v_i, self.participation[cid], p_m):
                self.state[cid] = IDLE
                self._log_halt(now, cid, "rho", v=v_i)
                if cid not in selected:  # a selected client gets a TRAIN below, which supersedes
                    out.append((cid, {"type": "HALT"}))
        # Clients that uploaded this round: continue if selected, otherwise stop.
        to_train: list[int] = []
        for cid in self.cids:
            if cid in uploaded or self.state[cid] == IDLE:
                if cid in selected:
                    to_train.append(cid)
                elif cid in uploaded:
                    self.state[cid] = IDLE
                    out.append((cid, {"type": "HALT"}))
        out.extend(self._train_msgs(to_train))
        self.uploads = {cid: 0 for cid in self.cids}
        return out

    # ---- update handling -----------------------------------------------------------------
    def _on_fresh_update(self, msg: Message, now: float) -> Outbox:
        """Mechanism (a): free uploads 1-2, Sinkhorn test for 3..σ_max-1, hard stop at σ_max.

        Args:
            msg: ``UPDATE`` whose ``base_version`` equals the current round.
            now: Server monotonic time.

        Returns:
            ``[(cid, HALT)]`` if the limit fires, else ``[]``.
        """
        cid, idx = msg["cid"], msg["upload_idx"]  # idx = this upload's ordinal in the round
        self.uploads[cid] += 1  # σ counter, reset at round end
        self._accept(msg, stale=False)
        if self.no_limit:  # Libra-NoL: neither Sinkhorn nor σ_max applies
            return []
        if idx >= self.sigma_max:
            self._log_halt(now, cid, "sigma_max")
            return [(cid, {"type": "HALT"})]
        if idx >= 3:  # uploads 1-2 are free (paper); distance test starts at the third
            dist = self._distance(msg["weights"])
            self.events.append(
                {"kind": "sinkhorn", "t": now, "cid": cid, "upload_idx": idx, "dist": dist}
            )
            if self.L is not None and dist >= self.L:
                self._log_halt(now, cid, "sinkhorn", dist=dist)
                return [(cid, {"type": "HALT"})]
        return []

    def _on_stale_update(self, msg: Message, now: float) -> Outbox:
        """Mechanism (b): admit a stale upload only if ϑ_i = 1, ρ_i = 0 and v_i < τ_max.

        Libra-SS replaces the participation gate by ``v_i <= τ_max // 2`` (our interpretation of
        "admit by staleness"; the paper's -SS ablation is not specified in detail).

        Args:
            msg: ``UPDATE`` with ``base_version`` older than the current round.
            now: Server monotonic time.

        Returns:
            ``[(cid, HALT)]``: a stale client stops after its one upload instead of re-uploading
            its old base model; it gets a fresh ``TRAIN`` at the next round end.
        """
        cid = msg["cid"]
        v_i = self.version - msg["base_version"]  # staleness v_i in rounds
        p_i = self.participation[cid]
        reason = None
        if v_i >= self.tau_max:
            reason = "tau_max"
        elif self.staleness_select:
            if v_i > self.tau_max // 2:
                reason = "staleness"
        elif self.theta[cid] == 0:
            reason = "theta"
        elif early_drop(
            self.tau_max, v_i, p_i, participation_threshold(self.participation, self.M)
        ):
            reason = "rho"
        if reason is None:
            self._accept(msg, stale=True)
        else:
            # Wasted client compute: the metrics layer counts these events.
            self.events.append({"kind": "reject", "t": now, "cid": cid, "v": v_i, "reason": reason})
        self.state[cid] = IDLE
        return [(cid, {"type": "HALT"})]

    def _accept(self, msg: Message, stale: bool) -> None:
        """Buffer an accepted update for aggregation and bump P_i.

        Args:
            msg: The ``UPDATE`` message.
            stale: Whether it came from an older round.
        """
        cid = msg["cid"]
        self._buffer.append((cid, msg["weights"], msg["n_samples"], stale))
        self.participation[cid] += 1  # P_i

    def _distance(self, local_weights: dict) -> float:
        """Sinkhorn distance between this round's global model and an uploaded local model.

        Args:
            local_weights: Uploaded ``state_dict`` (CPU tensors).

        Returns:
            The entropic OT cost compared against the threshold L.
        """
        self._probe.load_state_dict(local_weights)
        q = output_distribution(self._probe, self.x_val)
        return sinkhorn_distance(self._p_global, q, self.cost, self.eps, self.iters)

    # ---- round machinery ----------------------------------------------------------------
    def _aggregate(self, now: float) -> tuple[int, int]:
        """Sample-weighted average of all accepted uploads (fresh + admitted stale).

        Every accepted upload counts, so a fast device that uploaded twice contributes twice;
        that is the influence the Sinkhorn limit controls (decisions.md D13). An empty buffer
        keeps the old model.

        Args:
            now: Server monotonic time (for the log line).

        Returns:
            ``(n_fresh, n_stale)`` update counts.
        """
        n_stale = sum(1 for *_, stale in self._buffer if stale)
        n_fresh = len(self._buffer) - n_stale
        if not self._buffer:
            # Keep the old model; the agg event (n_updates=0) records the empty round.
            return 0, 0
        total = sum(n for _, _, n, _ in self._buffer)
        ref = self.global_model.state_dict()
        new_state = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in ref.items()}
        for _, weights, n, _ in self._buffer:
            for k in new_state:
                new_state[k] += weights[k].to(new_state[k].device).double() * (n / total)
        self.global_model.load_state_dict({k: new_state[k].to(ref[k].dtype) for k in ref})
        return n_fresh, n_stale

    def _select(self, now: float, initial: bool) -> None:
        """Choose this window's clients (c) and assign ϑ (b) from current P.

        Args:
            now: Server monotonic time.
            initial: True at round 0, when no losses exist yet.
        """
        pool = self.cids if self.random_select else sorted(self.good)
        if self.random_select or initial:
            # Round 0 has no loss information, so random (among A unless Libra-RS); Libra-RS
            # stays random forever.
            k = min(self.n_select, len(pool))
            self.selected = sorted(self._rng.sample(pool, k))
        else:
            self.selected = biased_selection(self.losses, self.good, self.n_select)
        self.theta = assign_admission(self.participation, self.M)
        self.events.append(
            {"kind": "select", "t": now, "version": self.version, "cids": list(self.selected)}
        )

    def _train_msgs(self, cids: list[int]) -> Outbox:
        """Build ``TRAIN`` messages carrying a CPU snapshot of the global model.

        Args:
            cids: Recipients; each is marked ``fresh`` on the current version.

        Returns:
            ``(cid, TRAIN)`` pairs.
        """
        if not cids:
            return []
        # One CPU snapshot shared by all recipients (weights are CPU tensors on the wire).
        snapshot = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        out: Outbox = []
        for cid in cids:
            self.state[cid] = FRESH
            self.train_version[cid] = self.version
            policy = {
                # Keep training and re-uploading after each E epochs until HALT / new TRAIN /
                # sigma_max uploads; Libra-NoL has no sigma_max (decisions.md D13).
                "multi_upload": True,
                "sigma_max": None if self.no_limit else self.sigma_max,
                "theta": self.theta[cid],
            }
            out.append(
                (
                    cid,
                    {
                        "type": "TRAIN",
                        "version": self.version,
                        "weights": snapshot,
                        "epochs": self.epochs,
                        "policy": policy,
                    },
                )
            )
        return out

    def _log_halt(self, now: float, cid: int, reason: str, **extra: float) -> None:
        """Append a ``halt`` event.

        Args:
            now: Server monotonic time.
            cid: The halted client.
            reason: ``sinkhorn`` / ``sigma_max`` / ``rho``.
            **extra: Extra fields such as the Sinkhorn ``dist``.
        """
        self.events.append({"kind": "halt", "t": now, "cid": cid, "reason": reason, **extra})

    def _n_classes(self) -> int:
        """Number of output classes, read off one forward pass on D_val."""
        with torch.no_grad():
            return int(self.global_model(self.x_val[:1]).shape[1])
