"""FedAvg server strategy: the synchronous baseline.

Sits in the server process of the flsim system, next to :mod:`flsim.strategies.seafl` and
:mod:`flsim.strategies.libra`: ``server.py`` owns the sockets and the single main loop and feeds
every message to this class. Like the others it is a pure state machine (no I/O, no clock of its
own) that returns the ``(cid, message)`` pairs to send (CLAUDE.md "Contracts", decisions.md D12).

Paper -> code map (FedAvg, McMahan et al., AISTATS 2017, Alg. 1 "FederatedAveraging")
=====================================================================================
    - initialise w_0 ................................. FedAvgStrategy.__init__()
    - m <- max(C*K, 1); S_t <- random set of m clients  FedAvgStrategy._select_round()
    - ClientUpdate(k, w_t) for k in S_t (parallel) ..... TRAIN messages built in _train_msgs()
      (the local SGD itself runs in client.py)
    - w_{t+1} <- sum_k (n_k / n) w_{t+1}^k ............. FedAvgStrategy._aggregate()
      (n = sum of n_k over S_t; reuses seafl.weighted_average)
    - round barrier: wait for *all* of S_t ............. FedAvgStrategy.on_update()

Interpretation choices: decisions.md D6 (clients per round = 12 to match Libra's C*N), D13
(events list is the one source of truth for metrics). The barrier is what makes this the
synchronous reference: one slow client in S_t stalls the whole round.
"""

from __future__ import annotations

import numpy as np
import torch

from flsim.strategies.seafl import StateDict, weighted_average

Message = dict
Outbox = list[tuple[int, Message]]


class FedAvgStrategy:
    """Synchronous FedAvg: pick m clients, wait for all of them, average, repeat.

    Attributes:
        global_model: The nn.Module w_t the server evaluates.
        version: t, number of completed rounds (also the base version stamped on every TRAIN).
        events: Event dicts (``select`` / ``agg``) drained by the server into events.jsonl.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: dict[int, int],
        clients_per_round: int,
        epochs: int,
        seed: int = 0,
    ) -> None:
        """Create the strategy.

        Args:
            model: Initial global model w_0 (kept as ``global_model``).
            n_samples: |D_k| per client id; its keys define the client population N.
            clients_per_round: m = C*N, size of the random subset S_t each round.
            epochs: E, local epochs requested in every TRAIN.
            seed: Seed of the client-sampling RNG.

        Raises:
            ValueError: If ``clients_per_round`` is not in ``[1, N]``.
        """
        if not 1 <= clients_per_round <= len(n_samples):
            raise ValueError(f"clients_per_round={clients_per_round} not in [1, {len(n_samples)}]")
        self.global_model = model
        self.version = 0  # t
        self.events: list[dict] = []
        self.n_samples = n_samples  # n_k
        self.m = clients_per_round  # m
        self.epochs = epochs  # E
        self.rng = np.random.default_rng(seed)
        self.selected: list[int] = []  # S_t
        self.reported: dict[int, Message] = {}  # UPDATEs of S_t received so far

    def start(self, now: float) -> Outbox:
        """Round 0: sample S_0 and send it w_0.

        Args:
            now: Server clock, stored in the ``select`` event.

        Returns:
            One TRAIN (version 0) per selected client.
        """
        return self._select_round(now)

    def on_update(self, msg: Message, now: float) -> Outbox:
        """Buffer an UPDATE; once every client of S_t has reported, aggregate and start t+1.

        Args:
            msg: UPDATE dict (``cid, base_version, weights, n_samples, ...``).
            now: Server clock, stored in events.

        Returns:
            TRAIN messages for the next round once the barrier is complete, else ``[]``.

        Raises:
            KeyError: If the sender is not in S_t or already reported (trust boundary is the
                transport; in a synchronous run either case is a bug).
        """
        cid = msg["cid"]
        if cid not in self.selected or cid in self.reported:
            raise KeyError(f"unexpected UPDATE from client {cid} in round {self.version}")
        self.reported[cid] = msg
        if len(self.reported) < len(self.selected):
            return []  # barrier: the synchronous round waits for its slowest member
        self._aggregate(now)
        return self._select_round(now)

    def on_tick(self, now: float) -> Outbox:
        """No timers: FedAvg is purely update-driven (the barrier is a message count).

        Args:
            now: Server clock (unused).

        Returns:
            Always an empty list.
        """
        return []

    def _aggregate(self, now: float) -> None:
        """w_{t+1} = sum_k (n_k / n) w_k over S_t, then bump the version.

        Args:
            now: Server clock, stored in the ``agg`` event.
        """
        msgs = [self.reported[cid] for cid in self.selected]
        sizes = np.asarray([m["n_samples"] for m in msgs], dtype=np.float64)  # n_k
        p = sizes / sizes.sum()  # n_k / n, the FedAvg weights
        states: list[StateDict] = [m["weights"] for m in msgs]
        self.global_model.load_state_dict(weighted_average(states, p))
        self.version += 1
        self.events.append(
            {
                "kind": "agg",
                "t": now,
                "version": self.version,
                "cids": list(self.selected),
                "n_updates": len(msgs),
            }
        )

    def _select_round(self, now: float) -> Outbox:
        """Sample S_t uniformly without replacement and build the TRAIN broadcast.

        Args:
            now: Server clock, stored in the ``select`` event.

        Returns:
            One TRAIN per selected client, all carrying the same CPU snapshot of w_t.
        """
        ids = sorted(self.n_samples)
        self.selected = sorted(int(c) for c in self.rng.choice(ids, size=self.m, replace=False))
        self.reported = {}
        self.events.append(
            {"kind": "select", "t": now, "version": self.version, "cids": list(self.selected)}
        )
        # One CPU snapshot shared by all recipients (weights are CPU tensors on the wire).
        snapshot = {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}
        return [
            (
                cid,
                {
                    "type": "TRAIN",
                    "version": self.version,
                    "weights": snapshot,
                    "epochs": self.epochs,
                    "policy": {},
                },
            )
            for cid in self.selected
        ]
