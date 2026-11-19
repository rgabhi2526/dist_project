"""The server main loop: the only code that touches strategy state.

Sits in the server process (decisions.md D2). ``run.py`` accepts the client connections and
builds a strategy; :func:`serve` then runs until the wall-time budget is spent:

    reader threads (transport.start_readers) --> queue.Queue --> THIS loop --> transport.send

Why a single-threaded consumer: the strategies are plain state machines without locks. Because
exactly one thread dequeues messages, calls ``on_update`` / ``on_tick`` and sends the replies,
there is no concurrent access to strategy state and no locking anywhere in the algorithm code.
Threads only move bytes between sockets and the queue.

The server also owns the single clock (``time.monotonic()``, decisions.md D8), the eval timer
(a fixed wall-clock period, not per aggregation) and all metric files (decisions.md D3, D13).
"""

from __future__ import annotations

import argparse
import copy
import logging
import queue
import time
from multiprocessing.connection import Connection

from torch.utils.data import DataLoader

from flsim import transport
from flsim.metrics import EvalLogger, Traffic, write_events_jsonl, write_summary
from flsim.model import evaluate

log = logging.getLogger("flsim.server")

POLL_S = 0.05  # queue timeout: bounds how late on_tick (deadlines) and the eval timer can fire
SLOW_SEND_S = 0.5  # a send that blocks this long means the client is not reading (warn)


def _dispatch(
    outbox: list[tuple[int, dict]],
    conns: dict[int, Connection],
    traffic: Traffic,
) -> None:
    """Send every ``(cid, msg)`` of a strategy reply and count it by direction.

    TRAIN counts as downstream model traffic; NOTIFY and HALT as control (decisions.md D8).

    Args:
        outbox: Messages returned by a strategy hook.
        conns: Connection of each client id.
        traffic: Counters updated in place.
    """
    for cid, msg in outbox:
        t_send = time.monotonic()
        nbytes = transport.send(conns[cid], msg)
        if (took := time.monotonic() - t_send) > SLOW_SEND_S:
            log.warning("send of %s to client %d blocked %.2fs", msg["type"], cid, took)
        traffic.bytes_down += nbytes
        if msg["type"] == "TRAIN":
            traffic.n_down += 1
        else:
            traffic.n_ctrl += 1


def serve(
    args: argparse.Namespace,
    strategy,  # duck-typed strategy (CLAUDE.md Contracts), deliberately no ABC
    conns: dict[int, Connection],
    test_loader: DataLoader,
    device: str,
) -> None:
    """Run one experiment for ``args.budget_s`` seconds of wall time.

    Args:
        args: Resolved config (uses ``out``, ``budget_s``, ``eval_every_s``, ``n_clients``).
        strategy: Object with ``start / on_update / on_tick``, ``global_model`` and ``events``.
        conns: Client id -> connection, from ``transport.accept_clients``.
        test_loader: Test data (D_val already excluded, decisions.md D5).
        device: Device used for evaluation (GPU on the Jarvis L4, decisions.md D3).

    Raises:
        RuntimeError: If a client disconnects before the budget ends. In our runs a dead client
            is a bug, and silently continuing would corrupt the comparison.
    """
    out_dir = args.out
    traffic = Traffic()
    evals = EvalLogger(f"{out_dir}/evals.csv")
    events_path = f"{out_dir}/events.jsonl"
    all_events: list[dict] = []
    accs: list[float] = []
    # Separate eval copy on `device`: the strategy's own model may be on another device and is
    # being replaced by aggregation, so we never evaluate it in place.
    eval_model = copy.deepcopy(strategy.global_model).to(device)

    def run_eval(now: float) -> None:
        """Evaluate the current global model and append one evals.csv row."""
        eval_model.load_state_dict(strategy.global_model.state_dict())
        acc, loss = evaluate(eval_model, test_loader, device)
        accs.append(acc)
        evals.log(now, strategy.version, acc, loss, traffic)

    def drain_events() -> None:
        """Move strategy.events to events.jsonl (the single source of truth, D13)."""
        fresh, strategy.events[:] = list(strategy.events), []
        all_events.extend(fresh)
        write_events_jsonl(events_path, fresh)

    t0 = time.monotonic()  # the one clock of the whole system (D8)
    q: queue.Queue = queue.Queue()
    transport.start_readers(conns, q)
    try:
        _dispatch(strategy.start(0.0), conns, traffic)
        next_eval = 0.0
        while (now := time.monotonic() - t0) < args.budget_s:
            if now >= next_eval:
                run_eval(now)
                next_eval += args.eval_every_s
                now = time.monotonic() - t0  # eval may be slow on CPU; keep `now` honest
            try:
                cid, msg, nbytes = q.get(timeout=POLL_S)
            except queue.Empty:
                pass
            else:
                if msg["type"] == "DISCONNECT":
                    raise RuntimeError(f"client {cid} disconnected before the budget ended")
                if msg["type"] == "UPDATE":
                    traffic.n_up += 1
                    traffic.bytes_up += nbytes
                    _dispatch(strategy.on_update(msg, time.monotonic() - t0), conns, traffic)
            _dispatch(strategy.on_tick(time.monotonic() - t0), conns, traffic)
            drain_events()
        run_eval(time.monotonic() - t0)  # final point at the end of the budget
        drain_events()
    except RuntimeError as err:
        log.error("%s", err)
        raise
    finally:
        for conn in conns.values():
            try:
                transport.send(conn, transport.shutdown())
            except OSError:
                pass  # that client is already gone
        evals.close()
        write_summary(f"{out_dir}/summary.json", accs, all_events, args.n_clients, traffic)
