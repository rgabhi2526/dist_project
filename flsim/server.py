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
MANY_CIDS = 8  # a select event naming more clients than this is logged as a count, not a list


def _log_event(ev: dict) -> None:
    """Log one strategy event as a single short line (INFO), Sinkhorn distances at DEBUG.

    Args:
        ev: An event dict as drained from ``strategy.events`` (kind in agg / reject / halt /
            notify / select / sinkhorn, decisions.md D13).
    """
    kind = ev["kind"]
    if kind == "agg":
        stale = ev.get("staleness")  # only SEAFL records per-update staleness s_k
        extra = f" mean_staleness={sum(stale) / len(stale):.1f}" if stale else ""
        n_upd = ev.get("n_updates", len(ev["cids"]))
        log.info("agg v=%d n_updates=%d%s", ev["version"], n_upd, extra)
    elif kind in ("reject", "halt"):
        log.info("%s cid=%d reason=%s", kind, ev["cid"], ev["reason"])
    elif kind == "notify":
        log.info("notify cid=%d", ev["cid"])
    elif kind == "select":
        cids = ev["cids"]
        shown = cids if len(cids) <= MANY_CIDS else f"{len(cids)} clients"
        log.info("select v=%d %s", ev["version"], shown)
    elif kind == "sinkhorn":
        log.debug("sinkhorn cid=%d upload=%d dist=%.4f", ev["cid"], ev["upload_idx"], ev["dist"])


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
        log.debug("sent %s to cid=%d (%d bytes)", msg["type"], cid, nbytes)
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
        kinds = [e["kind"] for e in all_events]
        log.info(
            "t=%5.1fs/%.0fs v=%d acc=%.3f loss=%.3f | up=%d down=%d ctrl=%d | "
            "agg=%d rej=%d halt=%d",
            now,
            args.budget_s,
            strategy.version,
            acc,
            loss,
            traffic.n_up,
            traffic.n_down,
            traffic.n_ctrl,
            kinds.count("agg"),
            kinds.count("reject"),
            kinds.count("halt"),
        )

    def drain_events() -> None:
        """Move strategy.events to events.jsonl (the single source of truth, D13)."""
        fresh, strategy.events[:] = list(strategy.events), []
        all_events.extend(fresh)
        for ev in fresh:
            _log_event(ev)
        write_events_jsonl(events_path, fresh)

    complete = False  # flipped only when the loop below ends by budget, never on error / Ctrl-C
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
                    log.debug(
                        "recv UPDATE cid=%d base_v=%d upload=%d (%d bytes)",
                        cid,
                        msg["base_version"],
                        msg["upload_idx"],
                        nbytes,
                    )
                    traffic.n_up += 1
                    traffic.bytes_up += nbytes
                    _dispatch(strategy.on_update(msg, time.monotonic() - t0), conns, traffic)
            _dispatch(strategy.on_tick(time.monotonic() - t0), conns, traffic)
            drain_events()
        run_eval(time.monotonic() - t0)  # final point at the end of the budget
        drain_events()
        complete = True
    except RuntimeError as err:
        log.error("%s", err)
        raise
    finally:
        for conn in conns.values():
            try:
                nbytes = transport.send(conn, transport.shutdown())
                log.debug("sent SHUTDOWN (%d bytes)", nbytes)
            except OSError:
                pass  # that client is already gone
        evals.close()
        # complete stays False on any exception *or* Ctrl-C (KeyboardInterrupt skips the
        # `complete = True` line), so experiments.sh re-runs an interrupted run on resume.
        write_summary(
            f"{out_dir}/summary.json", accs, all_events, args.n_clients, traffic, complete
        )
