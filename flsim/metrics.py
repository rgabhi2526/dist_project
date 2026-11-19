"""Metric writers and statistics: evals.csv, events.jsonl, summary.json, Jain's index.

Sits at the output end of the system: ``server.py`` calls :class:`EvalLogger` on every eval tick
and :func:`write_events_jsonl` as it drains ``strategy.events`` (decisions.md D13); at shutdown it
calls :func:`write_summary`. ``plots.py`` later reads only these files. Everything is flushed to
disk as it is produced so a crashed run keeps its partial results (decisions.md D3).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any

import numpy as np

EVAL_HEADER = ["t", "version", "acc", "loss", "n_up", "n_down", "n_ctrl", "bytes_up", "bytes_down"]


@dataclass
class Traffic:
    """Running message and byte counters kept by the server (communication metric, D8).

    Attributes:
        n_up: UPDATE messages received (client -> server).
        n_down: TRAIN messages sent (server -> client).
        n_ctrl: NOTIFY / HALT control messages sent.
        bytes_up: Payload bytes of all UPDATEs.
        bytes_down: Payload bytes of everything the server sent (TRAIN + control).
    """

    n_up: int = 0
    n_down: int = 0
    n_ctrl: int = 0
    bytes_up: int = 0
    bytes_down: int = 0


def jain_index(counts: list[int] | list[float]) -> float:
    """Jain's fairness index (sum x)^2 / (n * sum x^2) of participation counts.

    1.0 means perfectly equal participation, 1/n means one client did everything.

    Args:
        counts: Non-negative per-client counts x_k.

    Returns:
        Index in ``[1/n, 1]``. All-zero counts return 1.0 by convention: nobody participated,
        so nobody is treated worse than anybody else (and it avoids 0/0).
    """
    x = np.asarray(counts, dtype=np.float64)
    denom = len(x) * float((x**2).sum())
    if denom == 0.0:
        return 1.0
    return float(x.sum() ** 2 / denom)


def participation_counts(events: list[dict], n_clients: int) -> list[int]:
    """Count how many accepted contributions each client made, from ``agg`` events.

    Every strategy lists the contributing clients of an aggregation in ``cids`` (FedAvg: the
    round's set; SEAFL: the buffer; Libra: every accepted upload, so a fast client that uploaded
    twice appears twice, which is exactly Libra's P_i). Rejected uploads are ``reject`` events
    and are therefore not counted.

    Args:
        events: All events of a run.
        n_clients: N.

    Returns:
        List of length N, index = client id.
    """
    counts = [0] * n_clients
    for e in events:
        if e["kind"] == "agg":
            for cid in e["cids"]:
                counts[cid] += 1
    return counts


class EvalLogger:
    """Appends one row per eval tick to ``evals.csv`` and flushes it immediately (D3)."""

    def __init__(self, path: str | Path) -> None:
        """Create the file and write the header.

        Args:
            path: Destination ``evals.csv``; parent directory is created.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[str] = open(path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(EVAL_HEADER)
        self._fh.flush()

    def log(self, t: float, version: int, acc: float, loss: float, traffic: Traffic) -> None:
        """Write and flush one eval row.

        Args:
            t: Seconds since server start (the single clock, D8).
            version: Global model version evaluated.
            acc: Test accuracy in [0, 1].
            loss: Test cross-entropy.
            traffic: Cumulative counters at this moment.
        """
        self._writer.writerow(
            [f"{t:.3f}", version, f"{acc:.6f}", f"{loss:.6f}"]
            + [getattr(traffic, f) for f in EVAL_HEADER[4:]]
        )
        self._fh.flush()  # a crash must keep everything logged so far

    def close(self) -> None:
        """Close the file."""
        self._fh.close()


def write_events_jsonl(path: str | Path, events: list[dict]) -> None:
    """Append events, one JSON object per line.

    Args:
        path: ``events.jsonl``; created on first call.
        events: Event dicts drained from a strategy.
    """
    if not events:
        return
    # ponytail: reopens per call, ceiling is a few syscalls per loop iteration, upgrade path is
    # a held file handle if profiling ever shows it.
    with open(path, "a") as fh:
        for e in events:
            fh.write(json.dumps(e, default=str) + "\n")


def write_summary(
    path: str | Path,
    accs: list[float],
    events: list[dict],
    n_clients: int,
    traffic: Traffic,
) -> dict[str, Any]:
    """Write ``summary.json`` with the headline numbers of a run.

    Args:
        path: Destination file.
        accs: Test accuracy of every eval row, in time order.
        events: All events of the run.
        n_clients: N.
        traffic: Final traffic counters.

    Returns:
        The summary dict that was written.
    """
    counts = participation_counts(events, n_clients)
    summary = {
        "final_acc": accs[-1] if accs else None,
        "best_acc": max(accs) if accs else None,
        "participation_counts": counts,
        "participation_variance": float(np.var(counts)),
        "jain_index": jain_index(counts),
        "pct_never_participating": 100.0 * sum(c == 0 for c in counts) / n_clients,
        "total_msgs": traffic.n_up + traffic.n_down + traffic.n_ctrl,
        "total_bytes": traffic.bytes_up + traffic.bytes_down,
        "n_rejects": sum(e["kind"] == "reject" for e in events),
        "n_halts": sum(e["kind"] == "halt" for e in events),
        **{k: v for k, v in asdict(traffic).items()},
    }
    Path(path).write_text(json.dumps(summary, indent=2))
    return summary
