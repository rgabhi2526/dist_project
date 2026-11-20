"""Command-line configuration for every flsim process (server, clients, launcher).

Sits at the root of the system: ``run.py`` parses the flags once, dumps the resolved values to
``config.json`` (decisions.md D9) and passes them to the server and the N client processes.
Also hosts :func:`client_delays`, the shared device-heterogeneity model (requirements.md 3.3,
decisions.md D13) so that the server (Libra's deadline T, good-resource set A) and the clients
(injected sleeps) derive the *same* per-client speeds from the same seed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

# Preset-controlled arguments (decisions.md D6). Their parser default is None so that
# "the user did not pass the flag" is distinguishable from "the user passed the default".
_PRESET_KEYS = ("K", "stale_bound", "n_active")


def setup_logging(level: str) -> None:
    """Configure the root logger (same format in the server and in every client process).

    Spawned client processes start a fresh interpreter and do not inherit the parent's logging
    config, so each of them calls this itself. ``force=True`` replaces any earlier handler.

    Args:
        level: ``"INFO"`` or ``"DEBUG"`` (``--log-level``).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser holding every hyperparameter (decisions.md D9).

    Returns:
        A parser whose defaults follow decisions.md D6. Preset-controlled flags (K, beta,
        n_active) default to ``None`` and are resolved later by :func:`parse_args`.
    """
    p = argparse.ArgumentParser(description="flsim: SEAFL vs Libra vs FedAvg on real processes")

    # --- run identity ------------------------------------------------------------------
    p.add_argument("--algo", required=True, choices=["fedavg", "seafl", "libra"])
    p.add_argument("--preset", choices=["paper", "scaled"], default="scaled")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="results directory for this run")
    p.add_argument("--log-level", choices=["INFO", "DEBUG"], default="INFO", help="console log")

    # --- federation / data -------------------------------------------------------------
    p.add_argument("--n-clients", type=int, default=20, help="N, number of client processes")
    p.add_argument("--dataset", choices=["fmnist", "cifar10"], default="fmnist")
    p.add_argument("--data-root", default="data")
    p.add_argument("--dirichlet-beta", type=float, default=0.5, help="non-IID skew (small=skewed)")

    # --- local training (SGD, decisions.md D6) -----------------------------------------
    p.add_argument("--epochs", type=int, default=5, help="E, local epochs per round")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)

    # --- time model --------------------------------------------------------------------
    # 3.0 s ~ 3x the measured 1.1 s real epoch (Mac, ~3000 samples, 1 thread), so injected delay
    # dominates compute (decisions.md D1); the pilot finalises it (D15).
    p.add_argument("--base-delay", type=float, default=3.0, help="s/epoch of the fastest client")
    # Measured on the Mac; MUST be re-measured on Jarvis. Only used for Libra's auto deadline T.
    p.add_argument("--epoch-compute-s", type=float, default=1.1, help="real s per local epoch")
    p.add_argument("--time-scale", type=float, default=1.0, help="multiplies every injected delay")
    p.add_argument("--budget-s", type=float, default=1200, help="wall-time budget (decisions D8)")
    p.add_argument("--eval-every-s", type=float, default=5, help="eval timer period (D8)")

    # --- transport / devices -----------------------------------------------------------
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0, help="0 = let the OS pick a free port")
    p.add_argument(
        "--authkey", default="flsim", help="shared secret for multiprocessing.connection"
    )
    p.add_argument("--client-device", default="cpu")
    p.add_argument("--server-device", default="auto", help="auto = cuda if available else cpu")

    # --- SEAFL (and FedBuff / SEAFL^2 flags) -------------------------------------------
    p.add_argument("--K", type=int, default=None, help="buffer size (preset-controlled)")
    p.add_argument("--stale-bound", type=float, default=None, help="beta; 'inf' allowed")
    p.add_argument("--alpha", type=float, default=3.0, help="alpha, staleness-weight scale")
    p.add_argument("--mu", type=float, default=1.0, help="mu, importance-factor scale")
    p.add_argument("--theta", type=float, default=0.8, help="server mixing rate (vartheta)")
    p.add_argument("--n-active", type=int, default=None, help="active clients (preset-controlled)")
    p.add_argument("--uniform-weights", action="store_true", help="FedBuff: p_k = 1/K")
    p.add_argument("--partial", action="store_true", help="SEAFL^2: notify + partial uploads")

    # --- Libra -------------------------------------------------------------------------
    p.add_argument("--C", type=float, default=0.6, help="C, fraction of clients per round")
    p.add_argument("--tau-max", type=int, default=5, help="tau_max, max model staleness")
    p.add_argument("--sigma-max", type=int, default=5, help="sigma_max, max uploads per round")
    p.add_argument("--M", type=float, default=0.5, help="M, fraction of clients admitted stale")
    p.add_argument("--good-frac", type=float, default=0.7, help="fastest fraction = set A (D13)")
    p.add_argument(
        "--deadline-s", type=float, default=None, help="T; None = 1.2*E*(median delay + compute)"
    )
    p.add_argument("--sinkhorn-L", type=float, default=None, help="L; None = log only (pilot)")
    p.add_argument("--sinkhorn-eps", type=float, default=0.1, help="entropic regulariser (D6)")
    p.add_argument("--sinkhorn-iters", type=int, default=100)
    p.add_argument("--dval-size", type=int, default=500, help="|D_val| carved from the test set")
    p.add_argument("--select-every", type=int, default=10, help="rounds between re-selection")
    p.add_argument("--no-limit", action="store_true", help="Libra-NoL ablation")
    p.add_argument("--random-select", action="store_true", help="Libra-RS ablation")
    p.add_argument("--staleness-select", action="store_true", help="Libra-SS ablation")

    # --- FedAvg ------------------------------------------------------------------------
    p.add_argument("--clients-per-round", type=int, default=12, help="matches Libra's C*N (D6)")
    return p


def _apply_preset(args: argparse.Namespace) -> None:
    """Fill still-``None`` preset-controlled values in place (decisions.md D6).

    Explicit command-line flags are never overwritten: only ``None`` values are filled.

    Args:
        args: Parsed namespace with ``preset`` and ``n_clients`` set.
    """
    if args.preset == "scaled":
        # K = 5 keeps K/N close to the paper's ratio at N = 20; all clients stay active.
        values = {"K": 5, "stale_bound": 10.0, "n_active": args.n_clients}
    else:
        # ponytail: paper uses a subset of active clients; N // 2 is a guess, exact fraction to
        # be verified against the SEAFL paper (decisions.md D6), then fix here.
        values = {"K": 10, "stale_bound": 10.0, "n_active": args.n_clients // 2}
    for key in _PRESET_KEYS:
        if getattr(args, key) is None:
            setattr(args, key, values[key])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse flags, then resolve presets and ``--server-device auto``.

    Args:
        argv: Argument list (without program name); ``None`` reads ``sys.argv``.

    Returns:
        Fully resolved namespace: no preset-controlled value is ``None`` any more.
    """
    args = build_parser().parse_args(argv)
    _apply_preset(args)
    if args.server_device == "auto":
        import torch  # local import keeps `--help` and client start-up light

        args.server_device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def client_delays(n: int, seed: int, base: float, cap: float = 20.0) -> np.ndarray:
    """Draw the shared per-client compute-delay model (requirements.md 3.3, decisions.md D13).

    Delay of client k in seconds per local epoch is ``base * min(1 + Pareto(a=1.5), cap)``.
    ``numpy``'s ``pareto`` is the Lomax variant, so ``1 + pareto`` is the classical Pareto with
    minimum 1: the fastest possible client takes exactly ``base``. The cap (decisions.md D6)
    stops an extreme tail draw from producing a client that never finishes.

    Server and clients must call this with identical arguments; the seeded generator then
    returns identical draws in every process without any message exchange.

    Args:
        n: Number of clients N.
        seed: Run seed.
        base: Seconds per local epoch for the fastest client (already times ``time_scale``
            if the caller wants it applied).
        cap: Maximum slowdown factor relative to ``base``.

    Returns:
        Float array of shape ``(n,)``, each entry in ``[base, base * cap]``.
    """
    rng = np.random.default_rng(seed)
    return base * np.minimum(1.0 + rng.pareto(a=1.5, size=n), cap)


def _json_safe(obj: Any) -> Any:
    """Convert values ``json`` cannot encode (bytes, +/-inf) into strings.

    Args:
        obj: A value from the args namespace.

    Returns:
        A JSON-serialisable replacement; unknown types are returned unchanged.
    """
    if isinstance(obj, bytes):
        return obj.decode(errors="replace")
    if isinstance(obj, float) and obj in (float("inf"), float("-inf")):
        return "inf" if obj > 0 else "-inf"  # strict JSON has no Infinity literal
    return obj


def save_config(args: argparse.Namespace, out_dir: str | Path) -> None:
    """Write the resolved arguments to ``out_dir/config.json`` for provenance (decisions D9).

    Args:
        args: Resolved namespace from :func:`parse_args`.
        out_dir: Results directory; created if missing.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: _json_safe(v) for k, v in vars(args).items()}
    (out / "config.json").write_text(json.dumps(clean, indent=2, sort_keys=True))
