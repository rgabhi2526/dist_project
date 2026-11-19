"""Launcher: ``python -m flsim.run --algo ... --out results/<exp>/<algo>/<seed>``.

Sits at the top of the system (decisions.md D1, D9). One invocation is one experiment run: this
process becomes the server, spawns the N client processes, builds the strategy chosen by
``--algo`` and hands everything to :func:`flsim.server.serve`. All shared, seeded inputs
(Dirichlet partition, delays, test/D_val split) are derived here and in the clients from the same
seed, so no data or index lists are ever sent over the wire.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from flsim import transport
from flsim.client import client_main
from flsim.config import client_delays, parse_args, save_config
from flsim.data import dataset_labels, dirichlet_partition, load_dataset, split_test_dval
from flsim.model import make_model
from flsim.server import serve
from flsim.strategies.fedavg import FedAvgStrategy
from flsim.strategies.libra import LibraStrategy
from flsim.strategies.seafl import SEAFLStrategy

log = logging.getLogger("flsim.run")
JOIN_TIMEOUT_S = 15.0  # clients get this long to exit after SHUTDOWN before being terminated


def _watch_startup(procs: list[mp.Process], listener: Listener, done: threading.Event) -> None:
    """Close the listener if a client dies before connecting, so ``accept`` cannot hang.

    Args:
        procs: Spawned client processes.
        listener: The server listener; closing it makes a blocked ``accept`` raise.
        done: Set by the main thread once every client has connected.
    """
    while not done.wait(0.5):
        if any(p.exitcode not in (None, 0) for p in procs):
            listener.close()
            return


def _build_strategy(args, model, n_samples, x_val):
    """Create the strategy selected by ``--algo`` (CLAUDE.md "Contracts").

    Args:
        args: Resolved config.
        model: Initial global model w_0.
        n_samples: |D_k| per client id.
        x_val: Tensor of D_val inputs on the server device (Libra's Sinkhorn probe).

    Returns:
        A FedAvg, SEAFL or Libra strategy.
    """
    if args.algo == "fedavg":
        return FedAvgStrategy(model, n_samples, args.clients_per_round, args.epochs, args.seed)
    if args.algo == "seafl":
        return SEAFLStrategy(
            model,
            n_samples,
            K=args.K,
            stale_bound=args.stale_bound,
            alpha=args.alpha,
            mu=args.mu,
            theta=args.theta,
            n_active=args.n_active,
            epochs=args.epochs,
            uniform_weights=args.uniform_weights,
            partial=args.partial,
            seed=args.seed,
        )
    # Libra needs the same delay draws as the clients (good-resource set A, deadline T).
    delays = client_delays(args.n_clients, args.seed, args.base_delay * args.time_scale)
    if args.deadline_s is None:
        # ponytail: T = 1.2 x E x (median scaled delay + measured epoch compute); the 1.2 is a
        # margin, ceiling is that compute is one constant (--epoch-compute-s, Mac value) and
        # not per-client; upgrade path is the pilot-tuned --deadline-s (D6).
        args.deadline_s = float(
            1.2 * args.epochs * (float(np.median(delays)) + args.epoch_compute_s)
        )
    return LibraStrategy(
        model,
        n_samples,
        delays={cid: float(d) for cid, d in enumerate(delays)},
        x_val=x_val,
        epochs=args.epochs,
        C=args.C,
        tau_max=args.tau_max,
        sigma_max=args.sigma_max,
        M=args.M,
        good_frac=args.good_frac,
        deadline_s=args.deadline_s,
        sinkhorn_L=args.sinkhorn_L,
        sinkhorn_eps=args.sinkhorn_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        select_every=args.select_every,
        no_limit=args.no_limit,
        random_select=args.random_select,
        staleness_select=args.staleness_select,
        seed=args.seed,
        device=args.server_device,
    )


def main(argv: list[str] | None = None) -> None:
    """Run one experiment end to end.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)
    save_config(args, args.out)
    torch.manual_seed(args.seed)

    # Download once here, before spawning, so N clients do not race on the same files.
    train_set, test_set = load_dataset(args.dataset, args.data_root)
    parts = dirichlet_partition(
        dataset_labels(train_set), args.n_clients, args.dirichlet_beta, args.seed
    )
    n_samples = {cid: len(p) for cid, p in enumerate(parts)}
    test_idx, dval_idx = split_test_dval(len(test_set), args.dval_size, args.seed)
    test_loader = DataLoader(Subset(test_set, test_idx.tolist()), batch_size=1000)
    x_val = torch.stack([test_set[int(i)][0] for i in dval_idx]).to(args.server_device)
    model = make_model(args.dataset)

    listener = transport.listen(args.host, args.port, args.authkey.encode())
    port = listener.address[1]  # resolves --port 0
    ctx = mp.get_context("spawn")  # fresh interpreters: no fork-inherited torch thread state
    procs = [
        ctx.Process(target=client_main, args=(cid, vars(args), port), name=f"client-{cid}")
        for cid in range(args.n_clients)
    ]
    conns: dict = {}
    try:
        for p in procs:
            p.start()
        done = threading.Event()
        threading.Thread(target=_watch_startup, args=(procs, listener, done), daemon=True).start()
        try:
            conns = transport.accept_clients(listener, args.n_clients)
        except OSError as err:
            raise RuntimeError("a client process died during start-up") from err
        finally:
            done.set()
        strategy = _build_strategy(args, model, n_samples, x_val)
        save_config(args, args.out)  # re-save: deadline_s may have been resolved above
        serve(args, strategy, conns, test_loader, args.server_device)
    finally:
        # Never leave orphan clients behind, whatever went wrong.
        deadline = time.monotonic() + JOIN_TIMEOUT_S
        for p in procs:
            p.join(max(0.0, deadline - time.monotonic()))
            if p.is_alive():
                log.warning("terminating straggler %s", p.name)
                p.terminate()
                p.join(2.0)
        for conn in conns.values():
            conn.close()
        listener.close()


if __name__ == "__main__":
    main()
