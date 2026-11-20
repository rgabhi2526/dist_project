"""One client process: local training, injected heterogeneity, and uploads.

Sits at the leaf of the system (decisions.md D1, D2): ``run.py`` spawns N of these with
``multiprocessing`` ("spawn"). A client shares no memory with the server; it connects over TCP,
announces itself with HELLO and then reacts to the server's messages (CLAUDE.md "Contracts"):

    TRAIN    -> train E epochs from the received weights, upload, maybe train again (Libra)
    NOTIFY   -> SEAFL^2: stop after the current epoch and upload now (epochs_done < E)
    HALT     -> Libra: discard the current work and idle until the next TRAIN
    SHUTDOWN -> exit

The client never sends a timestamp (decisions.md D8): the server's ``time.monotonic()`` is the
only clock, so there is no cross-process clock skew to reason about.

Inbox thread (decisions.md D2): a daemon thread drains the socket into a local queue for the
whole life of the process. Without it, a client that is busy inside an epoch would not read its
socket, the kernel buffers (~400 KB TRAIN messages) would fill, and the server's blocking
``send`` would stall the single main loop on that one client. With it the server never waits on
a slow client; the training loop simply looks at the local queue after each epoch.
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from flsim import transport
from flsim.config import client_delays, setup_logging
from flsim.data import dataset_labels, dirichlet_partition, load_dataset
from flsim.model import make_model, train_epoch

Message = dict[str, Any]


def _inbox_reader(conn: Connection, inbox: queue.Queue) -> None:
    """Forward every server message into ``inbox`` until the connection ends.

    Args:
        conn: Connection to the server.
        inbox: Local queue consumed by the training / idle loops.
    """
    while True:
        try:
            msg, _nbytes = transport.recv(conn)
        except (EOFError, OSError):
            # Server gone (or socket closed at exit): treat as SHUTDOWN so the loop exits.
            inbox.put({"type": "SHUTDOWN"})
            return
        inbox.put(msg)


def _poll_inbox(inbox: queue.Queue) -> tuple[bool, Message | None]:
    """Read every message already waiting in the local inbox, without blocking.

    Args:
        inbox: Local queue filled by :func:`_inbox_reader`.

    Returns:
        ``(notified, interrupt)``: ``notified`` is True if a NOTIFY was seen; ``interrupt`` is
        the first HALT / TRAIN / SHUTDOWN (which ends the current training immediately, any later
        message stays queued for the idle loop), or None.
    """
    notified = False
    while True:
        try:
            msg = inbox.get_nowait()  # never wait: training must not be stalled
        except queue.Empty:
            return notified, None
        if msg["type"] == "NOTIFY":
            notified = True
        else:
            return notified, msg


def _train_session(
    conn: Connection,
    inbox: queue.Queue,
    cid: int,
    model: nn.Module,
    loader: DataLoader,
    n_samples: int,
    delay_s: float,
    lr: float,
    train_msg: Message,
    log: logging.Logger,
) -> Message | None:
    """Run one TRAIN message to completion (possibly several uploads) or until interrupted.

    A *local round* is E epochs followed by one UPDATE. With Libra's ``multi_upload`` policy a
    fast device keeps starting new local rounds from its current local model until the server
    HALTs it, sends a new TRAIN, or sigma_max uploads were made.

    Args:
        conn: Connection to the server (used for sending UPDATEs).
        inbox: Local queue of server messages.
        cid: Client id k.
        model: Local model, overwritten with the TRAIN weights.
        loader: Local data loader.
        n_samples: |D_k|, sent with every UPDATE as the aggregation weight.
        delay_s: Injected sleep per epoch (already multiplied by ``time_scale``).
        lr: SGD learning rate eta.
        train_msg: The TRAIN message that started this session.
        log: This client's ``client.NN`` logger.

    Returns:
        The message that interrupted the session (a new TRAIN or SHUTDOWN) and must be handled
        next, or None if the client should now idle (HALT, or no more uploads wanted).
    """
    model.load_state_dict(train_msg["weights"])
    base_version = train_msg["version"]  # t_k, the staleness basis the server reads back
    epochs = train_msg["epochs"]  # E
    policy = train_msg.get("policy") or {}
    sigma_max = policy.get("sigma_max")  # Libra's cap on uploads per TRAIN (None = unlimited)
    upload_idx = 0  # 1-based after the first upload; resets on every TRAIN (Contracts)
    log.info("TRAIN received v=%d policy=%s", base_version, policy)

    while True:
        losses: list[float] = []
        notified = False
        t_round = time.monotonic()  # local-round wall time includes the injected sleeps
        for _epoch in range(epochs):
            losses.append(train_epoch(model, loader, lr, "cpu"))
            log.debug("epoch %d/%d loss=%.4f", _epoch + 1, epochs, losses[-1])
            # D1: the injected delay *is* the device heterogeneity. Sleep (not busy-wait) so 20
            # processes do not steal each other's CPU; delay >> real compute keeps OS noise small.
            time.sleep(delay_s)
            # Checked once per epoch: the finest granularity at which SEAFL^2 / Libra can act.
            notified, interrupt = _poll_inbox(inbox)
            if notified:
                log.info("NOTIFY received (uploading after this epoch)")
            if interrupt is not None:
                log.info("%s received (ends this session)", interrupt["type"])
                # HALT: discard work and idle. TRAIN / SHUTDOWN: hand over to the main loop.
                return None if interrupt["type"] == "HALT" else interrupt
            if notified:
                break  # SEAFL^2: upload the partial result after this epoch
        upload_idx += 1
        transport.send(
            conn,
            transport.update(
                cid=cid,
                base_version=base_version,
                weights=model.state_dict(),
                n_samples=n_samples,
                loss=float(np.mean(losses)),
                epochs_done=len(losses),
                upload_idx=upload_idx,
            ),
        )
        log.info(
            "upload v=%d idx=%d epochs_done=%d loss=%.4f round=%.1fs",
            base_version,
            upload_idx,
            len(losses),
            float(np.mean(losses)),
            time.monotonic() - t_round,
        )
        keep_going = policy.get("multi_upload") and (sigma_max is None or upload_idx < sigma_max)
        if notified or not keep_going:
            return None


def client_main(cid: int, args_dict: dict[str, Any], port: int) -> None:
    """Entry point of a client process (top level so ``spawn`` can pickle it).

    Args:
        cid: Client id k in ``[0, N)``.
        args_dict: ``vars()`` of the resolved config namespace.
        port: Server TCP port (resolved by the launcher when ``--port 0``).
    """
    args = argparse.Namespace(**args_dict)
    setup_logging(args.log_level)  # spawn gives a fresh interpreter: logging is not inherited
    log = logging.getLogger(f"client.{cid:02d}")
    torch.set_num_threads(1)  # D3: one thread per client so 20 clients do not oversubscribe
    torch.manual_seed(args.seed * 100_003 + cid)  # distinct but reproducible per (seed, cid)
    np.random.seed((args.seed * 100_003 + cid) % 2**32)

    # The parent already downloaded the data, so this only reads from disk (no download race).
    train_set, _test_set = load_dataset(args.dataset, args.data_root)
    # Same call as the launcher: identical seed -> identical partition, no index exchange needed.
    parts = dirichlet_partition(
        dataset_labels(train_set), args.n_clients, args.dirichlet_beta, args.seed
    )
    shard = Subset(train_set, parts[cid].tolist())
    gen = torch.Generator().manual_seed(args.seed * 100_003 + cid)  # reproducible shuffling
    loader = DataLoader(
        shard, batch_size=args.batch_size, shuffle=True, generator=gen, num_workers=0
    )
    # Same draws as the server's (Libra's set A, deadline T); time_scale applied on top.
    delay_s = float(client_delays(args.n_clients, args.seed, args.base_delay)[cid]) * (
        args.time_scale
    )
    model = make_model(args.dataset)

    conn = transport.connect(args.host, port, args.authkey.encode())
    inbox: queue.Queue = queue.Queue()
    threading.Thread(target=_inbox_reader, args=(conn, inbox), daemon=True).start()
    try:
        transport.send(conn, transport.hello(cid))
        log.info("connected")
        msg: Message | None = None
        while True:
            if msg is None:
                msg = inbox.get()  # idle: block until the server speaks
            kind = msg["type"]
            if kind == "SHUTDOWN":
                log.info("shutdown")
                return
            if kind == "TRAIN":
                msg = _train_session(
                    conn, inbox, cid, model, loader, len(shard), delay_s, args.lr, msg, log
                )
            else:
                msg = None  # HALT / NOTIFY while idle: nothing to stop or finish, ignore
    except (EOFError, OSError):
        return  # server went away: nothing left to do
    finally:
        conn.close()
