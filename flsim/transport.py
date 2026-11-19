"""Transport layer: framed pickle messages over TCP, plus message constructors.

Where this sits in the system (decisions.md D2, D12):
    run.py spawns one server process and N client processes. They share no memory and talk only
    through this module. The server calls ``listen`` + ``accept_clients`` + ``start_readers``;
    each client calls ``connect`` and then ``send`` / ``recv``. Strategies never import this
    module: they build message dicts literally, and the server / client loops use the
    constructors below (which produce exactly the same dicts, per the CLAUDE.md Contracts table).

Wire format: ``conn.send_bytes(pickle.dumps(msg))``. We pickle ourselves (instead of
``conn.send``) so that the byte count of every message is known for free; that count is the
communication metric reported in the experiments.

Security: pickle executes arbitrary code on load. The mandatory ``authkey`` (an HMAC
challenge/response performed by ``multiprocessing.connection`` on connect) is therefore the only
barrier against a hostile peer. Only our own processes, started with the same authkey, may
connect; never expose the port to an untrusted network.
"""

from __future__ import annotations

import pickle
import queue
import threading
import time
from multiprocessing.connection import Client, Connection, Listener
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Raw send / receive
# ---------------------------------------------------------------------------


def send(conn: Connection, msg: dict[str, Any]) -> int:
    """Pickle ``msg`` and send it as one framed byte string.

    Args:
        conn: An open connection.
        msg: Plain dict with a ``"type"`` key (see the Contracts table in CLAUDE.md).

    Returns:
        Number of payload bytes sent (the communication metric).
    """
    data = pickle.dumps(msg)
    conn.send_bytes(data)
    return len(data)


def recv(conn: Connection) -> tuple[dict[str, Any], int]:
    """Receive one framed message and unpickle it.

    Args:
        conn: An open connection.

    Returns:
        ``(msg, nbytes)`` where ``nbytes`` is the payload size.

    Raises:
        EOFError: If the peer closed the connection.
    """
    data = conn.recv_bytes()
    return pickle.loads(data), len(data)


# ---------------------------------------------------------------------------
# Connection setup
# ---------------------------------------------------------------------------


def listen(host: str, port: int, authkey: bytes) -> Listener:
    """Open a listening socket (server side).

    Args:
        host: Interface to bind, e.g. ``"127.0.0.1"``.
        port: TCP port; 0 lets the OS pick one (read it back from ``listener.address``).
        authkey: Shared secret for the HMAC handshake; mandatory because we unpickle (see module
            docstring).

    Returns:
        A ``Listener`` bound to ``(host, port)``.
    """
    return Listener((host, port), authkey=authkey)


def connect(
    host: str, port: int, authkey: bytes, retries: int = 50, wait_s: float = 0.2
) -> Connection:
    """Connect to the server, retrying while it is not listening yet (client side).

    Clients and server are launched together, so a client may well start before the server has
    bound its port; refused connections are therefore expected and retried.

    Args:
        host: Server host.
        port: Server port.
        authkey: Shared secret; must equal the server's.
        retries: Maximum number of connection attempts.
        wait_s: Sleep between attempts, in seconds.

    Returns:
        An open ``Connection``.

    Raises:
        ConnectionError: If the server never accepted within ``retries`` attempts.
    """
    for attempt in range(retries):
        try:
            return Client((host, port), authkey=authkey)
        except ConnectionRefusedError:
            # Only "refused" is retried; an AuthenticationError means a real config bug and must
            # surface immediately instead of being retried 50 times.
            if attempt < retries - 1:
                time.sleep(wait_s)
    raise ConnectionError(f"could not connect to {host}:{port} after {retries} attempts")


def accept_clients(listener: Listener, n: int) -> dict[int, Connection]:
    """Accept ``n`` clients and identify each by its HELLO message (server side).

    Args:
        listener: Listener returned by ``listen``.
        n: Number of clients to wait for.

    Returns:
        Mapping client id -> connection.

    Raises:
        ValueError: If a client's first message is not a valid HELLO, or its cid repeats.
    """
    conns: dict[int, Connection] = {}
    # ponytail: accepts sequentially and blocks on each HELLO, ceiling is one stalled client
    # delaying start-up (fine: our own clients send HELLO immediately), upgrade path is a
    # per-connection timeout via conn.poll().
    for _ in range(n):
        conn = listener.accept()
        msg, _nbytes = recv(conn)
        if msg.get("type") != "HELLO" or not isinstance(msg.get("cid"), int):
            conn.close()
            raise ValueError(f"expected HELLO as first message, got {msg!r}")
        cid = msg["cid"]
        if cid in conns:
            conn.close()
            raise ValueError(f"duplicate client id {cid}")
        conns[cid] = conn
    return conns


def start_readers(conns: dict[int, Connection], q: queue.Queue) -> list[threading.Thread]:
    """Start one daemon reader thread per client (server side).

    Design (D2): reader threads ONLY enqueue ``(cid, msg, nbytes)`` tuples. The single server
    main loop is the only consumer of the queue and the only code that touches strategy state,
    so no locks are needed anywhere in the algorithm code. When a connection dies the reader
    pushes a synthetic DISCONNECT message and exits, so the main loop learns about it in order.

    ponytail: thread-per-client fine at N=20; use selectors if N grows to hundreds

    Args:
        conns: Mapping client id -> connection, from ``accept_clients``.
        q: Queue the server main loop reads from.

    Returns:
        The started threads (daemon, so they never block interpreter exit).
    """

    def reader(cid: int, conn: Connection) -> None:
        """Forward every message from one client into ``q`` until the connection ends."""
        while True:
            try:
                msg, nbytes = recv(conn)
            except (EOFError, OSError):
                # EOF = client exited; OSError = socket closed locally during shutdown.
                q.put((cid, {"type": "DISCONNECT", "cid": cid}, 0))
                return
            q.put((cid, msg, nbytes))

    threads = [
        threading.Thread(target=reader, args=(cid, conn), daemon=True, name=f"reader-{cid}")
        for cid, conn in conns.items()
    ]
    for t in threads:
        t.start()
    return threads


# ---------------------------------------------------------------------------
# Message constructors (exact field sets from the CLAUDE.md Contracts table)
# ---------------------------------------------------------------------------


def _cpu_copy(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a detached CPU clone of a state_dict.

    CPU because client processes are CPU-only and cannot unpickle CUDA tensors; clone because
    the sender may keep training the model, and an un-cloned tensor would alias (and later
    mutate) the bytes we are about to pickle.

    Args:
        state_dict: Model parameters, possibly on GPU.

    Returns:
        New dict with independent CPU tensors.
    """
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def hello(cid: int) -> dict[str, Any]:
    """Build the C->S HELLO message announcing a client id."""
    return {"type": "HELLO", "cid": cid}


def update(
    cid: int,
    base_version: int,
    weights: dict[str, torch.Tensor],
    n_samples: int,
    loss: float,
    epochs_done: int,
    upload_idx: int,
) -> dict[str, Any]:
    """Build the C->S UPDATE message.

    Args:
        cid: Client id.
        base_version: Global-model version the client started training from (staleness basis).
        weights: Trained state_dict (copied to CPU).
        n_samples: Local sample count (aggregation weight).
        loss: Local training loss.
        epochs_done: Epochs completed before uploading (may be < E after a NOTIFY).
        upload_idx: Running count of this client's uploads.
    """
    return {
        "type": "UPDATE",
        "cid": cid,
        "base_version": base_version,
        "weights": _cpu_copy(weights),
        "n_samples": n_samples,
        "loss": loss,
        "epochs_done": epochs_done,
        "upload_idx": upload_idx,
    }


def train(
    version: int,
    weights: dict[str, torch.Tensor],
    epochs: int,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the S->C TRAIN message.

    Args:
        version: Global-model version being sent.
        weights: Global state_dict (copied to CPU; the server model usually lives on GPU).
        epochs: Local epochs to run.
        policy: Optional algorithm-specific dict (Libra theta/rho, etc.); None if unused.
    """
    return {
        "type": "TRAIN",
        "version": version,
        "weights": _cpu_copy(weights),
        "epochs": epochs,
        "policy": policy,
    }


def notify() -> dict[str, Any]:
    """Build S->C NOTIFY (SEAFL^2: finish the current epoch and upload now)."""
    return {"type": "NOTIFY"}


def halt() -> dict[str, Any]:
    """Build S->C HALT (Libra: stop training until the next TRAIN)."""
    return {"type": "HALT"}


def shutdown() -> dict[str, Any]:
    """Build S->C SHUTDOWN (client process should exit)."""
    return {"type": "SHUTDOWN"}
