"""Tests for flsim.transport using real localhost sockets (no mocks).

Every blocking call runs in a daemon thread or has a timeout, so a failure cannot hang pytest.
"""

from __future__ import annotations

import pickle
import queue
import threading
import time
from multiprocessing import AuthenticationError
from multiprocessing.connection import Connection

import pytest
import torch

from flsim import transport

KEY = b"test-key"
HOST = "127.0.0.1"


def _state_dict() -> dict[str, torch.Tensor]:
    """Small deterministic state_dict."""
    return {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b": torch.ones(3)}


def test_round_trip_and_byte_count() -> None:
    """An UPDATE survives send/recv bit-exactly and both sides report len(pickle.dumps)."""
    listener = transport.listen(HOST, 0, KEY)
    port = listener.address[1]
    msg = transport.update(3, 7, _state_dict(), 100, 0.5, 2, 1)
    sent: list[int] = []

    def client() -> None:
        conn = transport.connect(HOST, port, KEY)
        sent.append(transport.send(conn, msg))
        conn.close()

    t = threading.Thread(target=client, daemon=True)
    t.start()
    server_conn = listener.accept()
    got, nbytes = transport.recv(server_conn)
    t.join(5)
    listener.close()

    assert nbytes > 0
    assert nbytes == len(pickle.dumps(msg)) == sent[0]
    assert {k: v for k, v in got.items() if k != "weights"} == {
        k: v for k, v in msg.items() if k != "weights"
    }
    for k, tensor in msg["weights"].items():
        assert torch.equal(got["weights"][k], tensor)


def test_connect_retries_until_server_listens() -> None:
    """connect() started before the listener exists still succeeds once it appears."""
    # Reserve a free port, release it, and only re-bind after a delay.
    probe = transport.listen(HOST, 0, KEY)
    port = probe.address[1]
    probe.close()
    holder: list[object] = []

    def late_server() -> None:
        time.sleep(0.3)
        lst = transport.listen(HOST, port, KEY)
        holder.append(lst)
        holder.append(lst.accept())

    t = threading.Thread(target=late_server, daemon=True)
    t.start()
    conn = transport.connect(HOST, port, KEY, retries=50, wait_s=0.05)
    t.join(5)
    assert isinstance(conn, Connection)
    conn.close()
    holder[0].close()  # type: ignore[attr-defined]


def test_connect_gives_up_with_clear_error() -> None:
    """With nothing listening, connect() raises ConnectionError after its retries."""
    probe = transport.listen(HOST, 0, KEY)
    port = probe.address[1]
    probe.close()
    with pytest.raises(ConnectionError, match="could not connect"):
        transport.connect(HOST, port, KEY, retries=2, wait_s=0.01)


def test_wrong_authkey_rejected() -> None:
    """A client with the wrong authkey fails the HMAC handshake; neither side hangs."""
    listener = transport.listen(HOST, 0, KEY)
    port = listener.address[1]
    server_errors: list[BaseException] = []

    def server() -> None:
        try:
            listener.accept()
        except BaseException as exc:  # noqa: BLE001 - we only record what the server saw
            server_errors.append(exc)

    t = threading.Thread(target=server, daemon=True)
    t.start()
    with pytest.raises(AuthenticationError):
        transport.connect(HOST, port, b"wrong-key")
    t.join(5)
    assert not t.is_alive()
    listener.close()


def test_accept_clients_and_readers() -> None:
    """3 clients HELLO + one message each, then close: queue gets 3 msgs and 3 DISCONNECTs."""
    listener = transport.listen(HOST, 0, KEY)
    port = listener.address[1]

    def client(cid: int) -> None:
        conn = transport.connect(HOST, port, KEY)
        transport.send(conn, transport.hello(cid))
        transport.send(conn, transport.update(cid, 0, _state_dict(), 10, 0.1, 1, 0))
        conn.close()

    for cid in (5, 2, 9):
        threading.Thread(target=client, args=(cid,), daemon=True).start()

    conns = transport.accept_clients(listener, 3)
    assert sorted(conns) == [2, 5, 9]
    q: queue.Queue = queue.Queue()
    threads = transport.start_readers(conns, q)
    for th in threads:
        th.join(5)
    listener.close()

    items = [q.get(timeout=5) for _ in range(6)]
    assert q.empty()
    updates = sorted(cid for cid, m, _ in items if m["type"] == "UPDATE")
    discs = sorted(cid for cid, m, n in items if m["type"] == "DISCONNECT" and n == 0)
    assert updates == [2, 5, 9]
    assert discs == [2, 5, 9]
    for cid, m, _ in items:
        assert m["cid"] == cid


def test_accept_clients_rejects_bad_first_message_and_duplicates() -> None:
    """A non-HELLO first message or a repeated cid raises ValueError."""
    listener = transport.listen(HOST, 0, KEY)
    port = listener.address[1]

    def bad_client() -> None:
        conn = transport.connect(HOST, port, KEY)
        transport.send(conn, transport.halt())

    threading.Thread(target=bad_client, daemon=True).start()
    with pytest.raises(ValueError, match="HELLO"):
        transport.accept_clients(listener, 1)

    keep: list[Connection] = []

    def dup_client() -> None:
        conn = transport.connect(HOST, port, KEY)
        keep.append(conn)  # keep the socket alive until the test ends
        transport.send(conn, transport.hello(1))

    for _ in range(2):
        threading.Thread(target=dup_client, daemon=True).start()
    with pytest.raises(ValueError, match="duplicate"):
        transport.accept_clients(listener, 2)
    listener.close()


def test_constructors_match_contracts() -> None:
    """Constructors produce exactly the field sets in the CLAUDE.md Contracts table."""
    sd = _state_dict()
    assert set(transport.hello(1)) == {"type", "cid"}
    assert set(transport.update(1, 0, sd, 5, 0.1, 1, 0)) == {
        "type",
        "cid",
        "base_version",
        "weights",
        "n_samples",
        "loss",
        "epochs_done",
        "upload_idx",
    }
    assert set(transport.train(1, sd, 2)) == {"type", "version", "weights", "epochs", "policy"}
    assert transport.train(1, sd, 2)["policy"] is None
    assert transport.notify() == {"type": "NOTIFY"}
    assert transport.halt() == {"type": "HALT"}
    assert transport.shutdown() == {"type": "SHUTDOWN"}


def test_weights_are_detached_cpu_copies() -> None:
    """train()/update() weights are CPU, detached, and do not alias the source tensors."""
    p = torch.ones(3, requires_grad=True)
    sd = {"p": p}
    for msg in (transport.train(0, sd, 1), transport.update(0, 0, sd, 1, 0.0, 1, 0)):
        w = msg["weights"]["p"]
        assert w.device.type == "cpu"
        assert not w.requires_grad
        assert w.data_ptr() != p.data_ptr()
