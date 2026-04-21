"""Live-feed integration client.

Connects to a :class:`midas_integrate.stream.Server`, feeds frames, gets
integrated 2D cakes back. Thin wrapper around the socket protocol.

Usage
-----
    >>> import midas_integrate as mi
    >>> with mi.stream.Client("127.0.0.1:60439") as c:
    ...     cake = c.send_frame(frame_array)
"""

from __future__ import annotations

import socket
from contextlib import AbstractContextManager
from typing import Optional, Union

import numpy as np

from . import protocol as proto

__all__ = ["Client", "StreamError"]


class StreamError(RuntimeError):
    """Raised when the server reports an error processing a frame."""


class Client(AbstractContextManager):
    """Client for ``midas_integrate.stream.Server``.

    Parameters
    ----------
    endpoint : str
        ``"host:port"`` or just ``"host"`` (defaults port to 60439).
    connect_timeout : float, default 5.0
        Seconds to wait for the initial socket connect.
    op_timeout : float | None, default None
        Per-operation socket timeout. None disables (wait forever).
    """

    def __init__(
        self,
        endpoint: str,
        *,
        connect_timeout: float = 5.0,
        op_timeout: Optional[float] = None,
    ):
        self.host, self.port = _parse_endpoint(endpoint)
        self.connect_timeout = connect_timeout
        self.op_timeout = op_timeout
        self._sock: Optional[socket.socket] = None
        self._seq = 0

    # ------------------------------------------------------------------
    # Context-manager / lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "Client":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.connect_timeout)
        try:
            sock.connect((self.host, self.port))
        except (OSError, socket.timeout) as e:
            sock.close()
            raise ConnectionError(
                f"failed to connect to {self.host}:{self.port} — {e}"
            ) from e
        sock.settimeout(self.op_timeout)
        self._sock = sock

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        except OSError:
            pass
        self._sock = None

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------
    def send_frame(self, frame: np.ndarray) -> np.ndarray:
        """Submit a 2D frame, block until the cake comes back."""
        if self._sock is None:
            raise RuntimeError("client not connected — call .connect() first")
        if frame.ndim != 2:
            raise ValueError(f"frame must be 2D, got {frame.ndim}-D")

        self._seq = (self._seq + 1) & 0xFFFFFFFF
        proto.write_frame(self._sock, proto.FrameMessage(seq=self._seq, frame=frame))

        resp = proto.read_profile(self._sock)
        if resp.seq != self._seq:
            raise proto.ProtocolError(
                f"seq mismatch: sent {self._seq}, got {resp.seq}"
            )
        if resp.status != 0:
            raise StreamError(resp.error or f"server status={resp.status}")
        if resp.cake is None:
            raise StreamError("server returned status=0 with no payload")
        return resp.cake

    def ping(self, *, timeout: float = 2.0) -> float:
        """Send PING, measure round-trip in seconds."""
        if self._sock is None:
            raise RuntimeError("client not connected")
        import time
        old = self._sock.gettimeout()
        try:
            self._sock.settimeout(timeout)
            self._seq = (self._seq + 1) & 0xFFFFFFFF
            t0 = time.perf_counter()
            proto.send_control(self._sock, proto.OP_PING, seq=self._seq)
            from .protocol import recv_exact, _REQ_STRUCT
            header = recv_exact(self._sock, _REQ_STRUCT.size)
            import struct
            _, _, op, _, _, _, _, _ = _REQ_STRUCT.unpack(header)
            if op != proto.OP_PONG:
                raise proto.ProtocolError(f"PING got op={op:#x}, wanted PONG")
            return time.perf_counter() - t0
        finally:
            self._sock.settimeout(old)

    def shutdown_server(self) -> None:
        """Ask the server to stop its accept-loop."""
        if self._sock is None:
            raise RuntimeError("client not connected")
        proto.send_control(self._sock, proto.OP_SHUTDOWN)


def _parse_endpoint(endpoint: str) -> tuple[str, int]:
    if ":" in endpoint:
        host, port_s = endpoint.rsplit(":", 1)
        return host, int(port_s)
    return endpoint, 60439
