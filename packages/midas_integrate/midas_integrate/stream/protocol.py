"""Wire protocol for the live-feed integration server.

Frame layout (all little-endian, fixed-size headers):

    Client → Server:
        ─────────────────────────────────────────────────────────
        | MAGIC (4B) | version (1B) | op (1B) | seq (4B)        |
        | ny (4B) | nz (4B) | dtype_code (1B) | reserved (3B)   |
        | payload (ny * nz * dtype_size bytes)                   |
        ─────────────────────────────────────────────────────────

    Server → Client:
        ─────────────────────────────────────────────────────────
        | MAGIC (4B) | version (1B) | op (1B) | seq (4B)        |
        | n_r (4B) | n_eta (4B) | status (1B) | reserved (3B)   |
        | payload (n_r * n_eta * 4 bytes, float32)  on status=0 |
        | error_string (length-prefix uint16 + bytes) on status!=0
        ─────────────────────────────────────────────────────────

Ops:
    0x01  FRAME        — client submits a raw 2D frame to integrate
    0x02  PROFILE      — server returns the 2D cake for that frame
    0x03  PING         — client health check; server replies with PONG
    0x04  PONG
    0x05  SHUTDOWN     — graceful server stop (loopback only)

dtype_code:
    0 uint16, 1 uint32, 2 int32, 3 int64, 4 float32, 5 float64

status:
    0 OK, non-zero = error (payload is a UTF-8 error string).
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

MAGIC = b"MIDS"                       # 4-byte protocol tag
VERSION = 1

OP_FRAME = 0x01
OP_PROFILE = 0x02
OP_PING = 0x03
OP_PONG = 0x04
OP_SHUTDOWN = 0x05

_REQ_STRUCT = struct.Struct("<4sBB I II B 3s")
_RESP_STRUCT = struct.Struct("<4sBB I II B 3s")

_DTYPE_TO_CODE: dict[np.dtype, int] = {
    np.dtype(np.uint16): 0,
    np.dtype(np.uint32): 1,
    np.dtype(np.int32): 2,
    np.dtype(np.int64): 3,
    np.dtype(np.float32): 4,
    np.dtype(np.float64): 5,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}


class ProtocolError(RuntimeError):
    """Raised when wire bytes don't match the expected layout."""


@dataclass
class FrameMessage:
    seq: int
    frame: np.ndarray           # 2D (ny, nz)

    def to_bytes(self) -> bytes:
        arr = np.ascontiguousarray(self.frame)
        if arr.ndim != 2:
            raise ValueError(f"frame must be 2D, got {arr.ndim}-D")
        code = _DTYPE_TO_CODE.get(arr.dtype)
        if code is None:
            raise ValueError(f"unsupported dtype: {arr.dtype}")
        ny, nz = arr.shape
        header = _REQ_STRUCT.pack(
            MAGIC, VERSION, OP_FRAME, self.seq, ny, nz, code, b"\0\0\0",
        )
        return header + arr.tobytes()


@dataclass
class ProfileMessage:
    seq: int
    cake: Optional[np.ndarray]  # (n_eta, n_r) float32, None on error
    status: int = 0
    error: str = ""

    def to_bytes(self) -> bytes:
        if self.status == 0 and self.cake is not None:
            arr = np.ascontiguousarray(self.cake, dtype=np.float32)
            n_eta, n_r = arr.shape
            header = _RESP_STRUCT.pack(
                MAGIC, VERSION, OP_PROFILE, self.seq,
                n_r, n_eta, self.status, b"\0\0\0",
            )
            return header + arr.tobytes()
        # Error path: zero-sized payload, then length-prefixed error string.
        err_bytes = self.error.encode("utf-8")
        header = _RESP_STRUCT.pack(
            MAGIC, VERSION, OP_PROFILE, self.seq,
            0, 0, self.status, b"\0\0\0",
        )
        return header + struct.pack("<H", len(err_bytes)) + err_bytes


# ---------------------------------------------------------------------------
# Socket I/O helpers — every read is bounded so a truncated stream raises
# ProtocolError rather than hanging.
# ---------------------------------------------------------------------------

def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes or raise ProtocolError on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ProtocolError(
                f"peer closed connection after {len(buf)}/{n} bytes"
            )
        buf.extend(chunk)
    return bytes(buf)


def read_request(sock: socket.socket) -> "FrameMessage | ControlMessage":
    """Read one client→server message: a FRAME (with payload) or a control op."""
    header = recv_exact(sock, _REQ_STRUCT.size)
    magic, version, op, seq, ny, nz, code, _ = _REQ_STRUCT.unpack(header)
    _check_header(magic, version, op, expected_op=None)

    if op == OP_FRAME:
        dtype = _CODE_TO_DTYPE.get(code)
        if dtype is None:
            raise ProtocolError(f"unknown dtype_code={code}")
        n_bytes = int(ny) * int(nz) * dtype.itemsize
        payload = recv_exact(sock, n_bytes)
        frame = np.frombuffer(payload, dtype=dtype).reshape(ny, nz)
        return FrameMessage(seq=int(seq), frame=frame)

    return ControlMessage(seq=int(seq), op=int(op))


@dataclass
class ControlMessage:
    """A non-FRAME message (PING / PONG / SHUTDOWN)."""
    seq: int
    op: int


def write_profile(sock: socket.socket, msg: ProfileMessage) -> None:
    sock.sendall(msg.to_bytes())


def read_profile(sock: socket.socket) -> ProfileMessage:
    header = recv_exact(sock, _RESP_STRUCT.size)
    magic, version, op, seq, n_r, n_eta, status, _ = _RESP_STRUCT.unpack(header)
    _check_header(magic, version, op, expected_op=OP_PROFILE)

    if status == 0:
        n_bytes = int(n_r) * int(n_eta) * 4
        payload = recv_exact(sock, n_bytes)
        cake = np.frombuffer(payload, dtype=np.float32).reshape(n_eta, n_r)
        return ProfileMessage(seq=int(seq), cake=cake, status=0)

    # Error path.
    err_len_bytes = recv_exact(sock, 2)
    (err_len,) = struct.unpack("<H", err_len_bytes)
    err_str = recv_exact(sock, err_len).decode("utf-8")
    return ProfileMessage(seq=int(seq), cake=None, status=int(status), error=err_str)


def write_frame(sock: socket.socket, msg: FrameMessage) -> None:
    sock.sendall(msg.to_bytes())


def send_control(sock: socket.socket, op: int, seq: int = 0) -> None:
    """Send a control-only message (PING / PONG / SHUTDOWN)."""
    header = _REQ_STRUCT.pack(
        MAGIC, VERSION, op, seq, 0, 0, 0, b"\0\0\0",
    )
    sock.sendall(header)


def _check_header(magic: bytes, version: int, op: int, *, expected_op: int | None) -> None:
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: expected {MAGIC!r}, got {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported version {version} (need {VERSION})")
    if expected_op is not None and op != expected_op:
        raise ProtocolError(f"unexpected op {op:#x}, wanted {expected_op:#x}")
