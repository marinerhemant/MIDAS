"""TCP frame server speaking the C ``IntegratorFitPeaksGPUStream`` protocol.

Lets a v2 geometry sit behind an existing beamline feeder unchanged: same
default port (60439), same 4-byte ``(dataset_num, dtype_code)`` header, same
hybrid uint16+overflow encoding (code 6), same three output files.

The wire format is *not* reimplemented here — the decoders are imported from
``midas_integrate``, so the two servers cannot drift apart. What this module
adds is the v2 integrate path and its (n_eta, n_r) → (n_r, n_eta) convention.

    from midas_integrate_v2.binning import SubpixelBinGeometry, integrate_subpixel
    geom = SubpixelBinGeometry.from_spec(spec, K=2)
    srv = V2FrameServer(spec=spec, geom=geom, integrate_fn=integrate_subpixel,
                        out_dir="./out")
    srv.start(); srv.serve_forever()
"""
from __future__ import annotations

import queue
import socket
import socketserver
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from midas_integrate.image import (
    DTYPE_CODE_HYBRID,
    NUMPY_DTYPE_FOR_CODE,
    bytes_per_pixel,
    decode_hybrid_payload,
)
from midas_integrate.server import HEADER_SIZE, DEFAULT_PORT, _read_exactly

from ..io.v1_outputs import (
    bin_weights,
    profile_1d_v1,
    r_axis_from_spec,
    write_int2d_bin,
    write_lineout_bin,
)

__all__ = ["V2FrameServer", "DEFAULT_PORT"]


@dataclass
class _Incoming:
    dataset_num: int
    dtype_code: int
    payload: bytes


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        srv = self.server.frame_server
        sock = self.request
        sock.settimeout(srv.client_timeout)
        try:
            while not srv.shutdown_event.is_set():
                header = _read_exactly(sock, HEADER_SIZE)
                if not header:
                    break
                dataset_num, dtype_code = struct.unpack("<HH", header)
                if dtype_code == DTYPE_CODE_HYBRID:
                    n_over = struct.unpack("<I", _read_exactly(sock, 4))[0]
                    body = _read_exactly(sock, srv.n_pixels * 2 + n_over * (4 + 8))
                    payload = struct.pack("<I", n_over) + body
                else:
                    payload = _read_exactly(
                        sock, srv.n_pixels * bytes_per_pixel(dtype_code))
                srv.frame_queue.put(_Incoming(dataset_num, dtype_code, payload))
        except (ConnectionError, OSError, socket.timeout):
            pass


class _TCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


@dataclass
class V2FrameServer:
    """Serve v2 integration over the C wire protocol.

    Args:
        spec: the v2 ``IntegrationSpec`` (supplies the R axis and bin counts).
        geom: any v2 bin geometry.
        integrate_fn: the matching ``integrate_*`` for that geometry.
        out_dir: where the three v1 files are written on stop.
    """
    spec: object
    geom: object
    integrate_fn: Callable
    out_dir: Path | str = "."
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    queue_size: int = 64
    client_timeout: float = 5.0
    write_2d: bool = True
    write_simple_mean: bool = True
    on_frame: Optional[Callable] = None

    shutdown_event: threading.Event = field(default_factory=threading.Event)
    frame_queue: "queue.Queue" = field(init=False)
    _server: Optional[_TCPServer] = field(default=None, init=False)
    _worker: Optional[threading.Thread] = field(default=None, init=False)
    _weights: Optional[torch.Tensor] = field(default=None, init=False)
    _cakes: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.frame_queue = queue.Queue(maxsize=self.queue_size)

    @property
    def n_pixels(self) -> int:
        return int(self.geom.n_pixels_y) * int(self.geom.n_pixels_z)

    # ── decode + integrate ────────────────────────────────────────────
    def _decode(self, item: _Incoming) -> np.ndarray:
        ny, nz = int(self.geom.n_pixels_y), int(self.geom.n_pixels_z)
        if item.dtype_code == DTYPE_CODE_HYBRID:
            return decode_hybrid_payload(item.payload, n_pixels_y=ny, n_pixels_z=nz)
        dt = np.dtype(NUMPY_DTYPE_FOR_CODE[item.dtype_code])
        # frombuffer aliases the read-only payload; torch.from_numpy on a
        # non-writable array is undefined behaviour, so take a copy.
        return np.frombuffer(item.payload, dtype=dt).reshape(nz, ny).copy()

    def _worker_loop(self):
        while True:
            item = self.frame_queue.get()
            if item is None:
                break
            try:
                arr = self._decode(item)
                img = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float64)
                cake = self.integrate_fn(img, self.geom, normalize=True)
                self._cakes.append(cake.detach().cpu())
                if self.on_frame is not None:
                    self.on_frame(item.dataset_num, cake)
            except Exception as exc:                       # keep serving
                print(f"[v2-server] frame {item.dataset_num} failed: {exc!r}")
            finally:
                self.frame_queue.task_done()

    # ── lifecycle ─────────────────────────────────────────────────────
    def start(self):
        # Image-independent, so pay for it once before the first frame lands
        # rather than inside the hot loop.
        self._weights = bin_weights(self.geom, self.integrate_fn)
        self._worker = threading.Thread(target=self._worker_loop,
                                        name="midas-v2-integrator",
                                        daemon=True)
        self._worker.start()
        self._server = _TCPServer((self.host, self.port), _Handler)
        self._server.frame_server = self
        threading.Thread(target=self._server.serve_forever,
                         name="midas-v2-acceptor", daemon=True).start()
        return self

    def stop(self) -> list[Path]:
        """Stop serving and flush the accumulated frames to the v1 files."""
        self.shutdown_event.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self.frame_queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=30)
        return self.flush()

    def flush(self) -> list[Path]:
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if not self._cakes:
            return []
        r_axis = r_axis_from_spec(self.spec)
        written = [write_lineout_bin(
            out / "lineout.bin", r_axis,
            [profile_1d_v1(c, self._weights, mode="area_weighted")
             for c in self._cakes])]
        if self.write_simple_mean:
            written.append(write_lineout_bin(
                out / "lineout_simple_mean.bin", r_axis,
                [profile_1d_v1(c, mode="simple_mean") for c in self._cakes]))
        if self.write_2d:
            written.append(write_int2d_bin(out / "Int2D.bin", self._cakes))
        return written
