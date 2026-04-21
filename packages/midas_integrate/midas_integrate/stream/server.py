"""Live-feed integration server.

Single-connection socket server: accepts a stream of frames over TCP,
runs ``Integrator`` on each, sends the 2D cake back. MSV — one connection
at a time, one worker. Multi-client + GPU backend land in v0.2.

Usage
-----
As a context manager (the typical pattern):

    >>> with Server(config, map_artifacts, port=60439) as srv:
    ...     srv.serve_forever(timeout_s=60)

Or with a manual thread:

    >>> srv = Server(config, map_artifacts, port=60439)
    >>> srv.start()          # background thread
    >>> ...                  # push frames from elsewhere
    >>> srv.shutdown()

Security: the server binds to ``127.0.0.1`` by default. Pass
``host="0.0.0.0"`` with eyes open — there's no auth in v0.1.0.
"""

from __future__ import annotations

import logging
import socket
import tempfile
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Optional

import numpy as np

from ..integrate import Integrator
from ..io import make_zarr_zip
from ..mapper import MapArtifacts
from .._config import IntegrationConfig
from . import protocol as proto

logger = logging.getLogger(__name__)

__all__ = ["Server"]


class Server(AbstractContextManager):
    """Background-thread integration server.

    Parameters
    ----------
    config : IntegrationConfig
        Same config used to build ``map_artifacts``.
    map_artifacts : MapArtifacts
        Pre-built Map.bin / nMap.bin from :meth:`Mapper.build`.
    host : str, default "127.0.0.1"
        Interface to bind. Only change when firewall / ACL is in place.
    port : int, default 60439
        TCP port (same as the GPU streaming default).
    backend : "cpu" | "gpu", default "cpu"
    """

    def __init__(
        self,
        config: IntegrationConfig,
        map_artifacts: MapArtifacts,
        *,
        host: str = "127.0.0.1",
        port: int = 60439,
        backend: str = "cpu",
        listen_all_allowed: bool = False,
    ):
        if host != "127.0.0.1" and not listen_all_allowed:
            raise ValueError(
                f"host={host!r} is non-loopback but listen_all_allowed=False. "
                f"v0.1.0 has no auth; only pass listen_all_allowed=True when "
                f"running behind a firewall / on a trusted network."
            )
        self.config = config
        self.map_artifacts = map_artifacts
        self.host = host
        self.port = port
        self.backend = backend
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._ready_event = threading.Event()

    # ------------------------------------------------------------------
    # Context-manager + thread lifecycle
    # ------------------------------------------------------------------
    def __enter__(self) -> "Server":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def start(self, *, wait_ready: bool = True, ready_timeout: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Server already running")
        self._thread = threading.Thread(
            target=self._accept_loop, name=f"midas-integrate-server:{self.port}",
            daemon=True,
        )
        self._thread.start()
        if wait_ready and not self._ready_event.wait(ready_timeout):
            raise TimeoutError(
                f"server did not start within {ready_timeout} s"
            )

    def shutdown(self, *, join_timeout: float = 5.0) -> None:
        self._shutdown_event.set()
        # Kick the accept-loop out of accept() by connecting to ourselves,
        # which is safe even if the listener already closed.
        if self._sock is not None:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as c:
                    c.settimeout(0.5)
                    c.connect((self.host, self.port))
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(join_timeout)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def serve_forever(self, *, timeout_s: float | None = None) -> None:
        """Block the current thread until ``shutdown()`` or timeout."""
        self.start()
        try:
            self._shutdown_event.wait(timeout_s)
        finally:
            self.shutdown()

    # ------------------------------------------------------------------
    # Accept / client-handling
    # ------------------------------------------------------------------
    def _accept_loop(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.bind((self.host, self.port))
        except OSError as e:
            logger.error("server bind failed on %s:%d — %s",
                         self.host, self.port, e)
            self._ready_event.set()  # so start()'s wait doesn't hang forever
            return
        self._sock.listen(1)
        self._sock.settimeout(0.5)
        self._ready_event.set()
        logger.info("server listening on %s:%d", self.host, self.port)

        while not self._shutdown_event.is_set():
            try:
                conn, addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            logger.info("client connected from %s", addr)
            with conn:
                self._handle_client(conn)
            logger.info("client disconnected")

    def _handle_client(self, conn: socket.socket) -> None:
        """Process frames from one connection until peer closes or SHUTDOWN."""
        integrator = Integrator(
            self.config, self.map_artifacts, backend=self.backend,
        )
        # One temp work-dir per client; cleanup on disconnect.
        with tempfile.TemporaryDirectory(prefix="midas-stream-") as tmp:
            tmp_path = Path(tmp)
            # Copy the Map.bin/nMap.bin into the work dir so the Integrator
            # subprocess finds them.
            import shutil as _sh
            _sh.copy(self.map_artifacts.map_bin, tmp_path / "Map.bin")
            _sh.copy(self.map_artifacts.n_map_bin, tmp_path / "nMap.bin")

            while not self._shutdown_event.is_set():
                try:
                    msg = proto.read_request(conn)
                except proto.ProtocolError as e:
                    logger.info("protocol error: %s", e)
                    break

                if isinstance(msg, proto.FrameMessage):
                    self._handle_frame(conn, msg, integrator, tmp_path)
                elif isinstance(msg, proto.ControlMessage):
                    if msg.op == proto.OP_PING:
                        proto.send_control(conn, proto.OP_PONG, seq=msg.seq)
                    elif msg.op == proto.OP_SHUTDOWN:
                        logger.info("client requested shutdown")
                        self._shutdown_event.set()
                        break
                    # PONG / unknown → ignore
                else:
                    logger.info("unknown message type: %r", msg)

    def _handle_frame(
        self,
        conn: socket.socket,
        msg: proto.FrameMessage,
        integrator: Integrator,
        work_dir: Path,
    ) -> None:
        """Integrate a single frame, send the cake back (or an error)."""
        zarr_zip = work_dir / f"frame_{msg.seq:06d}.zarr.zip"
        try:
            make_zarr_zip(msg.frame, self.config, zarr_zip)
            result = integrator.integrate(zarr_zip, n_cpus=1)
            cake = result.load_cake()
            I = np.asarray(cake["I"], dtype=np.float32)
            if I.ndim == 3:          # (n_frames, n_eta, n_r)
                I = I[0]
            reply = proto.ProfileMessage(seq=msg.seq, cake=I, status=0)
        except Exception as e:       # noqa: BLE001 — best-effort report-and-continue
            logger.exception("integration failed for seq=%d", msg.seq)
            reply = proto.ProfileMessage(
                seq=msg.seq, cake=None, status=1, error=str(e),
            )
        finally:
            # Clean up the ephemeral zarr.zip + any caked.hdf it produced.
            for p in (zarr_zip,
                      zarr_zip.with_name(zarr_zip.stem + ".caked.hdf")):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        proto.write_profile(conn, reply)
