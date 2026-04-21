"""Stream server + client loopback: protocol + end-to-end frame submission.

Two tiers:
  * Pure-protocol tests run everywhere (no binaries, no network).
  * Loopback tests start a Server on 127.0.0.1 and hit it with a Client;
    these need the MIDAS binaries installed (same gate as test_integrate_ceo2).
"""

from __future__ import annotations

import io
import socket
import time
from pathlib import Path

import numpy as np
import pytest

from midas_integrate import (
    IntegrationConfig,
    MapArtifacts,
    Mapper,
    MidasBinaryNotFoundError,
    make_zarr_zip,
    midas_bin,
    stream,
)
from midas_integrate.stream import protocol as proto


# ---------------------------------------------------------------------------
# Protocol — tested via in-memory byte buffers; no sockets needed.
# ---------------------------------------------------------------------------

class _MockSocket:
    """Minimal socket stand-in backed by a single ``BytesIO`` buffer.

    `sendall` appends to the buffer; `recv` reads from the current head.
    Good enough to exercise FrameMessage / ProfileMessage round-trips.
    """
    def __init__(self):
        self._buf = bytearray()
        self._head = 0

    def sendall(self, data):
        self._buf.extend(data)

    def recv(self, n):
        # Block forever is wrong for tests; just return what we have.
        chunk = bytes(self._buf[self._head:self._head + n])
        self._head += len(chunk)
        return chunk

    def sendmsg(self, buffers):   # noqa: D401 — match socket API
        total = 0
        for b in buffers:
            self._buf.extend(b)
            total += len(b)
        return total


class TestFrameMessageRoundTrip:
    def test_uint16_frame(self):
        frame = np.arange(64 * 48, dtype=np.uint16).reshape(64, 48)
        sock = _MockSocket()
        proto.write_frame(sock, proto.FrameMessage(seq=42, frame=frame))

        msg = proto.read_request(sock)
        assert isinstance(msg, proto.FrameMessage)
        assert msg.seq == 42
        np.testing.assert_array_equal(msg.frame, frame)
        assert msg.frame.dtype == np.uint16

    def test_float32_frame(self):
        rng = np.random.default_rng(0)
        frame = rng.uniform(0, 1000, (32, 32)).astype(np.float32)
        sock = _MockSocket()
        proto.write_frame(sock, proto.FrameMessage(seq=7, frame=frame))

        msg = proto.read_request(sock)
        assert isinstance(msg, proto.FrameMessage)
        np.testing.assert_array_equal(msg.frame, frame)

    def test_bad_magic_raises(self):
        sock = _MockSocket()
        # Header-sized payload with zeroed magic bytes.
        sock.sendall(b"\x00" * proto._REQ_STRUCT.size)
        with pytest.raises(proto.ProtocolError, match="bad magic"):
            proto.read_request(sock)

    def test_reject_3d_frame(self):
        frame = np.zeros((2, 3, 4), dtype=np.uint16)
        sock = _MockSocket()
        with pytest.raises(ValueError, match="frame must be 2D"):
            proto.write_frame(sock, proto.FrameMessage(seq=0, frame=frame))


class TestProfileMessageRoundTrip:
    def test_ok_cake(self):
        cake = np.arange(10 * 20, dtype=np.float32).reshape(10, 20)
        sock = _MockSocket()
        proto.write_profile(
            sock, proto.ProfileMessage(seq=5, cake=cake, status=0)
        )

        msg = proto.read_profile(sock)
        assert msg.seq == 5
        assert msg.status == 0
        np.testing.assert_array_equal(msg.cake, cake)

    def test_error_status_payload(self):
        sock = _MockSocket()
        proto.write_profile(
            sock,
            proto.ProfileMessage(
                seq=99, cake=None, status=3, error="integration failed",
            ),
        )
        msg = proto.read_profile(sock)
        assert msg.seq == 99
        assert msg.status == 3
        assert msg.error == "integration failed"
        assert msg.cake is None


class TestControlMessages:
    def test_ping_pong_round_trip(self):
        sock = _MockSocket()
        proto.send_control(sock, proto.OP_PING, seq=11)
        msg = proto.read_request(sock)
        assert isinstance(msg, proto.ControlMessage)
        assert msg.seq == 11
        assert msg.op == proto.OP_PING

    def test_shutdown(self):
        sock = _MockSocket()
        proto.send_control(sock, proto.OP_SHUTDOWN, seq=3)
        msg = proto.read_request(sock)
        assert isinstance(msg, proto.ControlMessage)
        assert msg.op == proto.OP_SHUTDOWN


# ---------------------------------------------------------------------------
# Client endpoint parsing
# ---------------------------------------------------------------------------

class TestEndpointParsing:
    def test_host_port(self):
        from midas_integrate.stream.client import _parse_endpoint
        assert _parse_endpoint("10.0.0.1:9000") == ("10.0.0.1", 9000)

    def test_default_port(self):
        from midas_integrate.stream.client import _parse_endpoint
        assert _parse_endpoint("example.beamline") == ("example.beamline", 60439)


# ---------------------------------------------------------------------------
# Server safety: non-loopback host must be opt-in.
# ---------------------------------------------------------------------------

class TestServerSafety:
    def test_rejects_non_loopback_by_default(self):
        cfg = IntegrationConfig(nr_pixels_y=64, nr_pixels_z=64)
        bogus = MapArtifacts(
            work_dir=Path("."),
            map_bin=Path("Map.bin"), n_map_bin=Path("nMap.bin"),
        )
        with pytest.raises(ValueError, match="listen_all_allowed=False"):
            stream.Server(cfg, bogus, host="0.0.0.0")

    def test_accepts_non_loopback_with_opt_in(self):
        cfg = IntegrationConfig(nr_pixels_y=64, nr_pixels_z=64)
        bogus = MapArtifacts(
            work_dir=Path("."),
            map_bin=Path("Map.bin"), n_map_bin=Path("nMap.bin"),
        )
        # Constructor succeeds — we don't actually bind in __init__.
        stream.Server(cfg, bogus, host="0.0.0.0", listen_all_allowed=True)


# ---------------------------------------------------------------------------
# Loopback end-to-end: server + client against bundled CeO2 Pilatus.
# ---------------------------------------------------------------------------

def _binaries_available() -> bool:
    try:
        midas_bin("MIDASDetectorMapper")
        midas_bin("MIDASIntegrator")
        return True
    except MidasBinaryNotFoundError:
        return False


def _free_port() -> int:
    """Ask the OS for a free TCP port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(not _binaries_available(),
                    reason="MIDASDetectorMapper + MIDASIntegrator not discoverable.")
def test_server_round_trips_real_frame(tmp_path):
    """Server integrates a frame, client gets back a real cake."""
    from midas_auto_calibrate import data as mac_data
    import tifffile

    if not mac_data.CEO2_PILATUS.exists():
        pytest.skip("bundled CeO2 data missing")

    # Build a map first (reused across frames).
    cfg = IntegrationConfig(
        lsd=657_436.9, ybc=685.5, zbc=921.0,
        wavelength=0.172973, pixel_size=172.0,
        nr_pixels_y=1475, nr_pixels_z=1679,
        r_bin_size=1.0, eta_bin_size=4.0,     # coarser bins = quicker
        r_min=50, r_max=1000,
    )
    artifacts = Mapper(cfg).build(tmp_path, n_cpus=2)

    frame = tifffile.imread(mac_data.CEO2_PILATUS).astype(np.uint32)

    port = _free_port()
    with stream.Server(cfg, artifacts, port=port) as srv:
        with stream.Client(f"127.0.0.1:{port}", op_timeout=120.0) as c:
            cake = c.send_frame(frame)
    # MIDAS stores IntegrationResult as (nR, nEta): with these knobs,
    # nR = (RMax-RMin)/RBinSize = 950 and nEta = 360/4 = 90.
    assert cake.ndim == 2
    n_r_expected = int((cfg.r_max - cfg.r_min) / cfg.r_bin_size)
    n_eta_expected = int((cfg.eta_max - cfg.eta_min) / cfg.eta_bin_size)
    assert cake.shape == (n_r_expected, n_eta_expected), (
        f"unexpected shape {cake.shape}, want "
        f"({n_r_expected}, {n_eta_expected})"
    )
    assert np.isfinite(cake).all()
    assert cake.max() > 100, "cake intensity suspiciously low"


@pytest.mark.skipif(not _binaries_available(),
                    reason="MIDASDetectorMapper + MIDASIntegrator not discoverable.")
def test_server_handles_multiple_sequential_frames(tmp_path):
    """One client session, N frames, each gets its own cake back."""
    cfg = IntegrationConfig(
        lsd=1_000_000, ybc=512, zbc=512,
        wavelength=0.172973, pixel_size=200.0,
        nr_pixels_y=1024, nr_pixels_z=1024,
        r_bin_size=2.0, eta_bin_size=5.0,
        r_min=50, r_max=500,
    )
    artifacts = Mapper(cfg).build(tmp_path, n_cpus=2)

    port = _free_port()
    rng = np.random.default_rng(0)
    with stream.Server(cfg, artifacts, port=port) as srv:
        with stream.Client(f"127.0.0.1:{port}", op_timeout=120.0) as c:
            # Three synthetic uniform-noise frames; just checking plumbing.
            shapes = []
            for i in range(3):
                frame = rng.integers(0, 100, (1024, 1024)).astype(np.uint16)
                cake = c.send_frame(frame)
                shapes.append(cake.shape)
        # All three returned the same binning shape.
        assert len(set(shapes)) == 1, f"inconsistent cake shapes: {shapes}"
