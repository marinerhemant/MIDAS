"""v2 must be able to stand in for the C integrator at a beamline.

Two gaps this covers: v2 wrote only CSV (the chain downstream reads
lineout.bin / Int2D.bin), and v2 had no socket server at all, so it could not
sit behind an existing feeder. The socket test drives the server with v1's own
``send_frame`` — if the two ever disagree on the wire format, this fails.
"""
from __future__ import annotations

import socket
import time

import numpy as np
import pytest
import torch

from midas_integrate_v2.binning.subpixel import (
    SubpixelBinGeometry, integrate_subpixel,
)
from midas_integrate_v2.io.v1_outputs import (
    bin_weights, profile_1d_v1, r_axis_from_spec,
    write_int2d_bin, write_lineout_bin, write_v1_outputs,
)
from midas_integrate.params import IntegrationParams
from midas_integrate_v2.compat.from_v1 import spec_from_v1_params


def _spec(NY=48, NZ=48):
    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2.0 + 0.37, BC_z=NZ / 2.0 - 0.41, RhoD=float(NY),
        RMin=1.0, RMax=13.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=30.0,
    )
    return spec_from_v1_params(p, requires_grad=False)


def _frame(spec, seed=0) -> torch.Tensor:
    g = np.random.default_rng(seed)
    return torch.from_numpy(
        g.uniform(10.0, 500.0, (spec.NrPixelsZ, spec.NrPixelsY))).to(torch.float64)


def _geom(spec):
    return SubpixelBinGeometry.from_spec(spec, K=2)


# ── file formats ─────────────────────────────────────────────────────────
def test_lineout_bin_is_interleaved_R_I_float64(tmp_path):
    spec = _spec()
    r = r_axis_from_spec(spec)
    prof = np.linspace(1.0, 2.0, len(r))
    p = write_lineout_bin(tmp_path / "lineout.bin", r, [prof, prof * 3])

    raw = np.fromfile(p, dtype=np.float64)
    assert raw.size == 2 * len(r) * 2                 # 2 frames
    blk = raw.reshape(2, len(r), 2)
    assert np.allclose(blk[0, :, 0], r)               # R column
    assert np.allclose(blk[0, :, 1], prof)
    assert np.allclose(blk[1, :, 1], prof * 3)


def test_int2d_bin_is_transposed_to_v1_orientation(tmp_path):
    spec = _spec()
    geom = _geom(spec)
    cake = integrate_subpixel(_frame(spec), geom, normalize=True)
    assert cake.shape == (spec.n_eta_bins, spec.n_r_bins)   # v2 orientation

    p = write_int2d_bin(tmp_path / "Int2D.bin", [cake])
    raw = np.fromfile(p, dtype=np.float64)
    assert raw.size == spec.n_r_bins * spec.n_eta_bins
    got = raw.reshape(spec.n_r_bins, spec.n_eta_bins)       # v1 orientation
    assert np.allclose(got, cake.numpy().T)


def test_write_v1_outputs_emits_all_three(tmp_path):
    spec = _spec()
    geom = _geom(spec)
    w = bin_weights(geom, integrate_subpixel)
    cake = integrate_subpixel(_frame(spec), geom, normalize=True)
    written = write_v1_outputs(tmp_path, [cake], spec=spec, weights=w)
    names = {p.name for p in written}
    assert names == {"lineout.bin", "lineout_simple_mean.bin", "Int2D.bin"}
    assert all(p.stat().st_size > 0 for p in written)


# ── profile semantics ────────────────────────────────────────────────────
def test_bin_weights_are_image_independent():
    spec = _spec()
    geom = _geom(spec)
    a = bin_weights(geom, integrate_subpixel)
    b = bin_weights(geom, integrate_subpixel)
    assert torch.equal(a, b)
    assert a.shape == (spec.n_eta_bins, spec.n_r_bins)
    assert float(a.sum()) > 0


def test_area_weighted_profile_of_a_flat_cake_is_that_constant():
    spec = _spec()
    geom = _geom(spec)
    w = bin_weights(geom, integrate_subpixel)
    cake = torch.full((spec.n_eta_bins, spec.n_r_bins), 7.0, dtype=torch.float64)
    prof = profile_1d_v1(cake, w, mode="area_weighted")
    occupied = (w > 0).any(dim=0).numpy()
    assert np.allclose(prof[occupied], 7.0)


def test_area_weighted_needs_weights():
    with pytest.raises(ValueError, match="needs weights"):
        profile_1d_v1(torch.zeros(3, 4), None, mode="area_weighted")


# ── the socket server, driven by v1's own client ─────────────────────────
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_accepts_v1_send_frame_and_writes_v1_files(tmp_path):
    from midas_integrate.server import send_frame
    from midas_integrate_v2.streaming.socket_server import V2FrameServer

    spec = _spec()
    geom = _geom(spec)
    port = _free_port()
    srv = V2FrameServer(spec=spec, geom=geom, integrate_fn=integrate_subpixel,
                        out_dir=tmp_path, host="127.0.0.1", port=port).start()
    try:
        img = _frame(spec, seed=3).numpy().astype(np.uint16)
        for n in (1, 2, 3):
            send_frame(np.ascontiguousarray(img), host="127.0.0.1",
                       port=port, dataset_num=n)
        deadline = time.time() + 30
        while len(srv._cakes) < 3 and time.time() < deadline:
            time.sleep(0.05)
        assert len(srv._cakes) == 3, "frames did not arrive over the socket"
    finally:
        written = srv.stop()

    names = {p.name for p in written}
    assert names == {"lineout.bin", "lineout_simple_mean.bin", "Int2D.bin"}

    raw = np.fromfile(tmp_path / "lineout.bin", dtype=np.float64)
    assert raw.size == 3 * spec.n_r_bins * 2
    blk = raw.reshape(3, spec.n_r_bins, 2)
    assert np.allclose(blk[0, :, 0], r_axis_from_spec(spec))
    # same frame three times -> identical profiles
    assert np.allclose(blk[0, :, 1], blk[1, :, 1])
    assert np.allclose(blk[1, :, 1], blk[2, :, 1])
    assert np.any(blk[0, :, 1] > 0)


def test_streamed_result_matches_a_direct_integrate(tmp_path):
    """The socket path must not perturb the numbers."""
    from midas_integrate.server import send_frame
    from midas_integrate_v2.streaming.socket_server import V2FrameServer

    spec = _spec()
    geom = _geom(spec)
    img = _frame(spec, seed=11).numpy().astype(np.uint16)

    direct = integrate_subpixel(torch.from_numpy(img.astype(np.float64)),
                                geom, normalize=True)
    expected = profile_1d_v1(direct, bin_weights(geom, integrate_subpixel),
                             mode="area_weighted")

    port = _free_port()
    srv = V2FrameServer(spec=spec, geom=geom, integrate_fn=integrate_subpixel,
                        out_dir=tmp_path, host="127.0.0.1", port=port).start()
    try:
        send_frame(np.ascontiguousarray(img), host="127.0.0.1", port=port,
                   dataset_num=1)
        deadline = time.time() + 30
        while not srv._cakes and time.time() < deadline:
            time.sleep(0.05)
        assert srv._cakes, "frame did not arrive"
    finally:
        srv.stop()

    got = np.fromfile(tmp_path / "lineout.bin",
                      dtype=np.float64).reshape(spec.n_r_bins, 2)[:, 1]
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-12)
