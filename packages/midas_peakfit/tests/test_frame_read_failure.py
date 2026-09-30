"""A frame that cannot be read must never become an empty frame.

Observed 2026-09-27 on copland (FF, /gdata GPFS, threaded producer, 16
threads): ``read_frame`` raised a TRANSIENT ``[Errno 13] Permission denied``
on the layer's .MIDAS.zip for 847 of 1441 frames. The producer caught it,
printed ``Frame N: failed to read (...); skipping`` and returned an empty
result, so the frame was written with 0 peaks, the stage reported
``✓ peakfit`` and the reconstruction came out with 3 grains instead of ~1600.

Contract pinned here:

* a transient OSError is retried, and a frame that succeeds on retry keeps
  every one of its peaks;
* a frame that still cannot be read makes ``run`` raise, before any peak
  file is written, with a message naming the frame -- in BOTH producers.
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path

import numpy as np
import pytest

from midas_peakfit import zarr_io
from midas_peakfit.compat.reference_decoder import read_ps
from midas_peakfit.orchestrator import run

# The frame the synthetic archive plants two peaks in.
_TARGET = 0


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Keep the retry schedule's shape but not its wall time."""
    monkeypatch.setattr(zarr_io, "READ_RETRY_DELAY_S", 0.0, raising=False)


def _flaky_zip_store(monkeypatch, n_failures):
    """Make opening the archive for ``_TARGET`` raise EACCES ``n_failures``
    times, the way GPFS did. Reads of other frames and metadata are untouched.

    The failure is injected at ``zarr.ZipStore`` construction because that is
    where the real EACCES came from (``zipfile`` opening the path), so the
    whole reader -- retry and all -- is exercised, not a stand-in for it.
    """
    real = zarr_io.zarr.ZipStore
    state = {"fail_left": n_failures, "failed": 0}
    armed = threading.local()   # per thread: the producer runs 2 threads

    def fake(path, *a, **kw):
        if getattr(armed, "on", False) and state["fail_left"] > 0:
            state["fail_left"] -= 1
            state["failed"] += 1
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, *a, **kw)

    real_read_frame = zarr_io.read_frame

    def arming_read_frame(path, frameNr):
        # Arm only for the target frame's read, so parse_zarr_params and
        # load_corrections (which also open the store) are not affected.
        armed.on = frameNr == _TARGET
        try:
            return real_read_frame(path, frameNr)
        finally:
            armed.on = False

    monkeypatch.setattr(zarr_io.zarr, "ZipStore", fake)
    import midas_peakfit.orchestrator as orch
    monkeypatch.setattr(orch, "read_frame", arming_read_frame)
    return state


def _run(zip_path: Path, out: Path):
    return run(
        str(zip_path),
        block_nr=0, n_blocks=1, num_procs=2,
        result_folder_cli=str(out),
        device="cpu", dtype="float64",
        producer="thread",
    )


def test_transient_read_failure_is_retried_and_keeps_peaks(
        synthetic_zarr, tmp_path, monkeypatch):
    state = _flaky_zip_store(monkeypatch, n_failures=2)
    summary = _run(synthetic_zarr, tmp_path)
    assert state["failed"] == 2, "the injected failure never fired"
    ps = read_ps(summary["ps_path"])
    assert ps.n_peaks[_TARGET] >= 2, (
        f"frame {_TARGET} came back with {ps.n_peaks[_TARGET]} peaks after a "
        "transient read failure: it was treated as an empty frame"
    )


def test_persistent_read_failure_fails_the_run(
        synthetic_zarr, tmp_path, monkeypatch):
    _flaky_zip_store(monkeypatch, n_failures=10 ** 6)
    with pytest.raises(Exception) as ei:
        _run(synthetic_zarr, tmp_path)
    msg = str(ei.value)
    assert "Permission denied" in msg and str(_TARGET) in msg, msg
    # Nothing that a caller (or midas_pipeline's "already exists; skip" cache)
    # could mistake for a finished peak search.
    assert not (tmp_path / "Temp" / "AllPeaks_PS.bin").exists()
    assert not (tmp_path / "Temp" / "AllPeaks_PX.bin").exists()


# ── The fork producer reads through a cached handle in its own worker. ──────
# macOS has no fork start method for the orchestrator's process pool, so the
# worker function is driven directly with the state its initializer builds.

class _FlakyData:
    def __init__(self, arr, n_failures):
        self.arr, self.fail_left = arr, n_failures

    def __getitem__(self, idx):
        if self.fail_left > 0:
            self.fail_left -= 1
            raise PermissionError(13, "Permission denied", "x.MIDAS.zip")
        return self.arr[idx]


def _init_worker_state(zip_path, tmp_path, n_failures):
    """Build the worker state exactly as orchestrator.run does."""
    from midas_peakfit import _producer_worker as w
    from midas_peakfit.geometry import compute_good_coords, load_ring_radii
    from midas_peakfit.orchestrator import _build_panels
    from midas_peakfit.preprocess import prepare_dark, prepare_flood, prepare_mask
    from midas_peakfit.zarr_io import load_corrections, parse_zarr_params

    p = parse_zarr_params(str(zip_path))
    p.ResultFolder = str(tmp_path)
    panels = _build_panels(p)
    load_corrections(str(zip_path), p)
    good = compute_good_coords(p, panels, load_ring_radii(p, p.ResultFolder))
    args = (p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
    dark, flood, mask = (prepare_dark(p.dark, *args),
                         prepare_flood(p.flood, *args),
                         prepare_mask(p.mask, *args))
    p_pickle = type(p)(**{**p.__dict__, "dark": None, "flood": None,
                          "mask": None, "residualMap": None})
    w.init_worker(str(zip_path), pickle.dumps(p_pickle), dark, flood, mask,
                  good, pickle.dumps(panels))
    real = np.asarray(w._state["data"][...])
    w._state["data"] = _FlakyData(real, n_failures)
    return w


def test_process_worker_retries_a_transient_failure(synthetic_zarr, tmp_path):
    w = _init_worker_state(synthetic_zarr, tmp_path, n_failures=2)
    idx, _om, n_regs, seeded, *_sat = w.process_frame_in_worker(_TARGET)
    assert idx == _TARGET
    assert n_regs >= 2 and len(seeded) >= 2, (
        "a frame that read fine on retry lost its regions"
    )


def test_process_worker_raises_on_persistent_failure(
        synthetic_zarr, tmp_path, monkeypatch):
    w = _init_worker_state(synthetic_zarr, tmp_path, n_failures=10 ** 6)
    # A reopen on retry must not cure a persistent fault by accident: make the
    # reopened handle fail too.
    real_zs = w.zarr.ZipStore

    def failing_zs(path, *a, **kw):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(w.zarr, "ZipStore", failing_zs)
    with pytest.raises(Exception) as ei:
        w.process_frame_in_worker(_TARGET)
    assert "Permission denied" in str(ei.value)
    assert f"(zarr index) {_TARGET} " in str(ei.value), str(ei.value)
    # The exception crosses the ProcessPool boundary by pickling.
    pickle.loads(pickle.dumps(ei.value))
    monkeypatch.setattr(w.zarr, "ZipStore", real_zs)
