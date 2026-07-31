"""Regression tests for the peak-fit reproducibility contract.

Background (1-ID GE5 FF scan, Au3_cubes_ff_000008, 2026-07-30): running the
same FF pipeline three times on identical input produced three different
Grains.csv. Checksumming every stage located the first divergence at
``Temp/AllPeaks_PS.bin`` while ``Temp/AllPeaks_PX.bin`` (the raw
connected-pixel sets) was byte-identical — i.e. thresholding and connected
components were deterministic and the batched LM peak FIT was not. Running
``midas_peakfit.orchestrator.run`` twice by itself on one fixed zarr moved
1167 of 8599 peaks.

Two mechanisms, both here:

A. ``RegionPool`` decided the batch quantum from LIVE free VRAM / host
   MemAvailable and re-keyed the cache on the live bucket count, and the
   consumer pulled *every* queued entry rather than one quantum. So which
   regions were solved together depended on machine state and on thread
   scheduling — and batch size selects the cuBLAS/MAGMA kernel, which moves
   the last bits of ``J^T J``.

B. ``lm.py`` set ``torch.backends.cuda.matmul.allow_tf32 = True`` at import,
   process-wide. The design note for that only covers the fp64 path (J cast
   to fp32 for the matmul, Cholesky still fp64); with the FF default
   ``--dtype float32`` it also caught the plain fp32 ``Jt @ J``, assembling
   the normal equations at a 10-bit mantissa.

These tests pin both behaviours. They are pure-CPU and need no GPU.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_peakfit import lm as lm_mod
from midas_peakfit.fit import _pixel_bucket
from midas_peakfit.lm import LMConfig
from midas_peakfit.pool import RegionPool
from midas_peakfit.seeds import SeededRegion

# NOTE: ``_tf32`` is imported lazily inside the tests that need it, not at
# module scope. These tests must be able to RUN against the pre-fix source
# (that is what makes them regression tests); a top-level import of a symbol
# the fix introduces would turn every failure into a collection error.


def _pool(**kw) -> RegionPool:
    return RegionPool(
        device=torch.device("cpu"),
        dtype=torch.float64,
        Ycen=1024.0, Zcen=1024.0,
        do_peak_fit=1, local_maxima_only=0,
        lm_config=LMConfig(),
        log_fn=lambda *_a, **_k: None,
        **kw,
    )


# ── A. batch quantum is decided once, and is a power of two ──────────────

def test_capacity_is_cached_per_shape_not_per_live_bucket_count():
    """The quantum must not move when the bucket population changes.

    Keying the cache on the live bucket count meant every new distinct
    region shape silently re-priced every existing bucket, reshuffling the
    grouping mid-run.
    """
    p = _pool()
    first = p._capacity_for_bucket(3, 128)

    # Populate many other buckets: under the old keying this changed n_live
    # and therefore the answer.
    for n_peaks in range(1, 40):
        p.buckets[(n_peaks, 256)] = [object()]

    assert p._capacity_for_bucket(3, 128) == first
    assert set(p._cached_capacity) == {(3, 128)}, (
        "capacity cache key must be (n_peaks, m_pixels) only"
    )


def test_capacity_is_a_power_of_two():
    """Quantizing absorbs ordinary run-to-run free-memory jitter, so the
    grouping does not change just because the machine was busier."""
    p = _pool()
    for n_peaks, m in [(1, 32), (2, 128), (5, 256), (14, 512), (33, 1024)]:
        cap = p._capacity_for_bucket(n_peaks, m)
        assert cap >= 1
        assert cap == 1 << int(round(math.log2(cap))), (n_peaks, m, cap)


def test_capacity_uses_the_construction_time_memory_snapshot():
    """Re-reading free memory per bucket let another process on the same
    card change our batch quantum."""
    p = _pool()
    snap = p._dev_free_snapshot
    calls = []
    p._free_memory_bytes = lambda: (calls.append(1), snap // 64)[1]
    cap_a = p._capacity_for_bucket(7, 256)
    cap_b = p._capacity_for_bucket(9, 256)
    assert calls == [], "capacity must not re-read live free memory"
    assert cap_a >= 1 and cap_b >= 1


# ── A'. the consumer pulls exactly one quantum ───────────────────────────

def _fake_region(n_peaks: int, n_pixels: int) -> SeededRegion:
    n_par = 1 + 8 * n_peaks
    return SeededRegion(
        region_id=0,
        n_peaks=n_peaks,
        n_pixels=n_pixels,
        raw_sum=float(n_pixels),
        threshold=1.0,
        mask_touched=0,
        maxY=np.zeros(n_peaks, dtype=np.int32),
        maxZ=np.zeros(n_peaks, dtype=np.int32),
        maxima_values=np.ones(n_peaks),
        pixels_y=np.zeros(n_pixels, dtype=np.int32),
        pixels_z=np.zeros(n_pixels, dtype=np.int32),
        z_values=np.ones(n_pixels),
        Rs=np.ones(n_pixels),
        Etas=np.zeros(n_pixels),
        x0=np.zeros(n_par),
        xl=-np.ones(n_par),
        xu=np.ones(n_par),
        peak_R=np.ones(n_peaks),
        peak_Eta=np.zeros(n_peaks),
        peak_M0=np.ones(n_peaks),
        peak_quality=np.zeros(n_peaks, dtype=np.int8),
    )


def test_consumer_flushes_exactly_one_quantum_per_bucket():
    """The race that made the pipeline irreproducible: the consumer used to
    take ``self.buckets[k]`` wholesale, so the chunk boundary landed wherever
    the producer had got to when the consumer was scheduled. Feed it MORE
    than one quantum and assert the flush is still exactly one quantum, in
    insertion order.
    """
    p = _pool()
    flushed: list[list[int]] = []
    p._flush_bucket_entries = lambda key, entries: flushed.append(
        [e.frame_idx for e in entries]
    )

    n_peaks, n_pixels = 2, 100
    key = (n_peaks, _pixel_bucket(n_pixels))
    cap = p._capacity_for_bucket(*key)

    # 2.5 quanta queued before the consumer ever wakes.
    total = int(cap * 2.5)
    for i in range(total):
        p.add_frame(i, 0.0, [_fake_region(n_peaks, n_pixels)])

    p.start()
    p.stop()

    sizes = [len(c) for c in flushed]
    # The pre-fix consumer took the whole queue in one go, so this is the
    # assertion that actually catches the regression: with 2.5 quanta queued
    # there must be 3 flushes of (cap, cap, cap//2), never a single flush of
    # everything.
    assert len(flushed) == 3, sizes
    assert sizes == [cap, cap, total - 2 * cap], sizes
    # Nothing lost, nothing reordered.
    assert [i for c in flushed for i in c] == list(range(total))


def test_drain_also_slices_at_quantum_boundaries():
    """The end-of-stream drain must not take the whole remainder.

    If it did, ``_flush_bucket_entries`` would split that remainder by
    ``_MAX_SOLVE_BATCH`` instead of by the quantum, so for any bucket whose
    quantum is smaller than _MAX_SOLVE_BATCH the tail chunk boundaries would
    depend on how far the consumer had got before the producer finished.
    """
    p = _pool()
    flushed: list[int] = []
    p._flush_bucket_entries = lambda key, entries: flushed.append(len(entries))
    # Force a small quantum so cap < _MAX_SOLVE_BATCH, which is the case the
    # whole-remainder drain got wrong.
    p._cached_capacity[(2, _pixel_bucket(100))] = 8

    for i in range(20):
        p.add_frame(i, 0.0, [_fake_region(2, 100)])
    p.start()
    p.stop()

    assert flushed == [8, 8, 4], flushed
    assert all(n <= 8 for n in flushed), flushed


def test_queued_bytes_returns_to_zero_after_drain():
    """The residency backstop is only meaningful if its accounting is
    balanced; a leak here would fire the non-reproducible partial-flush
    path on a long run."""
    p = _pool()
    p._flush_bucket_entries = lambda key, entries: None
    for i in range(37):
        p.add_frame(i, 0.0, [_fake_region(2, 100)])
    assert p._queued_bytes > 0
    p.start()
    p.stop()
    assert p._queued_bytes == 0


# ── B. TF32 is scoped, not a global import side effect ───────────────────

def test_importing_lm_does_not_enable_tf32_globally():
    """Importing a module must not change the precision of every other fp32
    matmul in the interpreter."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA; the global flag is a no-op here")
    torch.backends.cuda.matmul.allow_tf32 = False
    import importlib
    importlib.reload(lm_mod)
    assert torch.backends.cuda.matmul.allow_tf32 is False


def test_tf32_context_restores_previous_state():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from midas_peakfit.lm import _tf32
    for prev in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = prev
        with _tf32(True):
            assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is prev


def test_tf32_context_is_a_noop_when_disabled():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from midas_peakfit.lm import _tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    with _tf32(False):
        assert torch.backends.cuda.matmul.allow_tf32 is False


def test_fp32_lm_does_not_take_the_tf32_matmul_branch():
    """Source-level guard on the fp64 condition. In an fp32 LM there is no
    fp64 Cholesky downstream to absorb a TF32 J^T J, so the tf32 branch must
    stay gated on ``J.dtype == torch.float64``."""
    import inspect
    src = inspect.getsource(lm_mod.lm_solve)
    i = src.index('config.matmul_precision == "tf32"')
    window = src[i:i + 200]
    assert "J.dtype == torch.float64" in window
    assert "with _tf32(True):" in window


# ── end-to-end: same input twice, same fit ───────────────────────────────

def test_pool_is_bit_reproducible_across_two_identical_runs():
    """The property the whole fix exists for."""
    rng = np.random.default_rng(0)
    regions = []
    for _ in range(300):
        n_pixels = int(rng.integers(20, 60))
        sr = _fake_region(1, n_pixels)
        sr.z_values = rng.random(n_pixels) * 100.0
        sr.Rs = rng.random(n_pixels) * 10.0 + 1.0
        sr.Etas = rng.random(n_pixels) * 10.0
        sr.x0 = rng.random(9)
        sr.xl = sr.x0 - 1.0
        sr.xu = sr.x0 + 1.0
        regions.append(sr)

    def run() -> np.ndarray:
        p = _pool()
        p.start()
        for i, sr in enumerate(regions):
            p.add_frame(i, 0.0, [sr])
        p.stop()
        rows = [fo.rows for f in sorted(p.frame_outputs) for fo in p.frame_outputs[f]]
        return np.concatenate(rows, axis=0) if rows else np.zeros((0, 1))

    a, b = run(), run()
    assert a.shape == b.shape
    np.testing.assert_array_equal(a, b)
