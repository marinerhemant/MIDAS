"""End-to-end parity tests for the unified C indexer ``midas_indexer``.

Skipped at module level when:
- :func:`backend_c.available` returns False (binary not built), OR
- The required fixture binaries (``Data.bin`` / ``nData.bin``) are absent
  (they're gitignored on size grounds and rebuilt locally by helpers).

All tests run single-threaded (``OMP_NUM_THREADS=1``) for FP-deterministic
parity. PF runs take ~70s on the 5-grain fixture, so the whole module is
marked ``@pytest.mark.slow``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from midas_index import backend_c

pytestmark = pytest.mark.slow

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
PF_FIXTURE = DATA_DIR / "scanning_5grain_golden"
FF_FIXTURE = DATA_DIR / "ref_dataset_unified"


def _pf_fixture_ready() -> bool:
    return (
        PF_FIXTURE.is_dir()
        and (PF_FIXTURE / "Spots.bin").is_file()
        and (PF_FIXTURE / "Data.bin").is_file()
        and (PF_FIXTURE / "nData.bin").is_file()
        and (PF_FIXTURE / "golden" / "IndexBest_all.bin").is_file()
    )


def _ff_fixture_ready() -> bool:
    return (
        FF_FIXTURE.is_dir()
        and (FF_FIXTURE / "Spots.bin").is_file()
        and (FF_FIXTURE / "Data.bin").is_file()
        and (FF_FIXTURE / "nData.bin").is_file()
        and (FF_FIXTURE / "golden" / "IndexBest.bin").is_file()
    )


def _read_consolidated_header(path: Path):
    with path.open("rb") as f:
        raw = f.read()
    n_vox = int.from_bytes(raw[:4], "little", signed=True)
    n_per = np.frombuffer(raw[4 : 4 + 4 * n_vox], dtype=np.int32)
    offsets = np.frombuffer(
        raw[4 + 4 * n_vox : 4 + 12 * n_vox], dtype=np.int64
    )
    header_size = 4 + 12 * n_vox
    return n_vox, n_per, offsets, raw, header_size


def _run_midas_indexer(
    paramstest_src: Path,
    *,
    out_dir: Path,
    n_work: int,
    extra_lines: str = "",
) -> subprocess.CompletedProcess[bytes]:
    """Build a tmp staging dir that mirrors the fixture (via symlinks for
    the binaries + copies for the small CSVs), then invoke midas_indexer
    single-threaded. OutputFolder is staging_dir/Output so the binary's
    dirname(OutputFolder) → staging_dir → finds the symlinked inputs."""
    fixture_dir = paramstest_src.parent
    staging = out_dir
    staging.mkdir(parents=True, exist_ok=True)
    output_subdir = staging / "Output"
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Symlink fixture inputs into staging so the binary can mmap them.
    for fname in ("Spots.bin", "Data.bin", "nData.bin"):
        src = fixture_dir / fname
        if not src.is_file():
            continue
        dst = staging / fname
        if not dst.exists():
            os.symlink(src, dst)
    # Copy the small text fixtures (binary reads positions.csv / hkls.csv
    # from cwd, NOT dirname(OutputFolder) — so they live in `cwd`).
    for fname in ("positions.csv", "hkls.csv", "SpotsToIndex.csv"):
        src = fixture_dir / fname
        if src.is_file() and not (staging / fname).exists():
            shutil.copy2(src, staging / fname)

    # Rewrite paramstest.txt with OutputFolder = staging/Output, append
    # any extra param keys the caller provided.
    pp = staging / "paramstest.txt"
    txt = paramstest_src.read_text()
    new_lines = []
    for line in txt.splitlines():
        if line.startswith("OutputFolder "):
            new_lines.append(f"OutputFolder {output_subdir}")
        else:
            new_lines.append(line)
    if extra_lines:
        new_lines.extend(extra_lines.strip().splitlines())
    pp.write_text("\n".join(new_lines) + "\n")

    return backend_c.run_indexer(
        pp,
        n_work=n_work,
        num_procs=1,
        extra_env={"OMP_NUM_THREADS": "1"},
        cwd=staging,
    )


# ---------------------------------------------------------------------------
# PF parity (bit-identical with legacy IndexerScanningOMP)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not backend_c.available(),
                    reason="midas_indexer C binary not built")
@pytest.mark.skipif(not _pf_fixture_ready(),
                    reason=f"PF fixture binaries not present at {PF_FIXTURE} "
                           "(Data.bin / nData.bin gitignored; regen via "
                           "build.py)")
def test_pf_indexbest_all_bit_identical_with_legacy_golden(tmp_path):
    """PF mode of midas_indexer should produce IndexBest_all.bin
    BIT-IDENTICAL with the legacy IndexerScanningOMP golden, with
    SoftAttrMode=0 (default)."""
    out_dir = tmp_path / "pf"
    proc = _run_midas_indexer(
        PF_FIXTURE / "paramstest.txt", out_dir=out_dir, n_work=15,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    produced = out_dir / "Output" / "IndexBest_all.bin"
    golden = PF_FIXTURE / "golden" / "IndexBest_all.bin"
    assert produced.is_file()
    assert produced.read_bytes() == golden.read_bytes(), (
        "PF IndexBest_all.bin diverges from legacy golden"
    )


# ---------------------------------------------------------------------------
# FF tolerance (matches legacy IndexerOMP within fp tolerance)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not backend_c.available(),
                    reason="midas_indexer C binary not built")
@pytest.mark.skipif(not _ff_fixture_ready(),
                    reason=f"FF unified fixture not present at {FF_FIXTURE} "
                           "(regen via convert_ff_to_unified.py)")
def test_ff_passes_tolerance_gate_against_legacy_indexerOMP(tmp_path):
    """FF mode (nScans=1) of midas_indexer should match legacy IndexerOMP
    on OM/position/nMatches within fp tolerance for every voxel."""
    out_dir = tmp_path / "ff"
    # SpotsToIndex.csv has 100 spots — read it to be safe.
    n_spots = sum(1 for line in (FF_FIXTURE / "SpotsToIndex.csv").open()
                  if line.strip())
    proc = _run_midas_indexer(
        FF_FIXTURE / "paramstest.txt", out_dir=out_dir, n_work=n_spots,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")

    legacy = np.fromfile(FF_FIXTURE / "golden" / "IndexBest.bin",
                         dtype=np.float64).reshape(-1, 15)
    n_vox, n_sol, offsets, raw, hsz = _read_consolidated_header(
        out_dir / "Output" / "IndexBest_all.bin"
    )

    failures = 0
    for v in range(n_vox):
        legacy_zero = bool((legacy[v] == 0).all())
        if n_sol[v] == 0:
            assert legacy_zero, f"voxel {v}: unified missing solution"
            continue
        assert not legacy_zero, f"voxel {v}: unified produced solution legacy missed"
        off = int(offsets[v]) - hsz
        rec = np.frombuffer(raw[hsz + off : hsz + off + 128],
                            dtype=np.float64)
        # rec[0] = SpotID (unified-only); rec[1:16] aligns with legacy[0:15].
        om_ok = np.allclose(rec[1:10], legacy[v, 0:9], atol=1e-10)
        pos_ok = np.allclose(rec[10:13], legacy[v, 9:12], atol=1e-9)
        nm_ok = abs(rec[14] - legacy[v, 13]) <= 2
        nt_ok = abs(rec[13] - legacy[v, 12]) <= 2
        if not (om_ok and pos_ok and nm_ok and nt_ok):
            failures += 1
    assert failures == 0, f"{failures}/{n_vox} voxels fail tolerance gate"


# ---------------------------------------------------------------------------
# RingsToExcludeFraction PF sensitivity (N5 resolution)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not backend_c.available(),
                    reason="midas_indexer C binary not built")
@pytest.mark.skipif(not _pf_fixture_ready(),
                    reason="PF fixture binaries not present")
def test_pf_rings_to_exclude_fraction_changes_output_at_min_frac(tmp_path):
    """With MinMatchesToAcceptFrac > 0, activating RingsToExcludeFraction
    must change at least one voxel's solution count (N5: confirms PF
    accept gate uses FracCalc denominators when nRingsToReject > 0)."""
    baseline = tmp_path / "baseline"
    reject = tmp_path / "reject"
    threshold_bump = "MinMatchesToAcceptFrac 0.150000"

    proc1 = _run_midas_indexer(
        PF_FIXTURE / "paramstest.txt", out_dir=baseline, n_work=15,
        extra_lines=threshold_bump,
    )
    assert proc1.returncode == 0
    proc2 = _run_midas_indexer(
        PF_FIXTURE / "paramstest.txt", out_dir=reject, n_work=15,
        extra_lines=threshold_bump + "\nRingsToExcludeFraction 2",
    )
    assert proc2.returncode == 0

    _, n_base, _, _, _ = _read_consolidated_header(baseline / "Output" / "IndexBest_all.bin")
    _, n_rej, _, _, _ = _read_consolidated_header(reject / "Output" / "IndexBest_all.bin")
    changed = int((n_base != n_rej).sum())
    assert changed > 0, (
        f"RingsToExcludeFraction 2 produced byte-identical output ({changed} "
        "voxels changed) — PF accept gate is ignoring FracCalc denominators."
    )


# ---------------------------------------------------------------------------
# Soft attribution sidecar (Phase 8)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not backend_c.available(),
                    reason="midas_indexer C binary not built")
@pytest.mark.skipif(not _pf_fixture_ready(),
                    reason="PF fixture binaries not present")
def test_soft_attr_gaussian_produces_nontrivial_weight_distribution(tmp_path):
    """SoftAttrMode=gaussian with a beam-comparable FWHM must produce
    weights spanning a meaningful fraction of [0, 1], not a sliver."""
    out_dir = tmp_path / "soft"
    proc = _run_midas_indexer(
        PF_FIXTURE / "paramstest.txt", out_dir=out_dir, n_work=15,
        extra_lines="SoftAttrMode gaussian\nSoftAttrFwhm 2.5",
    )
    assert proc.returncode == 0

    sidecar = out_dir / "Output" / "IndexBest_weights_all.bin"
    assert sidecar.is_file(), "Phase 8 sidecar missing"

    n_vox, n_ids, offsets, raw, hsz = _read_consolidated_header(sidecar)
    total = int(n_ids.sum())
    weights = np.frombuffer(raw[hsz:], dtype=np.float64)
    assert len(weights) == total, "weight count must equal sum(nIDs)"
    # Sanity: weights in [0, 1].
    assert weights.min() >= 0.0
    assert weights.max() <= 1.0
    # Distribution check: at least 1% of weights should be in (0.5, 1.0]
    # (the active-beam region) AND at least 1% in (0, 0.5) (the tail).
    near_peak = int((weights > 0.5).sum())
    tail = int(((weights > 0) & (weights <= 0.5)).sum())
    assert near_peak >= total * 0.01, (
        f"only {near_peak}/{total} weights > 0.5 — Gaussian collapsed?"
    )
    assert tail >= total * 0.01, (
        f"only {tail}/{total} weights in (0, 0.5] — no tail attribution"
    )


def test_soft_attr_mode_zero_emits_all_ones_sidecar(tmp_path):
    """With SoftAttrMode=0 (default), the weights sidecar exists but
    every weight is 1.0. Verifies the no-soft-attr back-compat path."""
    if not backend_c.available():
        pytest.skip("midas_indexer C binary not built")
    if not _pf_fixture_ready():
        pytest.skip("PF fixture binaries not present")
    out_dir = tmp_path / "default"
    proc = _run_midas_indexer(
        PF_FIXTURE / "paramstest.txt", out_dir=out_dir, n_work=15,
    )
    assert proc.returncode == 0

    sidecar = out_dir / "Output" / "IndexBest_weights_all.bin"
    assert sidecar.is_file()
    n_vox, n_ids, offsets, raw, hsz = _read_consolidated_header(sidecar)
    weights = np.frombuffer(raw[hsz:], dtype=np.float64)
    assert bool(np.all(weights == 1.0)), (
        "SoftAttrMode=0 emitted non-1.0 weights"
    )
