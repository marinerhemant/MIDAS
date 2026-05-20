"""Slow E2E: PF reconstruction is invariant to scan acquisition order.

The same physical experiment, presented to the pipeline in monotonic vs
alternating acquisition order, must produce a byte-identical reconstruction.
The per-scan spot files are reused verbatim; only the (file-index <->
physical-Y) pairing changes. A correct indexer keys the scan filter off the
acquisition-order position while building the voxel grid from sorted
positions (see midas_index.indexer.run_scanning); the C IndexerScanningOMP
already does this, the Python backend was fixed to match.

This guards the WHOLE chain (binning -> indexing -> find_grains), so it
catches regressions in any downstream scan_nr consumer, not just the indexer.

Marked ``slow`` and **skips when the fixture data is absent** — it reuses the
per-scan CSVs produced by::

    python tests/test_pf_hedm.py -nCPUs 8 --no-cleanup

which live in ``FF_HEDM/Example/pfhedm_test/``. Two full pipeline runs take
~20 min on CPU; this is an opt-in divergence check, not a CI gate.

The fast always-on regression for the same bug is
``midas_index/tests/test_run_scanning.py::
test_run_scanning_preserves_acquisition_order_for_scan_filter``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

# Resolve the MIDAS root + the test data dir produced by test_pf_hedm.py.
_MIDAS = Path(__file__).resolve().parents[4]
_SRC = _MIDAS / "FF_HEDM" / "Example" / "pfhedm_test"
_N_SCANS = 15
_BEAM_UM = 5.0


def _fixture_present() -> bool:
    if not _SRC.exists():
        return False
    return all(
        (_SRC / f"InputAllExtraInfoFittingAll{i}.csv").exists()
        for i in range(_N_SCANS)
    ) and (_SRC / "paramstest.txt").exists() and (_SRC / "hkls.csv").exists()


def _alternating_order(n: int) -> list[int]:
    """Center-out order over sorted indices: n=15 -> [7,6,8,5,9,...,0,14]."""
    mid = n // 2
    order = [mid]
    step = 1
    while len(order) < n:
        if mid - step >= 0:
            order.append(mid - step)
        if mid + step < n:
            order.append(mid + step)
        step += 1
    return order


def _stage_layer(root: Path, sorted_pos: np.ndarray, perm: list[int]) -> tuple[Path, np.ndarray]:
    layer = root / "LayerNr_1"
    layer.mkdir(parents=True, exist_ok=True)
    shutil.copy(_SRC / "paramstest.txt", layer / "paramstest.txt")
    shutil.copy(_SRC / "hkls.csv", layer / "hkls.csv")
    positions_acq = sorted_pos[perm]
    for m in range(_N_SCANS):
        shutil.copy(
            _SRC / f"InputAllExtraInfoFittingAll{perm[m]}.csv",
            layer / f"InputAllExtraInfoFittingAll{m}.csv",
        )
    np.savetxt(layer / "positions.csv", positions_acq, fmt="%.6f")
    return layer, positions_acq


def _run(root: Path, positions_acq: np.ndarray) -> Path:
    from midas_pipeline.config import PipelineConfig, ScanGeometry, LayerSelection
    from midas_pipeline.pipeline import Pipeline

    cfg = PipelineConfig(
        result_dir=str(root),
        params_file=str(root / "LayerNr_1" / "paramstest.txt"),
        scan=ScanGeometry(
            scan_mode="pf", n_scans=_N_SCANS,
            scan_positions=positions_acq, beam_size_um=_BEAM_UM,
        ),
        device="cpu", dtype="float64", n_cpus=4,
        resume="none",
        only_stages=["binning", "indexing", "find_grains"],
        indexer_backend="python",
        layer_selection=LayerSelection(start=1, end=1),
    )
    cfg.recon.do_tomo = False
    Pipeline(cfg).run()
    return root / "LayerNr_1"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("MIDAS_RUN_PF_E2E") != "1",
    reason=(
        "Opt-in only (two ~10-min pipeline runs). Enable with "
        "MIDAS_RUN_PF_E2E=1, after generating the fixture via "
        "`python tests/test_pf_hedm.py -nCPUs 8 --no-cleanup`."
    ),
)
@pytest.mark.skipif(
    not _fixture_present(),
    reason=(
        "PF fixture absent. Generate with:\n"
        "  python tests/test_pf_hedm.py -nCPUs 8 --no-cleanup\n"
        f"(expects per-scan CSVs in {_SRC})"
    ),
)
def test_pf_reconstruction_invariant_to_scan_order(tmp_path: Path):
    from midas_index.io.consolidated import read_index_best_all

    sorted_pos = np.sort(np.loadtxt(_SRC / "positions.csv").ravel())
    assert sorted_pos.size == _N_SCANS

    mono_layer, mono_pos = _stage_layer(tmp_path / "mono", sorted_pos, list(range(_N_SCANS)))
    alt_layer, alt_pos = _stage_layer(tmp_path / "alt", sorted_pos, _alternating_order(_N_SCANS))
    # Sanity: the alternating order really is non-monotonic.
    assert np.any(np.diff(alt_pos) < 0)

    _run(tmp_path / "mono", mono_pos)
    _run(tmp_path / "alt", alt_pos)

    # 1. Per-voxel indexer records identical (orientation/pos/nMatches).
    res_m = read_index_best_all(mono_layer / "Output" / "IndexBest_all.bin")
    res_a = read_index_best_all(alt_layer / "Output" / "IndexBest_all.bin")
    assert res_m.n_voxels == res_a.n_voxels == _N_SCANS * _N_SCANS
    np.testing.assert_array_equal(res_m.n_sol_arr, res_a.n_sol_arr)
    assert res_m.vals.shape == res_a.vals.shape
    np.testing.assert_allclose(res_m.vals, res_a.vals, atol=1e-9, rtol=0)

    # 2. Grain map + unique grains + sinograms identical (downstream chain).
    for name in ("voxel_grid.csv", "UniqueOrientations.csv"):
        fm = mono_layer / "Output" / name
        fa = alt_layer / "Output" / name
        assert fm.exists() and fa.exists(), f"missing {name}"
        assert fm.read_bytes() == fa.read_bytes(), f"{name} differs across scan order"

    sinos_m = sorted((mono_layer / "Output").glob("sinos_*.bin"))
    sinos_a = sorted((alt_layer / "Output").glob("sinos_*.bin"))
    assert sinos_m and [p.name for p in sinos_m] == [p.name for p in sinos_a]
    for pm, pa in zip(sinos_m, sinos_a):
        assert pm.read_bytes() == pa.read_bytes(), f"{pm.name} differs across scan order"
