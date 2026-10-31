"""Slow E2E: the c-omp indexer and refiner on panels at different distances (#75).

The c-omp codes used to PARSE the per-panel geometry and EtaCoverage rows and
never read them (``IndexerUnified.c`` Phase 9b TODO, ``FitUnified.c`` kept
``DetParams`` in an array nothing used). On a four-panel layout with two panels
120 mm further out, measured on this generator (30 grains, seed 42, CPU):

=======================  =========================  ========================
c-omp binaries / seeds   default completeness       rings 1-2 only
=======================  =========================  ========================
before (HEAD)            0 grains                   11 of 30, 14.6 um median
per-panel, one ring      28 of 30, 2.7 um median    28 of 30, 0.84 um median
per-panel, all rings     30 of 30, 2.5 um median    (not run)
=======================  =========================  ========================

The two grains lost with one seed ring were indexed (completeness 1.0) but each
had only 2 ring-2 spots on the panels, so only 2 seeds, and process_grains drops
a cluster with fewer than MinNrSpots = 3 seeds. Seeding from every ring (the
multi-detector default) recovered all 30, and all 40 hydra-layout grains, on
every seed tried (8 standoff and 5 hydra seeds), with no unmatched entries.

(misorientation 0.05 deg in every case that found grains). This runs the full
chain at DEFAULT settings and asserts the second row. It needs binaries that
carry the per-panel support (``backend_c.supports_multidet()``); against older
binaries it is skipped, not failed, so a stale install does not look like a
regression in the code.

``slow``: ~4 min on CPU. Run with
``python -m pytest -m slow tests/integration/test_multidet_c_indexing.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("torch")
pytest.importorskip("zarr")
pytest.importorskip("midas_diffract.simulate_panel_zarrs")
pytest.importorskip("midas_peakfit")
pytest.importorskip("midas_transforms")
pytest.importorskip("midas_hkls")
pytest.importorskip("midas_process_grains")
ms = pytest.importorskip("midas_stress.orientation")
ix_c = pytest.importorskip("midas_index.backend_c")
fg_c = pytest.importorskip("midas_fit_grain.backend_c")

if not (ix_c.available() and fg_c.available()):
    pytest.skip("c-omp indexer / refiner binaries not built", allow_module_level=True)
if not (getattr(ix_c, "supports_multidet", lambda: False)()
        and getattr(fg_c, "supports_multidet", lambda: False)()):
    pytest.skip("installed c-omp binaries predate per-panel support; reinstall "
                "midas-index and midas-fit-grain", allow_module_level=True)

N_GRAINS = 30
LSD_UM = [3.3e6, 3.42e6, 3.42e6, 3.3e6]          # two standoff groups, 120 mm apart


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from midas_pipeline import Pipeline, PipelineConfig, ScanGeometry
    from midas_pipeline.testing import generate_pinwheel_synthetic_dataset

    root = tmp_path_factory.mktemp("multidet_c")
    zips, dj = generate_pinwheel_synthetic_dataset(
        out_dir=root / "sim", n_grains=N_GRAINS, seed=42, n_panels=4,
        use_hydra_geometry=False, panel_lsd_um=LSD_UM, omega_step_deg=-0.5,
        device="cpu")
    cfg = PipelineConfig(
        scan=ScanGeometry.ff(), result_dir=str(root / "run"),
        params_file=str(root / "sim" / "Parameters.txt"), detectors_json=str(dj),
        n_cpus=8, device="cpu", dtype="float64", convert_files=False)
    Pipeline(config=cfg).run()
    return root


def _score(root: Path):
    g = np.loadtxt(root / "run" / "LayerNr_1" / "Grains.csv", comments="%", ndmin=2)
    t = np.loadtxt(root / "sim" / "GrainsSim.csv", comments="%", ndmin=2)
    OMt, Pt = t[:, 1:10].reshape(-1, 3, 3), t[:, 10:13]
    per = {}
    for om, p in zip(g[:, 1:10].reshape(-1, 3, 3), g[:, 10:13]):
        m = np.degrees(np.asarray(ms.misorientation_om_batch(
            np.repeat(om[None], len(OMt), 0), OMt, 225), float))
        k = int(np.argmin(m))
        if m[k] < 1.0:
            per.setdefault(k, []).append((m[k], float(np.linalg.norm(p - Pt[k]))))
    mis = np.array([np.median([x[0] for x in v]) for v in per.values()])
    pos = np.array([np.median([x[1] for x in v]) for v in per.values()])
    return len(per), mis, pos


def test_grains_are_found_at_default_completeness(run):
    n, mis, _ = _score(run)
    assert n >= 0.9 * N_GRAINS, f"{n} of {N_GRAINS} truth grains recovered"
    assert np.median(mis) < 0.15


def test_no_entry_is_unmatched(run):
    """Every Grains.csv entry is one of the simulated grains (no ghosts)."""
    g = np.loadtxt(run / "run" / "LayerNr_1" / "Grains.csv", comments="%", ndmin=2)
    n, _, _ = _score(run)
    assert len(g) == n


def test_positions_are_not_biased_by_the_standoff(run):
    """With one global Lsd the displaced panels pulled positions by tens of um
    (14.6 um median on rings 1-2 here, 69 um on a 40-grain 4-panel set)."""
    _, _, pos = _score(run)
    assert np.median(pos) < 8.0, f"median position error {np.median(pos):.1f} um"
