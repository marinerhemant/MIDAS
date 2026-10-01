"""Slow E2E: multi-detector FF front half on a simulated 2-panel pinwheel (#75).

``generate_pinwheel_synthetic_dataset`` simulates one ``.MIDAS.zip`` per panel
with midas_diffract and records the spots it put on each panel in
``setup_truth.json``. This runs ``zip_convert -> hkl -> peakfit -> transforms ->
cross_det_merge -> global_powder`` over ``--detectors detectors.json`` and
checks the merged spot table against that truth.

Before #75 hkl / peakfit / transforms read a single zip, soft-skipped, and
``cross_det_merge`` died on ``LayerNr_1/Det_1/paramstest.txt``. The unit tests in
``tests/unit/test_multidet_routing.py`` cover the same routing with stand-in
backends; this one runs the real peak fitter and transforms.

Spot counts are compared to ``setup_truth.json`` within 1 %, not exactly: at
other seeds peakfit/transforms lose 1-3 of ~500 simulated spots before the merge.

Deliberately NOT asserted: a grain count. With two off-centre panels the outer
rings are barely on the detector, and with rings that poorly covered the c-omp
indexer accepts nothing (measured 0 of 81 seeds; the same data restricted to
rings 1-2 on a 4-panel layout gave 27 of 40 truth grains, median
misorientation 0.05 deg). That is an indexer / simulated-layout question, not a
routing one, and a regression test should not hang on it.

``slow``: ~1 min on CPU. Run with
``python -m pytest -m slow tests/integration/test_multidet_synthetic_pipeline.py``.
"""
from __future__ import annotations

import json

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

N_GRAINS = 30
SEED = 42
STAGES = ["zip_convert", "hkl", "peakfit", "transforms",
          "cross_det_merge", "global_powder"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from midas_pipeline import Pipeline, PipelineConfig, ScanGeometry
    from midas_pipeline.testing import generate_pinwheel_synthetic_dataset

    root = tmp_path_factory.mktemp("multidet")
    zips, dj = generate_pinwheel_synthetic_dataset(
        out_dir=root / "sim", n_grains=N_GRAINS, seed=SEED, n_panels=2,
        use_hydra_geometry=False, omega_step_deg=-0.5, device="cpu")
    params = root / "sim" / "Parameters.txt"       # written by the generator
    cfg = PipelineConfig(
        scan=ScanGeometry.ff(), result_dir=str(root / "run"),
        params_file=str(params), detectors_json=str(dj), n_cpus=8,
        device="cpu", dtype="float64", only_stages=STAGES, convert_files=False)
    Pipeline(config=cfg).run()
    return root


def test_every_panel_gets_its_own_outputs(run):
    layer = run / "run" / "LayerNr_1"
    for d in (1, 2):
        for f in ("hkls.csv", "Temp/AllPeaks_PS.bin", "InputAll.csv",
                  "InputAllExtraInfoFittingAll.csv", "paramstest.txt", "IDsHash.csv"):
            assert (layer / f"Det_{d}" / f).exists(), f"Det_{d}/{f}"


def test_merged_spots_are_the_panels_spots_and_nearly_all_the_simulated_ones(run):
    layer = run / "run" / "LayerNr_1"
    truth = json.loads((run / "sim" / "setup_truth.json").read_text())["n_spots_per_panel"]
    lines = (layer / "InputAll.csv").read_text().splitlines()
    assert lines[0].split()[-2:] == ["Ttheta", "DetID"]
    rows = np.array([ln.split() for ln in lines[1:]], dtype=float)
    assert rows.shape[1] == 9
    got = [int((rows[:, 8] == d).sum()) for d in (1, 2)]
    # The merge conserves each panel's spots exactly.
    per_panel = [len((layer / f"Det_{d}" / "InputAll.csv").read_text().splitlines()) - 1
                 for d in (1, 2)]
    assert got == per_panel
    # Against the simulation: peak finding recovers essentially every simulated
    # spot, not always every one. Measured on this generator: 3 of 7 (seed,
    # n_grains) settings recover all of them, the others lose 1-3 spots in
    # peakfit/transforms (before the merge), e.g. 354 of 355. Equality at the
    # pinned seed holds today, but is a property of the peak fitter, so allow 1%.
    for g, t_ in zip(got, truth):
        assert 0.99 * t_ <= g <= t_, f"merged per-panel spots {got} vs simulated {truth}"
    assert list(rows[:, 4]) == list(range(1, len(rows) + 1))


def _table(path):
    lines = path.read_text().splitlines()
    return lines[0].split(), np.array([ln.split() for ln in lines[1:]], dtype=float)


def test_merged_rows_are_the_panels_rows_unshuffled(run):
    """Equal counts could hide duplicated or swapped rows; compare the rows."""
    layer = run / "run" / "LayerNr_1"
    head, merged = _table(layer / "InputAll.csv")
    # Columns: 0 YLab 1 ZLab 2 Omega 3 GrainRadius 4 SpotID 5 RingNumber 6 Eta
    # 7 Ttheta 8 DetID. GrainRadius is rescaled by global_powder and SpotID is
    # renumbered globally; every other column is the per-panel value.
    cols = [0, 1, 2, 5, 6, 7]
    start = 0
    for d in (1, 2):
        _, panel = _table(layer / f"Det_{d}" / "InputAll.csv")
        rows = merged[start:start + len(panel)]
        assert (rows[:, 8] == d).all()
        np.testing.assert_array_equal(rows[:, cols], panel[:, cols])
        start += len(panel)
    assert start == len(merged)


def test_each_panels_spots_lie_in_its_own_eta_coverage(run):
    """The arcs are derived from the panel geometry by pixel enumeration, the
    spots from the simulator: a wrong tilt or centre convention puts spots
    outside (measured: tx sign flipped leaves 392 of 522 outside)."""
    from midas_pipeline.eta_coverage import parse_coverage_blocks
    layer = run / "run" / "LayerNr_1"
    cov = parse_coverage_blocks((layer / "paramstest.txt").read_text())
    _, merged = _table(layer / "InputAll.csv")
    tol = 1.0   # deg; arcs come from pixel centres, spots from sub-pixel positions
    for det, ring, eta in zip(merged[:, 8].astype(int), merged[:, 5].astype(int), merged[:, 6]):
        arcs = [a for a in cov[det] if a.ring_nr == ring]
        assert any(a.eta_lo_deg - tol <= eta <= a.eta_hi_deg + tol for a in arcs), \
            f"det {det} ring {ring} eta {eta:.2f} outside its coverage {arcs}"


def test_merged_ids_hash_gives_every_spot_its_own_ring(run):
    """process_grains reads each spot's ring (hence its d0) through IDsHash.csv
    by SpotID, so a shifted-wrong range gives the wrong strain reference."""
    from midas_process_grains.io.ids_hash import load_ids_hash
    layer = run / "run" / "LayerNr_1"
    _, merged = _table(layer / "InputAll.csv")
    ih = load_ids_hash(layer / "IDsHash.csv")
    got = [ih.ring_for_spot_id(int(i)) for i in merged[:, 4]]
    assert got == [int(r) for r in merged[:, 5]]
    assert all(ih.d_for_spot_id(int(i)) > 0 for i in merged[:, 4])


def test_merge_leaves_what_downstream_stages_read(run):
    layer = run / "run" / "LayerNr_1"
    assert (layer / "IDsHash.csv").exists()          # process_grains refuses without it
    text = (layer / "paramstest.txt").read_text()
    assert "EtaCoverage_Det1 " in text and "EtaCoverage_Det2 " in text
    assert text.count("\nDetParams ") == 2
