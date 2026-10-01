"""Multi-detector FF routing (issue #75).

``--detectors detectors.json`` carries one ``.MIDAS.zip`` per panel. Every
early FF stage (hkl, peakfit, transforms) has to run once per panel into
``Det_<id>/`` so ``cross_det_merge`` has something to merge; they used to read
only the single ``--zarr`` / layer-dir zip and soft-skip, after which the merge
died on ``Det_1/paramstest.txt``.

The heavy backends are replaced by stand-ins that write the files the real
ones write, so this checks the routing and the merge, not the numerics. The
real chain on a simulated 2-panel dataset is
``tests/integration/test_multidet_synthetic_pipeline.py``.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from midas_pipeline.config import PipelineConfig, ScanGeometry
from midas_pipeline.detector import DetectorConfig
from midas_pipeline.preflight import check_inputs
from midas_pipeline.stages import cross_det_merge, hkl, peakfit, transforms
from midas_pipeline.stages._base import StageContext, ff_zip_jobs

N_PX = 256
PX_UM = 200.0
LSD_UM = 1_000_000.0
# Beam centres sit near opposite edges of the panel, so each panel catches a
# different part of the rings below.
HKLS = (
    "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n"
    "1 1 1 2.35 1 0 0 0 2.7 5.4 10000.0\n"
    "2 0 0 2.04 2 0 0 0 3.1 6.2 12000.0\n"
)


def _ctx(tmp_path: Path, n_det: int = 2, *, make_zips: bool = True) -> StageContext:
    params = tmp_path / "Parameters.txt"
    params.write_text("Lsd 1000000\n")
    layer = tmp_path / "LayerNr_1"
    layer.mkdir()
    dets = []
    for i in range(1, n_det + 1):
        z = tmp_path / f"panel{i}.MIDAS.zip"
        if make_zips:
            z.write_bytes(b"zip")
        dets.append(DetectorConfig(det_id=i, zarr_path=str(z), lsd=LSD_UM,
                                   y_bc=20.0 if i == 1 else 236.0, z_bc=128.0,
                                   tx=90.0 * (i - 1)))
    cfg = PipelineConfig(result_dir=str(tmp_path), params_file=str(params),
                         scan=ScanGeometry.ff())
    return StageContext(config=cfg, detectors=dets, layer_nr=1, layer_dir=layer,
                        log_dir=layer / "midas_log")


# ---- ff_zip_jobs ----------------------------------------------------------

def test_jobs_one_zip_and_dir_per_panel(tmp_path):
    ctx = _ctx(tmp_path)
    jobs = ff_zip_jobs(ctx)
    assert [(z.name, d.name) for z, d in jobs] == [
        ("panel1.MIDAS.zip", "Det_1"), ("panel2.MIDAS.zip", "Det_2")]
    assert all(d.is_dir() for _, d in jobs)


def test_jobs_missing_panel_zip_is_a_hard_error(tmp_path):
    ctx = _ctx(tmp_path, make_zips=False)
    with pytest.raises(FileNotFoundError, match="detector 1 zip not found"):
        ff_zip_jobs(ctx)


def test_jobs_single_detector_unchanged(tmp_path):
    ctx = _ctx(tmp_path, n_det=1)
    z = ctx.layer_dir / "run.MIDAS.zip"
    z.write_bytes(b"zip")
    assert ff_zip_jobs(ctx) == [(z, ctx.layer_dir)]


def test_jobs_panel_without_zarr_path_is_a_hard_error(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.detectors[1].zarr_path = ""
    with pytest.raises(FileNotFoundError, match="detector 2 has no zarr_path"):
        ff_zip_jobs(ctx)


# ---- stage routing --------------------------------------------------------

def test_hkl_runs_per_panel_and_promotes_one_layer_list(tmp_path, monkeypatch):
    calls = []

    def fake_gen(zip_path, result_folder, **_):
        calls.append((Path(zip_path).name, Path(result_folder).name))
        (Path(result_folder) / "hkls.csv").write_text(HKLS)

    mod = types.ModuleType("midas_hkls.zarr_compat")
    mod.generate_hkls_from_zarr = fake_gen
    monkeypatch.setitem(sys.modules, "midas_hkls.zarr_compat", mod)
    pkg = types.ModuleType("midas_hkls")
    pkg.read_phase_basis = lambda _p: {}
    pkg.zarr_compat = mod
    monkeypatch.setitem(sys.modules, "midas_hkls", pkg)

    ctx = _ctx(tmp_path)
    hkl.run(ctx)
    assert calls == [("panel1.MIDAS.zip", "Det_1"), ("panel2.MIDAS.zip", "Det_2")]
    assert (ctx.layer_dir / "Det_1" / "hkls.csv").exists()
    assert (ctx.layer_dir / "Det_2" / "hkls.csv").exists()
    assert (ctx.layer_dir / "hkls.csv").exists()


def test_peakfit_runs_per_panel(tmp_path, monkeypatch):
    calls = []

    def fake_run(*, data_file, result_folder_cli, **_):
        calls.append((Path(data_file).name, Path(result_folder_cli).name))
        t = Path(result_folder_cli) / "Temp"
        t.mkdir(parents=True, exist_ok=True)
        (t / "AllPeaks_PS.bin").write_bytes(b"x")

    mod = types.ModuleType("midas_peakfit.orchestrator")
    mod.run = fake_run
    monkeypatch.setitem(sys.modules, "midas_peakfit.orchestrator", mod)
    monkeypatch.setitem(sys.modules, "midas_peakfit", types.ModuleType("midas_peakfit"))

    ctx = _ctx(tmp_path)
    peakfit.run(ctx)
    assert calls == [("panel1.MIDAS.zip", "Det_1"), ("panel2.MIDAS.zip", "Det_2")]
    # A second run resumes: nothing is recomputed.
    calls.clear()
    peakfit.run(ctx)
    assert calls == []


def test_peakfit_resumes_per_panel(tmp_path, monkeypatch):
    """Panel 1 done, panel 2 not: only panel 2 is fitted (and panel 1 is not
    allowed to short-circuit the loop)."""
    calls = []

    def fake_run(*, data_file, result_folder_cli, **_):
        calls.append(Path(result_folder_cli).name)
        t = Path(result_folder_cli) / "Temp"
        t.mkdir(parents=True, exist_ok=True)
        (t / "AllPeaks_PS.bin").write_bytes(b"x")

    mod = types.ModuleType("midas_peakfit.orchestrator")
    mod.run = fake_run
    monkeypatch.setitem(sys.modules, "midas_peakfit.orchestrator", mod)
    monkeypatch.setitem(sys.modules, "midas_peakfit", types.ModuleType("midas_peakfit"))

    ctx = _ctx(tmp_path)
    done = ctx.layer_dir / "Det_1" / "Temp"
    done.mkdir(parents=True)
    (done / "AllPeaks_PS.bin").write_bytes(b"x")
    peakfit.run(ctx)
    assert calls == ["Det_2"]


def test_transforms_resumes_per_panel(tmp_path, monkeypatch):
    seen = []

    class FakePipeline:
        @classmethod
        def from_zarr(cls, zarr, *, result_folder, **_):
            o = cls()
            o.out = Path(result_folder)
            return o

        def run(self):
            seen.append(self.out.name)

        def dump(self, out_dir):
            Path(out_dir, "InputAll.csv").write_text("YLab\n")
            Path(out_dir, "InputAllExtraInfoFittingAll.csv").write_text("YLab\n")

    mod = types.ModuleType("midas_transforms")
    mod.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "midas_transforms", mod)

    ctx = _ctx(tmp_path)
    d1 = ctx.layer_dir / "Det_1"
    d1.mkdir()
    (d1 / "InputAll.csv").write_text("YLab\n")
    (d1 / "InputAllExtraInfoFittingAll.csv").write_text("YLab\n")
    transforms.run(ctx)
    assert seen == ["Det_2"]


def test_transforms_runs_per_panel(tmp_path, monkeypatch):
    seen = []

    class FakePipeline:
        @classmethod
        def from_zarr(cls, zarr, *, result_folder, **_):
            o = cls()
            o.zarr, o.out = Path(zarr), Path(result_folder)
            return o

        def run(self):
            seen.append((self.zarr.name, self.out.name))

        def dump(self, out_dir):
            Path(out_dir, "InputAll.csv").write_text("YLab\n")
            Path(out_dir, "InputAllExtraInfoFittingAll.csv").write_text("YLab\n")

    mod = types.ModuleType("midas_transforms")
    mod.Pipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "midas_transforms", mod)

    ctx = _ctx(tmp_path)
    transforms.run(ctx)
    assert seen == [("panel1.MIDAS.zip", "Det_1"), ("panel2.MIDAS.zip", "Det_2")]
    assert (ctx.layer_dir / "Det_2" / "InputAllExtraInfoFittingAll.csv").exists()
    # The layer dir itself is not written by a per-panel stage.
    assert not (ctx.layer_dir / "InputAll.csv").exists()


# ---- cross_det_merge ------------------------------------------------------

def _write_panel(d: Path, n: int, *, with_detid: int | None = None,
                 ring_blocks=((1, 3), (2, 2)), ring_numbers=(1, 2)) -> None:
    """A panel as ``transforms`` leaves it: ring-sorted spots, local IDs 1..n."""
    d.mkdir(parents=True, exist_ok=True)
    head = "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta"
    head += " DetID" if with_detid else ""
    rows, sid = [], 1
    for ring, count in ring_blocks:
        for _ in range(count):
            r = f"1.0 2.0 3.0 4.0 {sid} {ring} 5.0 6.0"
            rows.append(r + (f" {with_detid}" if with_detid else ""))
            sid += 1
    assert sid - 1 == n
    (d / "InputAll.csv").write_text(head + "\n" + "\n".join(rows) + "\n")
    (d / "InputAllExtraInfoFittingAll.csv").write_text(head + "\n" + "\n".join(rows) + "\n")
    (d / "SpotsToIndex.csv").write_text("1\n")
    (d / "hkls.csv").write_text(HKLS)
    # Same layout as midas_transforms.write_ring_tables: start += count, end = start+count+1.
    (d / "IDsHash.csv").write_text("1 1 5 2.35\n2 4 7 2.04\n")
    (d / "paramstest.txt").write_text(
        f"Lsd {LSD_UM}\nBC 128 128\ntx 0\nty 0\ntz 0\npx {PX_UM}\n"
        f"NrPixelsY {N_PX}\nNrPixelsZ {N_PX}\nWidth 1500\n"
        + "".join(f"RingNumbers {r}\n" for r in ring_numbers)
        + "RingRadii 10000.0\nRingRadii 12000.0\n")


@pytest.mark.parametrize("panels_have_detid", [False, True])
def test_merge_writes_one_detid_column(tmp_path, panels_have_detid):
    ctx = _ctx(tmp_path)
    for det in ctx.detectors:
        _write_panel(ctx.detector_dir(det), 5,
                     with_detid=det.det_id if panels_have_detid else None)
    cross_det_merge.run(ctx)

    lines = (ctx.layer_dir / "InputAll.csv").read_text().splitlines()
    assert lines[0].split() == ["YLab", "ZLab", "Omega", "GrainRadius", "SpotID",
                                "RingNumber", "Eta", "Ttheta", "DetID"]
    rows = [ln.split() for ln in lines[1:]]
    assert len(rows) == 10 and all(len(r) == 9 for r in rows)
    assert [int(r[4]) for r in rows] == list(range(1, 11))     # global SpotIDs
    assert [int(r[8]) for r in rows] == [1] * 5 + [2] * 5      # panel of origin

    from midas_transforms.io.csv import read_inputall_csv_with_detid
    spots, det_id = read_inputall_csv_with_detid(ctx.layer_dir / "InputAll.csv")
    assert len(spots) == 10 and list(det_id) == [1] * 5 + [2] * 5


def test_merge_writes_layer_ids_hash_with_global_ranges(tmp_path):
    ctx = _ctx(tmp_path)
    for det in ctx.detectors:
        _write_panel(ctx.detector_dir(det), 5)
    cross_det_merge.run(ctx)

    from midas_process_grains.io.ids_hash import load_ids_hash
    ih = load_ids_hash(ctx.layer_dir / "IDsHash.csv")
    # Panel 1 holds global IDs 1-5 (ring 1: 1-3, ring 2: 4-5), panel 2 holds 6-10.
    ring_of = [ih.ring_for_spot_id(i) for i in range(1, 11)]
    assert ring_of == [1, 1, 1, 2, 2, 1, 1, 1, 2, 2]
    assert all(ih.d_for_spot_id(i) > 0 for i in range(1, 11))


def test_merge_takes_the_union_of_the_panels_rings(tmp_path):
    """A ring only the second panel reaches must survive into the merged table."""
    ctx = _ctx(tmp_path)
    _write_panel(ctx.detector_dir(ctx.detectors[0]), 5, ring_numbers=(1, 2))
    _write_panel(ctx.detector_dir(ctx.detectors[1]), 5, ring_numbers=(1, 2, 3))
    cross_det_merge.run(ctx)
    text = (ctx.layer_dir / "paramstest.txt").read_text()
    rings = [ln.split()[1] for ln in text.splitlines() if ln.startswith("RingNumbers ")]
    assert rings == ["1", "2", "3"]


def test_merge_drops_empty_ring_ranges_from_ids_hash(tmp_path):
    """A zero-spot ring (end = start + 1) must not tie with the next panel's start."""
    ctx = _ctx(tmp_path)
    for det in ctx.detectors:
        d = ctx.detector_dir(det)
        _write_panel(d, 5)
        # panel 1: ring 3 has no spots, its row would start at local ID 6
        (d / "IDsHash.csv").write_text("1 1 5 2.35\n2 4 7 2.04\n3 6 7 1.44\n")
    cross_det_merge.run(ctx)
    rows = [ln.split() for ln in (ctx.layer_dir / "IDsHash.csv").read_text().splitlines()]
    assert [r[0] for r in rows] == ["1", "2", "1", "2"]
    from midas_process_grains.io.ids_hash import load_ids_hash
    ih = load_ids_hash(ctx.layer_dir / "IDsHash.csv")
    assert [ih.ring_for_spot_id(i) for i in range(1, 11)] == [1, 1, 1, 2, 2, 1, 1, 1, 2, 2]


def test_merge_emits_eta_coverage_for_every_panel(tmp_path):
    ctx = _ctx(tmp_path)
    for det in ctx.detectors:
        _write_panel(ctx.detector_dir(det), 5)
    cross_det_merge.run(ctx)

    from midas_pipeline.eta_coverage import parse_coverage_blocks
    cov = parse_coverage_blocks((ctx.layer_dir / "paramstest.txt").read_text())
    assert sorted(cov) == [1, 2]
    assert {a.ring_nr for a in cov[1]} == {1, 2}
    # The two panels sit at opposite edges of the beam, so they cover different arcs.
    assert [(a.eta_lo_deg, a.eta_hi_deg) for a in cov[1]] != \
           [(a.eta_lo_deg, a.eta_hi_deg) for a in cov[2]]


# ---- preflight ------------------------------------------------------------

def test_preflight_accepts_per_panel_zips_without_raw_keys(tmp_path):
    ctx = _ctx(tmp_path)
    dj = tmp_path / "detectors.json"
    DetectorConfig.dump_many(ctx.detectors, dj)
    ctx.config.detectors_json = str(dj)
    assert check_inputs(ctx.config) == []


def test_preflight_names_the_panel_whose_zip_is_missing(tmp_path):
    ctx = _ctx(tmp_path)
    dj = tmp_path / "detectors.json"
    DetectorConfig.dump_many(ctx.detectors, dj)
    Path(ctx.detectors[1].zarr_path).unlink()
    ctx.config.detectors_json = str(dj)
    problems = check_inputs(ctx.config)
    assert len(problems) == 1 and "detector 2" in problems[0]
