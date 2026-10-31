"""cross_det_merge with overlapping panels: report always, drop only on request."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.config import PipelineConfig, ScanGeometry
from midas_pipeline.detector import DetectorConfig
from midas_pipeline.panel_overlap import C_RING
from midas_pipeline.stages import cross_det_merge
from midas_pipeline.stages._base import StageContext

LSD = 3.3e6
RADII = {1: 313_000.0, 2: 362_000.0, 3: 514_000.0, 4: 605_000.0}
HKLS = ("h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n"
        + "".join(f"1 1 1 {2.0 / r} {r} 0 0 0 2.7 5.4 {RADII[r]}\n" for r in RADII))
D0 = {1: 2.35, 2: 2.04, 3: 1.44, 4: 1.23}


def _table(rng, n, ring=None):
    ring = rng.integers(1, 5, n) if ring is None else ring
    rad = np.array([RADII[int(r)] for r in ring])
    eta = rng.uniform(-170, 170, n)
    ome = np.round(rng.uniform(-179, 179, n) / 0.5) * 0.5
    return rad, eta, ome, ring


def _rows(rad, eta, ome, ring, jitter=0.0, rng=None):
    y = -rad * np.sin(np.radians(eta))
    z = rad * np.cos(np.radians(eta))
    if jitter:
        y = y + rng.normal(0, jitter, y.size)
        z = z + rng.normal(0, jitter, z.size)
    eta_ = np.degrees(np.arctan2(-y, z))
    tth = np.degrees(np.arctan(np.hypot(y, z) / LSD))
    return y, z, eta_, tth


def _write_panel(d: Path, y, z, ome, ring, eta, tth, seeds):
    d.mkdir(parents=True, exist_ok=True)
    head = "YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta"
    lines = [f"{y[i]:.6f} {z[i]:.6f} {ome[i]:.6f} 50.000000 {i + 1} {int(ring[i])} "
             f"{eta[i]:.6f} {tth[i]:.6f}" for i in range(y.size)]
    (d / "InputAll.csv").write_text(head + "\n" + "\n".join(lines) + "\n")
    (d / "SpotsToIndex.csv").write_text("".join(f"{s}\n" for s in seeds))
    (d / "hkls.csv").write_text(HKLS)
    (d / "IDsHash.csv").write_text("".join(
        f"{r} 1 2 {D0[r]}\n" for r in sorted(D0)))
    (d / "paramstest.txt").write_text(
        f"Lsd {LSD}\nBC 100 100\ntx 0\nty 0\ntz 0\npx 200\nNrPixelsY 256\n"
        "NrPixelsZ 256\nWidth 1500\nOmegaStep -0.5\nRingNumbers 1\nRingNumbers 2\n"
        "RingNumbers 3\nRingNumbers 4\n")


def _ctx(tmp_path: Path):
    params = tmp_path / "Parameters.txt"
    params.write_text("Lsd 1\n")
    layer = tmp_path / "LayerNr_1"
    layer.mkdir()
    dets = [DetectorConfig(det_id=i, zarr_path="", lsd=LSD, y_bc=100.0, z_bc=100.0,
                           tx=90.0 * (i - 1)) for i in (1, 2)]
    cfg = PipelineConfig(result_dir=str(tmp_path), params_file=str(params),
                         scan=ScanGeometry.ff())
    return StageContext(config=cfg, detectors=dets, layer_nr=1, layer_dir=layer,
                        log_dir=layer / "midas_log")


def _setup(tmp_path, n_dup=200, n_own=100, seeds_dup=10):
    rng = np.random.default_rng(11)
    ctx = _ctx(tmp_path)
    rad, eta, ome, ring = _table(rng, n_dup)
    ya, za, ea, ta = _rows(rad, eta, ome, ring)
    yb, zb, eb, tb = _rows(rad, eta, ome, ring, jitter=12.0, rng=rng)
    rad2, eta2, ome2, ring2 = _table(rng, n_own)
    own_a = _rows(rad2, eta2, ome2, ring2)
    rad3, eta3, ome3, ring3 = _table(rng, n_own)
    own_b = _rows(rad3, eta3, ome3, ring3)
    cat = np.concatenate
    # seeds: the first `seeds_dup` duplicates on panel b, and 3 own spots of a
    _write_panel(ctx.detector_dir(ctx.detectors[0]), cat([ya, own_a[0]]),
                 cat([za, own_a[1]]), cat([ome, ome2]), cat([ring, ring2]),
                 cat([ea, own_a[2]]), cat([ta, own_a[3]]),
                 [n_dup + 1, n_dup + 2, n_dup + 3])
    _write_panel(ctx.detector_dir(ctx.detectors[1]), cat([yb, own_b[0]]),
                 cat([zb, own_b[1]]), cat([ome, ome3]), cat([ring, ring3]),
                 cat([eb, own_b[2]]), cat([tb, own_b[3]]),
                 list(range(1, seeds_dup + 1)))
    return ctx, n_dup, n_own


def _merged(ctx):
    lines = (ctx.layer_dir / "InputAll.csv").read_text().splitlines()
    return lines[0].split(), np.array([ln.split() for ln in lines[1:]], dtype=float)


def test_overlap_is_measured_and_reported_but_nothing_is_dropped(tmp_path, monkeypatch):
    monkeypatch.delenv("MIDAS_DEDUP_PANELS", raising=False)
    ctx, n_dup, n_own = _setup(tmp_path)
    cross_det_merge.run(ctx)
    _, rows = _merged(ctx)
    assert rows.shape[0] == 2 * (n_dup + n_own)              # kept
    csv = (ctx.layer_dir / "PanelOverlap.csv").read_text().splitlines()
    head, vals = csv[0].split(), csv[1].split()
    rec = dict(zip(head, vals))
    assert int(rec["det_a"]) == 1 and int(rec["det_b"]) == 2
    assert int(rec["n_pairs"]) >= 0.95 * n_dup
    assert float(rec["pos_sigma_um"]) < 40.0


def test_dedup_drops_one_record_per_duplicate_and_keeps_the_table_consistent(
        tmp_path, monkeypatch):
    monkeypatch.setenv("MIDAS_DEDUP_PANELS", "1")
    ctx, n_dup, n_own = _setup(tmp_path)
    cross_det_merge.run(ctx)
    head, rows = _merged(ctx)
    csv = dict(zip(*[l.split() for l in
                     (ctx.layer_dir / "PanelOverlap.csv").read_text().splitlines()[:2]]))
    n_pairs = int(csv["n_pairs"])
    assert rows.shape[0] == 2 * (n_dup + n_own) - n_pairs
    # earlier panel wins without an edge score: every dropped record is panel 2's
    assert (rows[:, 8] == 1).sum() == n_dup + n_own
    assert list(rows[:, 4]) == list(range(1, rows.shape[0] + 1))

    from midas_process_grains.io.ids_hash import load_ids_hash
    ih = load_ids_hash(ctx.layer_dir / "IDsHash.csv")
    assert [ih.ring_for_spot_id(int(i)) for i in rows[:, 4]] == [int(r) for r in rows[:, C_RING]]
    assert all(ih.d_for_spot_id(int(i)) > 0 for i in rows[:, 4])


def test_a_seed_on_a_dropped_duplicate_moves_to_its_kept_partner(tmp_path, monkeypatch):
    monkeypatch.setenv("MIDAS_DEDUP_PANELS", "1")
    ctx, n_dup, n_own = _setup(tmp_path, seeds_dup=10)
    ctx.config.seed_rings = "transforms"          # keep the per-panel seed files
    cross_det_merge.run(ctx)
    seeds = [int(x) for x in (ctx.layer_dir / "SpotsToIndex.csv").read_text().split()]
    # 10 seeds were duplicates seen from panel b (dropped copy -> panel a's id,
    # which are global ids 1..), plus panel a's own 3 seeds; no id appears twice.
    assert len(seeds) == len(set(seeds)) == 13
    kept_a_ids = set(range(1, n_dup + n_own + 1))
    assert set(seeds) <= kept_a_ids


def _seeds(ctx):
    return {int(x) for x in (ctx.layer_dir / "SpotsToIndex.csv").read_text().split()}


def test_default_seeds_every_spot_of_every_ring_inside_the_omega_window(tmp_path, monkeypatch):
    """One seed ring starves a grain on a panel layout (a grain that shows only
    two spots of that ring has two seeds, and process_grains drops a cluster
    with fewer than MinNrSpots seeds); the default seeds from every ring."""
    monkeypatch.delenv("MIDAS_DEDUP_PANELS", raising=False)
    monkeypatch.delenv("MIDAS_SEED_RINGS", raising=False)
    ctx, _, _ = _setup(tmp_path)
    cross_det_merge.run(ctx)
    _, rows = _merged(ctx)
    assert _seeds(ctx) == {int(r[4]) for r in rows if -90.0 <= r[2] <= 90.0}


def test_transforms_keeps_the_per_panel_seed_files(tmp_path, monkeypatch):
    monkeypatch.delenv("MIDAS_SEED_RINGS", raising=False)
    ctx, _, _ = _setup(tmp_path)
    ctx.config.seed_rings = "transforms"
    cross_det_merge.run(ctx)
    assert len(_seeds(ctx)) == 13                     # the panels' own seeds


def test_an_explicit_ring_list_seeds_only_those_rings(tmp_path, monkeypatch):
    monkeypatch.delenv("MIDAS_SEED_RINGS", raising=False)
    ctx, _, _ = _setup(tmp_path)
    ctx.config.seed_rings = [1, 2]
    cross_det_merge.run(ctx)
    _, rows = _merged(ctx)
    assert _seeds(ctx) == {int(r[4]) for r in rows
                           if int(r[C_RING]) in (1, 2) and -90.0 <= r[2] <= 90.0}


def test_seed_rings_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MIDAS_SEED_RINGS", "3,4")
    ctx, _, _ = _setup(tmp_path)
    cross_det_merge.run(ctx)
    _, rows = _merged(ctx)
    assert _seeds(ctx) == {int(r[4]) for r in rows
                           if int(r[C_RING]) in (3, 4) and -90.0 <= r[2] <= 90.0}
