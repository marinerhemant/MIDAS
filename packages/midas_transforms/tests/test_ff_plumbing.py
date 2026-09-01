"""Tests for FF-HEDM V-map plumbing (P5).

Covers:
- ``SampleGrid.from_grain_centroids``: build a 1-voxel-per-grain grid.
- ``predicted_spot_intensities`` in the three ``scan_axis`` modes:
    * ``"none"`` (compact FF, beam ignored)
    * ``"z"``   (FF height-scan, beam attenuates by v_z)
    * ``"pf"``  (regression: unchanged behavior)
- ``load_ff_grains_to_tensors``: round-trip a synthetic Grains.csv.
- ``load_ff_spots_to_tensors``: SpotID join with InputAllExtraInfo;
  grain_id→grain_idx mapping; unknown ring→-1.
- ``refine_vmap_joint`` recovers per-grain V on synthetic FF data
  (compact mode, ``scan_axis="none"``).
- FF refine with absorption: A wider tomographic SampleGrid + per-grain
  voxel attribution + use_absorption=True returns finite, positive V.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from midas_transforms.geometry import SampleGrid, TopHat
from midas_transforms.radius import (
    FFGrainTensors,
    FFSpotTensors,
    RingTable,
    load_ff_grains_to_tensors,
    load_ff_spots_to_tensors,
    predicted_spot_intensities,
    refine_vmap_joint,
)


def _t(x, **kw):  # noqa: ANN
    kw.setdefault("dtype", torch.float64)
    return torch.tensor(x, **kw)


# ----------------------------------------------------------- SampleGrid FF


def test_sample_grid_from_grain_centroids_default_ids():
    sg = SampleGrid.from_grain_centroids(
        centroids_um=[[0.0, 0.0, 0.0], [10.0, 5.0, 0.0], [-5.0, 15.0, 0.0]],
        voxel_size_um=2.0,
    )
    assert sg.n_voxels == 3
    assert sg.grain_map.tolist() == [0, 1, 2]
    assert bool(sg.sample_mask.all().item())
    # grain id 0 contains voxel 0; grain 99 contains nothing
    assert sg.voxels_in_grain(0).tolist() == [0]
    assert sg.voxels_in_grain(99).numel() == 0


def test_sample_grid_from_grain_centroids_custom_ids_and_mask():
    sg = SampleGrid.from_grain_centroids(
        centroids_um=[[0.0]*3, [1.0]*3],
        grain_ids=[7, 12],
        sample_mask=[True, False],
        voxel_size_um=1.0,
    )
    assert sg.grain_map.tolist() == [7, 12]
    assert sg.sample_mask.tolist() == [True, False]
    # Voxel 1 is masked out, so grain 12 has no in-sample voxels
    assert sg.voxels_in_grain(12).numel() == 0


def test_sample_grid_from_grain_centroids_shape_validation():
    with pytest.raises(ValueError, match="centroids_um"):
        SampleGrid.from_grain_centroids(centroids_um=[1.0, 2.0])
    with pytest.raises(ValueError, match="grain_ids"):
        SampleGrid.from_grain_centroids(
            centroids_um=[[0.0]*3, [1.0]*3], grain_ids=[7],
        )


# ----------------------------------------------------------- scan_axis modes


def _three_grain_setup():
    sg = SampleGrid.from_grain_centroids(
        centroids_um=[[0.0, 0.0, 0.0], [20.0, 30.0, 5.0], [50.0, 5.0, -3.0]],
        voxel_size_um=10.0,
    )
    V = _t([1.5, 2.0, 0.8])
    K = _t([3.0]); I_th = _t([7.0])
    ring = torch.tensor([0, 0, 0]); grain = torch.tensor([0, 1, 2])
    scan = _t([0.0, 0.0, 0.0])
    omega = _t([0.0, 0.5, 1.0])
    return sg, V, K, I_th, ring, grain, scan, omega


def test_scan_axis_none_ignores_beam():
    """Compact FF: every voxel contributes its full V regardless of beam/scan/ω."""
    sg, V, K, I_th, ring, grain, scan, omega = _three_grain_setup()
    I = predicted_spot_intensities(
        V, K, I_th, ring, grain, scan, omega, sg,
        TopHat(0.001),    # absurdly narrow beam — but should be ignored
        scan_axis="none",
    )
    expected = K[0] * I_th[0] * V
    assert torch.allclose(I, expected)


def test_scan_axis_z_uses_voxel_z():
    """FF height scan: fraction depends on v_z vs scan_pos via beam profile."""
    sg, V, K, I_th, ring, grain, scan, omega = _three_grain_setup()
    beam = TopHat(10.0)
    I = predicted_spot_intensities(
        V, K, I_th, ring, grain, scan, omega, sg, beam, scan_axis="z",
    )
    # Grain 0 at z=0 (TopHat [-5,5] · voxel [-5,5] = 10 / 10 = 1)
    # Grain 1 at z=5 (overlap [0,5] = 5 / 10 = 0.5)
    # Grain 2 at z=-3 (overlap [-5,2] = 7 / 10 = 0.7)
    expected = K[0] * I_th[0] * V * _t([1.0, 0.5, 0.7])
    assert torch.allclose(I, expected, atol=1e-12)


def test_scan_axis_pf_unchanged():
    """PF default — make sure adding scan_axis didn't regress PF behavior."""
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=5.0, grain_map=[0],
    )
    V = _t([1.0]); K = _t([2.0]); I_th = _t([3.0])
    I = predicted_spot_intensities(
        V, K, I_th, torch.tensor([0]), torch.tensor([0]),
        _t([0.0]), _t([math.pi/2]), sg, TopHat(20.0),
        scan_axis="pf",
    )
    # Wide beam fully covers voxel -> I = 2*3*1 = 6
    assert math.isclose(float(I[0]), 6.0)


def test_invalid_scan_axis_raises():
    sg = SampleGrid.from_arrays(
        voxel_positions=[[0.0, 0.0, 0.0]], voxel_size_um=5.0, grain_map=[0],
    )
    with pytest.raises(ValueError, match="scan_axis"):
        predicted_spot_intensities(
            _t([1.0]), _t([1.0]), _t([1.0]),
            torch.tensor([0]), torch.tensor([0]),
            _t([0.0]), _t([0.0]), sg, TopHat(5.0),
            scan_axis="weird",
        )


# ----------------------------------------------------------- FF I/O


# The Grains.csv / SpotMatrix.csv fixtures below are at the widths
# process_grains writes TODAY (53 and 28 columns), not the dead 21/12-column
# layouts. The old fixtures were 21 and 12 columns with no ``Matched == 0``
# row, which is exactly why a positional reader that mistook the lattice for
# the strain, and one that let phantom un-found reflections through, both
# passed this file for as long as they did.

#: 53-column Grains.csv header (midas_process_grains.io.csv.GRAINS_HEADER_COLS).
#: Written under the ``%ID`` token here — the spelling the non-c_parity writer
#: emits, and the one the previous midas_stress-backed reader rejected outright.
_GRAINS_53_COLS = (
    ["O11","O12","O13","O21","O22","O23","O31","O32","O33", "X","Y","Z",
     "a","b","c","alpha","beta","gamma",
     "DiffPos","DiffOme","DiffAngle","GrainRadius","Confidence"]
    + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
    + ["RMSErrorStrain","PhaseNr","Eul0","Eul1","Eul2",
       "DiffPosPre","DiffOmePre","DiffAnglePre",
       "DiffPosPost","DiffOmePost","DiffAnglePost"]
)

#: (GrainID, X, Y, Z, GrainRadius, Confidence). DiffPos / DiffOme are written
#: as deliberately different values below, so a reader that takes columns 19
#: and 20 as radius / confidence (which is what positional readers of the
#: 21-column layout do at this width) gets a visibly wrong answer.
_GRAINS_ROWS = (
    (1, 0.0, 0.0, 0.0, 5.0, 0.95),
    (2, 10.0, 5.0, 0.0, 8.0, 0.90),
    (3, -5.0, 15.0, 0.0, 3.0, 0.85),
)
_DIFF_POS = (251.3, 12.4, 88.0)
_DIFF_OME = (0.072, 0.910, 0.455)


def _write_synthetic_grains_csv(path: Path, token: str = "ID"):
    with path.open("w") as f:
        f.write("% provenance line (skipped)\n")
        f.write("%\tSpaceGroup:225\n")
        f.write("%" + token + " " + " ".join(_GRAINS_53_COLS) + "\n")
        for (gid, x, y, z, rad, conf), dpos, dome in zip(
                _GRAINS_ROWS, _DIFF_POS, _DIFF_OME):
            row = ([gid, 1,0,0, 0,1,0, 0,0,1, x, y, z,
                    3, 3, 3, 90, 90, 90,          # lattice, NOT strain
                    dpos, dome, 0.15, rad, conf]
                   + [100.0] * 9 + [200.0] * 9    # eFab / eKen, microstrain
                   + [12.5, 1, 0.1, 0.2, 0.3]
                   + [dpos + 30.0, dome + 0.02, 0.17]
                   + [dpos, dome, 0.15])
            assert len(row) == 53
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def test_load_ff_grains_round_trip(tmp_path: Path):
    p = tmp_path / "Grains.csv"
    _write_synthetic_grains_csv(p)
    g = load_ff_grains_to_tensors(p)
    assert isinstance(g, FFGrainTensors)
    assert g.grain_id.tolist() == [1, 2, 3]
    assert torch.allclose(g.position_um, _t([[0,0,0], [10,5,0], [-5,15,0]]))
    assert torch.allclose(g.radius_um, _t([5.0, 8.0, 3.0]))
    expected_vol = _t([(4/3)*math.pi*r**3 for r in (5.0, 8.0, 3.0)])
    assert torch.allclose(g.volume_um3, expected_vol)
    assert g.confidence is not None
    assert torch.allclose(g.confidence, _t([0.95, 0.90, 0.85]))
    # The specific wrong answers a positional 21-column reader gives at width
    # 53: column 19 is DiffPos and column 20 is DiffOme.
    assert not torch.allclose(g.radius_um, _t(list(_DIFF_POS)))
    assert not torch.allclose(g.confidence, _t(list(_DIFF_OME)))


@pytest.mark.parametrize("token", ["ID", "GrainID"])
def test_load_ff_grains_both_header_tokens(tmp_path: Path, token: str):
    """io/csv writes ``%ID``; c_parity_emit writes ``%GrainID``. Both are real
    files, and the previous midas_stress-backed reader raised on ``%ID``."""
    p = tmp_path / "Grains.csv"
    _write_synthetic_grains_csv(p, token=token)
    g = load_ff_grains_to_tensors(p)
    assert g.grain_id.tolist() == [1, 2, 3]
    assert torch.allclose(g.radius_um, _t([5.0, 8.0, 3.0]))


def test_load_ff_grains_matches_canonical_reader(tmp_path: Path):
    """Guard against drift from ``midas_process_grains.io.read``.

    That module is canonical; ``midas_transforms.io.grains_csv`` mirrors its
    rules locally because midas-process-grains depends on midas-transforms and
    importing it here would make the packaging graph circular. This test keeps
    the two honest wherever both are installed (always, in the monorepo)."""
    read = pytest.importorskip("midas_process_grains.io.read")
    p = tmp_path / "Grains.csv"
    _write_synthetic_grains_csv(p)
    g = load_ff_grains_to_tensors(p)
    t = read.read_grains_csv(p)
    assert g.grain_id.tolist() == t.ids.tolist()
    assert torch.allclose(g.radius_um, _t(t.grain_radius.tolist()))
    assert torch.allclose(g.confidence, _t(t.confidence.tolist()))
    assert torch.allclose(g.position_um, _t(t.positions.tolist()))


def test_load_ff_grains_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_ff_grains_to_tensors(tmp_path / "nonexistent.csv")


#: 28-column SpotMatrix header
#: (midas_process_grains.io.csv.SPOT_MATRIX_HEADER_EXPANDED).
_SPOTMATRIX_28_COLS = (
    ["SpotID","Omega","DetectorHor","DetectorVert","OmeRaw","Eta","RingNr",
     "YLab","ZLab","Theta","StrainError","Matched","theorSpotID","theorRingNr",
     "theorEta","YExp","ZExp","OmegaExp","DiffLen","DiffOme","InternalAngle",
     "YExpPost","ZExpPost","OmegaExpPost","DiffLenPost","DiffOmePost",
     "InternalAnglePost"]
)


def _write_synthetic_spotmatrix(path: Path, n_unmatched: int = 2):
    """28-column SpotMatrix with ``n_unmatched`` predicted-but-never-found rows.

    An un-found row carries ``-1`` in the two integer columns (SpotID, RingNr)
    and NaN in every observed column. Its grain_id is a REAL grain, so before
    the ``Matched`` filter it entered that grain's volume/K refinement as a
    phantom spot with ``intensity = 0`` and ``ring_idx = -1``.
    """
    obs = [
        # GrainID SpotID Omega DetHor DetVert OmeRaw Eta Ring YLab ZLab Theta StrErr
        (1, 101, 10.0, 0, 0, 10.0,  30.0, 1, 0, 0, 1.5, 0.0),
        (1, 102, 20.0, 0, 0, 20.0,  60.0, 1, 0, 0, 1.5, 0.0),
        (2, 103, 30.0, 0, 0, 30.0,  90.0, 2, 0, 0, 2.0, 0.0),
        (2, 104, 40.0, 0, 0, 40.0, -30.0, 9, 0, 0, 1.0, 0.0),  # ring 9 unknown
    ]
    with path.open("w") as f:
        f.write("%GrainID\t" + "\t".join(_SPOTMATRIX_28_COLS) + "\n")
        for r in obs:
            row = list(r) + [1] + [r[1], r[7], r[6]] + [1.0] * 12
            assert len(row) == 28
            # written with newline='\t\n': every data row ends in a tab
            f.write("\t".join(f"{v:.6g}" for v in row) + "\t\n")
        for k in range(n_unmatched):
            row = ([1, -1] + ["nan"] * 5 + [-1] + ["nan"] * 4 + [0]
                   + [900 + k, 3, 44.0] + ["nan"] * 12)
            assert len(row) == 28
            f.write("\t".join(str(v) for v in row) + "\t\n")


def _write_synthetic_inputall(path: Path, intensities):
    """Cols: ... SpotID(4) ... IntegratedIntensity(14)."""
    n = 15
    with path.open("w") as f:
        f.write("% header\n")
        for sid, integ in intensities:
            row = [0.0] * n
            row[4] = sid; row[14] = integ
            f.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def test_load_ff_spots_round_trip(tmp_path: Path):
    sm = tmp_path / "SpotMatrix.csv"; _write_synthetic_spotmatrix(sm)
    ie = tmp_path / "ie.csv"
    _write_synthetic_inputall(
        ie, [(101, 100.0), (102, 200.0), (103, 50.0), (104, 25.0)],
    )
    grains = load_ff_grains_to_tensors(_make_tmp_grains(tmp_path))
    rt = RingTable(
        ring_numbers=torch.tensor([1, 2], dtype=torch.int64),
        two_theta_deg=_t([3.0, 4.0]),
    )
    sp = load_ff_spots_to_tensors(sm, ie, ring_table=rt, grain_table=grains)
    # The two Matched == 0 rows are gone: 4 observations, not 6.
    assert sp.spot_id.tolist() == [101, 102, 103, 104]
    assert sp.grain_id.tolist() == [1, 1, 2, 2]
    assert sp.grain_idx.tolist() == [0, 0, 1, 1]  # MIDAS GrainID 1,2 -> rows 0,1
    assert sp.ring_number.tolist() == [1, 1, 2, 9]
    assert sp.ring_idx.tolist() == [0, 0, 1, -1]  # ring 9 unknown
    assert sp.intensity.tolist() == [100.0, 200.0, 50.0, 25.0]
    assert sp.omega_deg.tolist() == [10.0, 20.0, 30.0, 40.0]


def test_load_ff_spots_drops_unmatched_predictions(tmp_path: Path):
    """A ``Matched == 0`` row is a prediction that was never observed.

    It carries ``SpotID = RingNr = -1``, so it joins to no intensity
    (``0.0``) and to no ring (``ring_idx = -1``) — but its grain_idx is a real
    grain, so leaving it in adds a phantom zero-intensity spot to that grain's
    volume and K refinement.
    """
    from midas_transforms.radius.theoretical import _parse_spotmatrix_csv

    sm = tmp_path / "SpotMatrix.csv"
    _write_synthetic_spotmatrix(sm, n_unmatched=3)
    raw = _parse_spotmatrix_csv(sm)
    assert raw["n_rows_total"] == 7
    assert raw["n_rows_unmatched"] == 3
    assert -1 not in raw["spot_id"].tolist()
    assert -1 not in raw["ring_nr"].tolist()
    assert np.isfinite(raw["omega_deg"]).all()
    assert np.isfinite(raw["eta_deg"]).all()

    # matched_only=False is the opt-in escape hatch for the un-found population.
    raw_all = _parse_spotmatrix_csv(sm, matched_only=False)
    assert len(raw_all["spot_id"]) == 7
    assert -1 in raw_all["spot_id"].tolist()

    rt = RingTable(
        ring_numbers=torch.tensor([1, 2], dtype=torch.int64),
        two_theta_deg=_t([3.0, 4.0]),
    )
    grains = load_ff_grains_to_tensors(_make_tmp_grains(tmp_path))
    sp = load_ff_spots_to_tensors(sm, None, ring_table=rt, grain_table=grains)
    assert sp.spot_id.tolist() == [101, 102, 103, 104]
    # Before the filter these three rows all mapped to grain 1 (grain_idx 0).
    assert sp.grain_idx.tolist() == [0, 0, 1, 1]


def test_load_ff_spots_matches_canonical_reader(tmp_path: Path):
    """Same anti-drift guard as for Grains.csv (see that test's docstring)."""
    read = pytest.importorskip("midas_process_grains.io.read")
    from midas_transforms.radius.theoretical import _parse_spotmatrix_csv

    sm = tmp_path / "SpotMatrix.csv"
    _write_synthetic_spotmatrix(sm, n_unmatched=2)
    raw = _parse_spotmatrix_csv(sm)
    t = read.read_spot_matrix(sm)
    assert raw["spot_id"].tolist() == t.spot_id.tolist()
    assert raw["grain_id"].tolist() == t.grain_id.tolist()
    assert raw["ring_nr"].tolist() == t.ring_nr.tolist()
    assert raw["n_rows_total"] == t.n_rows_total
    assert raw["n_rows_unmatched"] == t.n_rows_unmatched
    np.testing.assert_allclose(raw["omega_deg"], t.omega)


def test_load_ff_spots_no_intensity_csv(tmp_path: Path):
    sm = tmp_path / "SpotMatrix.csv"; _write_synthetic_spotmatrix(sm)
    rt = RingTable(
        ring_numbers=torch.tensor([1, 2], dtype=torch.int64),
        two_theta_deg=_t([3.0, 4.0]),
    )
    sp = load_ff_spots_to_tensors(sm, None, ring_table=rt)
    # Intensities default to 0; grain_idx -1 (no grain table)
    assert sp.intensity.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert sp.grain_idx.tolist() == [-1, -1, -1, -1]


def _make_tmp_grains(tmp_path: Path) -> Path:
    p = tmp_path / "Grains.csv"
    _write_synthetic_grains_csv(p)
    return p


# ----------------------------------------------------------- end-to-end FF refine


def test_ff_compact_refine_recovers_per_grain_V():
    """5 FF grains with known per-grain V; refine_V should recover them."""
    torch.manual_seed(0)
    n_grains = 5
    centroids = torch.rand(n_grains, 3, dtype=torch.float64) * 50.0
    sg = SampleGrid.from_grain_centroids(centroids_um=centroids)

    V_true = torch.rand(n_grains, dtype=torch.float64) * 2.0 + 0.5
    K_true = _t([4.2]); I_th = _t([11.0])

    n_spots_per = 10
    spot_grain = torch.repeat_interleave(torch.arange(n_grains), n_spots_per)
    n_spots = n_grains * n_spots_per
    spot_ring  = torch.zeros(n_spots, dtype=torch.int64)
    spot_scan  = torch.zeros(n_spots, dtype=torch.float64)
    spot_omega = torch.rand(n_spots, dtype=torch.float64) * 2 * math.pi

    I_obs = predicted_spot_intensities(
        V_true, K_true, I_th, spot_ring, spot_grain,
        spot_scan, spot_omega, sg, TopHat(1.0), scan_axis="none",
    )
    V_init = torch.full_like(V_true, float(V_true.mean()))
    result = refine_vmap_joint(
        V_init=V_init, K_init=K_true.clone(),
        spot_observed_intensity=I_obs,
        spot_ring_idx=spot_ring, spot_grain_idx=spot_grain,
        spot_scan_pos_um=spot_scan, spot_omega_rad=spot_omega,
        sample_grid=sg, beam_profile=TopHat(1.0),
        theoretical_intensity_per_ring=I_th,
        scan_axis="none",
        refine_V=True, refine_K=False,
        max_iter=40, tolerance=1e-12,
    )
    assert torch.allclose(result.V_voxel, V_true, atol=1e-5)
    assert result.converged


def test_ff_height_scan_refine_recovers_per_grain_V():
    """FF tall sample: 5 grains at staggered z, beam scans over z. Refine V."""
    torch.manual_seed(1)
    n_grains = 5
    # Place grain centroids spanning z = 0..40 with random xy
    z_centers = torch.linspace(0.0, 40.0, n_grains, dtype=torch.float64)
    centroids = torch.stack([
        torch.rand(n_grains, dtype=torch.float64) * 30.0,
        torch.rand(n_grains, dtype=torch.float64) * 30.0,
        z_centers,
    ], dim=1)
    sg = SampleGrid.from_grain_centroids(centroids_um=centroids, voxel_size_um=8.0)

    V_true = _t([0.8, 1.2, 1.0, 1.5, 0.7])
    K_true = _t([2.0]); I_th = _t([5.0])
    beam = TopHat(10.0)  # 10 µm beam height

    # 8 scan positions per grain (16 per grain across 2 rings actually keep simple)
    scan_positions = torch.linspace(0.0, 40.0, 8, dtype=torch.float64)
    # 5 grains × 8 scans = 40 spots
    spot_grain = torch.repeat_interleave(torch.arange(n_grains), 8)
    spot_scan  = scan_positions.repeat(n_grains)
    spot_omega = torch.zeros_like(spot_scan)
    spot_ring  = torch.zeros_like(spot_grain)

    I_obs = predicted_spot_intensities(
        V_true, K_true, I_th, spot_ring, spot_grain,
        spot_scan, spot_omega, sg, beam, scan_axis="z",
    )
    # Filter zero-weight spots (scan position out of any grain's beam reach)
    keep = I_obs > 1e-12
    spot_grain, spot_scan, spot_omega, spot_ring, I_obs = (
        spot_grain[keep], spot_scan[keep], spot_omega[keep],
        spot_ring[keep], I_obs[keep],
    )

    V_init = torch.full_like(V_true, float(V_true.mean()))
    result = refine_vmap_joint(
        V_init=V_init, K_init=K_true.clone(),
        spot_observed_intensity=I_obs,
        spot_ring_idx=spot_ring, spot_grain_idx=spot_grain,
        spot_scan_pos_um=spot_scan, spot_omega_rad=spot_omega,
        sample_grid=sg, beam_profile=beam,
        theoretical_intensity_per_ring=I_th,
        scan_axis="z",
        refine_V=True, refine_K=False,
        max_iter=60, tolerance=1e-12,
    )
    assert torch.allclose(result.V_voxel, V_true, atol=1e-4)


def test_ff_compact_grad_flows_to_V():
    """Differentiability through scan_axis='none' path."""
    sg = SampleGrid.from_grain_centroids(
        centroids_um=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        voxel_size_um=5.0,
    )
    V = torch.tensor([1.0, 2.0], dtype=torch.float64, requires_grad=True)
    K = _t([3.0]); I_th = _t([1.0])
    I = predicted_spot_intensities(
        V, K, I_th,
        torch.tensor([0, 0]), torch.tensor([0, 1]),
        _t([0.0, 0.0]), _t([0.0, 0.0]),
        sg, TopHat(1.0), scan_axis="none",
    )
    I.sum().backward()
    # dI[0]/dV[0] = 3*1 = 3; dI[1]/dV[1] = 3*1 = 3; cross-terms 0
    assert torch.allclose(V.grad, _t([3.0, 3.0]))
