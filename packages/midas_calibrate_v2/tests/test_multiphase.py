"""Multi-phase calibration support, blend clustering, and the new gates."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_calibrate.params import CalibrationParams as V1Params

from midas_calibrate_v2.forward.doublets import (
    cluster_rings, detect_doublets, doublet_index_map,
)
from midas_calibrate_v2.loss.diagnostics import per_phase_summary, strain_summary
from midas_calibrate_v2.pipelines.diagnostics import (
    azimuth_coverage_gate, rho_d_scaling_gate,
)
from midas_calibrate_v2.pipelines.multi import build_multi_spec, MULTI_MODES
from midas_calibrate_v2.pipelines._common import FittedDataset
from midas_calibrate_v2.seed.calibrant import (
    resolve_calibrant, resolve_calibrants, phases_from_calibrants,
    lattice_uncertainty_lsd_ppm,
)


# ---------------------------------------------------------------- clustering

def test_pair_is_a_pair():
    c = cluster_rings(np.array([100.0, 110.0, 400.0]), min_separation_px=25.0)
    assert [(g.i, g.j) for g in c.pairs] == [(0, 1)]
    assert c.n_ary == []
    assert c.singletons == [2]


def test_chain_of_three_is_not_two_overlapping_pairs():
    """The bug this fixes: rings (k,k+1) and (k+1,k+2) both look like doublets,
    so the interior ring gets co-fitted twice at two different centres."""
    R = np.array([100.0, 110.0, 120.0, 400.0])
    c = cluster_rings(R, min_separation_px=25.0)
    assert c.pairs == []
    assert c.n_ary == [[0, 1, 2]]
    assert c.singletons == [3]

    # no ring appears in more than one emitted pair
    pairs = detect_doublets(R, min_separation_px=25.0)
    seen = [i for g in pairs for i in (g.i, g.j)]
    assert len(seen) == len(set(seen))

    partner, _ = doublet_index_map(R, min_separation_px=25.0)
    assert list(partner) == [-2, -2, -2, -1]     # -2 = inside a >=3 blend


def test_legacy_chaining_available_for_comparison():
    R = np.array([100.0, 110.0, 120.0])
    assert detect_doublets(R, min_separation_px=25.0) == []
    legacy = detect_doublets(R, min_separation_px=25.0, include_n_ary=True)
    seen = [i for g in legacy for i in (g.i, g.j)]
    assert len(seen) != len(set(seen))          # the historical double-count


def test_clustering_is_order_independent():
    R = np.array([400.0, 110.0, 100.0, 120.0])
    c = cluster_rings(R, min_separation_px=25.0)
    assert sorted(c.n_ary[0]) == [1, 2, 3]
    assert c.singletons == [0]


def test_empty_and_single_ring():
    assert cluster_rings(np.array([]), min_separation_px=25.0).pairs == []
    c = cluster_rings(np.array([100.0]), min_separation_px=25.0)
    assert c.singletons == [0]


# ------------------------------------------------------------- calibrant list

def test_resolve_calibrants_accepts_one_or_many():
    assert len(resolve_calibrants("CeO2")) == 1
    specs = resolve_calibrants(["CeO2", "LaB6"])
    assert [s["name"] for s in specs] == ["CeO2", "LaB6"]
    assert specs[1]["sg"] == 221


def test_duplicate_calibrant_names_are_disambiguated():
    specs = resolve_calibrants(["CeO2", "CeO2"])
    assert [s["name"] for s in specs] == ["CeO2", "CeO2#2"]


def test_resolve_calibrant_rejects_a_list_with_a_pointer():
    with pytest.raises(TypeError, match="resolve_calibrants"):
        resolve_calibrant(["CeO2", "LaB6"])


def test_phases_from_calibrants_shape():
    ph = phases_from_calibrants(["CeO2", "LaB6"])
    assert [p["name"] for p in ph] == ["CeO2", "LaB6"]
    assert all(len(p["lattice"]) == 6 for p in ph)


def test_custom_dict_keeps_its_name():
    spec = resolve_calibrant({"a": 4.0, "sg": 221, "name": "MyLaB6"})
    assert spec["name"] == "MyLaB6"


def test_lattice_uncertainty_maps_to_lsd_ppm():
    # da/a and dLsd/Lsd enter the ring radius identically.
    assert lattice_uncertainty_lsd_ppm(4.15689, 4.15689e-4) == pytest.approx(100.0)
    with pytest.raises(ValueError):
        lattice_uncertainty_lsd_ppm(0.0, 1e-4)


# ------------------------------------------------------------ per-phase report

def _phase_resid(n_a, n_b, mean_a, mean_b):
    r = torch.cat([torch.full((n_a,), mean_a, dtype=torch.float64),
                   torch.full((n_b,), mean_b, dtype=torch.float64)])
    pid = torch.cat([torch.zeros(n_a, dtype=torch.long),
                     torch.ones(n_b, dtype=torch.long)])
    return r, pid


def test_per_phase_summary_flags_disagreement():
    r, pid = _phase_resid(50, 50, 50.0, 200.0)
    out = per_phase_summary(r, pid, ("CeO2", "LaB6"))
    assert "CeO2" in out and "LaB6" in out
    assert "disagree" in out
    assert "4.00×" in out


def test_per_phase_summary_passes_when_phases_agree():
    r, pid = _phase_resid(50, 50, 50.0, 55.0)
    out = per_phase_summary(r, pid, ("CeO2", "LaB6"))
    assert "agree" in out


def test_per_phase_summary_single_phase_is_explicit():
    r = torch.full((10,), 50.0, dtype=torch.float64)
    pid = torch.zeros(10, dtype=torch.long)
    out = per_phase_summary(r, pid, ("CeO2",))
    assert "single calibrant" in out


def test_strain_summary_includes_phase_block_when_asked():
    r, pid = _phase_resid(30, 30, 40.0, 90.0)
    out = strain_summary(r, phase_idx=pid, phase_names=("CeO2", "LaB6"))
    assert "per-phase" in out


# -------------------------------------------------------------- multi spec

def _v1(**kw):
    p = V1Params(NrPixelsY=512, NrPixelsZ=512, pxY=200.0, pxZ=200.0,
                 Lsd=1e6, BC_y=256.0, BC_z=256.0, Wavelength=0.15,
                 SpaceGroup=225,
                 LatticeConstant=(5.41, 5.41, 5.41, 90.0, 90.0, 90.0),
                 MaxRingRad=250.0, RhoD=50000.0)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_same_detector_shares_tilts_and_leaves_position_per_image():
    ms = build_multi_spec([_v1(), _v1()], mode="same_detector")
    for t in ("tx", "ty", "tz"):
        assert t in ms.shared, f"{t} must be shared when the detector is one detector"
    for name in ("Lsd", "BC_y", "BC_z"):
        assert name in ms.per_image[0], f"{name} is the per-exposure sample position"
        assert name not in ms.shared


def test_independent_mode_is_unchanged():
    ms = build_multi_spec([_v1(), _v1()])
    for t in ("tx", "ty", "tz"):
        assert t not in ms.shared


def test_same_detector_augments_explicit_shared_names():
    ms = build_multi_spec([_v1(), _v1()], shared_names=["pxY"],
                          mode="same_detector")
    assert {"tx", "ty", "tz"}.issubset(set(ms.shared))
    assert "pxY" in ms.shared


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        build_multi_spec([_v1()], mode="nonsense")
    assert "same_detector" in MULTI_MODES


# ------------------------------------------------------------------- gates

def _fits_over_arc(span_deg, n=200, r_px=300.0, bc=256.0, rho_d=50000.0):
    eta = np.deg2rad(np.linspace(-span_deg / 2.0, span_deg / 2.0, n))
    Y = bc - r_px * np.cos(eta)
    Z = bc + r_px * np.sin(eta)
    t = lambda v: torch.as_tensor(v, dtype=torch.float64)
    return FittedDataset(
        Y_pix=t(Y), Z_pix=t(Z),
        ring_idx=torch.zeros(n, dtype=torch.long),
        snr=torch.full((n,), 10.0, dtype=torch.float64),
        ring_two_theta_deg=torch.full((n,), 3.0, dtype=torch.float64),
        rho_d=t(rho_d), weights=torch.ones(n, dtype=torch.float64))


def _unpacked(**kw):
    base = dict(Lsd=1e6, BC_y=256.0, BC_z=256.0, tx=0.0, ty=0.0, tz=0.0,
                pxY=200.0, pxZ=200.0)
    base.update(kw)
    return {k: torch.as_tensor(float(v), dtype=torch.float64)
            for k, v in base.items()}


def test_azimuth_gate_passes_on_full_rings():
    d = azimuth_coverage_gate(_fits_over_arc(359.0), _unpacked())
    assert d.severity == "ok"
    assert d.metrics["covered_fraction"] > 0.9


def test_azimuth_gate_fails_on_a_narrow_wedge_with_harmonics_refined():
    """The ge1 case: ~70 deg of azimuth while refining a1..a6."""
    unp = _unpacked(a1=1e-3, a2=1e-3, a3=1e-3, a4=1e-3, a5=1e-3, a6=1e-3)
    d = azimuth_coverage_gate(_fits_over_arc(70.0), unp)
    assert d.severity == "fail"
    assert d.metrics["covered_fraction"] < 0.25
    assert "second calibrant will NOT help" in d.message


def test_azimuth_gate_only_warns_when_harmonics_are_frozen():
    d = azimuth_coverage_gate(_fits_over_arc(70.0), _unpacked())
    assert d.severity == "warn"


def test_azimuth_gate_reports_longest_arc():
    d = azimuth_coverage_gate(_fits_over_arc(90.0), _unpacked())
    assert 80.0 <= d.metrics["longest_arc_deg"] <= 100.0


def test_rho_d_gate_flags_an_oversized_rhod():
    """RhoD 3.6x the outer ring is what killed iso_R4/iso_R6 on real data."""
    fits = _fits_over_arc(359.0, r_px=300.0, rho_d=2.0e6)   # ring is 60 kum
    unp = _unpacked(iso_R2=1e-3, iso_R4=1e-3, iso_R6=1e-3)
    d = rho_d_scaling_gate(fits, unp)
    assert d.severity == "fail"
    assert d.metrics["ratio"] > 3.0
    assert "no lever" in d.message


def test_rho_d_gate_ok_when_scaled_to_the_outer_ring():
    fits = _fits_over_arc(359.0, r_px=300.0, rho_d=300.0 * 200.0)
    d = rho_d_scaling_gate(fits, _unpacked(iso_R2=1e-3))
    assert d.severity == "ok"
    assert d.metrics["rho_max"] == pytest.approx(1.0, rel=1e-6)
