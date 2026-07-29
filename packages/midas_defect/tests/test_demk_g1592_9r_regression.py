"""Real-data regression: the g1592 / L3786 9R ladder through the polytype pipeline.

Locks in the hand-derived demk findings (session 2026-06-24) via the packaged
modules, on a small committed voxel fixture (per-label capped so the weak
high-order rungs survive). This is the second real-data anchor (the first is the
L1/L2346 matrix-twin regression). If the fixture is absent the test skips.

Headline numbers reproduced:
  * G/3 satellite resolves into TWO reflections (Ewald-artifact test), azimuth ~180
    (twin polarity), distinct omega.
  * the n*G/3 ladder decontaminates to 4 fundamentals (+-111, +-222) + 8 forbidden-
    gap 9R satellites (5G/3 stays a satellite -- not a 220).
  * modulation tilt beta ~3 deg; 9R well-ordered (high periodicity fraction).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from midas_defect.lattice import fcc_cu_crystal
from midas_defect.polytype import (
    build_satellite_ladder,
    decontaminate_ladder,
    fit_modulation_tilt,
    periodic_aperiodic_balance,
    resolve_satellite_doublet,
)

_FIX = Path(__file__).parent / "fixtures" / "demk_g1592_9r.npz"


@pytest.fixture(scope="module")
def g1592():
    if not _FIX.exists():
        pytest.skip(f"g1592 fixture not present: {_FIX}")
    d = np.load(_FIX, allow_pickle=True)
    OM = np.asarray(d["OM"], float)
    a = float(d["a_fcc"])
    B0 = 2 * math.pi / a
    G = math.sqrt(3) * B0
    U = OM.T  # package U @ G_crystal convention <- raw MIDAS Grains.csv OM transposed
    axis = U @ np.asarray(d["hkl_axis"], float)
    axis = axis / np.linalg.norm(axis)
    return dict(q=np.asarray(d["q"], float), I=np.asarray(d["intensity"], float),
                om=np.asarray(d["omega"], float), U=U, axis=axis, G=G, g3=G / 3.0)


def test_g3_satellite_resolves_to_two_twin_related_reflections(g1592):
    res = resolve_satellite_doublet(g1592["q"], g1592["I"], g1592["om"],
                                    g1592["axis"], g1592["g3"])
    assert res.n_members == 2
    assert res.verdict == "two-reflections"
    assert abs(res.azimuth_deg - 180.0) < 20.0
    assert res.is_twin_polarity
    assert res.metadata["omega_sep_deg"] > 4.0
    # honesty: no host/lamella identity leaked onto members
    for m in res.members:
        assert "parent" not in m and "twin" not in m


def test_ladder_decontaminates_to_fundamentals_and_satellites(g1592):
    cr = fcc_cu_crystal()
    lad = build_satellite_ladder(g1592["q"], g1592["I"], g1592["axis"], g1592["G"])
    lad = decontaminate_ladder(lad, g1592["U"], cr, tol_inv_A=0.25)
    cls = {r["n"]: r["classification"] for r in lad.rungs}
    for n in (3, -3, 6, -6):
        assert cls.get(n) == "fundamental", (n, cls)
    # the forbidden-gap rungs that survived the per-label cap are 9R satellites
    for n in (1, -1, 2, -2, 4, -4, 5, -5):
        if n in cls:
            assert cls[n] == "9R-satellite", (n, cls)
    assert lad.metadata["n_fundamentals"] == 4
    assert lad.metadata["n_satellites"] >= 6


def test_modulation_tilt_about_three_degrees(g1592):
    """Split |Delta_perp| ~ 0.105 at G/3, ~0.20 at 2G/3 -> beta ~3 deg."""
    g3 = g1592["g3"]
    splits = []
    orders = []
    for n in (1, 2):
        res = resolve_satellite_doublet(g1592["q"], g1592["I"], g1592["om"],
                                        g1592["axis"], n * g3)
        if res.n_members == 2:
            # transverse split = |perp_a - perp_b| projected; use member perp offsets
            pa = res.members[0]["perp_vec"]
            pb = res.members[1]["perp_vec"]
            splits.append(float(np.linalg.norm(np.asarray(pa) - np.asarray(pb))))
            orders.append(n)
    if len(orders) >= 2:
        out = fit_modulation_tilt(orders, splits, g3)
        assert 1.5 < out["beta_deg"] < 5.0


def test_well_ordered_9r_high_periodicity(g1592):
    b = periodic_aperiodic_balance(g1592["q"], g1592["I"], g1592["axis"], g1592["G"])
    # discrete satellites dominate the inter-fundamental signal (sharp 9R, faint relrod)
    assert b["periodicity_fraction"] > 0.8
    assert math.isfinite(b["satellite_to_fundamental"])
