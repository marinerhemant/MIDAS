"""Orientation refinement on the pair model of a two-component twin (PREREGISTER A27): synthetic blob centroids, known answer."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

pytest.importorskip("gemmi")
import simulate as sim                                              # noqa: E402

from midas_hkls import Lattice, SpaceGroup                          # noqa: E402
from midas_sx.geometry import Scan, make_geometry                   # noqa: E402
from midas_sx.orient import index_and_refine                        # noqa: E402
from midas_sx.predict import enumerate_hkl, twin_targets            # noqa: E402
from midas_sx.twin_orient import RZ90, index_and_refine_twin, refine_orientation_pairs   # noqa: E402

M = 0.3


@pytest.fixture(scope="module")
def scene():
    exp = sim.make_experiment(det_n=512, centre_rc=(255.5, 255.5), omega_start=-40.0, n_frames=320)
    scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
    geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
    lat = Lattice(3.83, 3.83 * 1.03, 20.5, 90, 90, 90)
    sg = SpaceGroup.from_hm("I m m m")
    U = R.random(random_state=np.random.RandomState(4)).as_matrix()
    hkl = enumerate_hkl(lat, 1.0, sg)
    tg = twin_targets(U, U @ RZ90, lat, geom, scan, hkl)
    # blobs: a blend is one blob at (1 - M) p1 + M p2 (equal F2), a resolved pair is two blobs
    blobs = []
    rng = np.random.default_rng(0)
    for t in tg.itertuples():
        if t.kind == "blend":
            blobs.append((0.7 * t.frame1 + 0.3 * t.frame2, 0.7 * t.row1 + 0.3 * t.row2, 0.7 * t.col1 + 0.3 * t.col2))
        elif t.kind == "c1":
            blobs.append((t.frame1, t.row1, t.col1))
        else:
            blobs.append((t.frame2, t.row2, t.col2))
    b = np.array(blobs) + rng.normal(0, [0.05, 0.05, 0.05], (len(blobs), 3))
    spots = pd.DataFrame(dict(frame=b[:, 0], row=b[:, 1], col=b[:, 2]))
    return exp, scan, geom, lat, U, hkl, spots, tg


def test_there_are_blends_and_resolved_components_in_the_scene(scene):
    tg = scene[-1]
    assert (tg.kind == "blend").sum() > 100 and (tg.kind == "c1").sum() > 10 and (tg.kind == "c2").sum() > 10


def test_pair_model_recovers_u_geometry_and_m_where_the_centroid_model_is_biased(scene):
    exp, scan, geom, lat, U, hkl, spots, tg = scene
    ax = np.array([0.3, -0.5, 0.8]); ax /= np.linalg.norm(ax)
    U0 = R.from_rotvec(np.radians(0.3) * ax).as_matrix() @ U
    g0 = replace(geom, bcz_px=geom.bcz_px + 0.4, bcy_px=geom.bcy_px - 0.3, lsd_um=geom.lsd_um * 1.008)
    mis = lambda A, B: float(np.degrees(np.linalg.norm(R.from_matrix(A @ B.T).as_rotvec())))
    base = index_and_refine(spots, lat, U0, g0, scan)
    twin = index_and_refine_twin(spots, lat, base.U, base.geom, scan, hkl)
    assert twin.converged, twin.reason
    assert mis(twin.U, U) < 0.005 and mis(twin.U, U) < 0.5 * mis(base.U, U)             # a known-answer improvement over the centroid model
    assert abs(twin.rounds[-1]["m"] - M) < 0.03
    assert abs(twin.geom.lsd_um / geom.lsd_um - 1.0) < 1e-3 and abs(twin.geom.bcz_px - geom.bcz_px) < 0.05


def test_refine_orientation_pairs_validates_its_input(scene):
    exp, scan, geom, lat, U, hkl, spots, tg = scene
    with pytest.raises(ValueError, match="need >= 12"):
        refine_orientation_pairs(U, geom, lat, scan, hkl[:5], hkl[:5], np.zeros(5, int), np.zeros(5), np.zeros(5), np.zeros(5))
