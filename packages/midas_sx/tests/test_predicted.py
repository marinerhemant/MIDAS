"""Predicted-position integration (reduce_scan(integrate_predicted=True), A39-FLOOR step 1c) against the independent simulation of test_workflow."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import replace

from test_workflow import _crystal, scene                                  # noqa: F401  (the module-scoped simulated scene)
from midas_sx.reduce import reduce_spots
from midas_sx.workflow import predicted_targets, reduce_scan


def test_default_is_the_blob_driven_reduction_bit_for_bit(scene):          # noqa: F811
    exp, st, res, raw, setup, U0 = scene
    setup = replace(setup, dist_mm=exp.dist_mm, centre_rc=exp.centre_rc)           # exact geometry: no orientation refinement here
    red = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False)
    obs, rej, counts = reduce_spots(red.spots, raw, red.geom, setup.scan(raw.shape[0]), _crystal(st).lattice, red.U, red.profile, dac=setup.dac, pol=setup.pol, mask=red.mask)
    pd.testing.assert_frame_equal(red.obs, obs)
    assert red.targets is None and "predicted" not in red.info


def test_predicted_mode_contains_every_blob_driven_observation_with_the_same_f2(scene):   # noqa: F811
    exp, st, res, raw, setup, U0 = scene
    setup = replace(setup, dist_mm=exp.dist_mm, centre_rc=exp.centre_rc)           # exact geometry: no orientation refinement here
    a = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False)
    b = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False, integrate_predicted=True)
    assert b.counts["no_crossing"] == 0                                       # targets sit EXACTLY on the crossing (the exact-root bracket bug)
    assert b.info["predicted"]["n_hkl_differs"] == 0 and b.targets is not None and len(b.targets) == b.counts["input"]
    m = b.obs.merge(a.obs, on=["h", "k", "l"], suffixes=("_p", "_b"))
    m = m[(m.frame_p - m.frame_b).abs() < 1.5]
    assert len(m) == len(a.obs)                                               # every blob-driven observation has its predicted counterpart
    assert np.allclose(m.F2_p, m.F2_b, rtol=1e-6) and np.allclose(m.I_p, m.I_b, rtol=1e-6)
    assert b.obs.blob_found.sum() >= len(a.obs)


def test_a_targets_own_blob_is_not_its_own_crowding_neighbour(scene):       # noqa: F811
    exp, st, res, raw, setup, U0 = scene
    setup = replace(setup, dist_mm=exp.dist_mm, centre_rc=exp.centre_rc)           # exact geometry: no orientation refinement here
    a = reduce_scan(raw, _crystal(st), res.U, setup, refine_orientation=False)
    t, nb = predicted_targets(a.spots, _crystal(st).lattice, _crystal(st).space_group, a.U, a.geom, setup.scan(raw.shape[0]))
    owned = int(t.blob_found.sum())
    assert owned > 0 and len(nb) == len(t) + len(a.spots) - owned             # owned blobs are replaced by their target; unowned blobs stay
