"""The default stratified trim bucketed by 2-theta, so it did nothing.

`single_pv.py` and `single_pv_2d.py` passed `ring_two_theta_deg` into
`stratified_trim`'s `eta_deg` POSITIONAL slot. 2-theta is constant within a
ring, so the `(ring_idx, eta_bucket, panel_idx)` cell key collapsed onto
`(ring_idx, panel_idx)`: the azimuthal stratification was not mis-bucketed, it
was INERT, and had been for as long as the stratified trim had existed.

The point of stratifying by azimuth is that a trim which is free to reject
whatever it likes will preferentially reject one side of a ring when the
geometry is still slightly wrong -- which is exactly the asymmetry a tilt or
beam-centre fit then has to absorb.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

# ─────────────────────────────────────────────── the stratified-trim slot bug
def test_stratified_trim_buckets_by_azimuth_not_by_two_theta():
    """single_pv.py and single_pv_2d.py passed `ring_two_theta_deg` into the
    `eta_deg` positional slot. 2-theta is constant within a ring, so the
    (ring, eta_bucket, panel) cell key collapsed onto (ring, panel) and the
    azimuthal stratification was not mis-bucketed -- it was INERT."""
    from midas_calibrate_v2.loss.robust_trim import azimuth_deg

    BCy, BCz = 1024.0, 1024.0
    Y, Z, ring = [], [], []
    for r_i, R in enumerate((300.0, 600.0)):
        th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        Y.append(BCy + R * np.cos(th)); Z.append(BCz + R * np.sin(th))
        ring.append(np.full(400, r_i))
    Y = torch.tensor(np.concatenate(Y)); Z = torch.tensor(np.concatenate(Z))
    rid = torch.tensor(np.concatenate(ring), dtype=torch.long)
    two_theta = torch.tensor(np.where(np.concatenate(ring) == 0, 5.0, 9.0))

    def n_cells(az):
        b = ((az + 180.0) % 360.0 / 45.0).floor().long().clamp(0, 7)
        return len({(int(a), int(c)) for a, c in zip(rid.tolist(), b.tolist())})

    assert n_cells(two_theta) == 2, "the old argument collapsed the cell key"
    assert n_cells(azimuth_deg(Y, Z, BCy, BCz)) == 16, "2 rings x 8 buckets"


@pytest.mark.parametrize("mod", ["single_pv", "single_pv_2d"])
def test_trim_call_sites_pass_an_azimuth(mod):
    m = __import__(f"midas_calibrate_v2.pipelines.{mod}", fromlist=["x"])
    src = inspect.getsource(m)
    assert "azimuth_deg(fits_ds.Y_pix" in src
    assert "r_pre, fits_ds.ring_idx, fits_ds.ring_two_theta_deg," not in src
