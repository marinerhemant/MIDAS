"""Regression test: the synthetic test scaffolding must support the strain
forward path of HEDMForwardModel.correct_hkls_latc.

The strain-coupling robustness study revealed that conftest.make_model()
was constructing the forward model without hkls_int, which silently
disabled the strain code path. Strain calls then raised
RuntimeError("correct_hkls_latc requires integer Miller indices ...")
which the surrounding study driver caught and reported as NaN.

This test exercises the strain forward path explicitly so any future
regression of make_model() / make_fcc_hkls() will fail at test time
rather than silently break a downstream study.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from conftest import make_model, random_orientation  # noqa: E402

from midas_grain_odf.forward_helpers import forward_orientations  # noqa: E402

DEG = math.pi / 180.0


def test_make_model_carries_hkls_int():
    """make_model() must produce a model with non-None hkls_int."""
    model = make_model()
    assert model.hkls_int is not None, (
        "tests/conftest.py:make_model() built a model without hkls_int; "
        "the strain forward path will raise. Pass hkls_int=hkls_int into "
        "HEDMForwardModel(...)."
    )
    assert model.hkls_int.shape[-1] == 3
    assert model.hkls_int.shape[0] == model.hkls.shape[0]


def test_strain_forward_runs_and_perturbs_spots():
    """A non-zero strain on a single grain must change predicted spot
    positions relative to the unstrained baseline. If the strain code
    path silently no-ops (or errors), this test fails.
    """
    model = make_model()
    R_avg = random_orientation(seed=11).to(torch.float64)
    position = torch.zeros(3, dtype=torch.float64)

    # Reference Cu lattice and a 0.3 % anisotropic strain on the a-axis.
    a0 = 3.61
    lat_unstrained = torch.tensor(
        [a0, a0, a0, 90.0, 90.0, 90.0], dtype=torch.float64,
    )
    lat_strained = torch.tensor(
        [a0 * 1.003, a0, a0, 90.0, 90.0, 90.0], dtype=torch.float64,
    )

    R = R_avg.unsqueeze(0)  # (1, 3, 3)
    spots_a = forward_orientations(
        model, R, position, lattice_params=lat_unstrained,
    )
    spots_b = forward_orientations(
        model, R, position, lattice_params=lat_strained,
    )

    valid = (spots_a.valid > 0.5) & (spots_b.valid > 0.5)
    assert int(valid.sum()) > 0, "no overlapping valid spots in test geometry"

    dy = (spots_a.y_pixel - spots_b.y_pixel)[valid]
    dz = (spots_a.z_pixel - spots_b.z_pixel)[valid]
    max_shift = torch.sqrt(dy ** 2 + dz ** 2).max().item()
    print(f"  strained vs unstrained: max spot shift = {max_shift:.3f} px")

    # 0.3% on the a-axis must move at least one valid spot by >1 px on a
    # 884 mm Lsd / 75 um pixel detector (see conftest.standard_ff_geometry).
    assert max_shift > 1.0, (
        f"strain forward path appears to be a no-op: "
        f"max spot shift = {max_shift:.3f} px (expected > 1)"
    )
