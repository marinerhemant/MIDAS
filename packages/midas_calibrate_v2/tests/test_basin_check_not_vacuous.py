"""``basin_check`` must be measuring something.

The gate compares the SEED geometry against the converged one, and reports a
fail when the refiner has walked out of the seed's basin. That only works if
``v1_init`` actually holds the pre-refinement values.

``autocalibrate`` refines its ``CalibrationParams`` **in place**. ``auto.py``
used to hand the same object to the gate afterwards, so the gate compared the
final geometry with itself: ``delta_Lsd_um`` and ``delta_BC_px`` were
identically 0.0 and the severity could only ever be ``ok``. A geometry that had
walked 800 mm away from its seed still collected a green tick.

That is worse than having no gate, because the surrounding code *acts* on it —
``first_time.py`` accepts the first attempt where ``basin_check`` is not a fail,
and ``SeedFallbackWarning``'s docstring cites the gate's zero drift as the
reason the warning has to exist at all.

These tests pin the two halves: the metrics must be real, and the gate must
still be able to fail.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ------------------------------------------------------- the gate itself

def _seed(Lsd=800_000.0, BC_y=1024.0, BC_z=1024.0):
    from midas_calibrate.params import CalibrationParams
    return CalibrationParams(NrPixelsY=2048, NrPixelsZ=2048, pxY=200.0,
                             pxZ=200.0, Lsd=Lsd, BC_y=BC_y, BC_z=BC_z)


def _unpacked(Lsd, BC_y, BC_z):
    return {"Lsd": torch.tensor(float(Lsd)),
            "BC_y": torch.tensor(float(BC_y)),
            "BC_z": torch.tensor(float(BC_z))}


def test_gate_fails_on_a_geometry_that_left_its_basin():
    from midas_calibrate_v2.pipelines.diagnostics import basin_check
    r = basin_check(_seed(), _unpacked(700_000.0, 1024.0, 1024.0))
    assert r.severity == "fail"
    assert r.metrics["delta_Lsd_um"] == pytest.approx(-100_000.0)


def test_gate_is_identically_zero_when_handed_the_same_geometry_twice():
    """The failure mode itself, made explicit: this is what a mutated-in-place
    ``v1_init`` produces, and it is indistinguishable from a perfect run."""
    from midas_calibrate_v2.pipelines.diagnostics import basin_check
    final = _seed(Lsd=700_000.0, BC_y=900.0, BC_z=1100.0)
    r = basin_check(final, _unpacked(final.Lsd, final.BC_y, final.BC_z))
    assert r.severity == "ok"
    assert r.metrics["delta_Lsd_um"] == 0.0
    assert r.metrics["delta_BC_px"] == 0.0


# ------------------------------------------- the pipeline actually feeding it

@pytest.mark.slow
def test_pipeline_reports_real_seed_to_map_drift():
    """End-to-end: run ``calibrate`` and require the reported drift to be the
    real seed-to-MAP distance, not zero.

    The seeder lands near but not on the answer, so a correctly-wired gate sees
    a small non-zero drift. Exactly 0.0 on both metrics means ``v1_init`` and
    the converged geometry are the same object again.
    """
    import math
    import warnings
    from midas_calibrate_v2 import calibrate
    from midas_integrate.geometry import pixel_to_REta, build_tilt_matrix

    N, px, LSD, WL, A = 512, 200.0, 400_000.0, 0.172973, 5.4116
    Yi, Zi = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float),
                         indexing="xy")
    R, _ = pixel_to_REta(Yi, Zi, Ycen=N / 2.0, Zcen=N / 2.0,
                         TRs=build_tilt_matrix(0.0, 0.0, 0.0), Lsd=LSD,
                         RhoD=float(N) * px, px=px,
                         **{f"p{k}": 0.0 for k in range(15)}, parallax=False)
    img = np.full_like(R, 25.0)
    for s2 in (3, 4, 8, 11, 12, 16, 19, 20, 24, 27):
        ratio = WL / (2.0 * (A / math.sqrt(s2)))
        R0 = LSD * math.tan(2.0 * math.asin(ratio)) / px
        if 25 < R0 < 0.72 * N:
            img += 4000.0 * np.exp(-((R - R0) ** 2) / (2 * 1.6 ** 2))
    img = np.random.default_rng(0).poisson(np.clip(img, 0, None)).astype(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = calibrate(img, wavelength=WL, pxY=px, pxZ=px, calibrant="CeO2",
                        refine_tilts=True, refine_distortion=False,
                        build_residual_corr=False, verbose=False, n_iter=2)

    gate = next((d for d in res.diagnostics if d.name == "basin_check"), None)
    assert gate is not None, "basin_check did not run"
    assert not (gate.metrics["delta_Lsd_um"] == 0.0
                and gate.metrics["delta_BC_px"] == 0.0), (
        "basin_check reported exactly zero drift on both metrics — v1_init is "
        "the refined geometry again, so the gate is comparing it with itself")
    assert gate.metrics["Lsd_seed"] != pytest.approx(res.Lsd, abs=1e-9)
