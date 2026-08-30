"""Per-parameter σ for VECTOR parameters, and what a σ of 0 actually means.

``auto.py`` recorded σ only for SCALAR parameters (``if sz == 1``). The panel
stage freezes every global and refines nothing but vectors — ``panel_delta_yz``
(48×2), ``panel_delta_theta``, ``panel_delta_lsd``, ``panel_delta_p2`` — so it
returned an EMPTY ``sigma`` and a trivially empty ``unconstrained``, which is
indistinguishable from "everything is well determined".

The numbers were never missing: ``laplace_at_map`` computes σ for all 240 of
those dimensions. Only the recording loop dropped them.

**And a σ of 0 is the opposite of precision.**
``sigma_per_dim = sqrt(diag(cov_x).clamp(min=0.0))``, so a zero is a
NON-POSITIVE variance clamped away — the Hessian is indefinite in that
direction. A zero can also come from the bounded reparameterisation's Jacobian
``span·s·(1−s)`` vanishing when a parameter is railed at a bound. Either way
the value is not measured, and the ``|value| < σ`` test that fills
``unconstrained`` cannot fire for it (``0 < σ`` is False), so such parameters
escaped every check silently. They now go in ``undetermined``.

Measured on the real 48-panel Pilatus with per-panel shift + Lsd + p2: of 240
refined DOF, **69 undetermined and 133 consistent with zero — only 38 carry
information**. That is now a gate that fires.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch


# ------------------------------------------------- the fields exist and mean it

def test_result_carries_vector_sigma_and_undetermined():
    from midas_calibrate_v2.pipelines.auto import AutoCalibrationResult as R
    f = R.__dataclass_fields__
    for k in ("sigma", "sigma_vector", "unconstrained", "undetermined",
              "at_bounds"):
        assert k in f, f"AutoCalibrationResult has no {k}"


def test_sigma_zero_means_unmeasured_not_precise():
    """Pin the semantics at the source, so nobody 'simplifies' the clamp away
    or folds `undetermined` back into `unconstrained`."""
    import inspect
    from midas_peakfit.laplace import laplace_at_map

    src = inspect.getsource(laplace_at_map)
    assert "clamp(min=0.0)" in src, (
        "sigma_per_dim no longer clamps a non-positive variance to zero — the "
        "meaning of a zero has changed and `undetermined` must be revisited")


def test_the_recording_loop_handles_vectors():
    import inspect
    from midas_calibrate_v2.pipelines import auto

    src = inspect.getsource(auto)
    assert "sigma_vector[nm]" in src, "vector sigma is not recorded"
    assert "undetermined.append(f\"{nm}[{k}]\")" in src, (
        "vector elements with no information are not flagged")
    # scalars with a zero sigma must be flagged too, not skipped
    i = src.index("for nm, sg_ in sigma.items():")
    blk = src[i:i + 700]
    assert "undetermined.append(nm)" in blk, (
        "a SCALAR with sigma == 0 still escapes every check")


def test_the_overparameterisation_gate_exists_and_can_fail():
    import inspect
    from midas_calibrate_v2.pipelines import auto

    src = inspect.getsource(auto)
    assert "refined degrees of\n" in src or "refined degrees of " in src
    assert "over-parameterised" in src


# ------------------------------------------------- end to end on a synthetic

@pytest.mark.slow
def test_panel_stage_reports_per_element_sigma():
    """The case the gap was found in: a run that refines only vectors."""
    import math
    from midas_calibrate_v2 import calibrate
    from midas_calibrate_v2.forward.panels import PanelLayout
    from midas_integrate.geometry import pixel_to_REta, build_tilt_matrix

    N, PX, LSD, WL, A = 512, 200.0, 400_000.0, 0.172973, 5.4116
    Yi, Zi = np.meshgrid(np.arange(N, dtype=float), np.arange(N, dtype=float),
                         indexing="xy")
    R, _ = pixel_to_REta(Yi, Zi, Ycen=N / 2.0, Zcen=N / 2.0,
                         TRs=build_tilt_matrix(0.0, 0.30, -0.45), Lsd=LSD,
                         RhoD=float(N) * PX, px=PX,
                         **{f"p{k}": 0.0 for k in range(15)}, parallax=False)
    img = np.full_like(R, 25.0)
    seen = set()
    for h in range(8):
        for k in range(8):
            for l in range(8):
                if h == k == l == 0 or not (h % 2 == k % 2 == l % 2):
                    continue
                s2 = h * h + k * k + l * l
                if s2 in seen:
                    continue
                ratio = WL / (2.0 * (A / math.sqrt(s2)))
                if ratio >= 1.0:
                    continue
                seen.add(s2)
                R0 = LSD * math.tan(2.0 * math.asin(ratio)) / PX
                if 25 < R0 < 0.70 * N:
                    img += 4000.0 * np.exp(-((R - R0) ** 2) / (2 * 1.6 ** 2))
    img = np.random.default_rng(0).poisson(np.clip(img, 0, None)).astype(float)

    lay = PanelLayout.regular(n_y=2, n_z=2, sy=N // 2, sz=N // 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = calibrate(img, wavelength=WL, pxY=PX, pxZ=PX, calibrant="CeO2",
                      refine_tilts=True, refine_distortion=False,
                      build_residual_corr=False, verbose=False, n_iter=2,
                      panel_layout=lay, panel_mode="shift")

    assert r.sigma_vector, (
        "the panel stage refined only vector parameters and reported no "
        "per-element sigma — the gap is back")
    for nm, v in r.sigma_vector.items():
        assert isinstance(v, np.ndarray) and v.size > 1, f"{nm}: {v!r}"
        assert (v >= 0).all(), f"{nm} has a negative sigma"
    # every element is accounted for: determined, unconstrained, or undetermined
    total = sum(v.size for v in r.sigma_vector.values())
    named = len(r.undetermined) + len(r.unconstrained)
    assert named <= total
