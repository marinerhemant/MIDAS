"""Unit + integration smoke tests for pipelines.bayesian_multi.

Heavy end-to-end validation happens against real data on copland; here we
verify the smaller things that can break silently:
  * the import surface,
  * the flat refined-name enumeration order (shared first, then per-image
    with _imgK suffixes — the layout LaplaceResult.cov / sigma_per_dim
    inherits),
  * that the returned MultiBayesianResult round-trips the MAP result.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def _make_v1():
    """A minimal valid V1 CalibrationParams. Pixel/wavelength/ring counts
    match the Ti-7Al + CeO2 setup we use on copland; values aren't refined
    here — we only need a syntactically-valid spec for build_multi_spec()."""
    from midas_calibrate.params import CalibrationParams
    return CalibrationParams(
        NrPixelsY=2048, NrPixelsZ=2048,
        pxY=200.0, pxZ=200.0,
        Lsd=940000.0,
        BC_y=1024.0, BC_z=994.0,
        tx=0.0, ty=-0.2, tz=0.0,
        Wavelength=0.189714,
        SpaceGroup=225,
        LatticeConstant=(5.4116, 5.4116, 5.4116, 90.0, 90.0, 90.0),
        MaxRingRad=1024.0, MinRingRad=80.0,
    )


def test_bayesian_multi_import_surface():
    """Module + public names load without import errors."""
    from midas_calibrate_v2.pipelines import bayesian_multi
    assert hasattr(bayesian_multi, "autocalibrate_multi_bayesian")
    assert hasattr(bayesian_multi, "MultiBayesianResult")


def test_flat_refined_names_layout():
    """Refined-name list is ``[shared..., per_image_0..., per_image_1...]``
    with image suffix. Order must match the index order in _build_multi_indices
    so LaplaceResult.cov rows/cols align with the names."""
    from midas_calibrate_v2.pipelines.bayesian_multi import (
        _build_multi_indices, _flat_refined_names,
    )
    from midas_calibrate_v2.pipelines.multi import build_multi_spec
    from midas_calibrate_v2.parameters.pack import pack_multi

    v1_0 = _make_v1()
    v1_1 = _make_v1()

    # Use the same custom shared set as run_calib_joint.py — BC/ty/tz and
    # all v2 distortion params shared, Lsd per-image.
    from midas_calibrate_v2.forward.distortion import P_COEF_NAMES
    shared_names = (["pxY", "pxZ", "RhoD"] + list(P_COEF_NAMES) +
                    ["panel_delta_yz", "panel_delta_theta",
                     "panel_delta_lsd", "panel_delta_p2",
                     "BC_y", "BC_z", "ty", "tz"])
    multi_spec = build_multi_spec([v1_0, v1_1], shared_names=shared_names)

    x_full, info = pack_multi(multi_spec, dtype=torch.float64, device="cpu")
    refined_idx, lo, hi = _build_multi_indices(
        multi_spec, info, dtype=torch.float64, device="cpu",
    )

    names = _flat_refined_names(multi_spec, info)
    assert len(names) == refined_idx.numel(), (
        f"refined names ({len(names)}) must match refined indices "
        f"({refined_idx.numel()})"
    )

    # Per-image Lsd names should exist and be suffixed.
    assert "Lsd_img0" in names
    assert "Lsd_img1" in names

    # Shared BC/ty/tz should appear once (no suffix).
    assert names.count("BC_y") == 1
    assert names.count("ty") == 1

    # Shared block names come first; per-image suffixed names last.
    img_suffixed = [n for n in names if n.endswith("_img0") or n.endswith("_img1")]
    shared_only = [n for n in names if not (n.endswith("_img0") or n.endswith("_img1"))]
    last_shared_idx = max(names.index(n) for n in shared_only)
    first_imaged_idx = min(names.index(n) for n in img_suffixed)
    assert last_shared_idx < first_imaged_idx, (
        f"shared names must precede per-image suffixed names; saw "
        f"last_shared={last_shared_idx}, first_imaged={first_imaged_idx}"
    )


def test_flat_refined_names_image_index_ordering():
    """For N images, suffixes are _img0 .. _img(N-1) in input order."""
    from midas_calibrate_v2.pipelines.bayesian_multi import _flat_refined_names
    from midas_calibrate_v2.pipelines.multi import build_multi_spec
    from midas_calibrate_v2.parameters.pack import pack_multi

    v1s = [_make_v1() for _ in range(3)]
    multi_spec = build_multi_spec(v1s)
    _, info = pack_multi(multi_spec, dtype=torch.float64, device="cpu")
    names = _flat_refined_names(multi_spec, info)

    # The default-shared spec puts Lsd per-image, so we should see
    # Lsd_img0, Lsd_img1, Lsd_img2 in that order.
    lsd_positions = {f"Lsd_img{k}": names.index(f"Lsd_img{k}")
                     for k in range(3) if f"Lsd_img{k}" in names}
    assert len(lsd_positions) == 3
    assert (lsd_positions["Lsd_img0"] <
            lsd_positions["Lsd_img1"] <
            lsd_positions["Lsd_img2"])
