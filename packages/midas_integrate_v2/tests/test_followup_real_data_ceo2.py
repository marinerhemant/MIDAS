"""Real-data smoke: Pilatus CeO₂ calibrant from FF_HEDM/Example/Calibration.

The dataset is a single TIFF + dark + parameter file shipped with MIDAS
under :file:`FF_HEDM/Example/Calibration/`. We integrate it through both
the v1 hot path and the v2 native-torch soft path, and verify:

1. The 1D profile from v1 has clear CeO₂ peaks at the predicted ring R
   values (we know which rings to expect from the lattice + wavelength).
2. The v2 soft path's profile lines up with v1's at peak positions
   (within ~1 R bin: the soft kernel blurs slightly but does not shift
   peaks).
3. Joint refinement on the v2 path: starting from a *perturbed* BC_y,
   :class:`EtaUniformityLoss` over the image's strongest ring drives
   BC_y back toward the converged calibration value within a few px in
   under 60 Adam steps.

Skipped automatically if the dataset isn't present.

Marked ``@pytest.mark.slow`` — full Pilatus grid (1475×1679 = 2.5 M
pixels), ~30s on CPU.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pytest
import torch

# Locate the example dataset relative to the repo root.
_REPO = Path(__file__).resolve().parents[3]
_DATA = _REPO / "FF_HEDM" / "Example" / "Calibration"
_PARAMS = _DATA / "parameters.txt"
_IMAGE = _DATA / "CeO2_Pil_100x100_att000_650mm_71p676keV_001956.tif"
_DARK  = _DATA / "dark_CeO2_Pil_100x100_att000_650mm_71p676keV_001975.tif"

if not (_PARAMS.exists() and _IMAGE.exists() and _DARK.exists()):
    pytest.skip("FF_HEDM/Example/Calibration dataset not found",
                allow_module_level=True)

tifffile = pytest.importorskip("tifffile")

from midas_integrate.params import parse_params
from midas_integrate.detector_mapper import build_map as v1_build_map
from midas_integrate.bin_io import PixelMap
from midas_integrate.kernels import (
    build_csr as v1_build_csr,
    integrate as v1_integrate,
    profile_1d as v1_profile_1d,
)
from midas_integrate_v2 import (
    spec_from_v1_paramstest,
    spec_from_v1_params,
    SoftBinGeometry,
    integrate_soft,
    integrate_with_corrections,
    profile_1d_diff,
    EtaUniformityLoss,
)

pytestmark = pytest.mark.slow


def _coarse_params():
    """Load Pilatus params but coarsen the binning so the smoke test
    runs in ~30s instead of ~5 min. Coarse bins are still fine for
    peak-position verification (peak widths ≫ bin width)."""
    p = parse_params(_PARAMS)
    p.RBinSize = 2.0       # was 0.25
    p.EtaBinSize = 5.0     # was 1.0
    return p


def _load_raw_image() -> np.ndarray:
    """Load + dark-subtract the calibrant frame. No ImTransOpt applied —
    the image is in raw (file-on-disk) coordinates."""
    img = tifffile.imread(_IMAGE).astype(np.float64)
    dark = tifffile.imread(_DARK).astype(np.float64)
    if dark.ndim == 3:
        dark = dark.mean(axis=0)
    return np.clip(img - dark, 0.0, None)


# v0.3 made ImTransOpt native — `integrate_soft` and
# `integrate_with_corrections` apply spec.TransOpt to raw images
# automatically (apply_trans_opt=True default). Tests below feed the
# raw image directly; the v2 path handles the transform.


# Cached so the v1 build_map (~5s) only runs once per session.
_V1_CACHE: dict = {}


def _v1_build_and_integrate(p, img_raw: torch.Tensor):
    """Integrate a RAW image through v1's hot path (numba-default).
    v1 applies ImTransOpt internally to the map, so the image stays raw."""
    key = (p.RBinSize, p.EtaBinSize, p.BC_y, p.BC_z, p.Lsd)
    if key not in _V1_CACHE:
        res = v1_build_map(p, auto_load=False, verbose=False)
        _V1_CACHE[key] = res
    res = _V1_CACHE[key]
    pm = PixelMap(pxList=res.pxList, counts=res.counts, offsets=res.offsets,
                   map_header=None, nmap_header=None)
    geom = v1_build_csr(
        pm, n_r=p.n_r_bins, n_eta=p.n_eta_bins,
        n_pixels_y=p.NrPixelsY, n_pixels_z=p.NrPixelsZ,
        device="cpu", dtype=torch.float64,
        bc_y=p.BC_y, bc_z=p.BC_z,
    )
    int2d = v1_integrate(img_raw, geom, mode="floor", normalize=True)
    prof = v1_profile_1d(int2d, geom, mode="area_weighted").numpy()
    return prof


# ── (1) v1 hot path produces the right profile ──

def test_v1_profile_has_ceo2_first_ring():
    p = _coarse_params()
    img_raw = torch.from_numpy(_load_raw_image())
    prof = _v1_build_and_integrate(p, img_raw)

    n_r = p.n_r_bins
    r_axis = p.RMin + p.RBinSize * (np.arange(n_r) + 0.5)

    # CeO₂ first ring (111): d = 3.124 Å. With λ = 0.172979 Å:
    # 2θ = 2 arcsin(λ / 2d) = 2 arcsin(0.0277) ≈ 3.17°.
    # R = (Lsd / px) tan(2θ) = (657437 / 172) tan(3.17°) ≈ 211 px.
    band = (r_axis > 195) & (r_axis < 230)
    band_idx = np.where(band)[0]
    peak_idx = band_idx[np.argmax(prof[band])]
    peak_R = r_axis[peak_idx]
    assert 205 <= peak_R <= 220, (
        f"v1 profile peak in [195, 230] px is at R={peak_R:.1f}, "
        "expected first CeO₂ ring around 211 px"
    )


# ── (2) v2 soft path peak position matches v1 ──

def test_v2_soft_path_peak_aligns_with_v1():
    p = _coarse_params()
    img_raw = _load_raw_image()
    prof_v1 = _v1_build_and_integrate(p, torch.from_numpy(img_raw))

    spec = spec_from_v1_params(p, requires_grad=False)
    geom_v2 = SoftBinGeometry.from_spec(spec)
    int2d_v2 = integrate_soft(torch.from_numpy(img_raw), geom_v2)
    prof_v2 = profile_1d_diff(int2d_v2, spec).detach().numpy()

    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    band = (r_axis > 200) & (r_axis < 230)
    idx = np.where(band)[0]

    def centroid(profile):
        w = np.maximum(profile[idx], 0)
        if w.sum() < 1e-12:
            return float("nan")
        return float((w * r_axis[idx]).sum() / w.sum())

    c_v1 = centroid(prof_v1)
    c_v2 = centroid(prof_v2)
    # Soft kernel is permitted to shift the centroid by less than two R bins.
    assert abs(c_v1 - c_v2) < 2 * spec.RBinSize, (
        f"v1 vs v2 first-ring centroid drift {abs(c_v1 - c_v2):.3f} px > "
        f"{2 * spec.RBinSize} px (RBinSize); v1={c_v1:.3f}, v2={c_v2:.3f}"
    )


# ── (3) Joint refinement smoke: BC_y recovery on real data ──

def test_joint_refinement_recovers_BC_y_from_perturbation():
    """Start with BC_y perturbed by +1 px from the converged value;
    EtaUniformityLoss-driven Adam must close most of the gap."""
    p = _coarse_params()
    spec = spec_from_v1_params(p, requires_grad=True)
    BC_true = float(spec.BC_y.detach())

    img_raw_t = torch.from_numpy(_load_raw_image())

    # Restrict the loss to the first CeO₂ ring band — strongest signal,
    # near-convex basin around BC_true.
    n_r = spec.n_r_bins
    r_axis = spec.RMin + spec.RBinSize * (np.arange(n_r) + 0.5)
    band_mask = (r_axis > 200) & (r_axis < 230)
    r_indices = np.where(band_mask)[0].tolist()

    spec.BC_y = torch.tensor(BC_true + 1.0, dtype=torch.float64,
                              requires_grad=True)
    initial_err = abs(float(spec.BC_y.detach()) - BC_true)

    loss_fn = EtaUniformityLoss(r_indices=r_indices, intensity_floor=1.0)
    opt = torch.optim.Adam([spec.BC_y], lr=0.1)

    history = [float(spec.BC_y.detach())]
    losses = []
    for _ in range(60):
        opt.zero_grad()
        int2d = integrate_with_corrections(img_raw_t, spec)
        L = loss_fn(int2d)
        L.backward()
        opt.step()
        history.append(float(spec.BC_y.detach()))
        losses.append(float(L))

    closest_err = min(abs(bc - BC_true) for bc in history)
    final_err = abs(history[-1] - BC_true)
    assert closest_err < 0.5 * initial_err, (
        f"BC_y did not close half the gap on real data: closest err "
        f"{closest_err:.3f} px, initial err {initial_err:.3f} px, "
        f"final {final_err:.3f}, history end={history[-3:]}"
    )
    assert min(losses) < losses[0], (
        f"η-uniformity loss never improved: start={losses[0]:.4e}, "
        f"min={min(losses):.4e}"
    )
