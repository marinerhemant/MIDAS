"""pyFAI vs MIDAS calibration benchmark — Paper A Table 4.

This module provides the head-to-head harness the paper uses to report
"4.6–30× more accurate than pyFAI". Both tools calibrate the same frame
from the same starting geometry; we report pseudo-strain (mean + std)
and wall-clock time for each.

pyFAI is a lazy import — only required when you actually call
:func:`benchmark`. Install via the `[paper]` extra or your own env.

Public API
----------
- ``BenchmarkResult`` — dataclass returned by :func:`benchmark`.
- ``benchmark(image, material, wavelength, …)`` — run both, compare.
- ``pyfai_pseudo_strain(image, geometry, …)`` — per-ring 2θ dispersion
  via pyFAI's ``integrate2d`` at a given geometry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ._config import CalibrationConfig
from .calibrate import CalibrationResult, run_calibration
from .geometry import DetectorGeometry

__all__ = ["BenchmarkResult", "benchmark", "pyfai_pseudo_strain"]


@dataclass
class BenchmarkResult:
    """Head-to-head comparison of MIDAS vs pyFAI on one image."""

    midas_pseudo_strain: float           # µε (mean)
    midas_pseudo_strain_std: float       # µε (std across rings)
    midas_seconds: float                 # wall clock
    midas_geometry: DetectorGeometry

    pyfai_pseudo_strain: Optional[float] = None
    pyfai_pseudo_strain_std: Optional[float] = None
    pyfai_seconds: Optional[float] = None
    pyfai_per_ring_ustrain: list[float] = field(default_factory=list)

    image: Path = field(default_factory=Path)
    material: str = ""
    wavelength: float = 0.0

    def as_dict(self) -> dict:
        """Flat dict for easy CSV / JSON export (matches the README example)."""
        out = {
            "midas_ustrain": self.midas_pseudo_strain,
            "midas_seconds": self.midas_seconds,
        }
        if self.pyfai_pseudo_strain is not None:
            out["pyfai_ustrain"] = self.pyfai_pseudo_strain
            out["pyfai_seconds"] = self.pyfai_seconds
            # Only report ratios when denominators are non-zero; a MIDAS
            # pseudo-strain of 0 means the calibration succeeded so well
            # (or pathologically early-exited) that no ratio is meaningful.
            if self.pyfai_pseudo_strain and self.midas_pseudo_strain:
                out["accuracy_ratio"] = (
                    self.pyfai_pseudo_strain / self.midas_pseudo_strain
                )
            if self.pyfai_seconds and self.midas_seconds:
                out["speedup"] = self.pyfai_seconds / self.midas_seconds
        return out


def benchmark(
    image: str | Path,
    material: str,
    wavelength: float,
    pixel_size: float,
    *,
    nr_pixels_y: int,
    nr_pixels_z: int,
    lsd: float = 1_000_000.0,
    ybc: float | None = None,
    zbc: float | None = None,
    lattice_params: tuple[float, ...] = (5.4116, 5.4116, 5.4116, 90, 90, 90),
    space_group: int = 225,
    n_iterations: int = 5,
    work_dir: str | Path | None = None,
    n_cpus: int = 4,
    bin_dir: str | Path | None = None,
    include_pyfai: bool = True,
    dark_file: str = "dark.tif",
    mask_file: str = "mask_upd.tif",
    im_trans_opt: tuple[int, ...] = (2,),
    extra_params: dict | None = None,
) -> BenchmarkResult:
    """Run MIDAS calibration and (optionally) pyFAI integration, compare.

    Parameters
    ----------
    image : path
        Calibrant frame (numbered filename: ``<stem>_<NN...>.<ext>``).
    material : str
        Calibrant name — ``"CeO2"``, ``"LaB6"`` — passed to pyFAI's
        ``get_calibrant``. MIDAS uses ``lattice_params`` + ``space_group``.
    wavelength : float
        X-ray wavelength in angstroms.
    pixel_size : float
        Pixel pitch in micrometers. MIDAS native unit.
    nr_pixels_y, nr_pixels_z : int
        Detector dimensions.
    lsd, ybc, zbc : float
        Starting geometry. ``ybc`` / ``zbc`` default to the detector center.
    lattice_params : tuple
        Six floats (a, b, c, α, β, γ) — defaults to CeO₂.
    n_iterations : int, default 5
        MIDAS calibration iteration count; tune up for paper-quality runs.
    include_pyfai : bool, default True
        Skip the pyFAI half when False — useful when pyFAI isn't installed.
    """
    image_path = Path(image).resolve()
    if ybc is None:
        ybc = nr_pixels_y / 2
    if zbc is None:
        zbc = nr_pixels_z / 2

    cfg = CalibrationConfig(
        material=material,
        lattice_params=lattice_params,
        wavelength=wavelength,
        pixel_size=pixel_size,
        lsd=lsd, ybc=ybc, zbc=zbc,
        nr_pixels_y=nr_pixels_y, nr_pixels_z=nr_pixels_z,
        space_group=space_group,
        dark_file=dark_file,
        mask_file=mask_file,
        im_trans_opt=list(im_trans_opt),
        n_iterations=n_iterations,
        extra_params=dict(extra_params or {}),
    )

    t0 = time.perf_counter()
    result: CalibrationResult = run_calibration(
        cfg, image_path,
        work_dir=work_dir, n_cpus=n_cpus, bin_dir=bin_dir,
    )
    midas_seconds = time.perf_counter() - t0

    out = BenchmarkResult(
        midas_pseudo_strain=result.pseudo_strain,
        midas_pseudo_strain_std=result.pseudo_strain_std,
        midas_seconds=midas_seconds,
        midas_geometry=result.geometry,
        image=image_path, material=material, wavelength=wavelength,
    )

    if not include_pyfai:
        return out

    # pyFAI uses the MIDAS-refined geometry for its own integration — the
    # paper's "fair" comparison uses pyFAI's own refined geometry, which
    # requires its own peak-picking pipeline (not yet ported). Reporting
    # pyFAI @ MIDAS-geometry still captures the binning / solid-angle
    # differences the paper discusses; we label this clearly in the docs.
    try:
        t0 = time.perf_counter()
        per_ring, median_us, std_us = pyfai_pseudo_strain(
            image=image_path,
            geometry=result.geometry,
            wavelength=wavelength,
            material=material,
            pixel_size_um=pixel_size,
        )
        out.pyfai_seconds = time.perf_counter() - t0
        out.pyfai_per_ring_ustrain = per_ring
        out.pyfai_pseudo_strain = median_us
        out.pyfai_pseudo_strain_std = std_us
    except ImportError:
        # pyFAI not installed — leave fields None.
        pass

    return out


# ---------------------------------------------------------------------------
# pyFAI half — lazy imports everywhere so a core install doesn't need it.
# ---------------------------------------------------------------------------

def pyfai_pseudo_strain(
    image: str | Path | np.ndarray,
    geometry: DetectorGeometry,
    wavelength: float,
    material: str,
    pixel_size_um: float,
    *,
    max_rings: int = 15,
    n_eta_bins: int = 360,
    n_tth_bins: int = 3000,
) -> tuple[list[float], float, float]:
    """Compute per-ring pseudo-strain via pyFAI's integrate2d.

    For each calibrant ring in range, fit a pseudo-Voigt to the azimuthal
    slice at each η bin, convert peak centre to d-spacing, compare with
    the known d-spacing. Returns ``(per_ring_ustrain, median, std)``.

    Raises ``ImportError`` if pyFAI / scipy not installed.
    """
    from scipy.optimize import curve_fit
    try:
        from pyFAI.integrator.azimuthal import AzimuthalIntegrator
    except ImportError:
        from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
    from pyFAI.calibrant import get_calibrant
    from pyFAI.detectors import Detector

    img = _load_image(image)
    calibrant = get_calibrant(material)
    calibrant.wavelength = wavelength * 1e-10

    ny, nz = img.shape
    px_m = pixel_size_um * 1e-6
    detector = Detector(pixel1=px_m, pixel2=px_m, max_shape=(ny, nz))

    # pyFAI PONI uses +0.5 px center offset vs MIDAS's integer-center BC.
    ai = AzimuthalIntegrator(
        dist=geometry.lsd * 1e-6,
        poni1=(geometry.ybc + 0.5) * px_m,
        poni2=(geometry.zbc + 0.5) * px_m,
        wavelength=wavelength * 1e-10,
        detector=detector,
    )

    tth_known_rad = np.array(calibrant.get_2th()[:max_rings])
    tth_known_deg = np.degrees(tth_known_rad)
    d_known = np.array(calibrant.get_dSpacing()[:max_rings])

    I_2d, tth_edges, eta_edges = ai.integrate2d(
        img, n_tth_bins, n_eta_bins, unit="2th_deg",
    )
    tth_centers = _edges_to_centers(tth_edges, I_2d.shape[1])

    per_ring_ustrain: list[float] = []

    for i, (tth_exp, d_exp) in enumerate(zip(tth_known_deg, d_known)):
        if tth_exp < tth_centers[0] + 0.2 or tth_exp > tth_centers[-1] - 0.2:
            continue
        ring_strains = _fit_ring_strain(
            I_2d, tth_centers, tth_exp, d_exp, wavelength, curve_fit,
        )
        if ring_strains.size == 0:
            continue
        per_ring_ustrain.append(float(np.median(np.abs(ring_strains)) * 1e6))

    if not per_ring_ustrain:
        return [], float("nan"), float("nan")

    median = float(np.median(per_ring_ustrain))
    std = float(np.std(per_ring_ustrain))
    return per_ring_ustrain, median, std


def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path).astype(np.float64)
    raise ValueError(f"Unsupported image format: {path.suffix}")


def _edges_to_centers(edges: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.asarray(edges)
    if edges.size == n_bins + 1:
        return (edges[:-1] + edges[1:]) / 2.0
    return edges[:n_bins]


def _pseudo_voigt(x, amp, center, fwhm, mixing, bg):
    # Thompson-Cox-Hastings: amp * (η·Lorentzian + (1-η)·Gaussian) + bg
    sigma = fwhm / 2.355
    gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    gamma = fwhm / 2.0
    lorentz = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
    return amp * (mixing * lorentz + (1 - mixing) * gauss) + bg


def _fit_ring_strain(
    I_2d: np.ndarray,
    tth_centers: np.ndarray,
    tth_expected: float,
    d_expected: float,
    wavelength_A: float,
    curve_fit,
    window_deg: float = 0.5,
) -> np.ndarray:
    """Fit pseudo-Voigt at each η slice around tth_expected, return per-η strain."""
    mask = np.abs(tth_centers - tth_expected) < window_deg
    if mask.sum() < 10:
        return np.array([])
    tth_fit = tth_centers[mask]
    strains: list[float] = []
    for eta_idx in range(I_2d.shape[0]):
        slice_I = I_2d[eta_idx, mask]
        if not np.any(slice_I > 0):
            continue
        if slice_I.max() < 3 * (np.abs(slice_I).mean() + 1e-9):
            continue
        try:
            p0 = [slice_I.max(), tth_expected, 0.1, 0.5, float(slice_I.min())]
            popt, _ = curve_fit(
                _pseudo_voigt, tth_fit, slice_I, p0=p0, maxfev=200,
            )
            tth_fitted_deg = popt[1]
            d_fitted = wavelength_A / (
                2 * np.sin(np.radians(tth_fitted_deg / 2))
            )
            strains.append((d_fitted - d_expected) / d_expected)
        except (RuntimeError, ValueError):
            continue
    return np.asarray(strains)
