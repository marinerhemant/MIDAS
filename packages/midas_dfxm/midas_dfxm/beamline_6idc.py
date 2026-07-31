"""6-ID-C polymer-optics DFXM beamline forward model (Qiao et al., RSI 91, 113703, 2020).

The 6-ID-C microscope replaces the beryllium CRL objective of the standard DFXM geometry
with SU-8 polymer optics: an x-ray prism-lens (XPL) condenser that illuminates the sample
uniformly from all angles, and a polymer CRL objective. This module models the XPL condenser
(the piece ``crl_abcd`` in :mod:`midas_dfxm.beamline` cannot represent), using the SU-8 index
from :mod:`midas_dfxm.polymer`. The vibration PSF (:mod:`midas_dfxm.vibration`) and the Talbot
wavefront sensor complete the instrument twin.

The XPL is two orthogonal arrays of ±45° SU-8 prisms (20 µm edges) that mimic a Fresnel lens,
displaced along the beam to cancel astigmatism (physical aperture 1.6 mm, axial length 68 mm,
design focal length ~130 mm at 20 keV). Because it is a Fresnel-segmented refractor its focal
length is set by the prism pitch (a design input here, not R/2δ), while the observables the
digital twin needs -- illumination NA, source-demagnified focal spot, SU-8 transmission, and
the characteristic square (±45°) wavefront -- follow from the geometry and the material index.
"""
from __future__ import annotations

import math

import torch

from .polymer import su8_index, _HC_KEV_A


def xpl_condenser(
    energy_keV: float = 20.0,
    *,
    usable_aperture_mm: float = 1.42,          # slit-defined usable aperture (paper: 1.42 x 1.55)
    focus_distance_mm: float = 2260.0,         # XPL-to-focus distance (paper Fig. 2b)
    focal_length_mm: float = 130.0,            # design focal length (prism-pitch set)
    prism_path_um: float = 300.0,              # SU-8 path a mid-aperture ray traverses
    source_size_um: float | None = None,       # BM source size (for the demagnified waist)
    source_distance_m: float = 62.5,           # source-to-condenser distance (paper)
) -> dict:
    """Observables of the SU-8 XPL condenser at ``energy_keV``.

    Returns a dict with ``NA`` (usable numerical aperture), ``focal_spot_um`` (source-demagnified,
    with the diffraction floor), ``transmission`` (SU-8 absorption over ``prism_path_um``),
    ``diffraction_floor_um`` and the SU-8 ``delta``/``beta``. NA is ``(usable_aperture/2) /
    focus_distance`` -- the paper's slit-scan definition (reproduces 3.2--3.5e-4).
    """
    idx = su8_index(energy_keV)
    lam_A = _HC_KEV_A / energy_keV
    lam_um = lam_A * 1e-4

    na = (usable_aperture_mm / 2.0) / focus_distance_mm            # slit-scan NA
    diff_floor_um = 0.886 * lam_um / (2.0 * na)                    # diffraction-limited waist
    demag = focus_distance_mm / (source_distance_m * 1000.0)
    if source_size_um is not None:
        waist = math.hypot(source_size_um * demag, diff_floor_um)
    else:
        waist = diff_floor_um
    transmission = math.exp(-idx["mu_lin_per_cm"] * prism_path_um * 1e-4)   # SU-8 absorption

    return {
        "NA": na,
        "focal_spot_um": waist,
        "diffraction_floor_um": diff_floor_um,
        "transmission": transmission,
        "demagnification": demag,
        "delta": idx["delta"],
        "beta": idx["beta"],
        "focal_length_mm": focal_length_mm,
        "energy_keV": energy_keV,
    }


def xpl_square_wavefront(size: int = 128, aperture_mm: float = 1.0, amplitude_nm: float = 131.0,
                        n_prisms: int = 8, dtype=torch.float64, device=None) -> torch.Tensor:
    """Characteristic ±45° square XPL wavefront (path-length nm) over the aperture.

    Two orthogonal 1D prism arrays impose a sawtooth (Fresnel) phase along each ±45° diagonal;
    their sum is the diamond/square pattern measured in Qiao et al. Fig. 2(d) (~±131 nm over
    1 mm). Returned as an ``(size, size)`` optical-path map for folding into a pupil/PSF
    (reuses the :mod:`midas_dfxm.aberration` machinery). Differentiable in ``amplitude_nm``.
    """
    xs = torch.linspace(-aperture_mm / 2, aperture_mm / 2, size, dtype=dtype, device=device)
    x, y = torch.meshgrid(xs, xs, indexing="xy")
    u = (x + y) / math.sqrt(2.0)                                   # +45 deg diagonal
    v = (x - y) / math.sqrt(2.0)                                   # -45 deg diagonal
    period = aperture_mm / n_prisms
    amp = torch.as_tensor(amplitude_nm, dtype=dtype, device=device)
    # triangular (Fresnel prism) profile along each diagonal, summed
    saw = lambda t: 2.0 * torch.abs(t / period - torch.round(t / period)) - 0.5
    return amp * (saw(u) + saw(v))
