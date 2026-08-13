"""Every sign, index and ordering convention XRD-CT depends on, in one place.

These are the things that produce a plausible-looking but wrong answer when
they are got wrong, which is why they live here rather than being spelled out
at each call site. Each one records how it was established.

Sources are the 2022/2023 MPE U3O8 beamtime and the C it was processed with:

* ``DT/src/IntegratorPeakFitOMP.c`` and ``DT/src/PeakFit.c`` -- canonical
  fit-output ordering
* ``mpe_nov22_dt/recon_peak_all_mul.py`` -- the surviving pipeline script
* ``DTnewversion/ps_dt_u3o8_600A_fileNr_161_215_rad_105_525.txt`` -- geometry
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "FIT_OUTPUT_NAMES",
    "ADDITIVE_FIT_OUTPUTS",
    "RECON_SIGN",
    "U3O8_ENERGY_KEV",
    "aps_1id_omega",
    "fit_output_index",
    "is_additive",
    "recon_size",
    "unsnake",
]


# ---------------------------------------------------------------- fit outputs
#: The 12 peak-fit outputs, in the order the C writes them.
#:
#: Canonical because two independent places in the C agree:
#: ``IntegratorPeakFitOMP.c`` (``valTypes[]``) and ``PeakFit.c`` (``Rfit[]``,
#: where ``Rfit[9] = CalcIntegratedIntensity(...)`` pins index 9).
#:
#: **Every legacy Python script gets this wrong.** Both ``DT/runDTrecon.py``
#: and ``recon_peak_all_mul.py`` omit ``MaxIntensityObs`` from slot 5, which
#: shifts every label from index 5 on. Files written by those scripts are
#: therefore mislabelled: a ``*_BGFit_*.bin`` actually holds
#: ``MaxIntensityObs``. Indices 0-4 are unaffected, which is why nobody
#: noticed -- ``RMEAN`` and ``MaxInt`` are what people looked at.
#:
#: When reading 2023 output, index by position and take the name from HERE.
FIT_OUTPUT_NAMES: Final[tuple[str, ...]] = (
    "RMEAN",                        # 0  peak centre, detector radius (px)
    "MixFactor",                    # 1  pseudo-Voigt mixing, 0 = pure Gaussian
    "SigmaG",                       # 2  Gaussian width
    "SigmaL",                       # 3  Lorentzian width -- see the note below
    "MaxInt",                       # 4  fitted peak amplitude
    "MaxIntensityObs",              # 5  observed maximum (NOT BGFit)
    "BGFit",                        # 6  fitted background
    "BGSimple",                     # 7  background from the window edges
    "MeanError",                    # 8  mean fit residual
    "FitIntegratedIntensity",       # 9  integral of the fitted profile
    "TotalIntensity",               # 10 raw sum over the window
    "TotalIntensityBackgroundCorr", # 11 raw sum minus background
)

#: Outputs that add along a ray, and so may be reconstructed directly.
#:
#: Radon inversion is linear. A quantity that is *not* additive (a peak
#: centre, a width, a mixing factor) has no meaning when back-projected: the
#: projection's fitted ``RMEAN`` is the intensity-weighted mean of the voxel
#: values along the ray, not their sum. Reconstruct those via the weighted
#: form -- ``recon(x * I) / recon(I)`` -- or fit after reconstructing.
ADDITIVE_FIT_OUTPUTS: Final[frozenset[str]] = frozenset({
    "TotalIntensity",
    "TotalIntensityBackgroundCorr",
    "FitIntegratedIntensity",
})

#: ``SigmaG`` and ``SigmaL`` are the SAME number in the legacy engine.
#: ``PeakFit.c`` sets both from ``x[2]``, so the "pseudo-Voigt" there is a
#: 5-parameter model with one shared width and the 12 outputs carry 11
#: distinct values. ``midas_peakfit`` fits independent widths; when comparing
#: against 2023 output, constrain them equal first, then relax and report the
#: difference rather than reading it as an improvement.
LEGACY_WIDTHS_ARE_SHARED: Final[bool] = True

#: The reconstruction is negated before peak fitting.
#:
#: ``recon_peak_all_mul.py`` does ``recons_reshape = -1 * transpose(...)``.
#: With ``doLog 0`` the engine back-projects diffracted intensity rather than
#: attenuation, so gridrec returns a negative-going image. Miss this and every
#: peak is inverted -- and a peak finder will still return numbers.
RECON_SIGN: Final[float] = -1.0

#: 0.136994 Å = 90.5 keV for the U3O8 beamtime. The parameter files comment
#: this as "55.618 keV (Ho-edge)", which is wrong -- 55.618 keV would be
#: 0.2230 Å. The wavelength is what the geometry was refined with and what
#: reproduces the 2023 output; the comment is stale. Confirmed 2026-08-13.
U3O8_ENERGY_KEV: Final[float] = 90.5


def fit_output_index(name: str) -> int:
    """Index of a fit output by name, using the canonical ordering."""
    try:
        return FIT_OUTPUT_NAMES.index(name)
    except ValueError:
        raise KeyError(
            f"{name!r} is not a fit output. Valid names: "
            f"{', '.join(FIT_OUTPUT_NAMES)}"
        ) from None


def is_additive(name: str) -> bool:
    """``True`` if *name* may be reconstructed directly (adds along a ray)."""
    fit_output_index(name)          # validates the name
    return name in ADDITIVE_FIT_OUTPUTS


# -------------------------------------------------------------------- omega
def aps_1id_omega(nominal_deg, *, negate: bool = True) -> np.ndarray:
    """Rotation angles in the sample frame, from the nominal motor values.

    The 1-ID aerotech stage turns the opposite way to the reconstruction
    convention, so every omega is negated. This is a standing site rule, not
    a per-dataset choice, and getting it wrong mirrors the reconstruction --
    which looks entirely reasonable until it is compared with anything else.

    Pass ``negate=False`` only for a beamline known not to need it, and record
    why at the call site.
    """
    ome = np.asarray(nominal_deg, dtype=np.float64)
    return -ome if negate else ome


# -------------------------------------------------------------------- snake
def unsnake(data: np.ndarray, *, axis: int = 0, frame_axis: int = 1) -> np.ndarray:
    """Reverse the frame order of every second entry along *axis*.

    A bidirectional ("snake") scan rotates alternate translations the opposite
    way, so their frames run backwards in omega. The 2022 scans set
    ``BadRotation 1`` for exactly this.

    Correcting a scan that did not need it -- or failing to correct one that
    did -- produces a reconstruction that looks fine and is wrong, so prefer
    :func:`~midas_dt.scan.detect_snake` over asserting it by hand.
    """
    out = np.array(data, copy=True)
    idx = [slice(None)] * out.ndim
    for i in range(1, out.shape[axis], 2):
        idx[axis] = i
        sl = tuple(idx)
        out[sl] = np.flip(out[sl], axis=frame_axis - (1 if frame_axis > axis else 0))
    return out


# ----------------------------------------------------------------- geometry
def recon_size(n_translations: int, *, extra_pad: bool = True) -> int:
    """Reconstruction edge length for a scan of *n_translations* columns.

    ``next_pow2(n)``, doubled when ``extra_pad`` -- matching
    ``ExtraPadForTomo`` in the legacy scripts. For the 2022 U3O8 scan
    (55 translations, ``ExtraPadForTomo 1``) this gives 128, which is the
    ``reconSize`` those runs used.
    """
    from midas_tomo.config import next_power_of_2

    if n_translations <= 0:
        raise ValueError(f"n_translations must be positive, got {n_translations}")
    n = next_power_of_2(n_translations)
    return 2 * n if extra_pad else n


@dataclass(frozen=True)
class ScanKnownLimits:
    """Limits that apply to a reduction, carried alongside its results.

    Attached to outputs so a map cannot be separated from the caveats that
    govern how far it can be trusted.
    """

    snake_corrected: bool
    omega_negated: bool
    self_absorption_corrected: bool = False
    texture_corrected: bool = False

    def warnings(self) -> list[str]:
        out = []
        if not self.self_absorption_corrected:
            out.append(
                "Self-absorption not corrected: phase fractions are biased "
                "towards the sample surface and are qualitative only."
            )
        if not self.texture_corrected:
            out.append(
                "Texture not corrected: the eta-integrated pattern is a powder "
                "pattern only if the voxel is randomly oriented."
            )
        if not self.omega_negated:
            out.append(
                "Omega was NOT negated. At 1-ID that mirrors the reconstruction."
            )
        return out
