"""Assembling sinograms from reduced frames, with sigma.

A sinogram here is intensity as a function of (rotation, translation) for one
detector bin. The reconstruction consumes one per bin, so a channel with
``n_r x n_eta`` bins yields that many sinograms and that many reconstructions.

Two things this module is careful about:

* **Snake correction happens exactly once**, here, and is recorded in the
  result. Applying it twice is the same as not applying it, and neither leaves
  a trace in the output.
* **Variance is assembled alongside**, not recomputed. Reconstruction is
  linear, so a variance sinogram propagates through it in closed form; a sigma
  re-estimated from the reconstructed intensity would be wrong wherever the
  illumination or the counting time varied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .channels import Channel
from .conventions import ScanKnownLimits, unsnake

__all__ = ["SinogramStack", "assemble"]

log = logging.getLogger(__name__)


@dataclass
class SinogramStack:
    """Sinograms for one channel, plus their variances and provenance.

    ``intensity`` is ``(n_bins, n_omega, n_translations)`` -- the layout
    ``midas_tomo.run_tomo_from_sinos`` wants, where the leading axis is the
    "slice" index it reconstructs independently.
    """

    intensity: np.ndarray
    variance: np.ndarray
    omega_deg: np.ndarray
    channel: Channel
    bin_shape: tuple[int, int]          # (n_eta, n_r) before flattening
    limits: ScanKnownLimits
    translations: np.ndarray | None = None

    def __post_init__(self):
        if self.intensity.shape != self.variance.shape:
            raise ValueError(
                f"intensity {self.intensity.shape} and variance "
                f"{self.variance.shape} must have the same shape"
            )
        if self.intensity.ndim != 3:
            raise ValueError(
                f"intensity must be 3-D (bin, omega, translation); got "
                f"{self.intensity.shape}"
            )
        n_om = self.intensity.shape[1]
        if len(self.omega_deg) != n_om:
            raise ValueError(
                f"omega has {len(self.omega_deg)} entries but the sinograms "
                f"have {n_om} rotation rows"
            )

    @property
    def n_bins(self) -> int:
        return int(self.intensity.shape[0])

    @property
    def n_omega(self) -> int:
        return int(self.intensity.shape[1])

    @property
    def n_translations(self) -> int:
        return int(self.intensity.shape[2])

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(self.variance)

    def bin_index(self, eta_bin: int, r_bin: int) -> int:
        """Flat bin index for an (eta, r) pair, matching the assembly order."""
        n_eta, n_r = self.bin_shape
        if not (0 <= eta_bin < n_eta and 0 <= r_bin < n_r):
            raise IndexError(
                f"(eta {eta_bin}, r {r_bin}) outside ({n_eta}, {n_r})"
            )
        return eta_bin * n_r + r_bin

    def describe(self) -> str:
        return (
            f"{self.channel.label}: {self.n_bins} sinograms of "
            f"{self.n_omega} x {self.n_translations} "
            f"(eta x r = {self.bin_shape[0]} x {self.bin_shape[1]}), "
            f"snake_corrected={self.limits.snake_corrected}"
        )


def assemble(
    intensity: np.ndarray,
    variance: np.ndarray,
    omega_deg: np.ndarray,
    channel: Channel,
    *,
    snake: bool,
    omega_negated: bool = True,
    translations: np.ndarray | None = None,
) -> SinogramStack:
    """Build a :class:`SinogramStack` from per-translation reductions.

    Parameters
    ----------
    intensity, variance : ndarray
        ``(n_translations, n_frames, n_eta, n_r)`` as produced by
        :meth:`~midas_dt.reduce.FrameReducer.reduce_translation`.
    snake : bool
        Whether alternate translations were scanned in the opposite rotation
        direction. Pass the result of
        :func:`~midas_dt.scan.detect_snake` rather than a hand-set flag.

    Notes
    -----
    The un-snaking is applied to intensity and variance together. Correcting
    one and not the other would leave sigma attached to the wrong pixel --
    which no test of the intensity alone would catch.
    """
    inten = np.asarray(intensity, dtype=np.float64)
    var = np.asarray(variance, dtype=np.float64)
    if inten.shape != var.shape:
        raise ValueError(
            f"intensity {inten.shape} and variance {var.shape} must match"
        )
    if inten.ndim != 4:
        raise ValueError(
            f"expected (translation, frame, eta, r); got {inten.shape}"
        )

    n_trans, n_frames, n_eta, n_r = inten.shape
    if len(omega_deg) != n_frames:
        raise ValueError(
            f"omega has {len(omega_deg)} entries but there are {n_frames} frames"
        )

    if snake:
        # Both arrays, together -- see the note above.
        inten = unsnake(inten, axis=0, frame_axis=1)
        var = unsnake(var, axis=0, frame_axis=1)
        log.info("un-snaked %d translations", n_trans // 2)

    # (translation, frame, eta, r) -> (eta*r, frame, translation)
    flat_i = inten.reshape(n_trans, n_frames, n_eta * n_r)
    flat_v = var.reshape(n_trans, n_frames, n_eta * n_r)
    sino_i = np.ascontiguousarray(np.transpose(flat_i, (2, 1, 0)))
    sino_v = np.ascontiguousarray(np.transpose(flat_v, (2, 1, 0)))

    return SinogramStack(
        intensity=sino_i, variance=sino_v,
        omega_deg=np.asarray(omega_deg, dtype=np.float64),
        channel=channel, bin_shape=(n_eta, n_r),
        limits=ScanKnownLimits(snake_corrected=bool(snake),
                               omega_negated=omega_negated),
        translations=translations,
    )
