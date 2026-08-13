"""Detector frames to (R, eta) intensity, with the error bar carried along.

Integration itself is `midas_integrate_v2`'s job; this module supplies the
scan-aware loop around it and makes variance a first-class output rather than
something recomputed later from a wrong assumption.

Why the variance matters here specifically: the standard XRD-CT stack
(pyFAI + nDTomo) produces a chemistry map with no error bar at all. Poisson
counting error is known exactly at the detector, and integration is a linear
combination of pixels, so it propagates in closed form. Everything downstream
-- sinograms, reconstruction, peak fits -- can then carry sigma instead of
inventing one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .channels import Channel
from .geometry import DTGeometry

__all__ = ["ReducedFrame", "FrameReducer", "poisson_variance"]

log = logging.getLogger(__name__)


def poisson_variance(counts: np.ndarray, *, dark: np.ndarray | None = None,
                     gain: float = 1.0, read_noise: float = 0.0) -> np.ndarray:
    """Per-pixel variance of a raw detector frame.

    For photon counting, ``var = counts / gain`` in detector units, plus a
    read-noise floor and, when a dark has been subtracted, that frame's own
    variance too -- subtracting a dark does not remove its noise, it adds it.

    Negative counts (over-subtraction) are clipped to zero *for the variance
    only*: a negative variance is meaningless, whereas a negative intensity is
    a legitimate fluctuation and is left alone.
    """
    counts = np.asarray(counts, dtype=np.float64)
    var = np.clip(counts, 0.0, None) / max(gain, 1e-12)
    if dark is not None:
        var = var + np.clip(np.asarray(dark, dtype=np.float64), 0.0, None) / max(gain, 1e-12)
    if read_noise:
        var = var + float(read_noise) ** 2
    return var


@dataclass
class ReducedFrame:
    """One frame integrated into a channel's (R, eta) grid."""

    intensity: np.ndarray      # (n_eta, n_r)
    variance: np.ndarray       # (n_eta, n_r), same units squared
    channel: Channel
    translation: int
    frame: int

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(self.variance)

    @property
    def lineout(self) -> np.ndarray:
        """Eta-collapsed 1-D profile: sum over azimuth at each radius."""
        return np.nansum(self.intensity, axis=0)

    @property
    def lineout_variance(self) -> np.ndarray:
        """Variance of :attr:`lineout`.

        Sum of independent bins, so variances add. This is where propagating
        rather than re-estimating pays: the eta bins have very different
        counts, and a sigma guessed from the summed intensity would be wrong
        wherever the illumination is uneven.
        """
        return np.nansum(self.variance, axis=0)


class FrameReducer:
    """Integrates frames into one channel's (R, eta) grid.

    Builds the integration geometry once and reuses it for every frame, which
    is the whole cost saving -- the (R, eta) map depends only on the detector
    geometry and the binning, not on the data.
    """

    def __init__(
        self,
        geometry: DTGeometry,
        channel: Channel,
        *,
        dark: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        gain: float = 1.0,
        read_noise: float = 0.0,
        device: str = "cpu",
    ):
        self.geometry = geometry
        self.channel = channel
        self.dark = None if dark is None else np.asarray(dark, dtype=np.float64)
        self.mask = mask
        self.gain = gain
        self.read_noise = read_noise
        self.device = device
        self._geom = None       # built lazily; importing torch is not free

    def _integration_geometry(self):
        if self._geom is None:
            try:
                import torch
                from midas_integrate_v2 import HardBinGeometry
            except ImportError as exc:
                raise ImportError(
                    "reduction needs midas-integrate-v2 (and torch). Install "
                    "with `pip install midas-dt[full]`."
                ) from exc
            spec = self.geometry.to_integration_spec(self.channel)
            # HardBinGeometry, not build_geometry(): the variance-aware
            # integrators take the hard-bin structure, while build_geometry()
            # returns the CSR-backed IntegrationGeometry used by integrate().
            # Its flat bin index is eta_bin * n_r + r_bin, which is the
            # ordering sinogram.assemble() and Reconstruction.voxel_pattern()
            # both assume.
            self._geom = HardBinGeometry.from_spec(spec, mask=self.mask)
            log.info("built hard-bin map for %s (%d x %d bins)",
                     self.channel.label, self.channel.n_eta, self.channel.n_r)
        return self._geom

    def reduce_frame(self, image: np.ndarray, *, translation: int = -1,
                     frame: int = -1) -> ReducedFrame:
        """Integrate one raw frame, propagating counting error.

        The dark is subtracted from the intensity and *added* to the variance.
        """
        import torch
        from midas_integrate_v2 import integrate_hard_with_variance

        raw = np.asarray(image, dtype=np.float64)
        var = poisson_variance(raw, dark=self.dark, gain=self.gain,
                               read_noise=self.read_noise)
        signal = raw - self.dark if self.dark is not None else raw

        geom = self._integration_geometry()
        inten, out_var = integrate_hard_with_variance(
            torch.as_tensor(signal, device=self.device),
            geom,
            variance_image=torch.as_tensor(var, device=self.device),
        )
        return ReducedFrame(
            intensity=np.asarray(inten.detach().cpu().numpy()),
            variance=np.asarray(out_var.detach().cpu().numpy()),
            channel=self.channel, translation=translation, frame=frame,
        )

    def reduce_translation(self, scan, translation: int, *,
                           frames: Sequence[int] | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate a whole translation.

        Returns ``(intensity, variance)`` shaped ``(n_frames, n_eta, n_r)``.
        Frames stream one at a time -- a translation is 14+ GB on the U3O8
        scan, so materialising it would defeat the memory-mapped reader.
        """
        idx = range(scan.n_frames) if frames is None else list(frames)
        first = self.reduce_frame(scan.frame(translation, next(iter(idx))),
                                  translation=translation, frame=0)
        n = len(idx) if not isinstance(idx, range) else len(list(idx))
        inten = np.empty((n, *first.intensity.shape), dtype=np.float64)
        var = np.empty_like(inten)
        inten[0], var[0] = first.intensity, first.variance
        for k, f in enumerate(list(idx)[1:], start=1):
            r = self.reduce_frame(scan.frame(translation, f),
                                  translation=translation, frame=f)
            inten[k], var[k] = r.intensity, r.variance
        return inten, var
