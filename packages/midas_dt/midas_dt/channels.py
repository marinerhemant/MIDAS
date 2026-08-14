"""Which parts of the detector become sinograms.

A :class:`Channel` is one radius window crossed with one azimuthal window,
plus how finely to bin inside it. Both reconstruction branches consume the
same list, which is what makes their results comparable window by window.

Successor to the legacy ``RMin``/``RMax``/``RBinSize`` +
``EtaMin``/``EtaMax``/``EtaBinSize`` + repeated ``RadiusToFit``/``EtaToFit``
keywords, but as an explicit list rather than global state plus a convention
about which repeats pair up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = ["Channel", "channels_from_legacy_params"]


@dataclass(frozen=True)
class Channel:
    """One (radius window x azimuthal window) region of the detector.

    Parameters
    ----------
    r_min, r_max : float
        Detector radius bounds, in pixels.
    eta_min, eta_max : float
        Azimuthal bounds in degrees, in ``[-180, 180]``.
    r_bin, eta_bin : float
        Bin sizes, pixels and degrees. These set how many sinograms the
        channel produces in the reconstruct-then-fit branch: ``n_r * n_eta``,
        one reconstruction each. Fine binning is what makes that branch
        expensive, so choose deliberately.
    label : str | None
        Name used in outputs. Defaults to a description of the window.
    n_peaks : int
        Peaks expected inside the radius window. ``> 1`` selects multi-peak
        fitting, the successor to the legacy ``Rcenters`` list.
    peak_centres : tuple[float, ...]
        Optional starting radii for those peaks. Must be inside the window and
        number ``n_peaks`` when given.
    """

    r_min: float
    r_max: float
    eta_min: float = -180.0
    eta_max: float = 180.0
    r_bin: float = 0.25
    eta_bin: float = 3.0
    label: str | None = None
    n_peaks: int = 1
    peak_centres: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if self.r_max <= self.r_min:
            raise ValueError(
                f"r_max must exceed r_min, got {self.r_min}..{self.r_max}"
            )
        if self.r_min < 0:
            raise ValueError(f"r_min must be non-negative, got {self.r_min}")
        if self.eta_max <= self.eta_min:
            raise ValueError(
                f"eta_max must exceed eta_min, got {self.eta_min}..{self.eta_max}"
            )
        if not (-180.0 <= self.eta_min and self.eta_max <= 180.0):
            raise ValueError(
                f"eta bounds must lie in [-180, 180], got "
                f"{self.eta_min}..{self.eta_max}"
            )
        if self.r_bin <= 0 or self.eta_bin <= 0:
            raise ValueError(
                f"bin sizes must be positive, got r_bin={self.r_bin}, "
                f"eta_bin={self.eta_bin}"
            )
        if self.n_peaks < 1:
            raise ValueError(f"n_peaks must be >= 1, got {self.n_peaks}")
        if self.peak_centres:
            if len(self.peak_centres) != self.n_peaks:
                raise ValueError(
                    f"peak_centres has {len(self.peak_centres)} entries but "
                    f"n_peaks is {self.n_peaks}"
                )
            outside = [c for c in self.peak_centres
                       if not (self.r_min <= c <= self.r_max)]
            if outside:
                raise ValueError(
                    f"peak_centres {outside} lie outside the radius window "
                    f"{self.r_min}..{self.r_max}"
                )
        if self.label is None:
            object.__setattr__(
                self, "label",
                f"rad_{self.r_min:g}_{self.r_max:g}_eta_{self.eta_min:g}_{self.eta_max:g}",
            )

    @property
    def n_r(self) -> int:
        """Radial bins in this channel.

        ``ceil``, not ``floor``: a window whose span is not an exact multiple of
        ``r_bin`` still gets a final partial bin, and the integrator produces
        it. These two disagreed until it was measured -- a 15-124.26 px window
        at 1 px reported 109 bins where the reducer returned 110.

        The count is not cosmetic. Anyone building a radius axis as
        ``linspace(r_min, r_max, n_r)`` -- the obvious thing to write, and what
        :meth:`radii` now exists to prevent -- would get an axis one short of
        the data and silently assign the wrong d-spacing to every bin.
        """
        return max(1, int(math.ceil(
            (self.r_max - self.r_min) / self.r_bin - 1e-9)))

    @property
    def n_eta(self) -> int:
        """Azimuthal bins in this channel. ``ceil``, as for :attr:`n_r`."""
        return max(1, int(math.ceil(
            (self.eta_max - self.eta_min) / self.eta_bin - 1e-9)))

    def radii(self) -> "np.ndarray":
        """Bin-centre radii, in pixels, matching what the integrator returns.

        Use this rather than building the axis by hand: it cannot drift from
        :attr:`n_r`, and getting the length wrong misassigns every d-spacing.
        """
        import numpy as np
        return np.linspace(self.r_min, self.r_max, self.n_r)

    @property
    def n_sinograms(self) -> int:
        """Sinograms the reconstruct-then-fit branch builds for this channel.

        Also the number of reconstructions it costs, which is the figure to
        look at before launching a fine-binned run.
        """
        return self.n_r * self.n_eta

    def describe(self) -> str:
        return (
            f"{self.label}: r {self.r_min:g}-{self.r_max:g} px / {self.r_bin:g}, "
            f"eta {self.eta_min:g}-{self.eta_max:g} deg / {self.eta_bin:g} "
            f"-> {self.n_r} x {self.n_eta} = {self.n_sinograms} sinograms"
        )

    @classmethod
    def centred(cls, radius: float, width: float, *, eta: float = 0.0,
                eta_width: float = 180.0, **kw) -> "Channel":
        """Build from a centre and half-width.

        Mirrors the legacy ``RadiusToFit <rad> <width>`` /
        ``EtaToFit <eta> <width>`` spelling, where the second number is a
        half-width, not a full width.
        """
        return cls(
            r_min=radius - width, r_max=radius + width,
            eta_min=max(-180.0, eta - eta_width),
            eta_max=min(180.0, eta + eta_width),
            **kw,
        )


def channels_from_legacy_params(params: dict) -> list[Channel]:
    """Build channels from a parsed legacy DT parameter file.

    Understands ``rads``/``Rwidth``, ``etas``/``etaWidth``, ``RBinSize``,
    ``EtaBinSize`` and ``Rcenters``/``multipeak``. The legacy files pair every
    radius with every eta, so this returns their cross product -- matching
    what ``recon_peak_all_mul.py`` did with its nested loops.
    """
    rads = params.get("rads") or []
    etas = params.get("etas") or [0.0]
    if not rads:
        raise ValueError("no 'rads' entries: nothing to build a channel from")

    r_width = float(params.get("Rwidth", 10.0))
    eta_width = float(params.get("etaWidth", 180.0))
    r_bin = float(params.get("RBinSize", 0.25))
    eta_bin = float(params.get("EtaBinSize", 3.0))
    centres = tuple(float(c) for c in (params.get("Rcenters") or ()))
    multipeak = bool(int(params.get("multipeak", 0)))

    out: list[Channel] = []
    for rad in rads:
        rad = float(rad)
        # Only keep the Rcenters that fall inside this window; the legacy
        # files list them globally across all radius windows.
        inside = tuple(c for c in centres
                       if rad - r_width <= c <= rad + r_width) if multipeak else ()
        for eta in etas:
            out.append(Channel.centred(
                rad, r_width, eta=float(eta), eta_width=eta_width,
                r_bin=r_bin, eta_bin=eta_bin,
                n_peaks=len(inside) if inside else 1,
                peak_centres=inside,
            ))
    return out
