"""Correct grain volumes for the volume the beam actually lit.

The defect
----------
``radius/core.py:180`` computes a grain volume as an intensity ratio against a
*gauge volume*::

    V_grain = 0.5 * m_hkl * dTheta * cos(Theta) * V_gauge * I_spot
              / (N_imgs * I_powder)

and ``V_gauge = Hbeam * pi * Rsample^2`` (``core.py:172``), overridden by
``Vsample`` when set. ``Hbeam`` and ``Rsample`` are deliberately **generous
search bounds** — by a hard project rule they are never the specimen — and
``midas_calibrate_v2`` writes ``Rsample 1000 / Hbeam 1000 / Vsample 50000000``
as template defaults. So the absolute scale of every reported grain size in
MIDAS is a canned constant that nothing measures.

Measured on the FF reference run (``ff_refiner_prepost/result/LayerNr_1``,
6112 grains): no ``Vsample`` line, so ``V_gauge = 2000 * pi * 2000^2 =
2.513e10 um^3``, and the sum of all grain volumes is **6.5 %** of it.

The correction
--------------
Writing the corrected estimator over the legacy one, everything cancels except::

    V_cor(s) / V_leg(s) = (V_illum / V_gauge) * C_cov(r) * f(s)/<f>_r

* ``V_illum / V_gauge`` is **global** — one number for the whole run. It needs
  no absorption model and it is where the factor of 5-20x in radius lives.
* ``f(s)/<f>_r`` is the per-spot part (absorption, beam profile). Note the
  division by the per-ring mean: **only the spread survives**, because the
  denominator ``I_powder`` is itself a sum of observed intensities and suffers
  every effect the numerator does.

That last point is the trap this module exists to prevent.

Powder double-counting
----------------------
``powder_int`` (``core.py:153-160``) is the sum of *observed* spot intensities
on a ring. Correcting ``I_spot`` while leaving it raw inflates every volume by
``<1/A>`` — uniformly, in the direction people expect, with no symptom.
At mu*D ~ 0.5 that is about 1.6x in volume and 17 % in radius, and it would
read as "the correction found bigger grains".

:func:`normalise_per_ring` is the guard: it enforces ``<f>_r == 1`` by
construction, so a *uniform* correction is exactly no correction. That identity
is asserted bit-exactly in the tests, which is why the function short-circuits
a constant ring rather than trusting ``f / mean(f)`` to round to 1.0.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

__all__ = [
    "GaugeVolume",
    "ShapeCorrection",
    "correct_grain_volumes",
    "normalise_per_ring",
    "volume_to_radius",
]

#: ``midas_calibrate_v2`` template defaults. Their presence means nobody chose
#: these values for this specimen — the number is the template's, not a
#: measurement of anything.
_TEMPLATE_RSAMPLE_UM = 1000.0
_TEMPLATE_HBEAM_UM = 1000.0
_TEMPLATE_VSAMPLE_UM3 = 50_000_000.0


# ------------------------------------------------------------- the gauge volume

@dataclass(frozen=True)
class GaugeVolume:
    """``V_gauge`` exactly as the pipeline computed it, with its provenance.

    Reproduces ``radius/core.py:168-174`` rather than paraphrasing it, so the
    ratio this module forms is against the number that actually divided into
    the reported volumes — not a plausible reconstruction of it.
    """

    hbeam_um: float
    rsample_um: float
    vsample_um3: float = 0.0
    disc_model: int = 0
    disc_area_um2: float = 0.0

    @property
    def value_um3(self) -> float:
        if self.disc_model == 1:
            return float(self.disc_area_um2)
        if self.vsample_um3 != 0:
            return float(self.vsample_um3)
        return float(self.hbeam_um * math.pi * self.rsample_um ** 2)

    @property
    def source(self) -> str:
        """Which of the three branches produced :attr:`value_um3`."""
        if self.disc_model == 1:
            return "DiscArea (DiscModel 1)"
        if self.vsample_um3 != 0:
            return "Vsample"
        return "Hbeam * pi * Rsample^2 (both are SEARCH BOUNDS)"

    @property
    def is_template_default(self) -> bool:
        """True when the value came from a calibration template, unchanged.

        Not an error — most runs are like this — but it means the number is
        the template's and the reported absolute grain sizes carry it.
        """
        if self.vsample_um3 == _TEMPLATE_VSAMPLE_UM3:
            return True
        return (
            self.vsample_um3 == 0
            and self.rsample_um == _TEMPLATE_RSAMPLE_UM
            and self.hbeam_um == _TEMPLATE_HBEAM_UM
        )

    @classmethod
    def from_param_file(cls, path: Union[str, Path]) -> "GaugeVolume":
        """Read the four keys out of a MIDAS parameter file.

        Absent keys take the same defaults the pipeline does (``Vsample 0``,
        ``DiscModel 0``), because absence is exactly how the FF reference run
        ends up on the search-bound branch.
        """
        vals: Dict[str, float] = {}
        wanted = {"hbeam", "rsample", "vsample", "discmodel", "discarea"}
        for line in Path(path).read_text(errors="replace").splitlines():
            tok = re.split(r"[\s;]+", line.strip())
            if len(tok) < 2:
                continue
            key = tok[0].lower()
            if key in wanted and key not in vals:
                try:
                    vals[key] = float(tok[1])
                except ValueError:
                    continue
        missing = {"hbeam", "rsample"} - set(vals)
        if missing and "vsample" not in vals and vals.get("discmodel", 0) != 1:
            raise ValueError(
                f"{path} has no {sorted(missing)} and no Vsample/DiscArea, so "
                "V_gauge cannot be reproduced. Without it there is nothing to "
                "take a ratio against and the correction is undefined."
            )
        return cls(
            hbeam_um=vals.get("hbeam", 0.0),
            rsample_um=vals.get("rsample", 0.0),
            vsample_um3=vals.get("vsample", 0.0),
            disc_model=int(vals.get("discmodel", 0)),
            disc_area_um2=vals.get("discarea", 0.0),
        )


# ------------------------------------------------- the powder-reference guard

def normalise_per_ring(
    f, ring_index=None, *, atol: float = 0.0
) -> np.ndarray:
    """Divide a per-spot correction by its per-ring mean, enforcing ``<f>_r = 1``.

    **This is the guard against powder double-counting**, which is the failure
    mode most likely to happen silently: the ``I_powder`` denominator is itself
    a sum of observed intensities, so any part of the correction common to a
    whole ring is already in it. Correcting the numerator alone multiplies
    every volume by ``<1/A>`` — uniform, plausible, and in the expected
    direction.

    A ring whose corrections are all equal returns **exactly** 1.0, not
    1.0 +/- 1 ulp: the identity "a uniform correction is no correction" is
    asserted bit-exactly downstream, and ``f / f.mean()`` does not guarantee it
    for an arbitrary constant. Rings with a real spread are normalised the
    ordinary way.

    Parameters
    ----------
    f
        Per-spot visibility/absorption factor, strictly positive.
    ring_index
        Ring label per spot. ``None`` treats every spot as one ring, which is
        correct only when the correction has no ring dependence — say so
        explicitly rather than letting it default.
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError(f"f must be 1-D, got shape {f.shape}")
    if f.size and not np.all(f > 0):
        n = int((f <= 0).sum())
        raise ValueError(
            f"{n} of {f.size} corrections are <= 0. f is a transmitted "
            "fraction and dividing by it is the whole operation; a zero here "
            "would make a grain infinitely large."
        )
    if ring_index is None:
        ring_index = np.zeros(f.shape, dtype=np.int64)
    ring_index = np.asarray(ring_index)
    if ring_index.shape != f.shape:
        raise ValueError(
            f"ring_index shape {ring_index.shape} != f shape {f.shape}"
        )

    out = np.empty_like(f)
    for r in np.unique(ring_index):
        sel = ring_index == r
        vals = f[sel]
        if float(np.ptp(vals)) <= atol:
            # Exactly no correction. See the docstring: this branch exists so
            # the "constant correction is bit-identical" invariant holds.
            out[sel] = 1.0
        else:
            out[sel] = vals / vals.mean()
    return out


# ------------------------------------------------------------- the correction

def volume_to_radius(volume_um3, *, disc_model: int = 0) -> np.ndarray:
    """Grain volume -> reported radius, matching ``core.py:186-189``.

    Sign-preserving cube root for the 3-D case (volumes can come out negative
    when a spot's intensity is below the local background), ``sqrt(V/pi)`` for
    the disc model.
    """
    v = np.asarray(volume_um3, dtype=np.float64)
    if disc_model == 1:
        return np.sqrt(np.abs(v) / math.pi) * np.sign(v)
    return np.sign(v) * np.abs(v) ** (1.0 / 3.0) * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)


@dataclass
class ShapeCorrection:
    """What :func:`correct_grain_volumes` did, and whether to believe it."""

    v_gauge_um3: float
    v_illum_um3: float
    volume_scale: float
    radius_scale: float
    n_spots: int
    per_spot_applied: bool
    packing_fraction: Optional[float] = None
    gauge_source: str = ""
    gauge_is_template_default: bool = False
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"V_gauge   {self.v_gauge_um3:.4g} um^3   [{self.gauge_source}]",
            f"V_illum   {self.v_illum_um3:.4g} um^3",
            f"volume x  {self.volume_scale:.4g}",
            f"radius x  {self.radius_scale:.4g}",
            f"per-spot  {'applied' if self.per_spot_applied else 'none (global only)'}",
        ]
        if self.packing_fraction is not None:
            lines.append(f"packing   {self.packing_fraction:.4f} of V_illum")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def correct_grain_volumes(
    volume_um3,
    *,
    gauge: Union[GaugeVolume, float],
    illuminated_volume_um3: float,
    spot_correction=None,
    ring_index=None,
    disc_model: int = 0,
):
    """Rescale legacy grain volumes onto the measured illuminated volume.

    Returns ``(volume_corrected, radius_corrected, report)``.

    ``spot_correction`` is ``f(s)`` — the transmitted/illuminated fraction for
    each spot. It is normalised per ring before use (see
    :func:`normalise_per_ring`), so supplying a constant is exactly a no-op and
    only its *spread* has any effect. Leave it ``None`` for the Phase-3 case:
    the global term is the whole first-order correction and needs no
    absorption model.

    ``packing_fraction`` in the report is ``sum(V_corrected) / V_illum``. It is
    a physical bound: the grains cannot occupy more than the volume that was
    lit. A value above 1 means the correction, the mask, or the indexed
    fraction is wrong, and it is reported rather than clipped.

    **What packing_fraction cannot tell you.** It is exactly invariant under
    the global term — ``sum(V*s)/(V_gauge*s) = sum(V)/V_gauge`` — so it is the
    same number before and after the rescale and carries no information about
    whether ``V_illum`` is right. On the FF reference run it reads 0.0653
    either way, which is the 6.5 % already measured against ``V_gauge``, not a
    second confirmation of it. It is a bound and a per-spot diagnostic, not a
    check on the shape.
    """
    v = np.asarray(volume_um3, dtype=np.float64)
    g = gauge if isinstance(gauge, GaugeVolume) else None
    v_gauge = float(g.value_um3 if g is not None else gauge)
    v_illum = float(illuminated_volume_um3)

    if not (v_gauge > 0):
        raise ValueError(f"V_gauge must be > 0; got {v_gauge}")
    if not (v_illum > 0):
        raise ValueError(
            f"V_illum must be > 0; got {v_illum}. An empty illuminated volume "
            "usually means the beam slab missed the sample in z — check "
            "slice0_z_um against the stage position."
        )

    scale = v_illum / v_gauge
    out = v * scale

    per_spot = spot_correction is not None
    if per_spot:
        f = normalise_per_ring(spot_correction, ring_index)
        if f.shape != v.shape:
            raise ValueError(
                f"spot_correction shape {f.shape} != volume shape {v.shape}"
            )
        out = out * f

    warnings: list = []
    if g is not None and g.is_template_default:
        warnings.append(
            f"V_gauge came from calibration-template defaults ({g.source}); "
            "the legacy absolute sizes carry the template constant."
        )
    if spot_correction is None:
        warnings.append(
            "global term only - no per-spot absorption/beam correction applied"
        )

    total = float(np.abs(out).sum())
    packing = total / v_illum
    if packing > 1.0:
        warnings.append(
            f"corrected grain volumes sum to {packing:.3g} x the illuminated "
            "volume, which is impossible. Suspect the mask, the threshold, or "
            "double-counted spots (one volume per SPOT, not per grain)."
        )

    report = ShapeCorrection(
        v_gauge_um3=v_gauge, v_illum_um3=v_illum,
        volume_scale=scale, radius_scale=scale ** (1.0 / 3.0),
        n_spots=int(v.size), per_spot_applied=per_spot,
        packing_fraction=packing,
        gauge_source=(g.source if g is not None else "supplied directly"),
        gauge_is_template_default=bool(g.is_template_default) if g else False,
        warnings=warnings,
    )
    return out, volume_to_radius(out, disc_model=disc_model), report
