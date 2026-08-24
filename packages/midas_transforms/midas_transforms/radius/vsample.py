"""Setting ``Vsample`` from a tomographic reconstruction.

The gap this closes
-------------------
``Vsample`` is the gauge volume that divides into every reported grain volume
(``radius/core.py:174``). Nothing has ever produced it. In practice it is
either absent — in which case the pipeline falls back to
``Hbeam * pi * Rsample^2``, built from two deliberately generous *search
bounds* — or it is a template constant. ``midas_calibrate_v2`` writes
``Vsample 50000000``.

A tomogram of the same specimen measures the missing quantity directly. This
module turns one into a ``Vsample`` value and, on request, writes it into the
FF parameter file with its provenance attached.

Why this is a safe integration, when the absorption path is not
---------------------------------------------------------------
``Vsample`` needs the specimen's **volume**, not its pose, and volume is
insensitive to almost everything the registration could get wrong:

* a **mirrored** mask has exactly the same volume (the failure mode that
  silently ruins path lengths does nothing here);
* the **rotation-axis position** barely matters — measured on Ce ht525 s2, an
  8 px shift error moved the cross-section by **0.25 %**;
* the **vertical registration** matters only if the cross-section varies with
  height, which is checkable and is checked below (Ce: 0.63 %).

What it still needs, and will not guess, is the **beam height**. That is a slit
setting, not a property of the specimen, and `Hbeam` is not it — by a hard
project rule `Hbeam` is a search bound. So the tomogram supplies the area and
the operator supplies the one number the tomogram cannot know.

The refusals
------------
Writing a bad ``Vsample`` is worse than leaving it unset, because a template
constant is at least obviously a constant. So:

* a **threshold-driven** volume is refused, but the bar is "is this number
  meaningful", not "is it perfect". The comparison is against what it replaces:
  a template constant or a pair of search bounds, which can be wrong by orders
  of magnitude. Refusing a value good to 16 % in favour of one that is off by
  2500x would be the wrong trade. So the spread is *recorded with the value*
  and only a spread above ``max_spread`` (25 % by default) is refused.
  Measured on Ce ht525 s2: the cross-section is reproducible to 0.6 % across
  four different artefact corrections and 0.25 % across an 8 px centring
  error, while a +/-40 % threshold sweep moves the volume by 16 % -- the
  specimen's edges sit in the beam penumbra, so the boundary is genuinely
  soft. That is a real +/-16 % uncertainty, and it belongs in the file rather
  than being grounds to write nothing.
* a **height-varying** cross-section is refused unless the vertical
  registration is supplied, because then it matters which slab the beam lit.
* an **omega-varying** illuminated volume is refused outright. ``Vsample`` is a
  scalar; if a narrow beam on a non-cylindrical specimen makes ``V_illum``
  depend on omega, no scalar is correct and the estimator needs the per-omega
  form it does not have.
"""
from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["VsampleResult", "vsample_from_shape", "write_vsample"]

log = logging.getLogger(__name__)


@dataclass
class VsampleResult:
    """A measured gauge volume, and whether it is fit to be written."""

    vsample_um3: float
    beam_height_um: float
    beam_width_um: Optional[float]
    usable: bool
    reasons: List[str] = field(default_factory=list)
    cross_section_um2: float = float("nan")
    cross_section_cv: float = float("nan")
    equivalent_diameter_um: float = float("nan")
    omega_modulation: float = float("nan")
    previous_vsample_um3: Optional[float] = None
    previous_source: str = ""
    beam_height_source: str = "operator-supplied"
    detail: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.usable

    @property
    def volume_scale(self) -> Optional[float]:
        """New / old gauge volume — the factor every grain volume moves by."""
        if not self.previous_vsample_um3:
            return None
        return self.vsample_um3 / self.previous_vsample_um3

    @property
    def radius_scale(self) -> Optional[float]:
        s = self.volume_scale
        return None if s is None else s ** (1.0 / 3.0)

    def provenance_lines(self) -> List[str]:
        """``#`` comment lines to write above the value."""
        lines = [
            "# Vsample MEASURED from a tomographic reconstruction of the same",
            "# specimen -- not a template constant and not a search bound.",
            f"#   written by midas_transforms.radius.vsample on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"#   cross-section     {self.cross_section_um2:.0f} um^2 "
            f"(equivalent diameter {self.equivalent_diameter_um:.1f} um)",
            f"#   cross-section CV  {self.cross_section_cv:.4f} along the "
            f"reconstructed height",
            "#   NOTE a MEASUREMENT WITH AN UNCERTAINTY, not an exact value --",
            "#        see threshold_spread below for how far the boundary moves.",
            f"#   beam height       {self.beam_height_um} um  "
            f"[{self.beam_height_source}]",
        ]
        if self.beam_width_um is not None:
            lines.append(f"#   beam width        {self.beam_width_um} um  "
                         f"(omega modulation {self.omega_modulation:.4f})")
        for k, v in self.detail.items():
            lines.append(f"#   {k}: {v}")
        if self.previous_vsample_um3:
            lines.append(
                f"#   replaces {self.previous_vsample_um3:.6g} um^3 "
                f"[{self.previous_source}] -- grain volumes move by "
                f"{self.volume_scale:.4g}x, radii by {self.radius_scale:.4g}x"
            )
        return lines

    def summary(self) -> str:
        out = [
            f"Vsample {self.vsample_um3:.6g} um^3",
            f"  cross-section    {self.cross_section_um2:.0f} um^2 "
            f"(equiv. diameter {self.equivalent_diameter_um:.1f} um)",
            f"  height CV        {self.cross_section_cv:.4f}",
            f"  beam height      {self.beam_height_um} um "
            f"[{self.beam_height_source}]",
        ]
        if self.beam_width_um is not None:
            out.append(f"  omega modulation {self.omega_modulation:.4f}")
        if self.previous_vsample_um3:
            out.append(
                f"  was             {self.previous_vsample_um3:.6g} um^3 "
                f"[{self.previous_source}]")
            out.append(
                f"  => grain volumes x{self.volume_scale:.4g}, "
                f"radii x{self.radius_scale:.4g}")
        out.append(f"  usable: {self.usable}"
                   + ("" if self.usable else " -- " + "; ".join(self.reasons)))
        return "\n".join(out)


def vsample_from_shape(
    shape,
    *,
    beam_height_um: float,
    beam_width_um: Optional[float] = None,
    beam_centre_z_um: Optional[float] = None,
    threshold_report: Optional[Dict[str, Any]] = None,
    omegas_deg: Optional[Sequence[float]] = None,
    uniformity_tol: float = 0.05,
    modulation_tol: float = 0.05,
    max_spread: float = 0.25,
    beam_height_source: str = "operator-supplied",
    param_file: Union[str, Path, None] = None,
) -> VsampleResult:
    """Measure the gauge volume from a :class:`SampleShape`.

    ``beam_height_um`` is required and is not inferrable — see the module
    docstring. ``beam_width_um`` may be omitted when the beam is wider than the
    specimen, which is the usual FF line-beam case; supply it and the
    omega-dependence is checked.

    ``param_file``, when given, is read (never written) so the result reports
    what the new value replaces.
    """
    if not (beam_height_um > 0):
        raise ValueError(
            f"beam_height_um must be > 0; got {beam_height_um}. It is a slit "
            "setting and cannot be taken from the tomogram, nor from Hbeam, "
            "which is a search bound."
        )

    occ = np.asarray(shape.occupancy, dtype=np.float64)
    px = float(shape.pixel_size_um)
    reasons: List[str] = []

    if occ.sum() <= 0:
        raise ValueError("the sample mask is empty; nothing to measure")

    # Per-slice cross-section: the quantity the beam height multiplies.
    area = occ.sum(axis=(1, 2)) * px * px
    lit = area[area > 0]
    if lit.size == 0:
        raise ValueError("no reconstructed slice contains sample")
    a_mean = float(lit.mean())
    cv = float(lit.std() / a_mean) if a_mean > 0 else float("inf")
    equiv_d = 2.0 * math.sqrt(a_mean / math.pi)

    detail: Dict[str, Any] = {
        "n_slices_with_sample": int(lit.size),
        "reconstructed_height_um": f"{occ.shape[0] * shape.slice_pitch_um:.1f}",
        "pixel_size_um": px,
    }
    src = shape.provenance.get("source")
    if src:
        detail["tomogram"] = src

    # --- the threshold gate -------------------------------------------------
    if threshold_report is None:
        reasons.append(
            "no threshold_sensitivity report was supplied, so it is not known "
            "whether this volume is threshold-driven. Sweep the threshold and "
            "pass the report."
        )
    else:
        spread = float(threshold_report.get("fractional_spread", float("nan")))
        detail["threshold_spread"] = f"{spread:.4f}"
        detail["volume_uncertainty"] = f"+/-{50 * spread:.0f} % (half the sweep)"
        detail["radius_uncertainty"] = (
            f"+/-{100 * ((1 + spread / 2) ** (1 / 3) - 1):.1f} %")
        if spread > max_spread:
            reasons.append(
                f"the volume moves {100 * spread:.0f} % across the threshold "
                f"sweep (limit {100 * max_spread:.0f} %), which is too much "
                "for the number to mean anything. Check for a soft boundary: "
                "a specimen whose edge sits in the beam penumbra reconstructs "
                "with a gradual edge."
            )

    # --- does it matter which slab the beam lit? ----------------------------
    if cv > uniformity_tol:
        if beam_centre_z_um is None:
            reasons.append(
                f"the cross-section varies by {cv:.3f} over the reconstructed "
                f"height (limit {uniformity_tol}), so which slab the beam lit "
                "matters. Supply beam_centre_z_um and a registered "
                "slice0_z_um, or reconstruct only the illuminated slab."
            )
    detail["cross_section_cv"] = f"{cv:.4f}"

    # --- can a scalar represent it at all? ---------------------------------
    modulation = 0.0
    if beam_width_um is not None:
        oms = np.asarray(omegas_deg if omegas_deg is not None
                         else np.arange(0.0, 180.0, 10.0), dtype=np.float64)
        v_om = shape.illuminated_volume_sinogram(
            oms, beam_height_um=beam_height_um, beam_width_um=beam_width_um,
            beam_centre_z_um=(beam_centre_z_um or 0.0),
        )
        if v_om.mean() > 0:
            modulation = float(v_om.std() / v_om.mean())
        if modulation > modulation_tol:
            reasons.append(
                f"the illuminated volume varies with omega by {modulation:.3f} "
                f"(limit {modulation_tol}): the beam is narrower than the "
                "specimen and the overlap changes as it turns. Vsample is a "
                "SCALAR, so no single value is correct here."
            )

    # --- the value ---------------------------------------------------------
    if beam_centre_z_um is not None:
        v = shape.illuminated_volume_um3(
            beam_height_um=beam_height_um, beam_width_um=beam_width_um,
            beam_centre_z_um=beam_centre_z_um,
        )
        detail["method"] = "beam slab intersected with the mask"
    else:
        v = a_mean * float(beam_height_um)
        detail["method"] = ("mean cross-section x beam height "
                            "(valid because the cross-section is uniform)")

    prev = prev_src = None
    if param_file is not None:
        from .shape_correction import GaugeVolume

        try:
            g = GaugeVolume.from_param_file(param_file)
            prev, prev_src = g.value_um3, g.source
            if g.is_template_default:
                prev_src += " -- A CALIBRATION-TEMPLATE DEFAULT"
        except (OSError, ValueError) as exc:
            log.warning("could not read %s: %s", param_file, exc)

    return VsampleResult(
        vsample_um3=float(v), beam_height_um=float(beam_height_um),
        beam_width_um=(None if beam_width_um is None else float(beam_width_um)),
        usable=not reasons, reasons=reasons,
        cross_section_um2=a_mean, cross_section_cv=cv,
        equivalent_diameter_um=equiv_d, omega_modulation=modulation,
        previous_vsample_um3=prev, previous_source=prev_src or "",
        beam_height_source=beam_height_source, detail=detail,
    )


def write_vsample(
    param_file: Union[str, Path],
    result: VsampleResult,
    *,
    backup: bool = True,
    force: bool = False,
) -> Path:
    """Patch ``Vsample`` into an FF parameter file, with provenance above it.

    Refuses an unusable result unless ``force`` — the whole point is that a
    measured gauge volume is better than a constant, and a threshold-driven one
    is not measured.

    The previous ``Vsample`` line, if any, is commented out rather than
    deleted, so the file still records what the run used to use.
    """
    if not result.usable and not force:
        raise ValueError(
            "refusing to write an unusable Vsample: "
            + "; ".join(result.reasons)
            + ". Fix the cause, or pass force=True and accept that the number "
              "is not a measurement."
        )

    path = Path(param_file)
    lines = path.read_text().splitlines()
    if backup:
        bak = path.with_suffix(path.suffix + ".before_vsample")
        shutil.copy2(path, bak)
        log.info("backed up %s -> %s", path, bak)

    out: List[str] = []
    replaced = False
    for ln in lines:
        tok = ln.strip().split()
        if tok and tok[0].lower() == "vsample" and not ln.lstrip().startswith("#"):
            out.append(f"# superseded by the measured value below: {ln.strip()}")
            replaced = True
        else:
            out.append(ln)

    out.append("")
    out.extend(result.provenance_lines())
    out.append(f"Vsample {result.vsample_um3:.6f}")
    if not replaced:
        out.append("# (no Vsample line existed before; the pipeline was using "
                   "Hbeam * pi * Rsample^2, both of which are search bounds)")

    path.write_text("\n".join(out) + "\n")
    return path
