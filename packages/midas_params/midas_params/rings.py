"""Ring geometry helper: which Bragg rings fall on the detector?

Given wavelength, detector geometry, and a crystal (space group + lattice),
enumerate reflections, compute 2θ, project onto the detector, and report
which rings are visible.

Value prop: users who've copied an Example file often leave RingThresh/
OverAllRingToIndex set for the Example's calibrant (e.g. CeO2), which has
nothing to do with their actual sample. This helper tells them:

  ring 1  (1 0 0)  2θ=4.52°  radius=79.0 mm  ✓ on detector
  ring 2  (1 1 0)  2θ=6.39°  radius=111.7 mm ✓ on detector
  ring 3  (2 0 0)  2θ=9.03°  radius=157.9 mm ✓ on detector
  ring 4  (2 1 1)  2θ=11.06° radius=193.5 mm × past RhoD

This is a minimum-dependency implementation: stdlib math only, cubic
lattices covered out of the box, lower-symmetry systems flagged as
"needs symmetry expansion" rather than guessed incorrectly.

Accurate multiplicity/extinction logic is explicitly NOT the goal here —
MIDAS's own `GetHKLList` is authoritative. This helper is for "is ring N
plausibly on my detector" wizard prompts, not for simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RingInfo:
    """Geometry summary for one Bragg ring."""

    ring_nr: int                   # 1-indexed in order of increasing 2θ
    hkl: tuple[int, int, int]      # a representative (h, k, l)
    d_spacing: float               # Å
    two_theta: float               # deg
    radius_um: float               # µm on the detector at Lsd
    on_detector: bool              # radius ≤ RhoD
    fits_ypx: bool | None = None   # radius ≤ NrPixelsY × px / 2 (if args supplied)


def _d_cubic(h: int, k: int, l: int, a: float) -> float:
    """d-spacing for cubic (a=b=c, α=β=γ=90°)."""
    s = h * h + k * k + l * l
    if s == 0:
        return float("inf")
    return a / math.sqrt(s)


def _d_hexagonal(h: int, k: int, l: int, a: float, c: float) -> float:
    """d-spacing for hexagonal (a=b, γ=120°)."""
    s = (4.0 / 3.0) * (h * h + h * k + k * k) / (a * a) + (l * l) / (c * c)
    if s == 0:
        return float("inf")
    return 1.0 / math.sqrt(s)


def _d_tetragonal(h: int, k: int, l: int, a: float, c: float) -> float:
    s = (h * h + k * k) / (a * a) + (l * l) / (c * c)
    if s == 0:
        return float("inf")
    return 1.0 / math.sqrt(s)


def _d_orthorhombic(h: int, k: int, l: int, a: float, b: float, c: float) -> float:
    """d-spacing for orthorhombic (α=β=γ=90°, a≠b≠c)."""
    s = (h * h) / (a * a) + (k * k) / (b * b) + (l * l) / (c * c)
    if s == 0:
        return float("inf")
    return 1.0 / math.sqrt(s)


def _d_monoclinic(h: int, k: int, l: int,
                   a: float, b: float, c: float, beta_deg: float) -> float:
    """d-spacing for monoclinic (α=γ=90°, β arbitrary, unique-axis b)."""
    beta = math.radians(beta_deg)
    sin_b = math.sin(beta)
    cos_b = math.cos(beta)
    if abs(sin_b) < 1e-9:
        return float("inf")
    s = (1.0 / (sin_b * sin_b)) * (
        (h * h) / (a * a)
        + (k * k * sin_b * sin_b) / (b * b)
        + (l * l) / (c * c)
        - (2 * h * l * cos_b) / (a * c)
    )
    if s <= 0:
        return float("inf")
    return 1.0 / math.sqrt(s)


def _d_triclinic(h: int, k: int, l: int,
                  a: float, b: float, c: float,
                  alpha_deg: float, beta_deg: float, gamma_deg: float) -> float:
    """d-spacing for triclinic (general case).

    Computed via the metric tensor G and d = 1/√(h·G*·h), where G* is the
    reciprocal metric tensor.
    """
    al = math.radians(alpha_deg)
    be = math.radians(beta_deg)
    ga = math.radians(gamma_deg)
    ca, cb, cg = math.cos(al), math.cos(be), math.cos(ga)
    sa, sb, sg = math.sin(al), math.sin(be), math.sin(ga)

    # Unit cell volume
    vol = a * b * c * math.sqrt(
        max(0.0, 1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    )
    if vol <= 0:
        return float("inf")

    # Reciprocal lattice parameters
    a_s = b * c * sa / vol
    b_s = a * c * sb / vol
    c_s = a * b * sg / vol
    cos_al_s = (cb * cg - ca) / (sb * sg)
    cos_be_s = (ca * cg - cb) / (sa * sg)
    cos_ga_s = (ca * cb - cg) / (sa * sb)

    s = (
        (h * h) * a_s * a_s
        + (k * k) * b_s * b_s
        + (l * l) * c_s * c_s
        + 2 * h * k * a_s * b_s * cos_ga_s
        + 2 * h * l * a_s * c_s * cos_be_s
        + 2 * k * l * b_s * c_s * cos_al_s
    )
    if s <= 0:
        return float("inf")
    return 1.0 / math.sqrt(s)


def _is_cubic(lattice: list[float]) -> bool:
    a, b, c, al, be, ga = lattice
    return (abs(a - b) < 1e-6 and abs(b - c) < 1e-6
            and abs(al - 90) < 1e-3 and abs(be - 90) < 1e-3 and abs(ga - 90) < 1e-3)


def _is_hexagonal(lattice: list[float]) -> bool:
    a, b, c, al, be, ga = lattice
    return (abs(a - b) < 1e-6
            and abs(al - 90) < 1e-3 and abs(be - 90) < 1e-3
            and abs(ga - 120) < 1e-3)


def _is_tetragonal(lattice: list[float]) -> bool:
    a, b, c, al, be, ga = lattice
    return (abs(a - b) < 1e-6 and abs(a - c) > 1e-6
            and abs(al - 90) < 1e-3 and abs(be - 90) < 1e-3 and abs(ga - 90) < 1e-3)


def _is_orthorhombic(lattice: list[float]) -> bool:
    _, _, _, al, be, ga = lattice
    return (abs(al - 90) < 1e-3 and abs(be - 90) < 1e-3 and abs(ga - 90) < 1e-3)


def _is_monoclinic(lattice: list[float]) -> bool:
    _, _, _, al, be, ga = lattice
    return (abs(al - 90) < 1e-3 and abs(ga - 90) < 1e-3 and abs(be - 90) > 1e-3)


def _extinct_fcc(h: int, k: int, l: int) -> bool:
    """FCC extinction: h, k, l all same parity (all even or all odd) = allowed.
    Otherwise systematically absent."""
    return not ((h % 2 == k % 2 == l % 2))


def _extinct_bcc(h: int, k: int, l: int) -> bool:
    """BCC extinction: h + k + l must be even."""
    return ((h + k + l) % 2) != 0


def _extinction_rule(space_group: int):
    """Return a function hkl -> bool (True = forbidden) for common cases.

    Comprehensive space-group extinction is NOT implemented here — MIDAS's
    GetHKLList is authoritative. We handle the common centerings (P/F/I)
    as a coarse filter so the wizard suggests plausible rings.
    """
    # Very rough bucketing by common space-group ranges
    # F-centered cubic: 225 (Fm-3m), 227 (Fd-3m), etc.
    fcc_sgs = {196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228}
    # I-centered cubic: 197, 199, 204, 211, 217, 220, 229, 230
    bcc_sgs = {197, 199, 204, 211, 217, 220, 229, 230}
    if space_group in fcc_sgs:
        return _extinct_fcc
    if space_group in bcc_sgs:
        return _extinct_bcc
    # Default: no extinction filter (P-primitive behavior)
    return lambda h, k, l: False


def enumerate_rings(
    wavelength: float,
    lsd_um: float,
    lattice: list[float],
    space_group: int,
    rho_d_um: float,
    nr_pixels_y: int | None = None,
    px_um: float | None = None,
    max_index: int = 8,
    max_rings: int = 30,
) -> list[RingInfo]:
    """Enumerate visible Bragg rings for a crystal on the given detector.

    Args:
        wavelength: X-ray wavelength in Å.
        lsd_um: Sample-detector distance in µm.
        lattice: [a, b, c, α, β, γ] in Å and degrees.
        space_group: Space group number (1–230).
        rho_d_um: Maximum ring radius (RhoD) in µm. Rings with radius > RhoD
            are flagged on_detector=False.
        nr_pixels_y, px_um: Optional — used to compute `fits_ypx`.
        max_index: Maximum |h|, |k|, |l| to enumerate (default 8).
        max_rings: Cap on returned rings (default 30).

    Returns rings sorted by increasing 2θ, numbered from 1.
    """
    if _is_cubic(lattice):
        a = lattice[0]
        d_fn = lambda h, k, l: _d_cubic(h, k, l, a)
    elif _is_hexagonal(lattice):
        a, _, c, *_ = lattice
        d_fn = lambda h, k, l: _d_hexagonal(h, k, l, a, c)
    elif _is_tetragonal(lattice):
        a, _, c, *_ = lattice
        d_fn = lambda h, k, l: _d_tetragonal(h, k, l, a, c)
    elif _is_orthorhombic(lattice):
        a, b, c, *_ = lattice
        d_fn = lambda h, k, l: _d_orthorhombic(h, k, l, a, b, c)
    elif _is_monoclinic(lattice):
        a, b, c, _, beta, _ = lattice
        d_fn = lambda h, k, l: _d_monoclinic(h, k, l, a, b, c, beta)
    else:
        # Triclinic / unknown — general formula works for everything
        a, b, c, al, be, ga = lattice
        d_fn = lambda h, k, l: _d_triclinic(h, k, l, a, b, c, al, be, ga)

    extinct = _extinction_rule(space_group)

    # For cubic symmetry, (h,k,l) permutations give identical d-spacing, so
    # we pick the sorted-descending tuple as canonical representative. For
    # lower symmetry, different permutations correspond to distinct rings,
    # so preserve the actual (|h|, |k|, |l|).
    if _is_cubic(lattice):
        rep_fn = lambda h, k, l: tuple(sorted((abs(h), abs(k), abs(l)), reverse=True))
    else:
        rep_fn = lambda h, k, l: (abs(h), abs(k), abs(l))

    # Group by unique d-spacing. Iterate non-negative octant; Friedel pair
    # (h,k,l)↔(-h,-k,-l) gives identical d so we can skip negatives.
    buckets: dict[float, tuple[int, int, int]] = {}
    for h in range(max_index + 1):
        for k in range(max_index + 1):
            for l in range(max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if extinct(h, k, l):
                    continue
                d = d_fn(h, k, l)
                if d <= 0 or not math.isfinite(d):
                    continue
                sin_theta = wavelength / (2.0 * d)
                if sin_theta > 0.99:
                    continue
                key = round(d, 6)
                candidate = rep_fn(h, k, l)
                # Prefer the lex-smallest representative within a bucket
                # (so cubic (1,1,1) wins over (0,0,2)-iteration-order).
                existing = buckets.get(key)
                if existing is None or candidate < existing:
                    buckets[key] = candidate

    entries = sorted(buckets.items(), key=lambda kv: -kv[0])  # descending d = ascending 2θ

    rings: list[RingInfo] = []
    for i, (d, hkl) in enumerate(entries, start=1):
        sin_theta = wavelength / (2.0 * d)
        theta = math.asin(sin_theta)
        two_theta = math.degrees(2.0 * theta)
        radius_um = lsd_um * math.tan(2.0 * theta)
        on_det = radius_um <= rho_d_um
        fits_y: bool | None = None
        if nr_pixels_y is not None and px_um is not None:
            detector_half_um = nr_pixels_y * px_um / 2.0
            fits_y = radius_um <= detector_half_um
        rings.append(RingInfo(
            ring_nr=i, hkl=hkl, d_spacing=d, two_theta=two_theta,
            radius_um=radius_um, on_detector=on_det, fits_ypx=fits_y,
        ))
        if len(rings) >= max_rings:
            break
    return rings


def format_ring_table(rings: list[RingInfo], use_color: bool = False) -> str:
    """Format a ring list for CLI display."""
    if use_color:
        green, yellow, red, reset, dim = "\033[32m", "\033[33m", "\033[31m", "\033[0m", "\033[2m"
    else:
        green = yellow = red = reset = dim = ""
    lines = [
        f"  Ring  (h k l)         d (Å)    2θ (°)    radius (mm)",
        f"  ----  ---------       ------   -------   -----------",
    ]
    for r in rings:
        hkl_s = f"({r.hkl[0]:>2} {r.hkl[1]:>2} {r.hkl[2]:>2})"
        if r.fits_ypx is False:
            marker, color = "× past detector half-Y", red
        elif not r.on_detector:
            marker, color = "× past RhoD", yellow
        else:
            marker, color = "✓", green
        lines.append(
            f"  {r.ring_nr:>3}   {hkl_s:<15} "
            f"{r.d_spacing:>6.3f}   {r.two_theta:>7.3f}   "
            f"{r.radius_um / 1000:>7.1f}    {color}{marker}{reset}"
        )
    return "\n".join(lines)


def recommend_rings(rings: list[RingInfo], max_recommend: int = 6) -> list[int]:
    """Pick up to N rings as a sensible starting RingThresh/Rings list.

    Policy: take the lowest-2θ rings that are on-detector. These have the
    strongest intensity and cleanest geometry.
    """
    return [r.ring_nr for r in rings if r.on_detector][:max_recommend]
