"""NF-HEDM ``hkls.csv`` writer — functional Python port of
``NF_HEDM/src/GetHKLList.c``.

The C executable produces an 11-column CSV consumed by ``MakeDiffrSpots``,
``FitOrientationOMP`` and the rest of the NF-HEDM pipeline:

    h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius

This module emits **functionally identical** output: the same set of rows
(same (h, k, l) family expansion + Friedel pairs, same d_min cutoff
:math:`d_\\min = \\lambda / (2\\sin(\\arctan(R_{\\max}/L_{sd})/2))`, same
:math:`\\epsilon = 10^{-4}` Å ring grouping, same B-matrix convention) with
the same per-row numerical values.

The only difference from the C bytes is the **intra-ring row ordering**:
the C uses ``qsort`` (unstable) whereas this writer is deterministic —
within a ring we keep the discovery order from the iteration grid.
Downstream code reads the full file and does not rely on intra-ring
order, so this difference is invisible to the pipeline.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence, TextIO, Tuple

import numpy as np

from .lattice import Lattice
from .space_group import SpaceGroup


# ---------------------------------------------------------------------------
#  B-matrix (lab-frame Cartesian reciprocal basis)
# ---------------------------------------------------------------------------

def _b_matrix(lattice: Lattice) -> np.ndarray:
    """B-matrix matching ``CorrectHKLsLatC`` in the C source.

    Direct port of the formula in ``NF_HEDM/src/GetHKLList.c:64-95``::

        BetaPr  = arccos((cos(γ)·cos(α) - cos(β)) / (sin(γ)·sin(α)))
        GammaPr = arccos((cos(α)·cos(β) - cos(γ)) / (sin(α)·sin(β)))
        Vol  = a·b·c·sin(α)·sin(β')·sin(γ)
        APr  = b·c·sin(α) / Vol
        BPr  = c·a·sin(β) / Vol
        CPr  = a·b·sin(γ) / Vol
        B    = | APr            BPr·cos(γ')         CPr·cos(β')      |
               |  0             BPr·sin(γ')        -CPr·sin(β')·cos(α)|
               |  0              0                  CPr·sin(β')·sin(α)|

    G_cart = B · (h, k, l)^T then has units of Å⁻¹.
    """
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha = math.radians(lattice.alpha)
    beta = math.radians(lattice.beta)
    gamma = math.radians(lattice.gamma)

    sa, ca = math.sin(alpha), math.cos(alpha)
    sb, cb = math.sin(beta), math.cos(beta)
    sg, cg = math.sin(gamma), math.cos(gamma)

    cos_gamma_pr = (ca * cb - cg) / (sa * sb)
    cos_beta_pr = (cg * ca - cb) / (sg * sa)
    gamma_pr = math.acos(max(-1.0, min(1.0, cos_gamma_pr)))
    beta_pr = math.acos(max(-1.0, min(1.0, cos_beta_pr)))
    sin_beta_pr = math.sin(beta_pr)

    vol = a * b * c * sa * sin_beta_pr * sg
    a_pr = b * c * sa / vol
    b_pr = c * a * sb / vol
    c_pr = a * b * sg / vol

    B = np.array([
        [a_pr,                      b_pr * math.cos(gamma_pr),  c_pr * math.cos(beta_pr)],
        [0.0,                       b_pr * math.sin(gamma_pr), -c_pr * sin_beta_pr * ca],
        [0.0,                       0.0,                        c_pr * sin_beta_pr * sa],
    ], dtype=np.float64)
    return B


# ---------------------------------------------------------------------------
#  Equivalent HKLs + Friedel pair expansion
# ---------------------------------------------------------------------------

def _expanded_hkl_set(
    space_group: SpaceGroup,
    h_max: int = 10,
    k_max: int = 10,
    l_max: int = 10,
) -> List[Tuple[int, int, int]]:
    """Walk the integer (h, k, l) box, collect non-systematically-absent
    reflections plus their full symmetry orbit and Friedel pair, dedup.

    The C ``GetHKLList.c`` uses ``Maxh = Maxk = Maxl = 10`` and the
    sginfo helpers ``IsSysAbsent_hkl`` / ``BuildEq_hkl``. The Friedel
    inclusion is the explicit ``j ∈ {-1, +1}`` doubling at line 311.

    The orbit returned by :meth:`SpaceGroup.equivalent_hkls` already
    includes Friedel pairs (centric structure factor convention), so
    the final set matches the C output set.
    """
    seen: set[Tuple[int, int, int]] = set()
    out: List[Tuple[int, int, int]] = []
    for h in range(-h_max, h_max + 1):
        for k in range(-k_max, k_max + 1):
            for l in range(-l_max, l_max + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if space_group.is_systematically_absent(h, k, l):
                    continue
                for hkl_eq in space_group.equivalent_hkls(h, k, l):
                    if hkl_eq in seen:
                        continue
                    seen.add(hkl_eq)
                    out.append(hkl_eq)
    return out


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def emit_nf_hkls_csv(
    space_group: SpaceGroup,
    lattice: Lattice,
    *,
    wavelength_A: float,
    lsd_um: float,
    max_ring_rad_um: float,
    epsilon: float = 1e-4,
    h_max: int = 10,
    k_max: int = 10,
    l_max: int = 10,
    fp: TextIO | None = None,
    atoms: "Sequence | None" = None,
) -> List[Tuple[float, ...]]:
    """Write an NF-HEDM ``hkls.csv`` matching ``GetHKLListNF`` row-for-row.

    Parameters
    ----------
    space_group : SpaceGroup
    lattice : Lattice
        Lattice constants (Å for a/b/c, deg for α/β/γ).
    wavelength_A : float
        X-ray wavelength in Å.
    lsd_um, max_ring_rad_um : float
        Sample-to-detector distance and detector "max ring radius"
        (both in microns; ratios are unit-free so this matches the C).
    epsilon : float
        D-spacing tolerance (Å) for grouping reflections into rings
        (matches ``Epsilon = 0.0001`` in the C source).
    h_max, k_max, l_max : int
        Half-extent of the integer HKL search box (default 10, matching
        the C source's ``Maxh = Maxk = Maxl = 10``).
    fp : TextIO, optional
        Open file (or other text-writable). If ``None``, no file is
        written — only the in-memory rows are returned.

    Returns
    -------
    list of 11-tuples
        Per-row ``(h, k, l, D-spacing, RingNr, g1, g2, g3, Theta, 2Theta, Radius)``,
        sorted by descending d-spacing. Useful for tests.
    """
    B = _b_matrix(lattice)

    # 1. Walk box, expand orbit + Friedel pair, dedup.
    hkl_set = _expanded_hkl_set(space_group, h_max, k_max, l_max)

    # 2. Compute G_cart and d-spacing for each.
    hkl_arr = np.asarray(hkl_set, dtype=np.float64)
    g_cart = (B @ hkl_arr.T).T                         # (N, 3)
    g_norm = np.linalg.norm(g_cart, axis=1)
    # 1/|g| = d-spacing.  Guard against zero (shouldn't happen post-(0,0,0) filter).
    d_spacing = np.where(g_norm > 0, 1.0 / g_norm, np.inf)

    # 3. Sort by d-spacing descending.
    order = np.argsort(-d_spacing, kind="stable")
    hkl_arr = hkl_arr[order]
    g_cart = g_cart[order]
    d_spacing = d_spacing[order]

    # 4. Apply d_min cutoff:  d_min = wl / (2·sin(atan(R/L)/2)).
    if max_ring_rad_um > 0 and lsd_um > 0:
        two_theta_max = math.atan(max_ring_rad_um / lsd_um)
        d_min = wavelength_A / (2.0 * math.sin(two_theta_max / 2.0))
        keep = d_spacing >= d_min
        hkl_arr = hkl_arr[keep]
        g_cart = g_cart[keep]
        d_spacing = d_spacing[keep]

    # 5. Assign RingNr by walking sorted list with epsilon tolerance.
    ring_nrs = np.zeros(len(d_spacing), dtype=np.int64)
    if len(d_spacing) > 0:
        ring_nrs[0] = 1
        ds_temp = d_spacing[0]
        ring_nr = 1
        for i in range(1, len(d_spacing)):
            if abs(d_spacing[i] - ds_temp) < epsilon:
                ring_nrs[i] = ring_nr
            else:
                ds_temp = d_spacing[i]
                ring_nr += 1
                ring_nrs[i] = ring_nr

    # 6. Compute Theta, 2Theta (degrees), Radius (microns).
    sin_theta = wavelength_A / (2.0 * d_spacing)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    theta = np.degrees(np.arcsin(sin_theta))
    two_theta = 2.0 * theta
    radius = lsd_um * np.tan(np.radians(two_theta))

    # 6b. Optional |F|^2 per reflection, from the ATOM BASIS.
    #
    # Space-group extinction rules alone cannot see basis-dependent
    # extinctions. For dhcp beta-Ce (P6_3/mmc with Ce at 2a + 2c) 126 of 736
    # reflections have |F|^2 = 0 and can never produce a spot -- yet they sat in
    # the NF confidence denominator, capping FracOverlap at 0.829 while fcc
    # (one atom at the origin, nothing extra extinct) could reach 1.000.
    #
    # |F|^2, NOT intensity: Lorentz-polarisation diverges at low 2-theta and
    # would swamp the normalisation, and the rotation-method Lorentz factor
    # CANCELS for per-frame peak detection anyway (Domega = w_rlp * L, so
    # I_peak = I_int/Domega is L-free; measured on nf_Ce_ht525_s2 -- spot peak
    # is flat in eta while 1/|sin eta| predicts 4.3x, and spot density on a ring
    # rises as sin eta as the same model requires).
    f2 = None
    if atoms is not None:
        import torch
        from .crystal import Crystal
        from .structure_factor import structure_factors

        cry = Crystal(lattice, space_group, list(atoms))
        F = structure_factors(cry.to_torch(), torch.as_tensor(hkl_arr.astype(int)))
        f2_arr = np.abs(np.asarray(F.detach().cpu().numpy()).ravel()) ** 2
        mx = float(f2_arr.max()) if f2_arr.size else 0.0
        f2 = f2_arr / mx if mx > 0 else f2_arr

    # 7. Build rows + write.
    rows: List[Tuple[float, ...]] = []
    for i in range(len(d_spacing)):
        row = (
            float(hkl_arr[i, 0]), float(hkl_arr[i, 1]), float(hkl_arr[i, 2]),
            float(d_spacing[i]), float(ring_nrs[i]),
            float(g_cart[i, 0]), float(g_cart[i, 1]), float(g_cart[i, 2]),
            float(theta[i]), float(two_theta[i]), float(radius[i]),
        )
        if f2 is not None:
            row = row + (float(f2[i]),)
        rows.append(row)

    if fp is not None:
        # The F2 column is APPENDED, never inserted: every existing reader
        # indexes hkls.csv by position and tolerates extra trailing tokens
        # (midas_nf_fitorientation.io.read_hkls requires >= 9 and reads 0-8),
        # so old readers are unaffected by its presence.
        head = "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius"
        fp.write(head + (" F2\n" if f2 is not None else "\n"))
        for row in rows:
            fp.write(
                f"{row[0]:.0f} {row[1]:.0f} {row[2]:.0f} {row[3]:.17g} {row[4]:.0f} "
                f"{row[5]:.17g} {row[6]:.17g} {row[7]:.17g} {row[8]:.17g} "
                f"{row[9]:.17g} {row[10]:.17g}"
                + (f" {row[11]:.17g}\n" if f2 is not None else "\n")
            )

    return rows


def write_nf_hkls_csv(
    path: str | Path,
    space_group: SpaceGroup,
    lattice: Lattice,
    *,
    wavelength_A: float,
    lsd_um: float,
    max_ring_rad_um: float,
    epsilon: float = 1e-4,
    h_max: int = 10,
    k_max: int = 10,
    l_max: int = 10,
    atoms: "Sequence | None" = None,
) -> int:
    """Convenience wrapper: write ``hkls.csv`` to ``path`` and return row count.

    ``atoms`` (optional): the unit-cell basis, e.g.
    ``[Atom("Ce", (0,0,0)), Atom("Ce", (1/3, 2/3, 0.25))]``. When supplied, an
    ``F2`` column (|F|^2 normalised to its maximum) is APPENDED so downstream
    code can drop or down-weight reflections the basis forbids. Omit it and the
    output is byte-identical to before.
    """
    path = Path(path)
    with open(path, "w") as fp:
        rows = emit_nf_hkls_csv(
            space_group, lattice,
            wavelength_A=wavelength_A,
            lsd_um=lsd_um,
            max_ring_rad_um=max_ring_rad_um,
            epsilon=epsilon,
            h_max=h_max, k_max=k_max, l_max=l_max,
            fp=fp, atoms=atoms,
        )
    return len(rows)
