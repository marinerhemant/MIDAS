"""Drop-in replacement for ``GetHKLListZarr``.

Reads lattice constant, wavelength, Lsd, RhoD/MaxRingRad, and SpaceGroup from
a MIDAS Zarr archive and writes ``hkls.csv`` with the legacy 11-column format::

    h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius

Order: one row per Friedel/Laue-equivalent reflection, sorted by descending
d-spacing. The B-matrix convention matches ``CorrectHKLsLatC`` in
``FF_HEDM/src/GetHKLListZarr.c``.
"""
from __future__ import annotations

from math import asin, atan, cos, degrees, pi, radians, sin, sqrt, tan
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .lattice import Lattice
from .space_group import SpaceGroup
from .hkl_gen import generate_hkls


def _read_zarr_minimal(zarr_path: str | Path) -> dict:
    """Read just the keys we need: LatticeParameter, Wavelength, Lsd, RhoD or
    MaxRingRad, SpaceGroup, ResultFolder. ``RhoD`` and ``MaxRingRad`` are
    treated as aliases (matches GetHKLListZarr.c lines 203–212)."""
    import zarr

    store = zarr.ZipStore(str(zarr_path), mode="r")
    try:
        root = zarr.group(store=store)
        ap = "analysis/process/analysis_parameters"

        def _flat(key):
            try:
                return np.asarray(root[f"{ap}/{key}"][...]).flatten()
            except KeyError:
                return None

        out: dict = {}
        latv = _flat("LatticeParameter")
        if latv is None or len(latv) < 6:
            raise KeyError("LatticeParameter not found in Zarr archive")
        out["LatC"] = tuple(float(x) for x in latv[:6])

        for key in ("Wavelength", "Lsd"):
            v = _flat(key)
            if v is None or len(v) < 1:
                raise KeyError(f"{key} not found in Zarr archive")
            out[key] = float(v[0])

        rho = _flat("RhoD")
        max_rr = _flat("MaxRingRad")
        if rho is not None and len(rho) >= 1 and float(rho[0]) > 0:
            out["MaxRingRad"] = float(rho[0])
        elif max_rr is not None and len(max_rr) >= 1 and float(max_rr[0]) > 0:
            out["MaxRingRad"] = float(max_rr[0])
        else:
            raise KeyError("Neither RhoD nor MaxRingRad found in Zarr archive")

        sg = _flat("SpaceGroup")
        if sg is None or len(sg) < 1:
            raise KeyError("SpaceGroup not found in Zarr archive")
        out["SpaceGroup"] = int(sg[0])

        # ResultFolder is a string scalar.
        try:
            rf_arr = root[f"{ap}/ResultFolder"][...]
            v = rf_arr.flat[0]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            out["ResultFolder"] = str(v)
        except KeyError:
            out["ResultFolder"] = ""
        return out
    finally:
        store.close()


def _b_matrix(LatC: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """Reciprocal basis in cartesian coordinates, matching CorrectHKLsLatC."""
    a, b, c, alpha, beta, gamma = LatC
    SinA, SinB, SinG = sin(radians(alpha)), sin(radians(beta)), sin(radians(gamma))
    CosA, CosB, CosG = cos(radians(alpha)), cos(radians(beta)), cos(radians(gamma))
    GammaPr = degrees(np.arccos((CosA * CosB - CosG) / (SinA * SinB)))
    BetaPr = degrees(np.arccos((CosG * CosA - CosB) / (SinG * SinA)))
    SinBetaPr = sin(radians(BetaPr))
    Vol = a * b * c * SinA * SinBetaPr * SinG
    APr = b * c * SinA / Vol
    BPr = c * a * SinB / Vol
    CPr = a * b * SinG / Vol
    B = np.zeros((3, 3), dtype=np.float64)
    B[0, 0] = APr
    B[0, 1] = BPr * cos(radians(GammaPr))
    B[0, 2] = CPr * cos(radians(BetaPr))
    B[1, 0] = 0.0
    B[1, 1] = BPr * sin(radians(GammaPr))
    B[1, 2] = -CPr * SinBetaPr * CosA
    B[2, 0] = 0.0
    B[2, 1] = 0.0
    B[2, 2] = CPr * SinBetaPr * SinA
    return B


def generate_hkls_from_zarr(
    zarr_path: str | Path,
    result_folder: Optional[str | Path] = None,
    atoms=None,
    cif_path: Optional[str | Path] = None,
    drop_forbidden: bool = False,
    forbidden_f2_threshold: float = 1e-6,
) -> Path:
    """Mimic ``GetHKLListZarr <zarr> [result_folder]``.

    Returns the path to the written ``hkls.csv``. If ``result_folder`` is None,
    falls back to ``ResultFolder`` from the Zarr archive, then the directory
    containing ``zarr_path``.

    Optional atom basis
    -------------------
    Give ``atoms`` (a list of :class:`midas_hkls.Atom`, e.g. from
    :func:`midas_hkls.parse_phase_atoms`) **or** ``cif_path``, and a 12th
    ``F2`` column is appended — |F|² per reflection, normalised to its
    maximum. With neither, the output is byte-identical to the historical
    11-column file.

    The two are mutually exclusive and passing both raises. Silently
    preferring one would let a stale ``PhaseAtom`` block override a CIF, or the
    reverse, with no way to tell from the output which structure produced the
    numbers.

    **The CIF supplies the basis only.** Lattice and space group still come
    from the Zarr — those are the *calibrated* values the rest of the
    reconstruction is built on, and quietly substituting a CIF's nominal cell
    would move every ring radius. A disagreement is logged, not applied.

    Appending is safe for every existing reader: the C indexer ``sscanf``s
    exactly 11 conversions (``IndexerUnified.c:2665``) and ignores trailing
    tokens, the C refiner likewise (``FitUnified.c:1562``), and the Python
    readers are positional with ``len(...) >= N`` guards. The one constraint is
    that ``numpy.loadtxt`` (``midas_transforms/radius/theoretical.py:82``)
    rejects a ragged file — so the column is written on every row or on none.
    """
    p = _read_zarr_minimal(zarr_path)
    if result_folder is not None:
        rf = Path(result_folder)
    elif p["ResultFolder"]:
        rf = Path(p["ResultFolder"])
    else:
        rf = Path(zarr_path).parent
    rf.mkdir(parents=True, exist_ok=True)

    LatC = p["LatC"]
    wl = p["Wavelength"]
    Lsd = p["Lsd"]
    MaxRingRad = p["MaxRingRad"]
    sg_num = int(p["SpaceGroup"])

    # Bragg cutoff: DsMin = wl / (2 sin(theta_max)), theta_max = atan(MaxRingRad/Lsd)/2
    theta_max_deg = degrees(atan(MaxRingRad / Lsd)) / 2.0
    s = sin(radians(theta_max_deg))
    if s <= 0:
        raise ValueError("Invalid MaxRingRad/Lsd: theta_max must be positive")
    d_min = wl / (2.0 * s)

    sg = SpaceGroup.from_number(sg_num)
    lat = Lattice(*LatC)

    if atoms is not None and cif_path is not None:
        raise ValueError(
            "Give atoms OR cif_path, not both. Preferring one silently would "
            "let a stale PhaseAtom block shadow a CIF (or the reverse) with "
            "nothing in the output to say which structure was used."
        )
    if cif_path is not None:
        from .io.cif import read_cif

        cry = read_cif(cif_path)
        atoms = list(cry.atoms)
        cif_lat = tuple(float(x) for x in (
            cry.lattice.a, cry.lattice.b, cry.lattice.c,
            cry.lattice.alpha, cry.lattice.beta, cry.lattice.gamma))
        if not np.allclose(cif_lat, LatC, rtol=1e-4, atol=1e-4):
            print(
                f"Note: {Path(cif_path).name} declares lattice {cif_lat} but "
                f"the Zarr's calibrated LatticeParameter is {LatC}. Using the "
                f"ZARR lattice for geometry (ring radii depend on it); the CIF "
                f"contributes only the atom basis for |F|^2."
            )
        if int(getattr(cry.space_group, "number", sg_num)) != sg_num:
            print(
                f"Note: {Path(cif_path).name} declares space group "
                f"{getattr(cry.space_group, 'number', '?')} but the Zarr says "
                f"{sg_num}. Using the ZARR space group."
            )

    # ASU representatives, sorted by d_spacing desc, with multiplicity expansion via equivalent_hkls.
    refs = generate_hkls(sg, lat, wavelength_A=wl, d_min=d_min)

    B = _b_matrix(LatC)

    rows: List[Tuple[int, int, int, float, float, float, float]] = []
    seen: set = set()
    for r in refs:
        for hkl in sg.equivalent_hkls(r.h, r.k, r.l):
            if hkl in seen:
                continue
            seen.add(hkl)
            g_init = np.array(hkl, dtype=np.float64)
            g_cart = B @ g_init
            mag = float(np.linalg.norm(g_cart))
            if mag <= 0:
                continue
            d = 1.0 / mag
            rows.append((hkl[0], hkl[1], hkl[2], d,
                         float(g_cart[0]), float(g_cart[1]), float(g_cart[2])))

    # Sort by d_spacing descending (stable, like qsort on doubles).
    rows.sort(key=lambda x: -x[3])
    # Drop anything below DsMin (matches the C truncation after sort).
    rows = [row for row in rows if row[3] >= d_min]

    # |F|^2 per reflection, in the SAME row order as `rows` — computed after
    # the sort and the d_min truncation so the join is positional-by-
    # construction rather than by a lookup that could mis-pair.
    f2 = None
    if atoms is not None and rows:
        from .structure_factor import f2_normalised

        f2 = f2_normalised(
            lat, sg, atoms,
            np.array([[r[0], r[1], r[2]] for r in rows], dtype=np.int64),
        )
        n_forbidden = int((f2 <= 1e-6).sum())
        print(
            f"hkls.csv: |F|^2 computed from {len(atoms)} atom(s); "
            f"{n_forbidden} of {len(rows)} reflections are basis-forbidden "
            f"(|F|^2 <= 1e-6). Max achievable completeness on this phase is "
            f"{1.0 - n_forbidden / len(rows):.4f} unless they are excluded."
        )

    # RingNr assignment via epsilon-grouping (matches C ``Epsilon = 0.0001``).
    epsilon = 1e-4
    head = "h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius"
    out_lines = [head + (" F2" if f2 is not None else "")]
    if rows:
        ring_nr = 1
        ds_anchor = rows[0][3]
        for i, (h, k, l, d, g1, g2, g3) in enumerate(rows):
            if i == 0:
                ring_nr = 1
                ds_anchor = d
            elif abs(d - ds_anchor) < epsilon:
                pass
            else:
                ds_anchor = d
                ring_nr += 1
            theta_deg = degrees(asin(wl / (2.0 * d)))
            two_theta_deg = 2.0 * theta_deg
            radius = Lsd * tan(radians(two_theta_deg))
            # %.17g-equivalent formatting via repr(float). Cast to float to
            # strip any numpy scalar wrapping.
            def f17(x):
                return repr(float(x))
            out_lines.append(
                f"{h:.0f} {k:.0f} {l:.0f} {f17(d)} {ring_nr:.0f} "
                f"{f17(g1)} {f17(g2)} {f17(g3)} "
                f"{f17(theta_deg)} {f17(two_theta_deg)} {f17(radius)}"
                + (f" {f17(f2[i])}" if f2 is not None else "")
            )

    # Optionally remove the basis-forbidden reflections outright.
    #
    # This happens AFTER RingNr assignment, never before. Ring numbers are
    # positional in the sorted-by-d list, so filtering first would renumber
    # every ring above the first gap — and RingThresh / RingNumbers /
    # OverAllRingToIndex in the parameter file all reference those numbers by
    # value. (Project rule: ring numbers come from the run's own hkls.csv and
    # are never regenerated.) Filtering afterwards leaves the surviving rows
    # carrying exactly the ring numbers they had.
    #
    # Removing rows rather than weighting them is what NF settled on
    # (midas_nf_pipeline/stages.py:162-203) and the reasoning carries over:
    # every downstream stage reads this one file, so dropping the rows fixes
    # the SEARCH as well as the reported completeness, with no change to any
    # of the four separate implementations of the matched-fraction ratio.
    if drop_forbidden:
        if f2 is None:
            raise ValueError(
                "drop_forbidden requires an atom basis (atoms= or cif_path=); "
                "without |F|^2 there is nothing to call forbidden."
            )
        body = out_lines[1:]
        kept = [ln for ln in body
                if ln.strip() and float(ln.split()[11]) > forbidden_f2_threshold]
        n_dropped = len(body) - len(kept)
        rings_before = {ln.split()[4] for ln in body if ln.strip()}
        rings_after = {ln.split()[4] for ln in kept}
        lost_rings = sorted(rings_before - rings_after, key=lambda x: int(float(x)))
        out_lines = [out_lines[0]] + kept
        print(
            f"DropForbiddenReflections: removed {n_dropped} of {len(body)} "
            f"reflections with |F|^2 <= {forbidden_f2_threshold:g} "
            f"(kept {len(kept)})."
        )
        if lost_rings:
            print(
                f"  NOTE: rings {', '.join(lost_rings)} are now absent from "
                f"hkls.csv entirely — every reflection on them is forbidden. "
                f"Remove them from RingThresh/RingNumbers, and check "
                f"OverAllRingToIndex is not one of them (the indexer finds "
                f"zero seed spots on a ring with no reflections)."
            )

    # The C printf uses %.17g (round-trip float). Python repr() of float64 is
    # the same shortest round-trip representation.
    out = rf / "hkls.csv"
    out.write_text("\n".join(out_lines) + "\n")
    return out
