"""Absolute sample-to-detector distance from spots seen at several distances.

Implements the handbook's §6i / §6i-bis triangulation.  A ray leaving the
sample reaches the detector at ``p(L) = BC(L) + L * d``, so between two
detector positions the pattern is a pure *scaling about the respective beam
centres*::

    (p2 - BC2) = k * (p1 - BC1)          k = L2 / L1
    L1 = dD / (k - 1)                    delta = L1 - DetZ1

Only the *difference* ``dD`` between detector positions is trustworthy from
the motor; the absolute distance must be measured (handbook hard rule 10).

Why the nulls and gates live in here
------------------------------------
Both of the mistakes this routine exists to prevent were made by hand, in a
one-off script, by someone who had just read the warning against them:

* the position-scrambled null was written as a whole-row permutation, which
  is a **no-op** when the matcher searches all (i, j) pairs -- the "null"
  silently re-ran the real analysis and returned contrast 1.00x;
* a pair that failed the y-vs-z gate was nearly discarded when it was in
  fact merely *starved* -- at 6 matched pairs it failed at 1534 um, at 18 it
  passed at 57 um.

So :func:`triangulate` runs its own controls and **refuses to return a
distance when they fail**.  A caller cannot forget them, and cannot
misimplement them.  ``TriangulationResult.ok`` is the only thing to branch on.

Handbook: §6i, §6i-bis.  Lab notebook: §7f (F1, F2, F4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["PairSolve", "TriangulationResult", "triangulate"]

# Gate thresholds, from handbook §6i:1004-1007.
_YZ_GATE_UM = 200.0        # y-only vs z-only L1 must agree to better than this
_MIN_PAIRS_FOR_GATE = 12   # below this a gate failure means STARVED, not BROKEN
_NULL_CONTRAST_MIN = 3.0   # real peak must stand this far above both nulls


@dataclass
class PairSolve:
    """One distance-pair solve, with everything needed to audit it."""

    ia: int
    ib: int
    delta_detz_um: float
    k: float
    n_peak: int
    n_pairs: int
    L_a_um: float
    delta_um: float
    k_y: float
    k_z: float
    yz_split_um: float
    starved: bool
    gate_ok: bool
    note: str = ""


@dataclass
class TriangulationResult:
    """Outcome of a triangulation.  Branch on :attr:`ok`, nothing else."""

    ok: bool
    reason: str
    L_um: Optional[List[float]] = None          # per detector position
    delta_um: Optional[float] = None            # L[0] - detz_um[0]
    pairs: List[PairSolve] = field(default_factory=list)
    null_contrast: Dict[str, float] = field(default_factory=dict)
    leave_one_out_std_um: Optional[float] = None
    spread_um: Optional[float] = None

    def report(self) -> str:
        out = [f"triangulation: {'OK' if self.ok else 'REJECTED'} -- {self.reason}"]
        for p in self.pairs:
            flag = "PASS" if p.gate_ok else ("STARVED" if p.starved else "FAIL")
            out.append(
                f"  {p.ia}->{p.ib}  dD={p.delta_detz_um:8.1f}  k={p.k:.4f}  "
                f"n_peak={p.n_peak:4d}/{p.n_pairs:5d}  L={p.L_a_um:9.1f}  "
                f"y-z={p.yz_split_um:8.1f}  {flag} {p.note}"
            )
        for name, c in self.null_contrast.items():
            out.append(f"  null[{name}]: contrast {c:.2f}x "
                       f"(must exceed {_NULL_CONTRAST_MIN})")
        if self.leave_one_out_std_um is not None:
            out.append(f"  leave-one-out std {self.leave_one_out_std_um:.2f} um")
        if self.ok:
            out.append(f"  => L = {[round(v, 1) for v in self.L_um]} um, "
                       f"delta = {self.delta_um:.1f} um")
        return "\n".join(out)


def _polar(spots: np.ndarray, bc: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    d = np.column_stack([spots[:, 0] - bc[0], spots[:, 1] - bc[1]])
    r = np.hypot(d[:, 0], d[:, 1])
    return d, r


def _ratios(
    spots_a: Dict[int, np.ndarray],
    spots_b: Dict[int, np.ndarray],
    bc_a: Sequence[float],
    bc_b: Sequence[float],
    *,
    r_min_px: float,
    ang_tol_deg: float,
    mode: str = "real",
    rng: Optional[np.random.Generator] = None,
    frame_shift: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair ``k``, ``k_y``, ``k_z`` for two spot dictionaries keyed by frame."""
    frames = sorted(set(spots_a) & set(spots_b))
    ks: List[float] = []
    kys: List[float] = []
    kzs: List[float] = []
    cos_tol = np.cos(np.radians(ang_tol_deg))
    for n, f in enumerate(frames):
        A = spots_a[f]
        # omega-shuffled null: pair with a physically unrelated frame
        B = spots_b[frames[(n + frame_shift) % len(frames)]] if frame_shift \
            else spots_b[f]
        if len(A) == 0 or len(B) == 0:
            continue
        dA, rA = _polar(A, bc_a)
        dB, rB = _polar(B, bc_b)
        if mode == "scramble":
            # Permute the two POSITION COMPONENTS INDEPENDENTLY.  Permuting
            # whole ROWS is a NO-OP: the matcher searches all (i, j) pairs, so
            # reordering B leaves the SET of B vectors unchanged and the null
            # reproduces the real answer exactly.  Handbook §6i:994-997.
            assert rng is not None
            dB = np.column_stack([dB[rng.permutation(len(dB)), 0],
                                  dB[rng.permutation(len(dB)), 1]])
            rB = np.hypot(dB[:, 0], dB[:, 1])
        ma, mb = rA > r_min_px, rB > r_min_px
        dA, rA, dB, rB = dA[ma], rA[ma], dB[mb], rB[mb]
        if len(rA) == 0 or len(rB) == 0:
            continue
        # match on RAY DIRECTION, not radius alone
        cos = (dA / rA[:, None]) @ (dB / rB[:, None]).T
        ii, jj = np.where(cos > cos_tol)
        if len(ii) == 0:
            continue
        ks.extend((rB[jj] / rA[ii]).tolist())
        with np.errstate(divide="ignore", invalid="ignore"):
            kys.extend((dB[jj, 1] / dA[ii, 1]).tolist())
            kzs.extend((dB[jj, 0] / dA[ii, 0]).tolist())
    return np.array(ks), np.array(kys), np.array(kzs)


def _peak(v: np.ndarray, lo: float = 1.0, hi: float = 3.0,
          w: float = 0.002) -> Tuple[float, int, int]:
    v = v[np.isfinite(v) & (v > lo) & (v < hi)]
    if v.size < 10:
        return float("nan"), 0, int(v.size)
    h, e = np.histogram(v, bins=np.arange(lo, hi, w))
    c = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    sel = v[np.abs(v - c) < 2 * w]
    return (float(np.median(sel)) if sel.size else float(c)), int(h.max()), int(v.size)


def accidental_match_rate(n_spots: float, ang_tol_deg: float) -> float:
    """Expected number of chance direction-matches per frame.

    Two directions agree to within ``t`` degrees by chance with probability
    ``~2t/360``, so ``N_a * N_b`` candidate pairs give ``N^2 * t / 180``.
    A peak that is not far above this is not a measurement (lab notebook
    §7f, F2 -- an entire matching statistic once turned out to BE its own
    chance expectation).
    """
    return n_spots * n_spots * ang_tol_deg / 180.0


def triangulate(
    spots_by_distance: Sequence[Dict[int, np.ndarray]],
    bc_by_distance: Sequence[Sequence[float]],
    detz_um: Sequence[float],
    *,
    r_min_px: float = 300.0,
    ang_tol_deg: float = 0.30,
    seed: int = 0,
) -> TriangulationResult:
    """Measure the absolute distance to each detector position.

    Parameters
    ----------
    spots_by_distance
        One dict per detector position, mapping frame index -> ``(N, 2)``
        array of spot centroids as ``(row, col)`` in RAW detector pixels.
        The same frame index must mean the same omega at every distance.
    bc_by_distance
        One ``(row, col)`` beam centre per detector position, RAW pixels.
        **Per distance** -- BC moves with distance by ``beta * L``, and using
        one shared centre biases ``k`` enough to shift ``L1`` by hundreds of
        microns (handbook §6i-bis).
    detz_um
        Motor readback per position.  Only DIFFERENCES are used.

    Returns
    -------
    TriangulationResult
        ``ok`` is False, with ``reason`` set, whenever a control fails.
        No distance is returned in that case -- by design.
    """
    n = len(spots_by_distance)
    if n < 2:
        return TriangulationResult(False, "need at least two detector distances")
    if not (len(bc_by_distance) == n == len(detz_um)):
        return TriangulationResult(False, "spots/BC/detz lengths disagree")

    rng = np.random.default_rng(seed)
    pairs: List[PairSolve] = []

    for ia in range(n):
        for ib in range(ia + 1, n):
            dD = float(detz_um[ib] - detz_um[ia])
            if dD == 0:
                continue
            k, ky, kz = _ratios(spots_by_distance[ia], spots_by_distance[ib],
                                bc_by_distance[ia], bc_by_distance[ib],
                                r_min_px=r_min_px, ang_tol_deg=ang_tol_deg)
            kp, npk, ntot = _peak(k)
            if not np.isfinite(kp) or kp <= 1.0:
                pairs.append(PairSolve(ia, ib, dD, float("nan"), npk, ntot,
                                       float("nan"), float("nan"), float("nan"),
                                       float("nan"), float("nan"),
                                       starved=True, gate_ok=False,
                                       note="no peak"))
                continue
            kyp, _, _ = _peak(ky)
            kzp, _, _ = _peak(kz)
            L = dD / (kp - 1.0)
            Ly = dD / (kyp - 1.0) if np.isfinite(kyp) and kyp > 1 else np.nan
            Lz = dD / (kzp - 1.0) if np.isfinite(kzp) and kzp > 1 else np.nan
            split = float(abs(Ly - Lz)) if np.isfinite(Ly) and np.isfinite(Lz) \
                else float("inf")
            starved = npk < _MIN_PAIRS_FOR_GATE
            gate_ok = split < _YZ_GATE_UM
            note = ""
            if not gate_ok and starved:
                note = (f"gate failed on only {npk} matches -- STARVED, not "
                        f"broken; sample more omega before discarding")
            pairs.append(PairSolve(
                ia, ib, dD, float(kp), npk, ntot,
                float(L - (detz_um[ia] - detz_um[0])),   # normalise to position 0
                float(L - detz_um[ia]), float(kyp), float(kzp), split,
                starved, gate_ok, note))

    usable = [p for p in pairs if p.gate_ok and not p.starved]
    if not usable:
        starved_any = any(p.starved for p in pairs)
        return TriangulationResult(
            False,
            "no pair passed the y-vs-z gate with enough matches"
            + (" (all starved -- sample more omega)" if starved_any else ""),
            pairs=pairs,
        )

    # --- controls, on the best-populated usable pair -----------------------
    best = max(usable, key=lambda p: p.n_peak)
    _, npk_real, _ = _peak(_ratios(
        spots_by_distance[best.ia], spots_by_distance[best.ib],
        bc_by_distance[best.ia], bc_by_distance[best.ib],
        r_min_px=r_min_px, ang_tol_deg=ang_tol_deg)[0])
    contrasts: Dict[str, float] = {}
    frames = sorted(set(spots_by_distance[best.ia]) & set(spots_by_distance[best.ib]))
    for name, kw in (
        ("omega-shuffled", dict(frame_shift=max(len(frames) // 2, 1))),
        ("position-scrambled", dict(mode="scramble", rng=rng)),
    ):
        kk, _, _ = _ratios(spots_by_distance[best.ia], spots_by_distance[best.ib],
                           bc_by_distance[best.ia], bc_by_distance[best.ib],
                           r_min_px=r_min_px, ang_tol_deg=ang_tol_deg, **kw)
        _, npk_null, _ = _peak(kk)
        contrasts[name] = float(npk_real / max(npk_null, 1))

    failed = [nm for nm, c in contrasts.items() if c < _NULL_CONTRAST_MIN]
    if failed:
        return TriangulationResult(
            False,
            f"null(s) did not die: {', '.join(failed)} "
            f"(contrast below {_NULL_CONTRAST_MIN}x)",
            pairs=pairs, null_contrast=contrasts,
        )

    L0 = float(np.average([p.L_a_um for p in usable],
                          weights=[p.n_peak for p in usable]))
    spread = float(max(p.L_a_um for p in usable) - min(p.L_a_um for p in usable))
    L = [L0 + (float(detz_um[i]) - float(detz_um[0])) for i in range(n)]
    return TriangulationResult(
        True, f"{len(usable)} pair(s) passed all gates and both nulls",
        L_um=L, delta_um=L0 - float(detz_um[0]),
        pairs=pairs, null_contrast=contrasts, spread_um=spread,
    )
