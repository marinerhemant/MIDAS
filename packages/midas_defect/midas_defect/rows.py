"""Reciprocal-lattice ROWS, and indexing a domain from one.

A row is a set of collinear spots through the origin at regular spacing. The
(00L) ladder of a layered material is one instance; this module handles rows
along any direction.

**Why rows matter for weak domains.** A row identifies a lattice direction
WITHOUT seeding from the brightest spots, so it reaches a minority domain that a
bright-core seeder cannot. On La3Ni2O7 2604 the second domain is 64-88x fainter
than the first: `find_seed_orientation` needed n_bright=150 to see it at all
(20 finds nothing), while a row finds it directly. A row also measures that
domain's c* directly, fixing two of three orientation degrees of freedom and
leaving a 1-D scan.

**Three things here were wrong until verification caught them, and the reasons
are worth keeping:**

* Parity is PER DIRECTION, from the space group -- not a blanket "even
  multiples only". That rule is right for an odd-sum step like (0,0,1) and wrong
  by a factor of 2 for every even-sum step, and different again for F-centring.
  With the blanket rule, synthetic rows along (1,0,0), (1,1,1), (0,1,2), (1,1,2)
  and (1,2,1) returned NOTHING; 1 of 8 directions worked.
* The spacing window must cover the OBSERVED ``n * |G|``, not ``|G|``.
* Identification must be able to REFUSE. Without a threshold, (0,0,1) absorbed
  80 % of all producible spacings and q-vectors scaled by 1.90 were still
  confidently labelled (0,0,1).

**And one guard that is not optional.** `omega_smear_duplicates` rejects the
main false-domain mechanism: a 3-D blob finder splitting one omega-smeared
reflection into pieces that then index as a separate grain. Verification found
3 of 4 "second domains" in a real raster were exactly this.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import numpy as np

from midas_hkls import Lattice

__all__ = ["Row", "candidate_row_spacings", "allowed_multiple_parity",
           "find_lattice_rows", "refine_row_spacing", "hkl_box_from_geometry",
           "index_from_ladder", "index_from_row", "index_from_pairs",
           "index_by_grid", "orientation_grid", "search_null",
           "cell_from_row",
           "row_scan_range", "match_mask",
    "unique_by_hkl", "refine_cell_orientation", "refine_lattice",
           "RefinedLattice",
           "omega_smear_duplicates"]


@dataclass
class Row:
    direction: np.ndarray      # unit vector
    spacing: float             # |q| step between adjacent rungs, 1/A
    steps: List[int]           # integer multiples observed
    n_rungs: int
    intensity: float
    rms_perp: float
    members: np.ndarray
    hkl_step: Optional[Tuple[int, int, int]] = None   # which lattice vector
    match_rel_2: float = float("inf")                 # runner-up |obs-pred|/pred
    match_rel: Optional[float] = None                 # |obs-pred|/pred
    spacing_resid: float = float("nan")               # RMS frac-index residual


def allowed_multiple_parity(hkl_step, space_group_number=139):
    """Smallest n such that n*(h,k,l) is an allowed reflection.

    NOT a blanket "even only". Verified from midas_hkls extinctions for SG 139
    (I4/mmm, h+k+l even): a step with ODD h+k+l shows only EVEN multiples, so
    the observed row spacing is 2|G_step|; a step with EVEN h+k+l shows EVERY
    multiple, so the spacing is |G_step| and halving it is a factor-2 error.
    (0,0,1) is odd-sum, which is why the blanket rule worked there and nowhere
    else -- and F-centred cells (Fmmm, S5) follow a different rule again.
    """
    h, k, l = hkl_step
    if space_group_number in (139,):                 # I-centred: h+k+l even
        return 2 if (h + k + l) % 2 else 1
    if space_group_number in (69,):                  # F-centred: h,k,l same parity
        return 1 if (h % 2 == k % 2 == l % 2) else 2
    return 1


def candidate_row_spacings(a, c, *, hmax=2, lmax=3, two_pi=True):
    """|G| of low-index reciprocal-lattice steps, ONE PER SYMMETRY ORBIT.

    A row's spacing is the |G| of the primitive step ALONG that row, so this is
    the lookup that turns an anonymous row into an identified lattice direction.

    The orbit reduction is not tidiness. In a tetragonal cell a = b, so (0,1,1)
    and (1,0,1) are the same reflection and have byte-identical |G|. Listing
    both makes every mixed-index step look like it has a rival that fits exactly
    as well, and a caller checking "is the runner-up clearly worse?" -- which it
    must, once the step fixes the cell and not merely a direction -- then
    rejects the identification as ambiguous. Measured at p=66: (0,1,1) came back
    rel = 0.0108 with runner-up 0.0108, and (0,1,3) 0.0118 with 0.0118, so both
    real rows were discarded for being indistinguishable from themselves.

    A row is an axis, so +/- of a step is the same row; the representative is
    ``(max(|h|,|k|), min(|h|,|k|), |l|)``. Two DIFFERENT orbits landing on the
    same spacing is a genuine ambiguity and both are kept.
    """
    k = 2*math.pi if two_pi else 1.0
    seen, out = set(), []
    for h in range(0, hmax+1):
        for kk in range(0, hmax+1):
            for l in range(0, lmax+1):
                if (h, kk, l) == (0, 0, 0):
                    continue
                if math.gcd(math.gcd(h, kk), l) != 1:    # primitive steps only
                    continue
                rep = (max(h, kk), min(h, kk), l)
                if rep in seen:
                    continue
                seen.add(rep)
                g = k*math.sqrt((rep[0]/a)**2 + (rep[1]/a)**2 + (rep[2]/c)**2)
                out.append((rep, g))
    return sorted(out, key=lambda z: z[1])


def _robust_common_factor(steps, *, min_frac=0.75, min_rungs=3, gmax=4):
    """Largest g > 1 whose multiples account for most of the observed steps.

    A sub-multiple spacing d/g describes exactly the same rungs with integers
    multiplied by g, so it never loses the "most distinct steps" contest -- and
    it WINS as soon as two or three interlopers happen to sit near the
    intermediate positions. Measured at La3Ni2O7 p=114: the (00L) ladder was
    returned as 16 rungs at d = 0.32810 with steps
    [-12,-10,-8,-6,-4,-3,-2,2,3,4,6,8,10,12,14,18] -- 14 of 16 even, the two
    odd ones (+/-3) the interlopers that bought it the win. Doubling gives
    0.6562, which is the spacing the ladder actually has. p=66 (2 odd of 9) and
    p=329 (3 odd of 15) show the same signature.

    A plain gcd cannot do this: gcd(..., 3, ...) = 1, so two outliers defeat it.
    Hence the robust version -- g must cover ``min_frac`` of the steps, not all
    of them, and must leave at least ``min_rungs`` distinct steps behind.

    Note this is about the PRIMITIVE spacing of the observed row. It is not the
    centring rule: the observed spacing of a (0,0,1) row in an I-centred cell is
    genuinely 2|G_001| because odd l is extinct, and that factor is applied
    separately by :func:`allowed_multiple_parity` at identification time.
    """
    st = [int(x) for x in steps if int(x) != 0]
    if len(st) < min_rungs:
        return 1
    for g in range(int(gmax), 1, -1):
        keep = [x for x in st if x % g == 0]
        if len(keep) >= min_frac*len(st) and len(set(keep)) >= min_rungs:
            return g
    return 1


def refine_row_spacing(proj, d0, *, tol=0.15, max_iter=6, max_drift=0.25,
                       robust=True, min_outlier_frac=0.25):
    """Least-squares row spacing from ALL rungs, refining a starting guess.

    A row's spacing was previously taken to BE one of the pairwise differences
    between rung projections -- a single noisy measurement -- and, because
    candidates were tried in ascending order and ties broken by the first, the
    winner among equally-good candidates was the SMALLEST. That is a minimum
    statistic: a 14-rung row ends up defined by its single worst pair. Measured
    on La3Ni2O7 at p=114, a 14-rung (00L) row was set by the one outlier pair
    (+6 -> +7) at d = 0.64643, giving c = 19.4397, while the least-squares slope
    of the same 14 rungs gives 0.65626 -> c = 19.1485. Every other pair lands in
    19.05-19.24. The 1.5 % error was enough to make the ladder-seeded search
    fail on a domain it otherwise indexes with 43 reflections.

    Rungs of a reciprocal-lattice row lie at integer multiples of the spacing
    through the origin, so with the integers assigned the estimator is the
    slope through zero, ``sum(n*proj) / sum(n*n)``, which weights the far rungs
    that carry the most information about the spacing. Integers are reassigned
    from the refined value and the fit repeated, so a starting guess that is
    slightly small (mis-assigning only the highest rungs, which then fail the
    tolerance and drop out) recovers rather than locking in its own error.

    ``max_drift`` refuses a refinement that moves more than that fraction from
    ``d0``: converging onto a sub- or super-multiple is a different row, not a
    better estimate of this one.

    Returns ``(spacing, ok_mask, rms_residual)``; ``spacing`` is ``d0``
    unchanged if the refinement could not be trusted.
    """
    proj = np.asarray(proj, float)
    d0 = float(d0)

    def _fit(subset):
        d, ok = d0, None
        for _ in range(max_iter):
            ratio = proj/d
            n = np.rint(ratio)
            sel = (np.abs(ratio - n) < tol) & (np.abs(n) > 0)
            if subset is not None:
                sel &= subset
            if int(sel.sum()) < 2:
                return None, None
            nn = n[sel]
            denom = float((nn*nn).sum())
            if denom <= 0:
                return None, None
            d_new = float((nn*proj[sel]).sum()/denom)
            if not np.isfinite(d_new) or d_new <= 0:
                return None, None
            if abs(d_new - d0) > max_drift*d0:
                return None, None
            converged = abs(d_new - d) <= 1e-12*max(d, 1.0)
            d, ok = d_new, sel
            if converged:
                break
        return d, ok

    d, ok = _fit(None)
    if ok is None:
        return d0, None, float("nan")

    if robust and int(ok.sum()) >= 5:
        # Outlier rejection runs on the CONVERGED fit, never during the
        # iteration. While d is still wrong the residuals are dominated by the
        # n*(D/d - 1) trend rather than by noise, so a robust filter applied
        # early throws away the highest-|n| rungs -- exactly the ones carrying
        # the most information about the spacing.
        ratio = proj/d
        rr = ratio - np.rint(ratio)
        vals = rr[ok]
        med = float(np.median(vals))
        sig = 1.4826*float(np.median(np.abs(vals - med)))
        if sig > 0:
            # TWO conditions, and the second is the load-bearing one. On a small
            # clean row the MAD badly underestimates sigma -- measured on a
            # 14-rung synthetic with true sigma 0.00914 the MAD gave 0.00444, so
            # "2.5 sigma" was really 1.2 sigma and rejected 4 of 14 good rungs.
            # A genuine contaminant is a large fraction of the inclusion window,
            # not the tail of a tight distribution: the real p=66 outlier sits
            # at 0.064 against a 0.15 window, while the worst point on the clean
            # synthetic row is 0.018.
            dev = np.abs(rr - med)
            keep = ~((dev > 2.5*sig) & (dev > min_outlier_frac*tol))
            subset = ok & keep
            if 3 <= int(subset.sum()) < int(ok.sum()):
                d2, ok2 = _fit(subset)
                if ok2 is not None and int(ok2.sum()) >= 3:
                    d, ok = d2, ok2

    resid = float(np.sqrt(np.mean((proj[ok]/d - np.rint(proj[ok]/d))**2)))
    return d, ok, resid


def find_lattice_rows(q, I, *, tol_perp=0.06, tol_step_rel=0.02,
                      min_rungs=3, n_seed=150, min_sep_deg=8.0,
                      spacing_min=0.30, spacing_max=8.0,
                      a=3.6116, c=19.2516, two_pi=True,
                      identify=True, space_group_number=139,
                      max_match_rel=0.05) -> List[Row]:
    """Rows of collinear, regularly spaced spots through the origin.

    Parity comes from the SPACE GROUP per direction (see
    :func:`allowed_multiple_parity`), not from a blanket even-only rule: that
    rule is correct for (0,0,1) and wrong by a factor 2 for every even-sum step,
    and wrong entirely for an F-centred cell.

    ``spacing_max`` must cover the OBSERVED spacing n*|G|, not |G|: with the old
    2.5 limit only 1 of 27 candidate steps was both reachable and correctly
    identified, and synthetic (1,0,0)/(1,1,1)/(0,1,2)/(1,1,2)/(1,2,1) rows
    returned nothing at all.

    ``max_match_rel`` REJECTS an identification that is not close. Without it
    (0,0,1) absorbed 80 % of all producible spacings and q-vectors scaled by
    1.90 were still labelled (0,0,1).
    """
    q = np.asarray(q, float); I = np.asarray(I, float)
    qn = np.linalg.norm(q, axis=1)
    good = qn > 1e-6
    order = [k for k in np.argsort(-I) if good[k]][:n_seed]
    cands = candidate_row_spacings(a, c, two_pi=two_pi) if identify else []
    out: List[Row] = []
    for k in order:
        u = q[k]/qn[k]
        if any(abs(float(np.dot(u, R.direction))) >
               math.cos(math.radians(min_sep_deg)) for R in out):
            continue
        proj = q @ u
        perp = np.linalg.norm(q - np.outer(proj, u), axis=1)
        near = np.flatnonzero(perp < tol_perp)
        if len(near) < min_rungs:
            continue
        # Candidate spacings come from the data's own pairwise differences, and
        # the candidate explaining the most distinct rungs wins. Two changes
        # from the original, both because the spacing feeds `cell_from_row` and
        # therefore fixes a lattice parameter:
        #
        #   * ties are broken by the SMALLEST fit residual, not by whichever
        #     candidate came first in ascending order. The old rule made the
        #     winner among equally-good candidates the smallest one, i.e. a
        #     minimum statistic over noisy pairwise differences.
        #   * the winner is then refined by least squares over all its rungs
        #     (`refine_row_spacing`) instead of being used raw. At p=114 the raw
        #     pair gave c = 19.4397 against 19.1485 from the slope of the same
        #     14 rungs -- 1.5 %, enough to break the ladder-seeded search.
        pv = np.sort(proj[near])
        best = None
        trial = sorted({abs(pv[i]-pv[j]) for i in range(len(pv))
                        for j in range(i+1, len(pv))
                        if spacing_min <= abs(pv[i]-pv[j]) <= spacing_max})
        for d in trial:
            d_ref, sel, resid = refine_row_spacing(proj[near], d)
            if sel is None:
                d_ref, resid = d, float("inf")
                ratio = proj[near]/d
                n = np.rint(ratio)
                sel = (np.abs(ratio - n) < 0.15) & (np.abs(n) > 0)
            steps = sorted(set(int(x) for x in np.rint(proj[near][sel]/d_ref)))
            if len(steps) < min_rungs:
                continue
            key = (-len(steps), resid)
            if best is None or key < best[0]:
                best = (key, d_ref, steps, near[sel], resid)
        if best is None:
            continue
        _, d, steps, mem, resid = best
        # Collapse a sub-multiple before anything downstream sees it: the
        # spacing feeds cell_from_row, so a factor of 2 here is a factor of 2 in
        # a lattice parameter.
        gfac = _robust_common_factor(steps, min_rungs=min_rungs)
        if gfac > 1:
            d2, sel2, resid2 = refine_row_spacing(proj[near], d*gfac)
            if sel2 is not None and int(sel2.sum()) >= min_rungs:
                steps2 = sorted(set(int(x) for x in np.rint(proj[near][sel2]/d2)))
                if len(steps2) >= min_rungs:
                    d, mem, resid, steps = d2, near[sel2], resid2, steps2
        nr = len(steps)
        row = Row(direction=u, spacing=float(d), steps=steps, n_rungs=nr,
                  intensity=float(I[mem].sum()),
                  rms_perp=float(np.sqrt((perp[mem]**2).mean())), members=mem,
                  spacing_resid=float(resid))
        if identify and cands:
            # the observed spacing is n_allowed * |G_step|, and n_allowed
            # depends on the STEP, so test each candidate on its own terms
            scored = []
            for hkl, gp in cands:
                nmul = allowed_multiple_parity(hkl, space_group_number)
                scored.append((abs(d - nmul*gp)/(nmul*gp), hkl))
            scored.sort()
            best_rel, best_id = scored[0]
            row.match_rel = float(best_rel)
            # Runner-up. Seeding a domain from a general row makes the step
            # identification load-bearing in a way it never was for (0,0,1):
            # the step fixes both the cell and the crystal direction being
            # aligned, so picking the wrong one of two near-equal candidates
            # produces a confidently wrong orientation. Record the second-best
            # so the caller can insist on a clear winner.
            row.match_rel_2 = float(scored[1][0]) if len(scored) > 1 else np.inf
            row.hkl_step = best_id if best_rel <= max_match_rel else None
        out.append(row)
    out.sort(key=lambda R: (-R.n_rungs, -R.intensity))
    return out


# --------------------------------------------------------------------------
# Indexing a domain from one row
# --------------------------------------------------------------------------
def _basis_from_cstar(u, phi_deg):
    """Orthonormal U with its third column along u, rotated phi about u."""
    u = np.asarray(u, float); u = u/np.linalg.norm(u)
    t = np.array([0., 0., 1.]) if abs(u[2]) < 0.9 else np.array([1., 0., 0.])
    e1 = np.cross(t, u); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    ph = math.radians(phi_deg)
    a1 = math.cos(ph)*e1 + math.sin(ph)*e2
    a2 = np.cross(u, a1)
    return np.column_stack([a1, a2, u])


def hkl_box_from_geometry(a, c, *, wavelength_A=0.42459, tth_max_deg=26.4):
    """Largest |h| and |l| the DETECTOR can actually reach. Do not guess these.

    d_min = lambda / (2 sin(tth_max/2)); h_max = a/d_min, l_max = c/d_min.
    For this geometry: d_min = 0.930 A -> h_max 4, l_max 21.

    A hand-picked lmax=16 silently DELETED the (1,0,L) family, which is the
    a<->b PARTNER of (0,1,L). Without the partner those reflections stop being
    a/b-sensitive, and p=329 domain 2 -- the domain that carries the entire
    splitting measurement -- went from 6-7 sensitive reflections to ZERO. The
    box was chosen for consistency with the objective, not from physics, and it
    destroyed the deliverable.
    """
    d_min = wavelength_A/(2*math.sin(math.radians(tth_max_deg)/2))
    return int(math.ceil(a/d_min)), int(math.ceil(c/d_min))


def cell_from_row(hkl_step, spacing, *, a0=3.6116, c0=19.2516,
                  space_group_number=139, two_pi=True):
    r"""Tetragonal cell implied by one row's observed spacing.

    The observed spacing is ``n * |G_step|`` with n from
    :func:`allowed_multiple_parity`, and

        |G_step| = k * sqrt((h^2 + k^2)/a^2 + l^2/c^2),   k = 2*pi or 1

    which is ONE equation in two unknowns. A pure (0,0,l) row therefore fixes c
    and says nothing about a; a pure (h,k,0) row fixes a and says nothing about
    c. For a mixed step the row is solved for whichever axis dominates \|G\|^2 at
    the nominal cell, holding the other at nominal: with a=3.61 and c=19.25 the
    in-plane term of a (0,1,1) row is 0.0767 against 0.0027 out of plane, so
    that row is 28x more informative about a than about c and solving it for c
    would put all of its error into the axis it barely constrains.

    Returns ``(a, c)``.
    """
    h, k, l = (int(x) for x in hkl_step)
    kk = 2*math.pi if two_pi else 1.0
    n = allowed_multiple_parity((h, k, l), space_group_number)
    g = float(spacing)/n
    if g <= 0:
        raise ValueError("spacing must be positive")
    inplane, axial = (h*h + k*k), (l*l)
    if inplane == 0:
        return a0, kk*math.sqrt(axial)/g
    if axial == 0:
        return kk*math.sqrt(inplane)/g, c0
    if inplane/a0**2 >= axial/c0**2:                  # row mostly constrains a
        rem = (g/kk)**2 - axial/c0**2
        if rem <= 0:
            return a0, c0
        return math.sqrt(inplane/rem), c0
    rem = (g/kk)**2 - inplane/a0**2
    if rem <= 0:
        return a0, c0
    return a0, math.sqrt(axial/rem)


def row_scan_range(hkl_step):
    """Distinct orientations left once one lattice ROW is pinned to a direction.

    Aligning a crystal direction to a measured row direction fixes two of three
    orientational degrees of freedom. What remains is a turn about that row, and
    how much of the turn is distinct is set by the tetragonal lattice's own
    proper symmetry (422, the rotations of holohedry 4/mmm):

      * rotations that FIX the axis make turns beyond 360/n redundant
      * if some rotation REVERSES the axis, then pinning the crystal direction
        along +u and along -u give symmetry-equivalent crystals and only one
        sign need be scanned; with no such rotation both signs are genuinely
        different crystals and both must be

    A row carries no direction, only an axis, so getting the second point wrong
    silently discards half the orientation space. For (0,0,1) -- the only step
    the pipeline used to handle -- the answer is 90 deg and one sign, which is
    exactly what :func:`index_from_ladder` hard-codes.

    Returns ``(range_deg, signs)``.
    """
    h, k, l = (int(x) for x in hkl_step)
    if (h, k, l) == (0, 0, 0):
        raise ValueError("(0,0,0) is not a row")
    # rotations of 422 that fix the axis (identity always does)
    if h == 0 and k == 0:
        n_about = 4                       # 4-fold along c*
    elif l == 0 and (k == 0 or h == 0 or h == k or h == -k):
        n_about = 2                       # 2-fold along <100> or <110>
    else:
        n_about = 1
    # rotations of 422 that reverse the axis
    has_flip = (l == 0) or (h == 0) or (k == 0) or (h == k) or (h == -k)
    return 360.0/n_about, ((1,) if has_flip else (1, -1))


def _rotation_taking(v_from, v_to):
    """Proper rotation carrying unit ``v_from`` onto unit ``v_to`` (Rodrigues)."""
    a_ = np.asarray(v_from, float); a_ = a_/np.linalg.norm(a_)
    b_ = np.asarray(v_to, float); b_ = b_/np.linalg.norm(b_)
    v = np.cross(a_, b_)
    c_ = float(np.dot(a_, b_))
    if c_ > 1.0 - 1e-12:
        return np.eye(3)
    if c_ < -1.0 + 1e-12:                 # antiparallel: half turn about any perp
        t = np.array([1.0, 0.0, 0.0]) if abs(a_[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax = np.cross(a_, t); ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2.0*(K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + (K @ K)/(1.0 + c_)


def _basis_from_axis(u_lab, g_hat, phi_deg):
    """Orientation putting crystal direction ``g_hat`` along lab ``u_lab``.

    ``phi_deg`` turns about ``u_lab``. Generalises :func:`_basis_from_cstar`,
    which is the ``g_hat = (0,0,1)`` case; that function is left in place and
    still used by :func:`index_from_ladder` so the (0,0,1) path stays exactly
    as it was, phi origin included, and old results stay reproducible.
    """
    u = np.asarray(u_lab, float); u = u/np.linalg.norm(u)
    R0 = _rotation_taking(g_hat, u)
    ph = math.radians(phi_deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    R = np.eye(3) + math.sin(ph)*K + (1.0 - math.cos(ph))*(K @ K)
    return R @ R0


def index_from_row(q, I, u_row, hkl_step, *, a=3.6116, c=19.2516, two_pi=True,
                   tol=0.10, phi_step=0.25, hmax=None, lmax=None,
                   exclude=None, space_group_number=139, return_margin=False):
    """Best orientation with lattice row ``hkl_step`` along ``u_row``.

    :func:`index_from_ladder` can only seed on a (00L) ladder. That is a real
    restriction, not a convenience: a crystallite whose (00L) row happens to lie
    outside the rotation wedge is invisible to it however strongly it
    diffracts. Measured on La3Ni2O7 at p=66, the spots left unexplained after
    the (00L)-seeded search form eight further lattice rows with steps (0,1,1),
    (1,1,0) twice, (0,1,3) and three unidentified -- real rows carrying real
    intensity that the search could not reach. The same restriction is why the
    S5 dataset, whose smaller rotation range shows no ladders at all, could not
    be indexed this way.

    Any identified row works here. The step fixes the crystal direction to
    align, :func:`cell_from_row` turns its spacing into a cell, and
    :func:`row_scan_range` says how much of the remaining turn is distinct and
    whether both senses of the row axis have to be tried.

    Returns ``(U, n_matched, phi)``, or with ``return_margin`` also the margin:
    n(best) - n(best rival at least 5 deg away, the opposite axis sense
    counting as always distant). A flat landscape and a sharp one look identical
    without it.
    """
    if hmax is None or lmax is None:
        hb, lb = hkl_box_from_geometry(a, c)
        hmax = hb if hmax is None else hmax
        lmax = lb if lmax is None else lmax
    q = np.asarray(q, float)
    keep = np.ones(len(q), bool) if exclude is None else ~np.asarray(exclude, bool)
    qq = q[keep]
    if len(qq) < 6:
        return (None, 0, 0.0, 0) if return_margin else (None, 0, 0.0)
    lat = Lattice(a=a, b=a, c=c, alpha=90., beta=90., gamma=90.)
    B = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
    if two_pi:
        B = B*2*math.pi
    g_hat = B @ np.asarray(hkl_step, float)
    ng = np.linalg.norm(g_hat)
    if ng < 1e-12:
        raise ValueError("degenerate hkl_step")
    g_hat = g_hat/ng
    span, signs = row_scan_range(hkl_step)
    u_row = np.asarray(u_row, float); u_row = u_row/np.linalg.norm(u_row)

    best = (None, 0, 0.0, 1)
    curve = []
    for sg in signs:
        for phi in np.arange(0.0, span, phi_step):
            U = _basis_from_axis(sg*u_row, g_hat, float(phi))
            h = np.linalg.solve(U @ B, qq.T).T
            hi = np.rint(h)
            ok = ((np.abs(h - hi).max(axis=1) < tol)
                  & (np.abs(hi[:, 0]) <= hmax) & (np.abs(hi[:, 1]) <= hmax)
                  & (np.abs(hi[:, 2]) <= lmax)
                  & (np.abs(hi).sum(axis=1) > 0))
            if space_group_number == 139:
                ok &= (np.abs(hi.sum(axis=1)) % 2 < 1e-6)     # I-centring
            n = int(ok.sum())
            curve.append((sg, float(phi), n))
            if n > best[1]:
                best = (U, n, float(phi), sg)
    if not return_margin:
        return best[0], best[1], best[2]
    second = 0
    for sg, phi, n in curve:
        if sg != best[3]:
            second = max(second, n)                # other sense: always a rival
            continue
        d = abs(phi - best[2]) % span
        if min(d, span - d) >= 5.0:
            second = max(second, n)
    return best[0], best[1], best[2], best[1] - second


def index_from_ladder(q, I, u_cstar, *, a=3.6116, c=19.2516, two_pi=True,
                      tol=0.10, phi_step=0.25, hmax=None, lmax=None,
                      exclude=None, return_margin=False):
    """Best orientation with c* along ``u_cstar``.

    Returns ``(U, n_matched, phi)``, or ``(U, n_matched, phi, margin)`` with
    ``return_margin``. The MARGIN is n(best) - n(best distinct peak >=5 deg
    away); without it a flat landscape looks identical to a sharp one in the
    output. Measured: domain 1 margins 14/16/14 at p=329/283/330 but only 2 at
    p=116, where BOTH domains had margins of 1-2 -- i.e. no discrimination at
    all, and invisible until it was asked for.

    Scans the single remaining freedom -- rotation about c* -- and counts how
    many observed spots fall within ``tol`` in fractional hkl. No seeding from
    bright cores, so a domain 88x fainter is found on the same footing as a
    dominant one.
    """
    if hmax is None or lmax is None:
        hb, lb = hkl_box_from_geometry(a, c)
        hmax = hb if hmax is None else hmax
        lmax = lb if lmax is None else lmax
    q = np.asarray(q, float)
    keep = np.ones(len(q), bool) if exclude is None else ~np.asarray(exclude, bool)
    qq = q[keep]
    if len(qq) < 6:
        return None, 0, 0.0
    lat = Lattice(a=a, b=a, c=c, alpha=90., beta=90., gamma=90.)
    B = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
    if two_pi:
        B = B*2*math.pi
    best = (None, 0, 0.0)
    curve = []
    for phi in np.arange(0.0, 90.0, phi_step):
        U = _basis_from_cstar(u_cstar, float(phi))
        UB = U @ B
        h = np.linalg.solve(UB, qq.T).T
        hi = np.rint(h)
        ok = ((np.abs(h - hi).max(axis=1) < tol)
              & (np.abs(hi[:, 0]) <= hmax) & (np.abs(hi[:, 1]) <= hmax)
              & (np.abs(hi[:, 2]) <= lmax)
              & (np.abs(hi).sum(axis=1) > 0)
              & (np.abs(hi.sum(axis=1)) % 2 < 1e-6))       # I-centring
        n = int(ok.sum())
        curve.append((float(phi), n))
        if n > best[1]:
            best = (U, n, float(phi))
    if not return_margin:
        return best
    # best distinct peak at least 5 deg away, on the 90 deg circle
    second = 0
    for phi, n in curve:
        d = abs(phi - best[2]) % 90.0
        if min(d, 90.0 - d) >= 5.0:
            second = max(second, n)
    return best[0], best[1], best[2], best[1] - second


def _weighted_A(q, hkl, sigma_rtn, axis):
    r"""Least squares for q = A h with a per-reflection anisotropic weight.

    The uncertainty on a spot is not isotropic. An error in omega rotates q
    about the rotation axis, so its contribution lies along z x q -- one known
    direction per reflection -- and detector-centroid error lies tangent to the
    Ewald sphere. Measured on La3Ni2O7 over 214 reflections in 9 domains: rms
    radial 0.0071, transverse 0.0145, normal 0.0094 1/A, a factor ~2 anisotropy
    pooled, 1.3-4.3 per domain. Treating the three directions as equally
    reliable throws information away; each reflection here contributes with
    weight 1/sigma^2 in its own (qhat, zhat x qhat, qhat x that) frame. On real
    data this halves the a/b uncertainty (0.281% vs 0.544% at one domain).

    THREE CORRECTIONS to an earlier version of this note, all from adversarial
    checks, recorded because each was wrong in a way that looked convincing:

    * The implied omega error is res_t / |z x q|, NOT res_t / |q|. Using |q|
      understated it by rms|q| / rms|z x q| = 1.27 and produced a spurious exact
      match with Delta_omega/sqrt(12). Solving res = J (drow, dcol, domega)
      properly gives an omega rms near 0.58 deg, and detector error is not
      negligible either (drow 1.13, dcol 1.44 px rms).
    * It is NOT mostly frame quantisation. Under a pure omega error
      res_t / |z x q| must be constant; measured it falls 2.5x across quartiles
      (Spearman p = 2e-5). Roughly half the transverse variance was a caller
      flooring the fractional intensity-weighted frame centroid that
      `find_blobs_3d` already returns; recovering it took transverse rms from
      0.0199 to 0.0145 and left the anisotropy near 2, not 1.
    * The survivor scales with |q| and does not track streak geometry, so it is
      an angular width of roughly 0.2 deg -- consistent with sample mosaic, and
      not something a finer omega step removes. Indexed reflections here span a
      MEDIAN of 6 omega frames: they are smear-limited, not step-limited.

    Solves the 9x9 normal equations for vec(A); n is small so the loop is fine.
    """
    q = np.asarray(q, float)
    hkl = np.asarray(hkl, float)
    axis = np.asarray(axis, float)
    axis = axis/np.linalg.norm(axis)
    # Each sigma may be a scalar OR a per-reflection array. Per-reflection
    # matters for the transverse term: an omega error scales with |z x q|, and
    # separately a reflection smeared over many frames has no well-defined
    # Bragg omega at all. Measured, the implied omega error is 0.27 deg for
    # reflections spanning <=8 frames and 1.11 deg for those spanning >=9 --
    # and for the wide ones the intensity centroid makes it WORSE, not better.
    # One global sigma_t therefore over-trusts the wide reflections and
    # under-trusts the narrow ones.
    def _as_vec(x):
        a = np.atleast_1d(np.asarray(x, float))
        if a.size == 1:
            return np.full(len(q), float(a[0]))
        if a.size != len(q):
            raise ValueError("per-reflection sigma must match len(q)")
        return a
    s_r, s_t, s_n = (_as_vec(x) for x in sigma_rtn)
    if min(s_r.min(), s_t.min(), s_n.min()) <= 0:
        raise ValueError("sigma_rtn entries must be positive")
    N = np.zeros((9, 9))
    b = np.zeros(9)
    for i in range(len(q)):
        nq = np.linalg.norm(q[i])
        if nq < 1e-12:
            continue
        qh = q[i]/nq
        t = np.cross(axis, qh)
        nt = np.linalg.norm(t)
        wi = np.array([1.0/s_r[i]**2, 1.0/s_t[i]**2, 1.0/s_n[i]**2])
        if nt < 1e-9:                      # q parallel to the axis: isotropic
            W = np.eye(3)*wi[0]
        else:
            t = t/nt
            n = np.cross(qh, t)
            E = np.column_stack([qh, t, n])
            W = E @ np.diag(wi) @ E.T
        h = hkl[i]
        hh = np.outer(h, h)
        for kk in range(3):
            for ll in range(3):
                N[3*kk:3*kk+3, 3*ll:3*ll+3] += W[kk, ll]*hh
        Wq = W @ q[i]
        for kk in range(3):
            b[3*kk:3*kk+3] += Wq[kk]*h
    if np.linalg.cond(N) > 1e12:
        return None
    try:
        x = np.linalg.solve(N, b)
    except np.linalg.LinAlgError:
        return None
    return x.reshape(3, 3)


@dataclass
class RefinedLattice:
    """A lattice fitted to indexed reflections, with nothing forced."""
    U: np.ndarray                 # orientation, columns = crystal axes in sample frame
    B: np.ndarray                 # reciprocal basis, columns a*, b*, c*
    a: float
    b: float
    c: float
    alpha: float                  # degrees
    beta: float
    gamma: float
    leverage: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    cond: float = float("nan")
    n: int = 0

    @property
    def ab_split(self) -> float:
        """2(b - a)/(b + a). The quantity the a/b measurement is after."""
        return 2.0*(self.b - self.a)/(self.b + self.a)


def refine_lattice(q, hkl, *, two_pi=True, min_reflections=6,
                   max_cell_change=0.10, a0=None, c0=None, constrain=None,
                   sigma_rtn=None, rotation_axis=(0.0, 0.0, 1.0)):
    r"""Fit the FULL lattice to reflections that already have indices.

    With indices assigned, ``q = A h`` is linear in ``A = U B``, so A is an
    ordinary least-squares solution and

        G* = A^T A

    is exactly the reciprocal metric tensor -- no matrix square root, no
    assumption that B is diagonal. Inverting it gives the direct metric and
    hence a, b, c and all three angles.

    WHY THIS MATTERS HERE, and why the earlier diagonal-B version was wrong:
    in a Ruddlesden-Popper subcell the Fmmm distortion is a GAMMA SHEAR, not
    a != b. Forcing B diagonal cannot express it. Verified synthetically:
    planting gamma = 90.5 deg and fitting with a diagonal B returned
    a_x = a_y = 3.61150 unchanged to 5e-5, with the whole distortion sitting in
    the off-diagonal term the fit then discarded. Real data showed the same
    off-diagonal at 0.00444 (p=66) and 0.00654 (p=329), i.e. gamma ~ 90.3-90.4
    deg, silently thrown away and reported as clean tetragonal. Since the a/b
    splitting IS the deliverable, the fit must not be the thing that removes it.

    The diagonal version also averaged two very unequally determined in-plane
    axes: at p=329 domain 2, a_x came from leverage 7 and a_y from leverage 68,
    and their mean was reported as a measurement. ``leverage`` (sum of h^2 per
    axis) is returned so a caller can refuse a poorly-conditioned axis instead.

    ``constrain="tetragonal"`` restores the old behaviour explicitly -- a = b
    from the two in-plane axes, all angles 90 -- for callers that need it. The
    default fits everything and lets the data say whether the cell is
    tetragonal.

    Returns a :class:`RefinedLattice`, or ``None`` if the fit was refused
    (too few reflections, coplanar indices, ill-conditioned, or a cell change
    beyond ``max_cell_change``).
    """
    q = np.asarray(q, float)
    hkl = np.asarray(hkl, float)
    if q.ndim != 2 or q.shape[1] != 3 or hkl.shape != q.shape:
        raise ValueError("q and hkl must both be (n, 3)")
    if len(q) < min_reflections:
        return None
    H = hkl.T
    if np.linalg.matrix_rank(H, tol=1e-9) < 3:
        return None                          # coplanar indices cannot fix a cell
    HHt = H @ H.T
    cond = float(np.linalg.cond(HHt))
    if cond > 1e10:
        return None
    if sigma_rtn is None:
        A = (q.T @ H.T) @ np.linalg.inv(HHt)
    else:
        A = _weighted_A(q, hkl, sigma_rtn, rotation_axis)
        if A is None:
            return None
    k = 2*math.pi if two_pi else 1.0
    Ar = A/k                                 # strip the 2pi so G is in Angstrom
    Gstar = Ar.T @ Ar
    try:
        G = np.linalg.inv(Gstar)
    except np.linalg.LinAlgError:
        return None
    d = np.diag(G)
    if np.any(d <= 0):
        return None
    la, lb, lc = (float(np.sqrt(x)) for x in d)

    def _ang(i, j, x, y):
        return math.degrees(math.acos(max(-1.0, min(1.0, G[i, j]/(x*y)))))

    alpha, beta, gamma = _ang(1, 2, lb, lc), _ang(0, 2, la, lc), _ang(0, 1, la, lb)
    if constrain == "tetragonal":
        la = lb = math.sqrt(0.5*(la*la + lb*lb))
        alpha = beta = gamma = 90.0
        B = np.diag([k/la, k/la, k/lc])
        U = A @ np.linalg.inv(B)
        Uu, _, Vt = np.linalg.svd(U)
        U = Uu @ np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Uu @ Vt)))]) @ Vt
    else:
        # B^T B = G*, B upper triangular: exactly orthogonal U by construction,
        # no polar decomposition and no orthogonalisation error.
        try:
            B = np.linalg.cholesky(Gstar).T * k
        except np.linalg.LinAlgError:
            return None
        U = A @ np.linalg.inv(B)
    if a0 is not None and abs(la - a0) > max_cell_change*a0:
        return None
    if c0 is not None and abs(lc - c0) > max_cell_change*c0:
        return None
    lev = tuple(float((hkl[:, i]**2).sum()) for i in range(3))
    return RefinedLattice(U=U, B=B, a=la, b=lb, c=lc, alpha=alpha, beta=beta,
                          gamma=gamma, leverage=lev, cond=cond, n=len(q))


def refine_cell_orientation(q, hkl, *, two_pi=True, min_reflections=6,
                            max_cell_change=0.10, a0=None, c0=None):
    """Tetragonal-constrained wrapper over :func:`refine_lattice`.

    Kept because callers that genuinely want a forced tetragonal cell should
    say so at the call site. Prefer :func:`refine_lattice` -- forcing a = b and
    gamma = 90 discards the a/b splitting and the gamma shear, which is the
    thing being measured. Returns ``(U, a, c)`` or ``(None, a0, c0)``.
    """
    res = refine_lattice(q, hkl, two_pi=two_pi, min_reflections=min_reflections,
                         max_cell_change=max_cell_change, a0=a0, c0=c0,
                         constrain="tetragonal")
    if res is None:
        return None, a0, c0
    return res.U, res.a, res.c


_ROT422 = (
    np.eye(3),
    np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),      # 4+ about c
    np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]),     # 2 about c
    np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]),      # 4- about c
    np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]),     # 2 about a
    np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]),     # 2 about b
    np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]]),      # 2 about [110]
    np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]]),    # 2 about [1-10]
)


def _misorientation_422(Ua, Ub, sgnum=139):
    """Smallest angle between two orientations, modulo crystal symmetry (degrees).

    Without the symmetry reduction two descriptions of the SAME crystal look
    like rivals up to 90 degrees apart, which makes every margin test on a
    well-determined domain report zero.

    ``midas_stress.misorientation_om`` is the CANONICAL implementation and the
    house rule routes misorientation there. This keeps a hard-coded 422 table
    only as a FAST PATH -- it is called once per candidate orientation inside
    :func:`index_from_pairs`, where the canonical routine is ~8x slower. The two
    agree to 9e-14 degrees over 200 random pairs and
    ``test_misorientation_matches_midas_stress`` pins that, so the table cannot
    drift. Any space group other than 139 delegates.

    NOTE ``misorientation_om`` returns RADIANS (and an axis); this returns
    DEGREES, matching the rest of this module.
    """
    if sgnum != 139:
        from midas_stress import misorientation_om
        return math.degrees(float(misorientation_om(Ua, Ub, sgnum)[0]))
    best = 180.0
    for R in _ROT422:
        M = Ua.T @ (Ub @ R)
        c = max(-1.0, min(1.0, (float(np.trace(M)) - 1.0)/2.0))
        best = min(best, math.degrees(math.acos(c)))
    return best


@dataclass
class ConvergedDomain:
    """A cell and the reflection set it was fitted to, guaranteed consistent.

    ``lat`` is the least-squares fit to ``q[claim]`` / ``hkl[claim]`` EXACTLY as
    returned. Nothing else in this dataclass is safe to pair with a cell taken
    from elsewhere -- see :func:`refine_to_convergence`.
    """
    lat: "RefinedLattice"
    claim: np.ndarray      # bool, one True per accepted reflection (deduped)
    frag: np.ndarray       # bool, before unique_by_hkl -- what to consume
    hkl: np.ndarray
    residual: np.ndarray
    n_iter: int


def refine_to_convergence(q, U0, *, a0, c0, avail=None, tol_sigma=7.0,
                          sigma_rtn=(0.0071, 0.0145, 0.0094), B0=None,
                          max_iter=4, rotation_axis=(0.0, 0.0, 1.0),
                          min_reflections=5):
    """Iterate refine -> rematch until the cell and its reflections agree.

    A seeded search has to bootstrap its FIRST match from some other domain's
    cell -- there is no cell for the new domain yet. What it must not do is
    stop there. Two distinct bugs come from stopping, and both were live in a
    production analysis script until refuters found them:

    **Seed-cell anchoring.** If the reflection SET is chosen with a neighbour's
    cell and never re-chosen, the fitted cell can be pulled toward the
    neighbour's. Iterate because selecting reflections with a foreign cell is
    WRONG, not because of a measured effect size -- see the warning below, every
    attempt to size this on real data has been confounded.

    **A reported cell that was never fitted to the reported reflections.**
    Assigning the candidate INSIDE the loop reports a cell fitted to the
    PREVIOUS iteration's claim, or to a REJECTED round. 26% of one run's stored
    ``c`` values were not the LS fit to their own stored hkl list (median
    0.0041 A) -- against a 0.035 A effect under study.

    So: iterate on the domain's OWN cell, keep a round only on a STRICT gain
    (more reflections, or an equal count at lower residual, which makes the loop
    monotone and safe to run blind), and refit once on the converged set before
    returning. The postcondition is the point of this function --
    ``result.lat`` IS the fit to ``result.claim``.

    **Do NOT credit this function with removing anchoring, and beware how you
    measure anchoring at all.** A gate of the form ``|c/c_seed - 1| < tol`` is a
    truncation band around the REGRESSOR of the obvious test, and manufactures a
    positive c-on-c slope with zero anchoring present: under a y-permutation
    null a 1% seed-referenced gate gave +0.128 +- 0.042, a nominal-referenced
    gate +0.002 +- 0.053. Measured on identity-matched domains over a common
    support, and in a 2x2 re-running all four arms one change at a time:

                          old |c/c_seed-1|<1%   new |c/c_nom-1|<1.5%
        no loop (pre-fix)      +0.2260               +0.0836
        loop + refit           +0.2073               +0.1247

    the LOOP alone is -0.019 [-0.098,+0.055] / +0.041 [-0.045,+0.135] -- zero,
    or positive. The GATE alone is -0.142 [-0.236,-0.059]. The buggy code with
    only the gate swapped (+0.084) BEATS the fixed pipeline (+0.125).
    Reference cell gates to a FIXED NOMINAL cell, and change a gate and a loop
    ONE AT A TIME.

    **Why this function still earns its place:** it moves cells OFF the seed,
    which is the point, and a seed-referenced gate then rejects them -- 43 pair
    domains lost (532 -> 489) under the old gate. 4.9% of ungated
    independently-seeded domains sit >1% from the seed cell, so that bound was
    clipping a real population. This loop and a nominal gate ship together.

    Returns ``None`` if no lattice can be fitted at all.
    """
    q = np.asarray(q, float)
    avail = np.ones(len(q), bool) if avail is None else np.asarray(avail, bool)
    U, a, c, B = np.asarray(U0, float), float(a0), float(c0), B0
    claim = frag = hkl = resid = None
    n_iter = 0
    for _ in range(int(max_iter)):
        cl, hk, rs = match_mask(q, U, B=B, a=a, c=c, tol_sigma=tol_sigma,
                                sigma_rtn=sigma_rtn, return_residual=True,
                                rotation_axis=rotation_axis)
        cl &= avail
        fr = cl.copy()
        cl = unique_by_hkl(cl, hk, rs)
        if claim is not None:
            better_n = int(cl.sum()) > int(claim.sum())
            better_r = (float(np.mean(rs[cl])) <
                        float(np.mean(resid[claim])) - 1e-12) \
                if int(cl.sum()) == int(claim.sum()) else False
            if not (better_n or better_r):
                break
        if int(cl.sum()) < int(min_reflections):
            if claim is None:
                return None
            break
        claim, frag, hkl, resid = cl, fr, hk, rs
        cand = refine_lattice(q[claim], hkl[claim], a0=a, c0=c,
                              sigma_rtn=sigma_rtn, rotation_axis=rotation_axis,
                              min_reflections=min_reflections)
        if cand is None:
            break
        U, a, c, B = cand.U, cand.a, cand.c, cand.B
        n_iter += 1
    if claim is None or int(claim.sum()) < int(min_reflections):
        return None
    # THE POSTCONDITION. Refit on the set actually being returned, so the cell
    # and the reflections cannot disagree however the loop exited.
    final = refine_lattice(q[claim], hkl[claim], a0=a, c0=c,
                           sigma_rtn=sigma_rtn, rotation_axis=rotation_axis,
                           min_reflections=min_reflections)
    if final is None:
        return None
    return ConvergedDomain(lat=final, claim=claim, frag=frag, hkl=hkl,
                           residual=resid, n_iter=n_iter)


def search_null(q, I, B, search, *, n_rep=40, alpha=0.05, seed=0, **kw):
    r"""What the SAME search returns on data with no crystal in it.

    This is the gate every seeding method here was missing, and the reason two
    of them failed. A search that scores N orientations and keeps the best has N
    chances at a coincidence, so the question is never "does this orientation
    explain 12 reflections" but "does this SEARCH find 12 on structureless data
    of the same kind".

    Both obvious references are wrong. The RUNNER-UP fails the moment a second
    real grain is present -- each grain's nearest rival is the other, the margin
    collapses to ~0, and the search stops after one domain. A PERCENTILE of the
    score distribution is too low a bar once N is large. And randomising the
    ORIENTATION while keeping the real spots is not a null either: with a few
    thousand tries one lands within a degree of a real grain, and the "null"
    then contains 67-79 against real grains at 96.

    So randomise the SPOTS: keep every \|q\| exactly, replace every direction.
    That preserves the radial structure a search exploits -- shell occupancy,
    the number of candidate hkl per spot -- and destroys only the mutual angles
    a crystal imposes. Then run the caller's own search on it, unchanged.

    ``search`` is called as ``search(q_scrambled, I, B, **kw)`` and must return
    an INT: the best reflection count that search achieved. Returning a count
    rather than a structure keeps this agnostic to which seeder is being
    calibrated -- pair, grid or anything later -- since each returns a different
    shape.

    Returns ``(threshold, per_replicate_best)``; a candidate must EXCEED the
    threshold.
    """
    q = np.asarray(q, float)
    qn = np.linalg.norm(q, axis=1)
    rng = np.random.default_rng(seed)
    best = []
    for _ in range(n_rep):
        v = rng.normal(size=q.shape)
        v /= np.linalg.norm(v, axis=1)[:, None]
        got = search(v*qn[:, None], I, B, **kw)
        best.append(int(got))
    return float(np.quantile(best, 1.0 - alpha)), best


def index_from_pairs(q, I, B, *, a=3.6116, c=19.2516, exclude=None,
                     tol_q=0.05, dq_shell=0.030, d_angle=1.2,
                     n_anchor=40, q_max_anchor=3.2, space_group_number=139,
                     min_reflections=8, min_sep_deg=5.0):
    r"""Seed a domain from a PAIR of spots, with the cell already known.

    Row seeding needs a row, and in the residual left by earlier domains rows
    are scarce: at La3Ni2O7 p=66 only 3-rung stubs survive, and they die at the
    nmatch and margin gates, so the search returned one domain from a pattern
    that visibly held more. Two reflections determine an orientation, and pairs
    of spots are plentiful where rows are not.

    This is only available once some domain has fixed the cell -- which is the
    point: the first domain is found ab initio from a row, and every later one
    can then be sought this way.

    Two prunings keep it cheap, both from the known cell:

    * a spot's \|q\| admits only the few hkl whose \|G\| matches within
      ``dq_shell``
    * a PAIR is consistent only if the angle between the two q vectors equals
      the angle between the two candidate G vectors within ``d_angle``

    At p=66 that leaves 224 orientations to score out of ~10^5 raw combinations,
    and finds a 17-reflection domain where the row search found none.

    Returns ``(U, n_matched, seed_pair, margin)``; ``U`` is None if nothing
    reached ``min_reflections``. The margin is n(best) - n(best rival at least
    ``min_sep_deg`` away), the same guard row seeding uses -- a pair-seeded
    domain accepted on its raw count alone can be a flat landscape.
    """
    q = np.asarray(q, float)
    I = np.asarray(I, float)
    B = np.asarray(B, float)
    avail = (np.ones(len(q), bool) if exclude is None
             else ~np.asarray(exclude, bool))
    if int(avail.sum()) < 6:
        return None, 0, None, 0
    qn = np.linalg.norm(q, axis=1)
    hb, lb = hkl_box_from_geometry(a, c)
    hkls, gmag, gvec = [], [], []
    for h in range(-hb, hb + 1):
        for k in range(-hb, hb + 1):
            for l in range(-lb, lb + 1):
                if (h, k, l) == (0, 0, 0):
                    continue
                if space_group_number == 139 and (h + k + l) % 2:
                    continue
                g = B @ np.array([h, k, l], float)
                hkls.append((h, k, l)); gvec.append(g)
                gmag.append(float(np.linalg.norm(g)))
    gmag = np.asarray(gmag)
    idx = np.flatnonzero(avail)
    cand = {int(i): np.flatnonzero(np.abs(gmag - qn[i]) < dq_shell) for i in idx}
    # Anchors must NOT be the brightest spots alone. A weak crystallite can
    # diffract ~100x less than a dominant one, so an intensity-ranked shortlist
    # is blind to exactly the domains this search exists to find. Half the
    # anchors are the brightest, half are drawn evenly across the rest of the
    # intensity-sorted list, so a faint domain gets anchors of its own.
    ranked = [int(i) for i in idx[np.argsort(-I[idx])] if qn[i] < q_max_anchor]
    n_top = max(1, n_anchor//2)
    anchors = ranked[:n_top]
    rest = ranked[n_top:]
    if rest:
        n_rest = min(n_anchor - len(anchors), len(rest))
        if n_rest > 0:
            step = max(1, len(rest)//n_rest)
            anchors += rest[::step][:n_rest]
    best = (None, 0, None)
    cands = []
    for ai in range(len(anchors)):
        ia = anchors[ai]
        if not len(cand[ia]):
            continue
        for bi in range(ai + 1, len(anchors)):
            ib = anchors[bi]
            if not len(cand[ib]):
                continue
            cos_obs = float(np.dot(q[ia], q[ib]))/(qn[ia]*qn[ib])
            ang_obs = math.degrees(math.acos(max(-1.0, min(1.0, cos_obs))))
            for j1 in cand[ia]:
                g1 = gvec[j1]; n1 = gmag[j1]
                for j2 in cand[ib]:
                    g2 = gvec[j2]; n2 = gmag[j2]
                    cth = float(np.dot(g1, g2))/(n1*n2)
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, cth))))
                    if abs(ang - ang_obs) > d_angle:
                        continue
                    U = _u_from_two_vectors(g1, g2, q[ia], q[ib])
                    if U is None:
                        continue
                    cl, hk, rs = match_mask(q, U, B=B, a=a, c=c, tol_q=tol_q,
                                            return_residual=True)
                    cl &= avail
                    n = int(unique_by_hkl(cl, hk, rs).sum())
                    if n > best[1]:
                        best = (U, n, (hkls[j1], hkls[j2]))
                    cands.append((n, U))
    if best[1] < min_reflections:
        return None, best[1], best[2], 0
    # MARGIN, the same protection row seeding has and this branch lacked.
    # A flat landscape -- many unrelated orientations scoring nearly as well as
    # the best -- is not a solution, and without this test a pair-seeded domain
    # was accepted on its raw count alone. The rival must be a genuinely
    # DIFFERENT orientation: at least `min_sep_deg` away from the winner.
    second = 0
    for n, U in cands:
        if U is best[0]:
            continue
        if _misorientation_422(best[0], U) >= min_sep_deg:
            second = max(second, n)
    return best[0], best[1], best[2], best[1] - second


def _u_from_two_vectors(g1, g2, q1, q2):
    """Rotation carrying crystal vectors g1, g2 onto lab vectors q1, q2."""
    def frame(v1, v2):
        n1 = np.linalg.norm(v1)
        if n1 < 1e-12:
            return None
        e1 = v1/n1
        e2 = v2 - e1*float(np.dot(e1, v2))
        n2 = np.linalg.norm(e2)
        if n2 < 1e-9:
            return None
        e2 = e2/n2
        return np.column_stack([e1, e2, np.cross(e1, e2)])
    Fc, Fl = frame(np.asarray(g1, float), np.asarray(g2, float)), \
        frame(np.asarray(q1, float), np.asarray(q2, float))
    if Fc is None or Fl is None:
        return None
    return Fl @ Fc.T


def orientation_grid(step_deg=3.0, space_group_number=139):
    r"""Orientations covering SO(3) at ``step_deg``, reduced by the Laue group.

    Cubochoric-style sampling is overkill here; a uniform axis grid crossed with
    a rotation angle is adequate and easy to audit. The 422 proper group has
    order 8, so only 1/8 of SO(3) is distinct for a tetragonal lattice.
    """
    axes = []
    n_th = max(2, int(round(180.0/step_deg)))
    for it in range(n_th + 1):
        th = math.pi*it/n_th
        st = math.sin(th)
        n_ph = max(1, int(round(2*math.pi*st/math.radians(step_deg))))
        for ip in range(n_ph):
            ph = 2*math.pi*ip/n_ph
            axes.append((st*math.cos(ph), st*math.sin(ph), math.cos(th)))
    order = 8 if space_group_number in (139,) else 1
    amax = 2*math.pi/order
    n_a = max(1, int(round(amax/math.radians(step_deg))))
    out = []
    for ax in axes:
        ax = np.asarray(ax, float)
        nn = np.linalg.norm(ax)
        if nn < 1e-12:
            continue
        ax = ax/nn
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        for ia in range(n_a):
            an = amax*ia/n_a
            out.append(np.eye(3) + math.sin(an)*K + (1-math.cos(an))*(K @ K))
    return np.array(out)


def _local_rotations(U, span_deg, step_deg):
    """Small perturbations of U -- a LOCAL grid, not a filtered global one."""
    out = [U]
    n = max(1, int(round(span_deg/step_deg)))
    for ax in np.eye(3):
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        for k in range(-n, n + 1):
            if k == 0:
                continue
            an = math.radians(k*step_deg)
            R = np.eye(3) + math.sin(an)*K + (1 - math.cos(an))*(K @ K)
            out.append(R @ U)
    return out


def index_by_grid(q, B, *, a=3.6116, c=19.2516, exclude=None, tol_q=0.05,
                  tol_sigma=None, sigma_rtn=(0.0071, 0.0145, 0.0094),
                  step_deg=1.5, refine_step_deg=0.25, space_group_number=139,
                  min_reflections=8, min_margin=3, max_domains=10):
    r"""Exhaustive orientation search with the cell known.

    DOES NOT WORK FOR SUBDOMINANT GRAINS -- do not wire this into a pipeline
    without reading this paragraph. On a planted control of three equal grains
    it returns the dominant one exactly (96/96 reflections, 0.57 deg from truth)
    and then invents spurious domains instead of the other two: at a 6 deg grid,
    1 real + 2 spurious at ~45 deg; at 1.5 deg, 1 real + 5 spurious at 23-52 deg.
    Two things defeat it, and they pull against each other. A reflection at
    \|q\| matches only while the orientation error times \|q\| stays inside
    tol_q, so at \|q\| = 5 and tol_q = 0.05 the grid must be finer than
    ~0.6 deg; but a grid that fine is millions of orientations, and searching
    millions of orientations finds 12-14 chance matches on any spot cloud, which
    the margin gate (a percentile of the score distribution) is far too weak to
    reject. Refining the grid adds trials faster than it adds resolution.
    It DOES correctly return nothing on a random cloud, so it is safe as a
    dominant-grain finder and as a negative control. Kept for that, and as the
    record of why exhaustive orientation search is the wrong tool here.

    Pair seeding needs TWO anchors from the same crystal. In a residual holding
    a few hundred reflections shared among many small grains that is a poor bet,
    and measured on La3Ni2O7 the pair search simply stopped -- its candidates
    dying at margin 0-2 -- while ~205 Bragg-like, lattice-structured spots per
    position stayed unindexed. A grid over SO(3) needs no anchor: it asks every
    orientation in turn.

    Two stages, because they want different things. The COARSE pass scores the
    whole grid with an isotropic \|dq\| test, vectorised over orientations --
    it only has to rank, and the anisotropic test costs a Python loop. The
    winner is then refined on a LOCAL grid and re-scored with the caller's real
    criterion (``tol_sigma`` if given).

    Each accepted domain must clear ``min_reflections`` AND beat its nearest
    distinct rival by ``min_margin``, the same gate row and pair seeding use, so
    a flat landscape is rejected here too.

    Returns ``[(U, n_reflections, margin), ...]``, strongest first.
    """
    q = np.asarray(q, float)
    B = np.asarray(B, float)
    avail = (np.ones(len(q), bool) if exclude is None
             else ~np.asarray(exclude, bool))
    grid = orientation_grid(step_deg, space_group_number)
    hb, lb = hkl_box_from_geometry(a, c)
    Binv = np.linalg.inv(B)
    found = []

    def score_all(mask, chunk=4000):
        r"""Reflections explained, per grid orientation. Isotropic, vectorised.

        REQUIRED GRID RESOLUTION. A reflection at \|q\| matches only while the
        orientation error times \|q\| stays inside tol_q, so at \|q\| = 5 and
        tol_q = 0.05 the grid must be finer than ~0.6 deg. A 6 deg grid finds
        only the dominant grain -- verified on a planted 3-grain control, which
        returned 1 real grain and 2 spurious peaks 45 deg from anything. Hence
        the chunking: sub-degree grids run to millions of orientations and the
        (n_orient x n_spot x 3) array will not fit otherwise.
        """
        qs = q[mask]
        if not len(qs):
            return np.zeros(len(grid), int)
        out = np.empty(len(grid), np.int32)
        for lo in range(0, len(grid), chunk):
            G = grid[lo:lo + chunk]
            qh = np.einsum("nji,mj->nmi", G, qs)         # U^T q
            h = np.einsum("ij,nmj->nmi", Binv, qh)
            hi = np.rint(h)
            dq = np.linalg.norm(qh - np.einsum("ij,nmj->nmi", B, hi), axis=2)
            ok = ((dq < tol_q)
                  & (np.abs(hi[:, :, 0]) <= hb) & (np.abs(hi[:, :, 1]) <= hb)
                  & (np.abs(hi[:, :, 2]) <= lb)
                  & (np.abs(hi).sum(axis=2) > 0))
            if space_group_number == 139:
                ok &= (np.abs(hi.sum(axis=2)) % 2 < 1e-6)
            out[lo:lo + chunk] = ok.sum(axis=1)
        return out

    for _ in range(max_domains):
        if int(avail.sum()) < min_reflections:
            break
        sc = score_all(avail)
        top = int(np.argmax(sc))
        if sc[top] < min_reflections:
            break
        # The margin asks "is this peak above the NOISE FLOOR", not "is it
        # above the next peak". Using the runner-up breaks as soon as a second
        # real grain is present: each grain's nearest rival is the other one,
        # the margin collapses to ~0, and the search stops after one domain --
        # measured on a planted 3-grain control, which returned 1. A high
        # percentile of the score distribution is robust to a handful of
        # genuine peaks while still catching a landscape with no peak at all.
        tr = np.einsum("ji,njk->nik", grid[top],
                       np.einsum("nij,sjk->nsik", grid, _ROT422).reshape(
                           len(grid), 8, 3, 3).transpose(1, 0, 2, 3).reshape(
                           -1, 3, 3)).reshape(8, len(grid), 3, 3
                                              ).trace(axis1=2, axis2=3)
        ang = np.degrees(np.arccos(np.clip((tr.max(axis=0) - 1)/2, -1, 1)))
        far = ang >= 5.0
        rival = int(np.percentile(sc[far], 99.0)) if far.any() else 0
        best = (grid[top], -1)
        for V in _local_rotations(grid[top], 1.5*step_deg, refine_step_deg):
            cl, hk, rs = match_mask(q, V, B=B, a=a, c=c, tol_q=tol_q,
                                    tol_sigma=tol_sigma, sigma_rtn=sigma_rtn,
                                    return_residual=True)
            n = int(unique_by_hkl(cl & avail, hk, rs).sum())
            if n > best[1]:
                best = (V, n)
        U, n = best
        if n < min_reflections or (n - rival) < min_margin:
            break
        cl, hk, rs = match_mask(q, U, B=B, a=a, c=c, tol_q=tol_q,
                                tol_sigma=tol_sigma, sigma_rtn=sigma_rtn,
                                return_residual=True)
        found.append((U, n, n - rival))
        avail &= ~(cl & avail)
    return found


def match_mask(q, U, *, a=3.6116, c=19.2516, two_pi=True, tol=0.10,
               hmax=None, lmax=None, return_residual=False, tol_q=None,
               B=None, tol_sigma=None, sigma_rtn=(0.0071, 0.0145, 0.0094),
               rotation_axis=(0.0, 0.0, 1.0)):
    r"""Spots explained by ``U``, inside the SAME hkl box the objective scored.

    Without the box the claim set was drawn from a larger space than
    ``index_from_ladder`` ever searched, so up to 7 % of claimed spots sat
    outside it (l up to 18) -- the two were inconsistent.

    ``return_residual`` also returns max\|h - round(h)\| per spot, which is what
    :func:`unique_by_hkl` needs to pick one spot per reflection. Note that the
    returned mask can claim the SAME hkl several times when a streak has been
    cut into fragments; pass the result through :func:`unique_by_hkl` before
    counting reflections or fitting a cell.
    """
    if hmax is None or lmax is None:
        hb, lb = hkl_box_from_geometry(a, c)
        hmax = hb if hmax is None else hmax
        lmax = lb if lmax is None else lmax
    if B is None:
        lat = Lattice(a=a, b=a, c=c, alpha=90., beta=90., gamma=90.)
        B = np.asarray(lat.reciprocal_cartesian_vectors(), float).T
        if two_pi:
            B = B*2*math.pi
    else:
        B = np.asarray(B, float)      # caller supplied a full (possibly
                                      # non-diagonal) reciprocal basis
    UB = U @ B
    h = np.linalg.solve(UB, np.asarray(q, float).T).T
    hi = np.rint(h)
    if tol_sigma is not None:
        # Anisotropic acceptance, matching the anisotropic error. An isotropic
        # tol_q has to be set by the WORST direction, so it rejects reflections
        # that are displaced transversely by an ordinary amount -- measured, 5-10
        # of the 30 brightest unexplained spots per position are reflections of
        # a domain already found, lost this way. Accept on Mahalanobis distance
        # in the (radial, transverse, normal) frame instead.
        s_r, s_t, s_n = (float(x) for x in sigma_rtn)
        axis = np.asarray(rotation_axis, float)
        axis = axis/np.linalg.norm(axis)
        qa = np.asarray(q, float)
        pred = (UB @ hi.T).T
        res = qa - pred
        nq = np.linalg.norm(qa, axis=1)
        d2 = np.zeros(len(qa))
        for i in range(len(qa)):
            if nq[i] < 1e-12:
                d2[i] = np.inf; continue
            qh = qa[i]/nq[i]
            t = np.cross(axis, qh); nt = np.linalg.norm(t)
            if nt < 1e-9:
                d2[i] = float(np.dot(res[i], res[i]))/s_r**2
                continue
            t = t/nt; n = np.cross(qh, t)
            d2[i] = ((float(res[i] @ qh)/s_r)**2 + (float(res[i] @ t)/s_t)**2
                     + (float(res[i] @ n)/s_n)**2)
        resid = np.sqrt(d2)
        within = resid < tol_sigma
    elif tol_q is None:
        resid = np.abs(h - hi).max(axis=1)
        within = resid < tol
    else:
        # A fractional-index tolerance is ANISOTROPIC: tol=0.10 is 0.10*|c*| =
        # 0.033 1/A along c but 0.10*|a*| = 0.174 1/A in plane, five times
        # looser, so the match count is flat in `a` across +/-3% and cannot
        # constrain it at all. A distance in reciprocal space is isotropic and
        # has a physical scale -- the measured q residual rms is 0.037 1/A,
        # instrument-limited by the omega binning.
        dq = np.linalg.norm(np.asarray(q, float) - (UB @ hi.T).T, axis=1)
        resid = dq
        within = dq < tol_q
    claim = ((within)
             & (np.abs(hi[:, 0]) <= hmax) & (np.abs(hi[:, 1]) <= hmax)
             & (np.abs(hi[:, 2]) <= lmax)
             & (np.abs(hi).sum(axis=1) > 0)
             & (np.abs(hi.sum(axis=1)) % 2 < 1e-6))
    if return_residual:
        return claim, hi.astype(int), resid
    return claim, hi.astype(int)


def unique_by_hkl(claim, hkl, residual):
    """Keep ONE claimed spot per reflection: the best-fitting one.

    ``find_blobs_3d`` splits a long streak into fragments and each fragment then
    satisfies :func:`match_mask` for the same (h,k,l). Measured on La3Ni2O7 in a
    DAC that inflated per-domain claim counts by 10-19 % -- (0,0,6) claimed three
    times at one position -- and, worse, made a least-squares over all claimed
    spots weight the brightest, most streaked rows several times over. That is
    precisely where a spurious a/b splitting would come from, so this runs before
    any count or fit, not after.

    ``omega_smear_duplicates`` does not cover this: it compares a NEW domain's
    claims against PREVIOUSLY accepted domains, never within one domain.

    The survivor is the fragment whose fractional index is closest to integer,
    i.e. the piece of the streak sitting nearest the exact Bragg condition.

    Parameters
    ----------
    claim, hkl, residual
        As returned by ``match_mask(..., return_residual=True)``.

    Returns
    -------
    numpy.ndarray
        Boolean mask, a subset of ``claim`` with no repeated hkl.
    """
    claim = np.asarray(claim, bool)
    hkl = np.asarray(hkl, int)
    residual = np.asarray(residual, float)
    if hkl.ndim != 2 or hkl.shape[1] != 3:
        raise ValueError("hkl must be (n, 3)")
    if not (claim.shape[0] == hkl.shape[0] == residual.shape[0]):
        raise ValueError("claim, hkl and residual must have the same length")
    out = np.zeros_like(claim)
    idx = np.flatnonzero(claim)
    if not idx.size:
        return out
    best = {}
    for i in idx:
        key = (int(hkl[i, 0]), int(hkl[i, 1]), int(hkl[i, 2]))
        j = best.get(key)
        if j is None or residual[i] < residual[j]:
            best[key] = i
    out[list(best.values())] = True
    return out


def omega_smear_duplicates(hkl_new, rc_new, frame_new,
                           hkl_old, rc_old, frame_old,
                           *, px_tol=80.0, frame_tol=20):
    """Which of a candidate's spots are an EXISTING domain's reflection,
    cut into blobs along omega and re-indexed.

    THIS IS THE MAIN FALSE-DOMAIN MECHANISM. At p=116, 5 of 8 'domain 2' spots
    carried the IDENTICAL hkl to a domain-1 spot 5.9-68 px and 6-17 frames away,
    and the two 'independent ladders' there shared the (002) reflection 5.9 px
    apart. `find_blobs_3d` split one omega-smeared reflection and the indexer
    called the pieces a second grain.

    A genuine second domain may share an hkl LABEL with the first, but not at
    the same place on the detector.
    """
    dup = np.zeros(len(hkl_new), bool)
    if len(hkl_old) == 0:
        return dup
    ho = np.abs(np.asarray(hkl_old))
    for i, (h, rc, f) in enumerate(zip(np.abs(np.asarray(hkl_new)),
                                       rc_new, frame_new)):
        same = np.all(ho == h, axis=1)
        if not same.any():
            continue
        d = np.hypot(rc_old[same, 0] - rc[0], rc_old[same, 1] - rc[1])
        df = np.abs(frame_old[same] - f)
        if np.any((d < px_tol) & (df <= frame_tol)):
            dup[i] = True
    return dup
