"""Defect statistical tests on the diffuse field.

Three tests that turn the classified diffuse signal into defect statements, each
ported from the validated demk re-analysis and made phase-agnostic (driven by a
`midas_hkls.Crystal`):

  - `forbidden_reflection_test`  — intensity at systematically-absent positions
    vs allowed, with a matched off-lattice control. Excess ⇒ APB /
    selection-rule-breaking defect. (FCC demk: ≈ 0, no excess.)
  - `rod_family_enrichment`      — for diffuse rod-window voxels, is Δq (offset
    from the nearest Bragg point) aligned with a low-index direction family more
    than an isotropic null predicts? (FCC demk: only ⟨111⟩ enriched.)
  - `fault_probability_alpha`    — per-grain α-proxy = I_rod / (I_rod + I_Bragg)
    along the fault family. (FCC demk: median ≈ 0.005.)

All three use scipy cKDTree for the (discrete, off-gradient-path) spatial sums —
the contract-sanctioned home for numpy/scipy. The "allowed" / "forbidden" sets
come from `SpaceGroup.is_systematically_absent`, so the FCC all-even/all-odd rule
(or any other) is never hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import math
import numpy as np

from .bragg_diffuse import enumerate_hkls


# --- cubic direction families (defaults for the rod test) -------------------
CUBIC_FAMILIES: Dict[str, np.ndarray] = {
    "<111>": np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], float),
    "<110>": np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
                       [0, 1, 1], [0, 1, -1]], float),
    "<100>": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float),
}


def _orientations(orientations) -> np.ndarray:
    U = np.asarray(orientations, dtype=np.float64)
    if U.ndim == 2 and U.shape[-1] == 9:
        U = U.reshape(-1, 3, 3)
    elif U.shape == (3, 3):
        U = U.reshape(1, 3, 3)
    return U


def _g_crystal(hkls: np.ndarray, crystal) -> np.ndarray:
    """Crystal-frame G = (2π h/a, 2π k/a, 2π l/c) for an orthogonal cell."""
    a = float(crystal.lattice.a)
    c = float(crystal.lattice.c)
    twopi = 2.0 * math.pi
    G = np.empty_like(hkls, dtype=np.float64)
    G[:, 0] = twopi * hkls[:, 0] / a
    G[:, 1] = twopi * hkls[:, 1] / a
    G[:, 2] = twopi * hkls[:, 2] / c
    return G


def _forbidden_hkls(crystal, *, q_max_inv_A: float, q_min_inv_A: float) -> np.ndarray:
    """Systematically-absent (h,k,l) in the |q| band — the 'forbidden' set."""
    sg = crystal.space_group
    a = float(crystal.lattice.a)
    c = float(crystal.lattice.c)
    twopi = 2.0 * math.pi
    h_max = int(math.ceil(q_max_inv_A * max(a, c) / twopi)) + 1
    out = []
    for h in range(-h_max, h_max + 1):
        for k in range(-h_max, h_max + 1):
            for l in range(-h_max, h_max + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                qmag = twopi * math.sqrt((h / a) ** 2 + (k / a) ** 2 + (l / c) ** 2)
                if not (q_min_inv_A < qmag < q_max_inv_A):
                    continue
                if sg.is_systematically_absent(h, k, l):
                    out.append((h, k, l))
    return np.asarray(out, dtype=np.int64)


@dataclass
class ForbiddenTest:
    forbidden_over_allowed_median: float
    control_over_allowed_median: float
    excess_median: float                 # forbidden − control (>0 ⇒ real)
    n_grains_excess: int                  # grains with forbidden > 1.5× control
    n_grains: int
    per_grain_ratio: np.ndarray
    per_grain_control: np.ndarray


def forbidden_reflection_test(
    q_sample: np.ndarray,
    intensity: np.ndarray,
    orientations,
    crystal,
    *,
    samp_radius_inv_A: float = 0.04,
    q_min_inv_A: float = 1.5,
    q_max_inv_A: float = 8.0,
    seed: int = 0,
) -> ForbiddenTest:
    """Selection-rule test: intensity at forbidden vs allowed reflections.

    For each grain, sum the diffuse intensity within ``samp_radius`` of its
    allowed reflections, its forbidden (systematically-absent) reflections, and
    a matched off-lattice random control at the forbidden |q| values. Excess of
    forbidden over control flags an APB / stacking defect that breaks the
    selection rule.
    """
    from scipy.spatial import cKDTree

    q = np.asarray(q_sample, dtype=np.float64)
    I = np.asarray(intensity, dtype=np.float64)
    U = _orientations(orientations)
    tree = cKDTree(q)

    def I_at(pts: np.ndarray) -> np.ndarray:
        nb = tree.query_ball_point(pts, samp_radius_inv_A, workers=-1)
        return np.array([I[ii].sum() for ii in nb])

    allowed_hkl = enumerate_hkls(crystal, q_max_inv_A=q_max_inv_A)
    # restrict allowed to the same |q| band as forbidden for a fair ratio
    Ga_all = _g_crystal(allowed_hkl, crystal)
    qa = np.linalg.norm(Ga_all, axis=1)
    Ga = Ga_all[(qa > q_min_inv_A) & (qa < q_max_inv_A)]
    Gf = _g_crystal(_forbidden_hkls(crystal, q_max_inv_A=q_max_inv_A,
                                    q_min_inv_A=q_min_inv_A), crystal)

    rng = np.random.default_rng(seed)
    ratios, ctrls = [], []
    for g in range(U.shape[0]):
        Pa = (U[g] @ Ga.T).T
        ia = I_at(Pa).mean()
        if Gf.shape[0] == 0:
            jf = 0.0
            ic = 0.0
        else:
            Pf = (U[g] @ Gf.T).T
            jf = I_at(Pf).mean()
            rv = rng.normal(size=Pf.shape)
            rv /= np.linalg.norm(rv, axis=1, keepdims=True)
            Pc = rv * np.linalg.norm(Pf, axis=1, keepdims=True)
            ic = I_at(Pc).mean()
        ratios.append(jf / max(ia, 1e-9))
        ctrls.append(ic / max(ia, 1e-9))
    ratios = np.asarray(ratios)
    ctrls = np.asarray(ctrls)
    return ForbiddenTest(
        forbidden_over_allowed_median=float(np.median(ratios)),
        control_over_allowed_median=float(np.median(ctrls)),
        excess_median=float(np.median(ratios) - np.median(ctrls)),
        n_grains_excess=int((ratios > 1.5 * np.maximum(ctrls, 1e-9)).sum()),
        n_grains=int(U.shape[0]),
        per_grain_ratio=ratios,
        per_grain_control=ctrls,
    )


@dataclass
class RodEnrichment:
    enrichment: Dict[str, float]          # family -> observed/null frac-within-window
    observed_frac: Dict[str, float]
    null_frac: Dict[str, float]
    n_candidates: int


def rod_family_enrichment(
    q_sample: np.ndarray,
    intensity: np.ndarray,
    orientations,
    crystal,
    *,
    families: Optional[Dict[str, np.ndarray]] = None,
    dq_window_inv_A: Tuple[float, float] = (0.12, 0.6),
    q_min_inv_A: float = 1.5,
    angle_window_deg: float = 15.0,
    q_max_inv_A: float = 8.5,
    bright_percentile: float = 98.0,
    seed: int = 0,
) -> RodEnrichment:
    """Are diffuse rods aligned with a low-index direction family?

    For every diffuse voxel whose offset from the nearest Bragg point lies in
    ``dq_window``, measure the angle of Δq to each grain's direction families,
    and compare the intensity-weighted fraction within ``angle_window`` to a
    matched isotropic null (which bakes in each family's direction count). An
    enrichment > 1 means rods preferentially align with that family.

    ``bright_percentile`` pre-selects the bright voxels where rods actually live
    (the validated demk analysis used the top ~2 %, i.e. 98.0). Without it the
    faint, radial dislocation-asterism background around strong reflections
    swamps the genuine inter-Bragg fault rods. Pass 0 to disable.

    .. warning::
       This nearest-Bragg metric is **confounded by reciprocal-lattice geometry**
       and is a *screening* tool, not a definitive fault-plane test. The Δq
       direction to the nearest reflection is dominated by the lattice
       connectivity (for FCC the reciprocal lattice is BCC, with its own
       ⟨100⟩/⟨110⟩/⟨111⟩ structure), so on real data the enrichment tracks the
       strong-reflection / inter-reflection directions rather than the fault
       normals (demk full-res: ⟨100⟩ ≫ ⟨111⟩). For a clean fault-rod test,
       sample intensity along each grain's *predicted* fault rods vs
       perpendicular controls (the explicit per-grain method) — that is the
       authoritative discriminator.
    """
    from scipy.spatial import cKDTree

    if families is None:
        families = CUBIC_FAMILIES
    q = np.asarray(q_sample, dtype=np.float64)
    I = np.asarray(intensity, dtype=np.float64)
    if bright_percentile and bright_percentile > 0:
        bright = I >= np.percentile(I, bright_percentile)
        q = q[bright]
        I = I[bright]
    U = _orientations(orientations)
    ng = U.shape[0]

    hkls = enumerate_hkls(crystal, q_max_inv_A=q_max_inv_A)
    G = _g_crystal(hkls, crystal)
    nref = G.shape[0]
    P = np.concatenate([(U[g] @ G.T).T for g in range(ng)], axis=0)
    tree = cKDTree(P)
    dist, ii = tree.query(q, k=1, workers=-1)
    gidx = ii // nref

    qmag = np.linalg.norm(q, axis=1)
    lo, hi = dq_window_inv_A
    cand = (dist >= lo) & (dist <= hi) & (qmag > q_min_inv_A)
    dq = q[cand] - P[ii[cand]]
    dqn = dq / np.linalg.norm(dq, axis=1, keepdims=True)
    gc = gidx[cand]
    w = I[cand]

    rng = np.random.default_rng(seed)
    rv = rng.normal(size=dqn.shape)
    rv /= np.linalg.norm(rv, axis=1, keepdims=True)

    def min_angle(dirs_vecs: np.ndarray, fam: np.ndarray) -> np.ndarray:
        ang = np.empty(len(dirs_vecs))
        for g in range(ng):
            sel = gc == g
            if not sel.any():
                continue
            D = (U[g] @ fam.T).T
            D = D / np.linalg.norm(D, axis=1, keepdims=True)
            c = np.abs(dirs_vecs[sel] @ D.T)
            ang[sel] = np.degrees(np.arccos(np.clip(c.max(1), 0, 1)))
        return ang

    enr, obs, nul = {}, {}, {}
    tot_w = max(w.sum(), 1e-30)
    for nm, fam in families.items():
        a_obs = min_angle(dqn, fam)
        a_nul = min_angle(rv, fam)
        fo = float((w * (a_obs < angle_window_deg)).sum() / tot_w)
        fn = float((w * (a_nul < angle_window_deg)).sum() / tot_w)
        obs[nm] = fo
        nul[nm] = fn
        enr[nm] = fo / max(fn, 1e-9)
    return RodEnrichment(enrichment=enr, observed_frac=obs, null_frac=nul,
                         n_candidates=int(cand.sum()))


@dataclass
class FaultRodAlignment:
    """Explicit per-grain fault-rod test: along-rod vs perpendicular intensity."""
    along_over_perp_median: float
    along_over_perp_mean: float
    frac_grains_enriched: float       # fraction with ratio > enriched_threshold
    per_grain_ratio: np.ndarray
    n_grains: int


def fault_rod_alignment(
    q_sample: np.ndarray,
    intensity: np.ndarray,
    orientations,
    crystal,
    *,
    fault_dirs: Optional[np.ndarray] = None,
    bragg_tol_inv_A: float = 0.10,
    q_min_inv_A: float = 1.5,
    q_max_inv_A: float = 8.0,
    samp_radius_inv_A: float = 0.06,
    offsets_inv_A: Optional[np.ndarray] = None,
    enriched_threshold: float = 1.2,
    seed: int = 0,
) -> FaultRodAlignment:
    """Authoritative ⟨111⟩ fault-rod test (the explicit per-grain method).

    For each grain, sample the diffuse field along that grain's *own* predicted
    fault-rod directions through each Bragg point (along-rod) vs perpendicular
    control directions of the same offset magnitude (perp). along/perp > 1 ⇒
    real {111} fault rods. Unlike `rod_family_enrichment`, this removes the
    nearest-Bragg assignment confound — it asks directly whether intensity
    extends along the predicted rod axis more than transverse to it.

    Returns the per-grain along/perp ratio distribution (demk: median ≈ 0.95,
    ~40 % of grains > 1.2 ⇒ real but heterogeneous faulting).
    """
    from scipy.spatial import cKDTree
    from .bragg_diffuse import predicted_reflection_points

    if fault_dirs is None:  # FCC {111} normals (4 unique)
        fault_dirs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]],
                              float) / math.sqrt(3.0)
    if offsets_inv_A is None:
        offsets_inv_A = np.arange(0.15, 0.55, 0.05)
    q = np.asarray(q_sample, dtype=np.float64)
    I = np.asarray(intensity, dtype=np.float64)
    U = _orientations(orientations)
    ng = U.shape[0]

    # diffuse field = voxels off the lattice (dist > tol), q > q_min
    P_all = predicted_reflection_points(U, crystal, q_max_inv_A=q_max_inv_A).numpy()
    dist, _ = cKDTree(P_all).query(q, k=1, workers=-1)
    qmag = np.linalg.norm(q, axis=1)
    diff = (dist > bragg_tol_inv_A) & (qmag > q_min_inv_A)
    qd = q[diff]; wd = I[diff]
    tree = cKDTree(qd)

    # per-grain reflections
    from .bragg_diffuse import enumerate_hkls
    G = _g_crystal(enumerate_hkls(crystal, q_max_inv_A=q_max_inv_A), crystal)

    def sample(pts):
        nb = tree.query_ball_point(pts, samp_radius_inv_A, workers=-1)
        return np.array([wd[ii].sum() for ii in nb])

    ratios = []
    for gi in range(ng):
        B = (U[gi] @ G.T).T
        B = B[np.linalg.norm(B, axis=1) < q_max_inv_A]
        if len(B) == 0:
            continue
        dirs = (U[gi] @ (fault_dirs * math.sqrt(3.0)).T).T
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        al, pe = [], []
        for v in dirs:
            tmp = np.array([1., 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1., 0])
            e1 = np.cross(v, tmp); e1 /= np.linalg.norm(e1)
            e2 = np.cross(v, e1)
            for t in offsets_inv_A:
                al.append(B + t * v); al.append(B - t * v)
                pe.append(B + t * e1); pe.append(B - t * e1)
                pe.append(B + t * e2); pe.append(B - t * e2)
        ai = sample(np.concatenate(al, 0)).mean()
        pi = sample(np.concatenate(pe, 0)).mean()
        if pi > 0:
            ratios.append(ai / pi)
    ratios = np.asarray(ratios)
    if len(ratios) == 0:
        return FaultRodAlignment(float("nan"), float("nan"), float("nan"),
                                 ratios, 0)
    return FaultRodAlignment(
        along_over_perp_median=float(np.median(ratios)),
        along_over_perp_mean=float(ratios.mean()),
        frac_grains_enriched=float((ratios > enriched_threshold).mean()),
        per_grain_ratio=ratios,
        n_grains=int(len(ratios)),
    )


@dataclass
class FaultAlpha:
    alpha_median: float
    alpha_mean: float
    faulted_third_median: float           # top tertile
    low_two_thirds_median: float
    per_grain_alpha: np.ndarray
    n_grains_scored: int


def fault_probability_alpha(
    q_sample: np.ndarray,
    intensity: np.ndarray,
    orientations,
    crystal,
    *,
    fault_family: Optional[np.ndarray] = None,
    samp_radius_inv_A: float = 0.04,
    rod_offsets_inv_A: Optional[np.ndarray] = None,
    seed: int = 0,
) -> FaultAlpha:
    """Per-grain fault-probability proxy α = I_rod / (I_rod + I_Bragg).

    For each grain's fault-family reflections (FCC: {111}), integrate diffuse
    intensity in the rod along the reflection direction (off the Bragg core) vs
    the Bragg-core intensity. In the small-α kinematic limit the rod fraction
    scales with the stacking-fault probability.
    """
    from scipy.spatial import cKDTree

    if fault_family is None:
        fault_family = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                                 [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]],
                                float)
    if rod_offsets_inv_A is None:
        rod_offsets_inv_A = np.arange(0.10, 0.45, 0.03)
    q = np.asarray(q_sample, dtype=np.float64)
    I = np.asarray(intensity, dtype=np.float64)
    U = _orientations(orientations)
    G = _g_crystal(np.asarray(fault_family, dtype=np.int64), crystal)
    tree = cKDTree(q)

    def Isum(pts: np.ndarray) -> float:
        nb = tree.query_ball_point(pts, samp_radius_inv_A, workers=-1)
        return float(np.array([I[i].sum() for i in nb]).sum())

    alphas = []
    for g in range(U.shape[0]):
        B = (U[g] @ G.T).T
        dirs = B / np.linalg.norm(B, axis=1, keepdims=True)
        Ib = Isum(B)
        rodpts = []
        for j in range(len(B)):
            for t in rod_offsets_inv_A:
                rodpts.append(B[j] + t * dirs[j])
                rodpts.append(B[j] - t * dirs[j])
        Ir = Isum(np.asarray(rodpts)) / (2 * len(rod_offsets_inv_A))
        if Ib > 0:
            alphas.append(Ir / (Ir + Ib))
    alphas = np.asarray(alphas)
    if len(alphas) == 0:
        return FaultAlpha(float("nan"), float("nan"), float("nan"),
                          float("nan"), alphas, 0)
    hi = alphas > np.percentile(alphas, 67)
    return FaultAlpha(
        alpha_median=float(np.median(alphas)),
        alpha_mean=float(alphas.mean()),
        faulted_third_median=float(np.median(alphas[hi])) if hi.any() else float("nan"),
        low_two_thirds_median=float(np.median(alphas[~hi])) if (~hi).any() else float("nan"),
        per_grain_alpha=alphas,
        n_grains_scored=int(len(alphas)),
    )
