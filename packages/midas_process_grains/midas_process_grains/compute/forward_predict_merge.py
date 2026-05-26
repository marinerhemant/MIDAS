"""Forward-prediction-based merge primitive with variant-agreement.

Motivation
----------
Misori-only clustering can fail on highly twinned datasets in two ways:

1. **Giant component**: chains of near-FZ-boundary candidates merge a
   physically heterogeneous population into one component (observed on
   xzhang LMO: ~250k alive candidates collapse into a single 247k-member
   "grain").
2. **Refiner asymmetry**: indexer-refiner pipelines do not produce
   symmetric matched-spot lists across same-grain candidates (cand A
   has spot S in its matched list; cand B has spot S in its matched
   list with a different attribution; mutual-spot edge tests fail).

This module sidesteps both problems by treating the **forward model**
as the source of truth. For each alive candidate, the canonical
:class:`midas_diffract.forward.HEDMForwardModel` predicts the 2V spot
positions on the indexing ring (V variants × 2 Friedel branches),
**naturally labelled by variant index**. Predictions snap to the
nearest detected spot via a 3D KDTree on (Y, Z, ω·scale); the
attribution map records "candidate C claims spot S as variant V."

Edge rule (per candidate pair):

* **Merge edge**: ``agree_count >= K_AGREE`` AND ``disagree_count == 0``
* **Twin edge**: ``disagree_count >= K_TWIN``

The pair-level discriminator is symmetric by construction
(both candidates predict the same V-axis from their FZ-canonical OM).

Empirical K_AGREE
-----------------
On Ni FCC (Indrajeet, 57k alives) ``K_AGREE=4`` produces 12,576
components vs misori-v4's 11,591 — same-grain agreement is recovered
with no giant component (max-size 117 vs misori's 1k+). On highly-
twinned LMO (xzhang, 250k alives) the same ``K_AGREE=4`` removes the
247k-member giant component without otherwise changing topology.

Performance
-----------
Vectorised 3D KDTree snap + ``defaultdict(int)`` pair-loop. ~5 s
on Indrajeet (57k alives, 478k valid preds) and ~60 s on xzhang
(250k alives, 4.3M valid preds), no Python-loop in the hot path.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


__all__ = [
    "ForwardPredictAttribution",
    "ForwardPredictGraph",
    "compute_forward_predict_attributions",
    "build_forward_predict_graph",
    "build_forward_predict_graph_multi_ring",
    "forward_predict_merge_components",
    "select_k_agree_auto",
    "select_om_spread_tol_auto",
    "split_components_by_om_spread",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class ForwardPredictAttribution:
    """Per-(candidate, detected_spot) variant attribution from forward-prediction.

    Each row represents a single attribution: candidate ``cand_idx[i]``
    forward-predicts a spot on the indexing ring as variant
    ``variant_idx[i]`` that snaps to the detected ring spot at
    ``spot_idx[i]`` (whose ``InputAll.SpotID`` is ``spot_id[i]``).
    """

    cand_idx: np.ndarray       # (M,) int32
    spot_idx: np.ndarray       # (M,) int32 — index into the detected ring spot list
    variant_idx: np.ndarray    # (M,) int8 — variant on indexing ring
    spot_id: np.ndarray        # (M,) int32 — InputAll SpotID for the snapped detection
    n_alive: int
    n_variants: int
    snap_rate: float           # fraction of valid predictions that snapped


@dataclass
class ForwardPredictGraph:
    """Pair-level agree/disagree counts on the indexing ring.

    Each pair ``(pair_a[p], pair_b[p])`` has been observed at least once
    (either agree or disagree). Pair indices are unordered (``pair_a < pair_b``).
    """

    pair_a: np.ndarray         # (P,) int32 — smaller candidate id
    pair_b: np.ndarray         # (P,) int32 — larger candidate id
    agree_count: np.ndarray    # (P,) int16
    disagree_count: np.ndarray # (P,) int16


# ---------------------------------------------------------------------------
# Stage A — predict + snap
# ---------------------------------------------------------------------------


def compute_forward_predict_attributions(
    om_fz: np.ndarray,
    positions: np.ndarray,
    *,
    g_crystal_ring: np.ndarray,
    theta_deg_ring: np.ndarray,
    geometry,
    detected_y_um: np.ndarray,
    detected_z_um: np.ndarray,
    detected_omega_deg: np.ndarray,
    detected_spot_id: np.ndarray,
    y_tol_um: float = 800.0,
    omega_tol_deg: float = 0.5,
    device: Optional[object] = None,
) -> ForwardPredictAttribution:
    """Forward-predict ring spots for each FZ-canonical OM and snap to detections.

    Parameters
    ----------
    om_fz : (N, 3, 3) float64
        FZ-canonicalised orientation matrices.
    positions : (N, 3) float64
        Grain centres in lab µm (matches OPF columns 11:14).
    g_crystal_ring : (V, 3) float64
        Crystal-frame g-vectors for the variants on the indexing ring.
    theta_deg_ring : (V,) float64
        Bragg angle in degrees per variant.
    geometry : :class:`midas_diffract.forward.HEDMGeometry`
        Detector + scan geometry (must include wavelength, BC, px, tilts).
    detected_y_um, detected_z_um : (D,) float64
        Detected ring-spot positions in lab µm (InputAll YLab / ZLab).
    detected_omega_deg : (D,) float64
        Detected ω in degrees.
    detected_spot_id : (D,) int64
        InputAll SpotID per detected spot (carried through for downstream
        spot-attribution joins).
    y_tol_um : float, optional
        Snap radius in (Y, Z, ω·scale) µm. Defaults to 800 µm (≈4 px at
        200 µm pixel) which matches the historical xzhang/Indrajeet
        diagnostics.
    omega_tol_deg : float, optional
        Effective ω matching tolerance — multiplied into the KDTree
        scaling so that ``Y_TOL/OMEGA_TOL`` µm per degree maps ω into
        the 3D distance metric. Defaults to 0.5°.
    device : torch.device or None
        Forwarded to ``HEDMForwardModel`` (None → CPU).

    Returns
    -------
    ForwardPredictAttribution
    """
    import torch  # local import: keeps the module importable without torch
    from midas_diffract.forward import HEDMForwardModel
    from scipy.spatial import cKDTree

    n_alive = int(om_fz.shape[0])
    n_v = int(g_crystal_ring.shape[0])
    if n_alive == 0 or n_v == 0:
        return ForwardPredictAttribution(
            cand_idx=np.zeros(0, dtype=np.int32),
            spot_idx=np.zeros(0, dtype=np.int32),
            variant_idx=np.zeros(0, dtype=np.int8),
            spot_id=np.zeros(0, dtype=np.int32),
            n_alive=n_alive,
            n_variants=n_v,
            snap_rate=0.0,
        )

    dev = device if device is not None else torch.device("cpu")
    model = HEDMForwardModel(
        hkls=torch.from_numpy(g_crystal_ring.astype(np.float64)),
        thetas=torch.from_numpy(np.deg2rad(theta_deg_ring.astype(np.float64))),
        geometry=geometry,
        device=dev,
    )
    om_t = torch.from_numpy(np.ascontiguousarray(om_fz, dtype=np.float64))
    omega_rad, eta_rad, two_theta, valid_b = model.calc_bragg_geometry(
        orientation_matrices=om_t,
    )
    pos_t = torch.from_numpy(np.ascontiguousarray(positions, dtype=np.float64))
    spots = model.project_to_detector(
        omega=omega_rad, eta=eta_rad, two_theta=two_theta,
        positions=pos_t, valid=valid_b,
    )
    y_pix = spots.y_pixel.detach().cpu().numpy()
    z_pix = spots.z_pixel.detach().cpu().numpy()
    valid = spots.valid.detach().cpu().numpy().astype(bool)
    om_d = np.degrees(omega_rad.detach().cpu().numpy())
    if y_pix.ndim == 3:
        # Distance dimension (D=1 for FF)
        y_pix = y_pix[0]; z_pix = z_pix[0]; valid = valid[0]

    # midas_diffract concatenates branches as [branch_a, branch_b] along the
    # orientation axis, so shape is (2N, M). Stack (NOT reshape) preserves order.
    yp = np.stack([y_pix[:n_alive], y_pix[n_alive:]], axis=1)   # (N, 2, V)
    zp = np.stack([z_pix[:n_alive], z_pix[n_alive:]], axis=1)
    vv = np.stack([valid[:n_alive], valid[n_alive:]], axis=1)
    om = np.stack([om_d[:n_alive], om_d[n_alive:]], axis=1)

    # Pixels → µm in InputAll convention (flip_y=True). The y_BC/z_BC come
    # from the HEDMGeometry instance.
    px = float(geometry.px) if hasattr(geometry, "px") else float(geometry.pixel_um)
    y_um = (float(geometry.y_BC) - yp) * px
    z_um = (zp - float(geometry.z_BC)) * px

    # Flatten (N, 2, V) → (N, 2V); the variant index in the flat axis is
    # ``arange(V)`` tiled twice (branch 0 and branch 1 share variants).
    n_preds_per = 2 * n_v
    y_flat = y_um.reshape(n_alive, n_preds_per)
    z_flat = z_um.reshape(n_alive, n_preds_per)
    o_flat = om.reshape(n_alive, n_preds_per)
    v_flat = vv.reshape(n_alive, n_preds_per)
    variant_idx_flat = np.tile(np.arange(n_v, dtype=np.int16), 2)

    # Wrap ω into [-180, 180]
    for _ in range(4):
        o_flat = np.where(o_flat < -180.0, o_flat + 360.0, o_flat)
        o_flat = np.where(o_flat >  180.0, o_flat - 360.0, o_flat)

    # 3D KDTree on (Y, Z, ω·scale): scale ω so 1° ≈ y_tol/omega_tol µm,
    # making the L2 ball in (Y, Z, ω·scale) equivalent to a (Y_TOL, Z_TOL,
    # OMEGA_TOL) box for typical near-on-tolerance hits.
    omega_scale = y_tol_um / max(omega_tol_deg, 1e-9)
    det_yzo = np.column_stack([
        detected_y_um.astype(np.float64),
        detected_z_um.astype(np.float64),
        detected_omega_deg.astype(np.float64) * omega_scale,
    ])
    det_tree = cKDTree(det_yzo)

    pred_valid_mask = v_flat.reshape(-1)
    pred_yzo = np.column_stack([
        y_flat.reshape(-1),
        z_flat.reshape(-1),
        o_flat.reshape(-1) * omega_scale,
    ])
    dist, nearest_det = det_tree.query(pred_yzo, k=1, distance_upper_bound=y_tol_um)
    hit = (dist < y_tol_um) & pred_valid_mask
    snap_rate = float(hit.sum() / max(pred_valid_mask.sum(), 1))

    cand_idx_flat = np.repeat(np.arange(n_alive, dtype=np.int32), n_preds_per)
    var_idx_flat = np.tile(variant_idx_flat, n_alive)

    hit_cand = cand_idx_flat[hit]
    hit_var = var_idx_flat[hit].astype(np.int8)
    hit_det = nearest_det[hit].astype(np.int32)
    # If multiple predictions of a candidate snap to the same detection
    # (Friedel pair), keep the first occurrence's variant.
    ord_ = np.lexsort((hit_det, hit_cand))
    hc = hit_cand[ord_]; hv = hit_var[ord_]; hd = hit_det[ord_]
    if len(hc) > 0:
        keep_mask = np.empty(len(hc), dtype=bool)
        keep_mask[0] = True
        keep_mask[1:] = (hc[1:] != hc[:-1]) | (hd[1:] != hd[:-1])
        hc = hc[keep_mask]; hv = hv[keep_mask]; hd = hd[keep_mask]
    hs = detected_spot_id[hd].astype(np.int32)

    return ForwardPredictAttribution(
        cand_idx=hc, spot_idx=hd, variant_idx=hv, spot_id=hs,
        n_alive=n_alive, n_variants=n_v, snap_rate=snap_rate,
    )


# ---------------------------------------------------------------------------
# Stage B — agree/disagree pair counts
# ---------------------------------------------------------------------------


def build_forward_predict_graph(
    attrib: ForwardPredictAttribution,
) -> ForwardPredictGraph:
    """Build the pair-level agree/disagree counts from an attribution map.

    For each detected spot claimed by ≥2 candidates, every pair contributes
    +1 to ``agree`` if they attribute the spot to the same variant index,
    or +1 to ``disagree`` otherwise.
    """
    # Group attributions by spot_idx
    if attrib.cand_idx.size == 0:
        return ForwardPredictGraph(
            pair_a=np.zeros(0, dtype=np.int32),
            pair_b=np.zeros(0, dtype=np.int32),
            agree_count=np.zeros(0, dtype=np.int16),
            disagree_count=np.zeros(0, dtype=np.int16),
        )

    order = np.argsort(attrib.spot_idx, kind="stable")
    spot_sorted = attrib.spot_idx[order]
    cand_sorted = attrib.cand_idx[order]
    var_sorted = attrib.variant_idx[order]

    # Find run-length boundaries
    diffs = np.diff(spot_sorted)
    starts = np.concatenate(([0], np.flatnonzero(diffs) + 1))
    ends = np.concatenate((starts[1:], [len(spot_sorted)]))

    agree = defaultdict(int)
    disagree = defaultdict(int)
    for s, e in zip(starts, ends):
        n = e - s
        if n < 2:
            continue
        cands = cand_sorted[s:e]
        vars_ = var_sorted[s:e]
        for i in range(n):
            ca, va = int(cands[i]), int(vars_[i])
            for j in range(i + 1, n):
                cb, vb = int(cands[j]), int(vars_[j])
                key = (ca, cb) if ca < cb else (cb, ca)
                if va == vb:
                    agree[key] += 1
                else:
                    disagree[key] += 1

    pair_keys = set(agree.keys()) | set(disagree.keys())
    n_pairs = len(pair_keys)
    pair_a = np.empty(n_pairs, dtype=np.int32)
    pair_b = np.empty(n_pairs, dtype=np.int32)
    a_arr = np.empty(n_pairs, dtype=np.int16)
    d_arr = np.empty(n_pairs, dtype=np.int16)
    for idx, key in enumerate(pair_keys):
        pair_a[idx] = key[0]
        pair_b[idx] = key[1]
        a_arr[idx] = min(agree.get(key, 0), np.iinfo(np.int16).max)
        d_arr[idx] = min(disagree.get(key, 0), np.iinfo(np.int16).max)

    return ForwardPredictGraph(
        pair_a=pair_a, pair_b=pair_b,
        agree_count=a_arr, disagree_count=d_arr,
    )


# ---------------------------------------------------------------------------
# Stage C — connected components / twin edges
# ---------------------------------------------------------------------------


def forward_predict_merge_components(
    graph: ForwardPredictGraph,
    n_alive: int,
    *,
    k_agree: int = 4,
    k_twin: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Connected-component clustering on the merge subgraph.

    Parameters
    ----------
    graph : ForwardPredictGraph
    n_alive : int
        Total number of alive candidates (= length of returned labels).
    k_agree : int, optional
        Minimum same-variant agreement count to fire a merge edge.
        Empirically ``4`` is the smallest value that breaks the
        chain-fusion giant component on highly-twinned datasets.
    k_twin : int or None
        Minimum cross-variant disagreement count to fire a twin edge.
        Defaults to ``k_agree``.

    Returns
    -------
    labels : (n_alive,) int64
        Cluster id per candidate, ``0..n_components-1``.
    twin_edges : (E_twin, 2) int32
        Unordered pairs (smaller, larger) of cluster ids connected by a
        twin edge. Self-loops removed.
    """
    if k_twin is None:
        k_twin = k_agree

    merge_mask = (graph.agree_count >= k_agree) & (graph.disagree_count == 0)
    twin_mask = graph.disagree_count >= k_twin

    parent = np.arange(n_alive, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(graph.pair_a[merge_mask], graph.pair_b[merge_mask]):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    roots = np.fromiter((find(i) for i in range(n_alive)), dtype=np.int64, count=n_alive)
    _, labels = np.unique(roots, return_inverse=True)
    labels = labels.astype(np.int64)

    # Twin edges between cluster pairs
    if twin_mask.any():
        ta = labels[graph.pair_a[twin_mask]]
        tb = labels[graph.pair_b[twin_mask]]
        keep = ta != tb
        ta, tb = ta[keep], tb[keep]
        small = np.minimum(ta, tb)
        large = np.maximum(ta, tb)
        pairs = np.column_stack([small, large])
        twin_edges = np.unique(pairs, axis=0).astype(np.int32)
    else:
        twin_edges = np.zeros((0, 2), dtype=np.int32)

    return labels, twin_edges


# ---------------------------------------------------------------------------
# Auto-K selection
# ---------------------------------------------------------------------------


def build_forward_predict_graph_multi_ring(
    attributions: list,
) -> ForwardPredictGraph:
    """Union per-ring attribution maps into a single agree/disagree graph.

    For each ring, ``variant_idx`` is unique only within that ring;
    we offset by ``sum(prior_n_variants)`` so the same numerical variant
    on different rings is recognised as DIFFERENT (else two ring-3
    variant-0's would falsely "agree"). Pair counts are then the sum
    across all rings.
    """
    if not attributions:
        return ForwardPredictGraph(
            pair_a=np.zeros(0, dtype=np.int32),
            pair_b=np.zeros(0, dtype=np.int32),
            agree_count=np.zeros(0, dtype=np.int16),
            disagree_count=np.zeros(0, dtype=np.int16),
        )
    # Offset variant ids per ring so they're globally unique
    rebased = []
    offset = 0
    for attrib in attributions:
        if attrib.cand_idx.size == 0:
            offset += attrib.n_variants
            continue
        rebased.append(ForwardPredictAttribution(
            cand_idx=attrib.cand_idx,
            spot_idx=attrib.spot_idx,
            variant_idx=(attrib.variant_idx.astype(np.int16) + offset).astype(np.int16),
            spot_id=attrib.spot_id,
            n_alive=attrib.n_alive, n_variants=attrib.n_variants,
            snap_rate=attrib.snap_rate,
        ))
        offset += attrib.n_variants
    if not rebased:
        return ForwardPredictGraph(
            pair_a=np.zeros(0, dtype=np.int32),
            pair_b=np.zeros(0, dtype=np.int32),
            agree_count=np.zeros(0, dtype=np.int16),
            disagree_count=np.zeros(0, dtype=np.int16),
        )
    # Concatenate per-ring attributions and offset spot_idx so different
    # rings don't get falsely merged on spot_id collisions
    cand = np.concatenate([a.cand_idx for a in rebased])
    var = np.concatenate([a.variant_idx for a in rebased])
    sid = np.concatenate([a.spot_id for a in rebased])
    # Per-ring spot offsets — append ring index high bits so spots from
    # different rings can never collide on the same detected ID
    spot_off = []
    base = 0
    for a in rebased:
        spot_off.append(a.spot_idx.astype(np.int64) + base)
        base += int(a.spot_idx.max() if a.spot_idx.size else 0) + 1
    spot = np.concatenate(spot_off).astype(np.int32)
    fused = ForwardPredictAttribution(
        cand_idx=cand, spot_idx=spot, variant_idx=var, spot_id=sid,
        n_alive=rebased[0].n_alive,
        n_variants=int(offset),
        snap_rate=float(np.mean([a.snap_rate for a in rebased])),
    )
    return build_forward_predict_graph(fused)


def select_om_spread_tol_auto(
    labels: np.ndarray,
    *,
    om_fz_quat: np.ndarray,
    space_group: int,
    floor_deg: float = 0.5,
    ceil_deg: float = 5.0,
) -> float:
    """Data-driven OM-spread tolerance via misori-histogram antimode.

    Strategy: sample up to 200 multi-cand components, compute the
    within-component pairwise misorientation distribution, and find the
    smallest local minimum in the log-misori histogram between two
    local maxima (the same antimode-finder logic as
    :mod:`compute.adaptive`). Clamped to ``[floor_deg, ceil_deg]``.

    If the histogram is unimodal (refiner noise only, no chain-fusion
    second mode), returns ``floor_deg``. If it's degenerate (no
    multi-cand components), returns ``floor_deg`` too — the OM-split
    has nothing to do.
    """
    import torch
    from scipy.ndimage import gaussian_filter1d
    from midas_stress.orientation import misorientation_quat_batch

    order = np.argsort(labels, kind="stable")
    lbl_sorted = labels[order]
    breaks = np.concatenate([
        [0], np.flatnonzero(np.diff(lbl_sorted)) + 1, [len(labels)],
    ])
    multi_starts = []
    for k in range(len(breaks) - 1):
        if breaks[k+1] - breaks[k] >= 2:
            multi_starts.append((int(breaks[k]), int(breaks[k+1])))
    if not multi_starts:
        return floor_deg
    rng = np.random.default_rng(0)
    if len(multi_starts) > 200:
        idx = rng.choice(len(multi_starts), 200, replace=False)
        multi_starts = [multi_starts[i] for i in idx]

    om = np.ascontiguousarray(om_fz_quat, dtype=np.float64)
    miso_all: list[float] = []
    for s, e in multi_starts:
        members = order[s:e]
        n_m = len(members)
        if n_m > 30:
            members = members[rng.choice(n_m, 30, replace=False)]
            n_m = 30
        ii, jj = np.triu_indices(n_m, k=1)
        qa = torch.from_numpy(np.ascontiguousarray(om[members[ii]]))
        qb = torch.from_numpy(np.ascontiguousarray(om[members[jj]]))
        miso_all.extend(np.rad2deg(misorientation_quat_batch(qa, qb, space_group).numpy()).tolist())
    if not miso_all:
        return floor_deg
    miso_arr = np.asarray(miso_all)
    log_m = np.log10(np.clip(miso_arr, 1e-3, None))
    hist, edges = np.histogram(log_m, bins=80)
    mids = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
    is_max = np.zeros_like(smooth, dtype=bool)
    is_max[1:-1] = (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:])
    is_min = np.zeros_like(smooth, dtype=bool)
    is_min[1:-1] = (smooth[1:-1] < smooth[:-2]) & (smooth[1:-1] < smooth[2:])
    max_idx = np.flatnonzero(is_max)
    min_idx = np.flatnonzero(is_min)
    if len(max_idx) < 2:
        return floor_deg
    lo_i, hi_i = int(max_idx[0]), int(max_idx[-1])
    between = (min_idx > lo_i) & (min_idx < hi_i)
    if not between.any():
        return floor_deg
    cand = min_idx[between]
    valley = int(cand[np.argmin(smooth[cand])])
    tol = float(10 ** mids[valley])
    return float(np.clip(tol, floor_deg, ceil_deg))


def split_components_by_om_spread(
    labels: np.ndarray,
    *,
    om_fz_quat: np.ndarray,
    space_group: int,
    om_tol_deg: float = 1.0,
) -> np.ndarray:
    """Split each merge component into sub-clusters of OM-consistent members.

    Why
    ---
    The agree/disagree pair graph only sees pairs that share at least
    one snapped detection. Union-find takes the transitive closure of
    merge edges, so two candidates ``A`` and ``C`` that have NO shared
    snaps can end up in the same component via a chain ``A↔B↔C``. The
    auto-K rule prevents the catastrophic chain-fusion (the 247k giant
    component on xzhang) but not the moderate one — on Indrajeet,
    ~20 % of multi-candidate components span >5° of symmetry-aware
    misorientation.

    Fix
    ---
    For each component, build a single-link sub-clustering on the
    pairwise symmetry-aware misorientation matrix at threshold
    ``om_tol_deg`` (default 1.0°, ≈ 5× typical refiner noise). Members
    that don't link to anyone else under this gate become their own
    sub-cluster. Singletons and 2-member components are skipped (single
    pair already has at most one link).

    Returns
    -------
    new_labels : (n_alive,) int64
        Cluster labels after the split. Singleton components keep their
        original label; multi-cand components may be expanded into one
        or more labels.
    """
    import torch
    from midas_stress.orientation import misorientation_quat_batch

    n_alive = len(labels)
    if n_alive == 0:
        return labels.copy()

    # Identify multi-member components
    order = np.argsort(labels, kind="stable")
    lbl_sorted = labels[order]
    breaks = np.concatenate([
        [0],
        np.flatnonzero(np.diff(lbl_sorted)) + 1,
        [n_alive],
    ])
    n_components = len(breaks) - 1
    new_labels = np.copy(labels).astype(np.int64)
    next_label = int(labels.max() + 1) if n_alive else 0

    om_fz_quat = np.ascontiguousarray(om_fz_quat, dtype=np.float64)

    for k in range(n_components):
        s, e = int(breaks[k]), int(breaks[k+1])
        if e - s < 2:
            continue  # singleton — nothing to split
        members = order[s:e]  # global indices in this component
        n_m = len(members)
        # All-pairs symmetry-aware misori, vectorised through midas_stress
        ii, jj = np.triu_indices(n_m, k=1)
        qa = torch.from_numpy(np.ascontiguousarray(om_fz_quat[members[ii]]))
        qb = torch.from_numpy(np.ascontiguousarray(om_fz_quat[members[jj]]))
        miso_deg = np.rad2deg(misorientation_quat_batch(qa, qb, space_group).numpy())
        # Single-link sub-clustering under om_tol_deg
        parent = np.arange(n_m, dtype=np.int64)
        def find_local(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        keep = miso_deg < om_tol_deg
        for li, lj in zip(ii[keep], jj[keep]):
            ra, rb = find_local(int(li)), find_local(int(lj))
            if ra != rb:
                parent[ra] = rb
        local_roots = np.fromiter((find_local(i) for i in range(n_m)), dtype=np.int64)
        _, local_subs = np.unique(local_roots, return_inverse=True)
        if local_subs.max() == 0:
            continue  # one sub-cluster — no split needed
        original_label = int(labels[members[0]])
        # Sub-cluster 0 keeps the original label; the rest get fresh labels.
        for sub in range(1, int(local_subs.max()) + 1):
            sub_members = members[local_subs == sub]
            new_labels[sub_members] = next_label
            next_label += 1
        # sub == 0 already has original_label

    # Compact label IDs so they remain 0..n_components-1
    _, new_labels = np.unique(new_labels, return_inverse=True)
    return new_labels.astype(np.int64)


def select_k_agree_auto(
    graph: ForwardPredictGraph,
    n_alive: int,
    *,
    k_min: int = 3,
    k_max: int = 12,
    giant_fraction: float = 0.01,
    giant_floor: int = 100,
    min_alive_for_auto: int = 100,
    min_pairs_for_auto: int = 10,
    fallback_k: int = 4,
) -> int:
    """Pick the smallest ``K_AGREE`` that breaks the chain-fusion giant component.

    Empirically the merge-edge graph contains a chain-fusion artifact
    for small ``K_AGREE``: weak two- or three-spot variant agreements
    chain together across the FZ to merge physically distinct grains
    into a single component. The artifact disappears abruptly above a
    dataset-dependent threshold (``K=4`` for cubic-Ni-style data,
    ``K=5`` for heavily-twinned LMO-style data). This routine returns
    the smallest ``K`` in ``[k_min, k_max]`` for which the largest
    connected component is bounded by
    ``max(giant_floor, n_alive * giant_fraction)``.

    Guard rails for small / sparse datasets:
      - ``n_alive < min_alive_for_auto`` (default 100): chain-fusion
        cannot manifest at scale; returns ``fallback_k``.
      - ``len(graph.pair_a) < min_pairs_for_auto`` (default 10): graph
        too sparse to discriminate; returns ``fallback_k``.
      - Every K in ``[k_min, k_max]`` fails the threshold: returns
        ``k_max``.

    Parameters
    ----------
    graph : ForwardPredictGraph
    n_alive : int
    k_min, k_max : int
        Inclusive search bounds.
    giant_fraction : float
        Default 0.01 (1%).
    giant_floor : int
        Absolute floor on biggest-allowed component. Default 100.
    min_alive_for_auto, min_pairs_for_auto, fallback_k : guard-rail args.

    Returns
    -------
    int
        Selected ``K_AGREE``.
    """
    if n_alive < min_alive_for_auto:
        return fallback_k
    if int(graph.pair_a.size) < min_pairs_for_auto:
        return fallback_k
    threshold = max(giant_floor, int(n_alive * giant_fraction))
    for K in range(k_min, k_max + 1):
        labels, _ = forward_predict_merge_components(graph, n_alive, k_agree=K)
        _, sizes = np.unique(labels, return_counts=True)
        biggest = int(sizes.max()) if sizes.size else 0
        if biggest <= threshold:
            return K
    return k_max
