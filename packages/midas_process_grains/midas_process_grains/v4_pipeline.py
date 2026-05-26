"""Single-entry-point v4 pipeline: Stages 1+2+3+4+5+6 + hierarchical emit.

Usage
-----

    from midas_process_grains.v4_pipeline import run_v4_pipeline
    paths = run_v4_pipeline(
        layer_dir="/path/to/LayerNr_1",
        out_dir="/path/to/v4_output",
        trust_scheme="strict",
        n_seeds=57192,  # rows in OrientPosFit.bin
    )

Returns a dict of paths to emitted artifacts (GrainsV4.csv,
GrainsV4_families.csv, GrainsV4.meta.json, grain_audit.csv).

Notes
-----
* Pass-1 misori clustering uses the patched
  :func:`compute.adaptive.derive_misori_tol` (smart antimode + 2-offset
  bucket prefilter). For unimodal datasets (no clean antimode) the
  threshold defaults to ``floor_deg = 0.05°``.
* Stage 6 (NNLS grain-size recompute) only runs on split clusters;
  unsplit grains keep their legacy GrainRadius.
* Stage 5 (twin / sub-grain labeling) runs on the final leaf grain
  table after all the above.

This is the canonical v4 entrypoint and the one a future ``--mode
physics`` CLI flag will dispatch to.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

from midas_stress.orientation import (
    orient_mat_to_quat, quat_to_orient_mat, fundamental_zone,
    make_symmetries, misorientation_quat_batch,
)


def _grab_paramstest_scalar(ps_path: Path, key: str) -> Optional[float]:
    for ln in open(ps_path):
        m = re.match(rf"^{key}\s+([^;#]+);?", ln)
        if m:
            try:
                return float(m.group(1).strip())
            except ValueError:
                return None
    return None


def _grab_paramstest_vsample(ps_path: Path) -> Optional[Dict[str, float]]:
    """Extract sample-volume parameters (Vsample / Rsample / Hbeam / DiscModel
    / DiscArea) from a paramstest.txt file.

    Returns a dict with keys ``Vsample``, ``Rsample``, ``Hbeam``,
    ``DiscModel``, ``DiscArea`` (all float). Missing keys default to 0.

    Used by :func:`_compute_radius_correction` to apply a multiplicative
    correction on inherited ``GrainRadius_naive`` when the paramstest
    ``Vsample`` differs from the ``Hbeam·π·Rsample²`` fallback that
    midas-transforms ``calc_radius`` used at C-pipeline time.
    """
    out = {"Vsample": 0.0, "Rsample": 0.0, "Hbeam": 0.0,
           "DiscModel": 0.0, "DiscArea": 0.0}
    try:
        with open(ps_path) as f:
            text = f.read()
        for key in out:
            m = re.search(rf"^{key}\s+([^;#]+);?", text, re.M)
            if m:
                try:
                    out[key] = float(m.group(1).strip().split()[0])
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass
    return out


def _back_solve_legacy_vgauge(layer_dir: Path, log) -> Optional[float]:
    """Back-solve the Vgauge that was used to compute ``GrainVolume`` in
    ``Radius_StartNr_*_EndNr_*.csv``.

    The legacy ``calc_radius`` formula is
    ``GrainVolume = 0.5 · m_hkl · ΔΘ · cos(Θ) · Vgauge · IntInt
                    / (NImgs · PowderInt)``,
    so we can invert it row-by-row and take the median.

    Returns Vgauge in µm³, or ``None`` if no Radius CSV / hkls.csv exists.
    """
    rad_files = list(layer_dir.glob("Radius_StartNr_*.csv"))
    hkl_file = layer_dir / "hkls.csv"
    if not rad_files or not hkl_file.exists():
        return None
    rad_file = rad_files[0]
    try:
        ring_counts: Dict[int, int] = {}
        with open(hkl_file) as fh:
            fh.readline()       # header
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    rn = int(float(parts[4]))
                    ring_counts[rn] = ring_counts.get(rn, 0) + 1
                except ValueError:
                    continue
        arr = np.loadtxt(rad_file, skiprows=1, max_rows=50000)
        if arr.size == 0:
            return None
        IntInt        = arr[:, 1]
        DeltaOmega_d  = arr[:, 11]
        NImgs         = arr[:, 12]
        RingNr        = arr[:, 13].astype(int)
        GrainVolume   = arr[:, 14]
        PowderInt     = arr[:, 16]
        Theta_d       = arr[:, 9]
        Eta_d         = arr[:, 10]
        d2r = np.pi / 180.0
        sin_th = np.sin(Theta_d * d2r); cos_th = np.cos(Theta_d * d2r)
        sin_dom = np.sin(DeltaOmega_d * d2r); cos_dom = np.cos(DeltaOmega_d * d2r)
        sin_eta = np.abs(np.sin(Eta_d * d2r))
        arg = np.clip(sin_th * cos_dom + cos_th * sin_eta * sin_dom, -1.0, 1.0)
        deltaTheta = d2r * (np.degrees(np.arcsin(arg)) - Theta_d)
        m_hkl = np.array([ring_counts.get(int(r), 1) for r in RingNr],
                         dtype=np.float64)
        ok = ((IntInt > 0) & (np.abs(deltaTheta) > 1e-12)
              & (cos_th > 0) & (PowderInt > 0))
        if not ok.any():
            return None
        Vg = (GrainVolume[ok] * NImgs[ok] * PowderInt[ok]) / (
            0.5 * m_hkl[ok] * deltaTheta[ok] * cos_th[ok] * IntInt[ok]
        )
        v_legacy = float(np.median(np.abs(Vg)))
        log(f"[v4] back-solved legacy Vgauge from {rad_file.name}: "
            f"{v_legacy:.4e} µm³  (from {ok.sum():,} rows)")
        return v_legacy
    except Exception as e:
        log(f"[v4] back-solve of legacy Vgauge failed ({e}); falling back to cylinder formula")
        return None


def _compute_radius_correction(ps_path: Path, log,
                                layer_dir: Optional[Path] = None) -> float:
    """Return the multiplicative correction factor for ``GrainRadius_naive``
    when the user-set ``Vsample`` or ``DiscArea`` differs from the Vgauge
    actually used in the legacy ``calc_radius`` calculation.

    Truth preference (descending):
      1. ``DiscArea`` if ``DiscModel == 1`` is set in paramstest (thin foil)
      2. ``Vsample``  if non-zero
      3. else no correction (returns 1.0)

    The legacy ``calc_radius`` formula uses
    ``Vgauge = Vsample (from upstream paramfile) if !=0 else Hbeam·π·Rsample²``.
    We back-solve the actual Vgauge from ``Radius_*.csv`` (ground truth);
    if that file is unavailable, fall back to the cylinder formula.

    Returns ``1.0`` if no correction needed; otherwise
    ``(V_truth / Vgauge_legacy)^(1/3)``.
    """
    p = _grab_paramstest_vsample(ps_path)

    # Truth preference: DiscArea (thin foil) > Vsample > none
    if p["DiscModel"] == 1.0 and p["DiscArea"] > 0:
        v_true = float(p["DiscArea"])
        log(f"[v4] DiscModel=1 detected; using DiscArea = {v_true:.3e} µm³ "
            f"as V_sample_true")
    elif p["Vsample"] > 0:
        v_true = float(p["Vsample"])
    else:
        return 1.0   # No truth supplied; nothing to correct

    v_legacy = (_back_solve_legacy_vgauge(layer_dir, log)
                if layer_dir is not None else None)
    if v_legacy is None or v_legacy <= 0:
        if p["DiscModel"] == 1.0 and p["DiscArea"] > 0:
            v_legacy = float(p["DiscArea"])
        else:
            v_legacy = float(p["Hbeam"]) * np.pi * float(p["Rsample"]) ** 2
        if v_legacy <= 0:
            return 1.0
        log(f"[v4] using cylinder-formula legacy Vgauge = {v_legacy:.4e} µm³")
    corr = (v_true / v_legacy) ** (1.0 / 3.0)
    log(f"[v4] Vsample correction: V_truth={v_true:.3e}µm³  "
        f"Vgauge_legacy={v_legacy:.3e}µm³  → R factor = {corr:.4f}")
    return corr


def _grab_paramstest_lattice(ps_path: Path) -> Optional[np.ndarray]:
    """Extract (a, b, c, α, β, γ) from paramstest LatticeParameter line."""
    try:
        for ln in open(ps_path):
            m = re.match(r"^LatticeParameter\s+([^;#]+);?", ln)
            if m:
                vals = m.group(1).split()
                if len(vals) >= 6:
                    return np.array([float(v) for v in vals[:6]], dtype=np.float64)
    except Exception:
        pass
    return None


def _grab_paramstest_lattice_c_over_a(ps_path: Path) -> Optional[float]:
    """Extract c/a from a paramstest ``LatticeParameter a b c α β γ`` line.

    Returns None if the line is missing or unparseable. Hexagonal cells
    have α = β = 90°, γ = 120°; the c/a ratio is c divided by a (= b).
    """
    try:
        for ln in open(ps_path):
            m = re.match(r"^LatticeParameter\s+([^;#]+);?", ln)
            if m:
                vals = m.group(1).split()
                if len(vals) >= 3:
                    a = float(vals[0]); c = float(vals[2])
                    if a > 0:
                        return c / a
    except Exception:
        pass
    return None


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def run_v4_pipeline(
    *,
    layer_dir: Union[str, Path],
    out_dir:   Union[str, Path],
    n_seeds:   Optional[int] = None,
    trust_scheme: str = "strict",
    indexing_rings: Optional[list[int]] = None,
    space_group: Optional[int] = None,
    omega_min_deg: float = -180.0,
    omega_max_deg: float = +180.0,
    min_n_unique_hkls: int = 2,
    merge_primitive: str = "misori",
    k_agree: Optional[int] = None,
    fp_y_tol_um: float = 800.0,
    fp_omega_tol_deg: float = 0.5,
    fp_om_split_tol_deg: Optional[float] = 1.0,
    compute_position_sigma: bool = False,
    position_sigma_max_grains: Optional[int] = None,
    compute_strain: bool = False,
    drop_recovery_floor: float = 0.0,
    drop_budget_tolerance: float = 1.0,
    force_keep_distinct_enabled: bool = True,
    force_keep_distinct_misori_deg: float = 1.0,
    force_keep_distinct_sigma: float = 3.0,
    orphan_reclaim_enabled: bool = True,
    orphan_reclaim_min_unique_spots: int = 5,
    orphan_reclaim_min_unique_fraction: float = 0.5,
    twin_aware_merge: bool = False,
    twin_merge_misori_deg: float = 2.0,
    twin_merge_position_um: float = 200.0,
    twin_merge_mode: str = "combined",
    verbose: bool = True,
) -> Dict[str, Path]:
    """Run the full v4 physics-bounded clustering pipeline.

    Parameters
    ----------
    layer_dir : Path
        Directory containing ``Results/OrientPosFit.bin``,
        ``Results/ProcessKey.bin``, ``InputAllExtraInfoFittingAll.csv``,
        ``hkls.csv``, ``paramstest.txt``.
    out_dir : Path
        Where to write the v4 grain table + audit CSVs.
    n_seeds : int, optional
        Number of rows in OrientPosFit.bin. If None, inferred from file size.
    trust_scheme : str
        ``"strict"`` (default), ``"loose"``, or ``"coverage_only"`` —
        see :mod:`compute.trust_tiers`.
    indexing_rings : list[int], optional
        Which detector rings the indexer used. If None, inferred from the
        seed-spot's ring numbers in InputAll.csv.
    merge_primitive : str, optional
        ``"misori"`` (default) — Pass-1 clusters use the symmetry-aware
        misori threshold from
        :func:`compute.adaptive.derive_misori_tol`. ``"forward_predict"``
        — Pass-1 clusters come from
        :func:`compute.forward_predict_merge.forward_predict_merge_components`
        on the indexing ring (variant-agreement evidence; symmetric by
        construction; immune to refiner-asymmetric matched lists). The
        forward-predict primitive correctly breaks chain-fusion giant
        components observed on heavily-twinned LMO data.
    k_agree : int, optional
        Minimum same-variant agreement count for a forward-predict merge
        edge. If ``None``, auto-selected by
        :func:`compute.forward_predict_merge.select_k_agree_auto`
        (smallest K such that the largest component is below
        ``max(100, n_alive / 100)``). Ignored unless
        ``merge_primitive == "forward_predict"``.
    fp_y_tol_um : float, optional
        Forward-predict snap radius (default 800 µm). Ignored unless
        ``merge_primitive == "forward_predict"``.
    fp_omega_tol_deg : float, optional
        Forward-predict ω matching tolerance (default 0.5°). Sets the
        anisotropic scaling of the 3D KDTree. Ignored unless
        ``merge_primitive == "forward_predict"``.
    fp_om_split_tol_deg : float or None, optional
        After the agree/disagree connected-components pass, split any
        component whose internal symmetry-aware misorientation exceeds
        this tolerance (default 1.0°). Catches the chain-fusion mode
        where ``A↔B↔C`` are transitively merged via non-overlapping
        evidence chains while ``A`` and ``C`` have no shared snaps.
        Set to ``None`` to disable the post-split. Ignored unless
        ``merge_primitive == "forward_predict"``.

    Returns
    -------
    Dict mapping artifact-name → :class:`Path` of the emitted file.
    """
    # derive_misori_tol is no longer called directly — the smart antimode
    # selection lives inside ``_pass1_misori``. We keep the docstring
    # reference intentional: the misori-primitive helper is the canonical
    # consumer of that algorithm.
    from .compute.hkl_ingest import (
        load_seed_hkls, read_hkls_csv, read_inputall_minimal,
    )
    from .compute.hkl_expected import (
        GeometryConfig, predict_expected_hkls,
        expected_visible_variants_per_grain,
    )
    from .compute.cluster_physics import split_clusters_by_physics
    from .compute.trust_tiers import assign_tiers, SCHEMES, tier_summary
    from .compute.twin_label import label_twins, label_subgrains
    from .compute.grain_size_recompute import recompute_grain_sizes
    from .io.hierarchical import emit_v4_grains, V4HierarchicalEmissionConfig
    from .io.binary import read_orient_pos_fit

    layer_dir = Path(layer_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a, **k: print(*a, **k, flush=True)) if verbose else (lambda *a, **k: None)
    t_total = time.time()

    # -- Inputs --
    ps = layer_dir / "paramstest.txt"
    SG = int(space_group or _grab_paramstest_scalar(ps, "SpaceGroup") or 225)
    LSD = (_grab_paramstest_scalar(ps, "LsdFit")
           or _grab_paramstest_scalar(ps, "Lsd"))
    if LSD is None:
        raise RuntimeError(f"paramstest.txt missing Lsd / LsdFit at {ps}")

    opf = read_orient_pos_fit(layer_dir)
    if n_seeds is None:
        n_seeds = opf.shape[0]
    alive_idx = np.flatnonzero(opf[:, 0] != 0)
    n_alive = len(alive_idx)
    log(f"[v4] layer={layer_dir}  SG={SG}  LSD={LSD:.1f} µm")
    log(f"[v4] alive candidates: {n_alive:,} / {n_seeds:,}")

    OM_raw = opf[alive_idx, 1:10].astype(np.float64).reshape(-1, 3, 3)
    # Position cols 11-13 (col 10 is SpId sentinel, per io/binary.py
    # ORIENT_POS_FIT_LAYOUT["position"] = slice(11, 14))
    positions = opf[alive_idx, 11:14].astype(np.float64)
    rep_radius_naive = opf[alive_idx, 25].astype(np.float64)
    confidence = opf[alive_idx, 26].astype(np.float64)

    # The c-omp refiner (FitUnified.c) writes ``meanRadius=1.0`` as a
    # placeholder to OrientPosFit col 25 — the legacy FitPosOrStrainsOMP
    # writes the spot-averaged GrainRadius. Detect and flag; we'll
    # recompute from per-spot GrainRadius once ``spot_sets`` are loaded.
    _radius_naive_is_placeholder = bool(np.all(rep_radius_naive == 1.0))
    if _radius_naive_is_placeholder:
        log("[v4]   OrientPosFit col 25 is all 1.0 — c-omp refiner placeholder; "
            "will recompute rep_radius_naive from per-spot GrainRadius after "
            "ProcessKey.bin load")

    # ── Vsample-based correction on inherited GrainRadius ──
    # The legacy midas-transforms ``calc_radius`` writes a per-spot R based
    # on ``Vgauge = Vsample if Vsample!=0 else Hbeam·π·Rsample²``. If the
    # user has added a TRUE ``Vsample`` to paramstest (e.g. because the
    # cylinder fallback was wrong for a rectangular sample), apply a
    # multiplicative correction in cube-root space so the v4 leaf reports
    # physically-correct R / V.
    radius_correction = _compute_radius_correction(ps, log, layer_dir=layer_dir)
    # Cache Vsample params for Stage 8.5 drop policy + Stage 9 vol check
    v_params = _grab_paramstest_vsample(ps)
    # Truth preference: DiscArea (thin foil, DiscModel=1) > Vsample > none.
    if v_params["DiscModel"] == 1.0 and v_params["DiscArea"] > 0:
        v_truth = float(v_params["DiscArea"])
        disc_model_active = True
    elif v_params["Vsample"] > 0:
        v_truth = float(v_params["Vsample"])
        disc_model_active = False
    else:
        v_truth = 0.0
        disc_model_active = False
    if radius_correction != 1.0 and not _radius_naive_is_placeholder:
        rep_radius_naive = rep_radius_naive * radius_correction
        log(f"[v4]   applied R correction = {radius_correction:.4f} "
            f"to {rep_radius_naive.size:,} candidate radii")

    # -- Stage 1: seed-(h,k,l) recovery with FZ canonicalization --
    t1 = time.time()
    seed_tbl = load_seed_hkls(layer_dir, opf=opf)
    seed_h    = seed_tbl.seed_h[alive_idx]
    seed_k    = seed_tbl.seed_k[alive_idx]
    seed_l    = seed_tbl.seed_l[alive_idx]
    seed_ring = seed_tbl.seed_ring[alive_idx]
    seed_ok   = seed_tbl.seed_alive[alive_idx]
    log(f"[v4] Stage 1 (seed-hkl): {seed_ok.sum():,} recovered "
        f"({100*seed_ok.mean():.1f}%)  [{time.time()-t1:.1f}s]")

    if indexing_rings is None:
        indexing_rings = sorted({int(r) for r in np.unique(seed_ring[seed_ok])})
    log(f"[v4]   indexing rings used: {indexing_rings}")

    # -- Load hkls + InputAll once; build OM_fz + GeometryConfig early so
    # both the (optional) forward-predict Pass-1 primitive and the Stage-2
    # expected-hkl predictor share the same geometry instance.
    hkls = read_hkls_csv(layer_dir / "hkls.csv")
    inputall = read_inputall_minimal(layer_dir / "InputAllExtraInfoFittingAll.csv")
    q_fz = fundamental_zone(orient_mat_to_quat(torch.from_numpy(OM_raw)), SG)
    OM_fz = np.asarray(quat_to_orient_mat(q_fz)).reshape(-1, 3, 3)
    geom = GeometryConfig(
        lsd_um=LSD,
        omega_min_deg=omega_min_deg, omega_max_deg=omega_max_deg,
        wavelength_a=_grab_paramstest_scalar(ps, "Wavelength") or 0.2,
        pixel_um=_grab_paramstest_scalar(ps, "px") or 200.0,
        y_BC=_grab_paramstest_scalar(ps, "YBCFit") or 1024.0,
        z_BC=_grab_paramstest_scalar(ps, "ZBCFit") or 1024.0,
        omega_step_deg=abs(_grab_paramstest_scalar(ps, "OmegaStep") or 0.25),
        wedge_deg=_grab_paramstest_scalar(ps, "Wedge") or 0.0,
        tx_deg=_grab_paramstest_scalar(ps, "txFit") or 0.0,
        ty_deg=_grab_paramstest_scalar(ps, "tyFit") or 0.0,
        tz_deg=_grab_paramstest_scalar(ps, "tzFit") or 0.0,
        n_pixels_y=int(_grab_paramstest_scalar(ps, "NrPixelsY") or 2048),
        n_pixels_z=int(_grab_paramstest_scalar(ps, "NrPixelsZ") or 2048),
        min_eta_deg=_grab_paramstest_scalar(ps, "ExcludePoleAngle") or 6.0,
        use_midas_diffract=True,
    )

    # -- Pass-1 clustering: either misori (default) or forward-predict --
    t2 = time.time()
    if merge_primitive == "forward_predict":
        p1_cluster, fp_twin_edges, fp_attrib, theta_star = _pass1_forward_predict(
            OM_fz=OM_fz, positions=positions, hkls=hkls,
            indexing_rings=indexing_rings, inputall=inputall,
            geom=geom, k_agree=k_agree,
            y_tol_um=fp_y_tol_um, omega_tol_deg=fp_omega_tol_deg,
            om_split_tol_deg=fp_om_split_tol_deg,
            qs_fz=q_fz.numpy() if hasattr(q_fz, "numpy") else np.asarray(q_fz),
            space_group=SG,
            log=log,
        )
        n_p1 = int(p1_cluster.max() + 1) if len(p1_cluster) else 0
        log(f"[v4] Pass-1 (forward-predict): {n_p1:,} clusters  "
            f"twin-pair edges: {len(fp_twin_edges):,}  [{time.time()-t2:.1f}s]")
    elif merge_primitive == "misori":
        p1_cluster, fp_twin_edges, fp_attrib, theta_star = _pass1_misori(
            OM_raw=OM_raw, SG=SG, n_alive=n_alive, log=log,
        )
        n_p1 = int(p1_cluster.max() + 1) if len(p1_cluster) else 0
        log(f"[v4] Pass-1 (misori): {n_p1:,} clusters  [{time.time()-t2:.1f}s]")
    else:
        raise ValueError(
            f"merge_primitive must be 'misori' or 'forward_predict', got {merge_primitive!r}"
        )

    # -- Pass-1.5: twin-aware cluster merge (optional) --
    # Collapses Pass-1 clusters that are alt-indexings or twin variants of
    # the same physical parent grain. For heavily-twinned materials (LMO,
    # MnO₂, hex alloys) forward-predict cannot merge twin variants because
    # their predicted spots are physically distinct. Pass-1.5 catches this
    # using the symmetry group's twin operators applied to per-cluster mean
    # orientations.
    if twin_aware_merge and n_p1 > 1:
        from .compute.cluster_merge import compute_cluster_merges
        t_p15 = time.time()
        c_over_a_p15 = _grab_paramstest_lattice_c_over_a(ps)
        # Cluster-representative OM/position via best-diff-pos seed.
        diff_pos_tmp = opf[alive_idx, 22]
        rep_tmp = np.zeros(n_p1, dtype=np.int64)
        for c in range(n_p1):
            members = np.flatnonzero(p1_cluster == c)
            if len(members) == 0:
                continue
            rep_tmp[c] = members[np.argmin(diff_pos_tmp[members])]
        OM_per_p1 = OM_fz[rep_tmp]
        pos_per_p1 = positions[rep_tmp]
        merge_res = compute_cluster_merges(
            cluster_orientation_matrices=OM_per_p1,
            cluster_positions_um=pos_per_p1,
            space_group=SG, c_over_a=c_over_a_p15,
            tol_misori_deg=twin_merge_misori_deg,
            tol_position_um=twin_merge_position_um,
            mode=twin_merge_mode,
        )
        p1_cluster = merge_res.parent_cluster_id[p1_cluster]
        n_p1 = merge_res.n_out_parents
        log(f"[v4] Pass-1.5 (twin-aware merge): {merge_res.n_in_clusters:,} "
            f"→ {n_p1:,} clusters "
            f"(direct merges: {merge_res.n_merges_direct:,}, "
            f"twin merges: {merge_res.n_merges_twin:,})  "
            f"[{time.time()-t_p15:.1f}s]")

    # NOTE: The original Pass-1 misori block (qs/sym pair enumeration + θ*
    # selection + union-find connected components) has been extracted to the
    # private helper ``_pass1_misori``. The forward-predict alternative lives
    # in ``_pass1_forward_predict``. Both produce a ``(n_alive,)`` int64
    # ``p1_cluster`` label array that the rest of the pipeline consumes
    # without modification.

    # -- Matched-spot sets from ProcessKey.bin (off-by-one safe) --
    t3 = time.time()
    pk_path = layer_dir / "Results" / "ProcessKey.bin"
    pk_rows = os.path.getsize(pk_path) // (5000 * 4)
    PK = np.memmap(pk_path, dtype=np.int32, mode="r", shape=(pk_rows, 5000))
    spot_sets = []
    for c in alive_idx:
        if c < pk_rows:
            row = PK[c]
            spot_sets.append({int(x) for x in row[row != 0].tolist()})
        else:
            spot_sets.append(set())
    log(f"[v4]   spot-set load: {len(spot_sets):,}  [{time.time()-t3:.1f}s]")

    # -- Recompute rep_radius_naive from per-spot GrainRadius when the
    # refiner wrote a placeholder (c-omp midas_fitgrain writes 1.0). The
    # legacy FitPosOrStrainsOMP writes the spot-averaged R itself; here we
    # match that by averaging the per-spot GrainRadius column from
    # ``InputAllExtraInfoFittingAll.csv`` over each candidate's spot_set.
    if _radius_naive_is_placeholder:
        _t_rr = time.time()
        _inp = pd.read_csv(layer_dir / "InputAllExtraInfoFittingAll.csv",
                           sep=r"\s+", engine="c")
        _inp.columns = [c.lstrip("%") for c in _inp.columns]
        _per_spot_R = dict(zip(_inp["SpotID"].astype(int),
                                _inp["GrainRadius"].astype(float)))
        for _i, _ss in enumerate(spot_sets):
            if _ss:
                _vals = [_per_spot_R[_s] for _s in _ss if _s in _per_spot_R]
                if _vals:
                    rep_radius_naive[_i] = float(np.mean(_vals))
        _good = rep_radius_naive[rep_radius_naive > 0]
        if radius_correction != 1.0:
            rep_radius_naive = rep_radius_naive * radius_correction
        log(f"[v4]   recomputed rep_radius_naive from per-spot GrainRadius "
            f"(median={np.median(_good):.2f} µm, "
            f"correction={radius_correction:.4f})  [{time.time()-_t_rr:.1f}s]")

    # -- Stage 2: expected-hkl prediction per Pass-1 consensus --
    # (OM_fz, hkls, inputall, geom were already loaded above so that the
    # forward-predict Pass-1 primitive could reuse them; the Stage-2
    # predictor consumes the SAME geometry instance — no duplication.)
    t4 = time.time()
    diff_pos = opf[alive_idx, 22]
    rep_per_p1 = np.zeros(n_p1, dtype=np.int64)
    for c in range(n_p1):
        members = np.flatnonzero(p1_cluster == c)
        if len(members) == 0: continue
        rep_per_p1[c] = members[np.argmin(diff_pos[members])]
    OM_consensus = OM_fz[rep_per_p1]
    exp_tbl = predict_expected_hkls(OM_consensus, hkls, indexing_rings, geom)
    vis_per_p1 = expected_visible_variants_per_grain(exp_tbl, n_grains=n_p1)
    log(f"[v4] Stage 2 (expected-hkl): median visible variants per cluster = "
        f"{int(np.median(vis_per_p1))}  [{time.time()-t4:.1f}s]")
    n_expected_per_p1 = {int(c): int(vis_per_p1[c]) for c in range(n_p1)}

    # -- Stage 3: physics-bounded split --
    t5 = time.time()
    phys = split_clusters_by_physics(
        pass1_cluster_id=p1_cluster,
        seed_h=seed_h, seed_k=seed_k, seed_l=seed_l, seed_alive=seed_ok,
        positions=positions, spot_sets=spot_sets, om_fz=OM_fz,
        n_expected_per_pass1=n_expected_per_p1,
    )
    n_final = phys.n_final_grains
    n_split = int((phys.grain_splits_emerged > 0).sum())
    log(f"[v4] Stage 3 (physics split): {n_final:,} final grains "
        f"({n_split:,} from splits)  [{time.time()-t5:.1f}s]")

    # ---- MATCHED-SPOT-BASED observed-hkls (the correct primitive) ----
    # The seed-based count (phys.grain_n_unique_hkls) is the number of
    # candidates' distinct seed-(h,k,l) per grain. It UNDER-counts the
    # actual evidence: refined candidates carry a matched-spot list
    # (from ProcessKey.bin) where the refiner accepted many MORE
    # indexing-ring spots as constraints on the orientation, beyond
    # just the ones the indexer reached as seeds.
    #
    # The correct count: recover the seed-(h,k,l) variant of EVERY
    # matched spot via Stage 1's recovery (using the grain's consensus
    # FZ-canonical OM), then count distinct variants per grain.
    # Empirically gives 97-100% gold-eligible on Indrajeet Ni / Ti-7Al /
    # Xuan SS LMO / peakfit (vs <16% under the seed-based primitive).
    t5b = time.time()
    log(f"[v4] Computing matched-spot-based observed_hkls ...")
    n_matched_hkls = _compute_matched_hkls_per_grain(
        phys, alive_idx, spot_sets, om_fz=OM_fz, hkls=hkls,
        inputall_df=inputall, lsd_um=LSD,
        indexing_rings=set(indexing_rings),
    )
    # Replace observed count + recompute coverage
    phys.grain_n_unique_hkls_seed = phys.grain_n_unique_hkls.copy()  # keep diagnostic
    phys.grain_n_unique_hkls = n_matched_hkls
    n_exp_safe = np.maximum(phys.grain_n_expected_hkls, 1)
    phys.grain_hkl_coverage = n_matched_hkls / n_exp_safe
    if n_matched_hkls.size:
        log(f"[v4]   matched-based observed: median={int(np.median(n_matched_hkls))} "
            f"vs seed-based median={int(np.median(phys.grain_n_unique_hkls_seed))}  "
            f"[{time.time()-t5b:.1f}s]")
    else:
        log(f"[v4]   (no final grains; matched-based count skipped)  "
            f"[{time.time()-t5b:.1f}s]")

    # ---- Minimum-evidence filter (analogue of legacy MinNrSpots) ----
    # A "grain" with fewer than ``min_n_unique_hkls`` distinct matched
    # variants is a single-spot indexing artifact. Under the matched-
    # spot primitive this filter is much MORE permissive (essentially
    # all v4 grains have >=2 matched variants).
    keep_mask = phys.grain_n_unique_hkls >= int(min_n_unique_hkls)
    n_kept = int(keep_mask.sum())
    n_filtered = n_final - n_kept
    log(f"[v4] Min-evidence filter (hkl_n_observed >= {min_n_unique_hkls}): "
        f"kept {n_kept:,}, removed {n_filtered:,} (single-/few-spot artifacts)")
    # Re-emit the per-grain arrays restricted to kept grains
    keep_idx = np.flatnonzero(keep_mask)
    # Update phys-result sliced views
    class _Sub:
        pass
    phys_kept = _Sub()
    phys_kept.n_final_grains = n_kept
    phys_kept.grain_pass1_parent = phys.grain_pass1_parent[keep_idx]
    phys_kept.grain_n_candidates = phys.grain_n_candidates[keep_idx]
    phys_kept.grain_n_unique_hkls = phys.grain_n_unique_hkls[keep_idx]
    phys_kept.grain_n_unique_hkls_seed = phys.grain_n_unique_hkls_seed[keep_idx]
    phys_kept.grain_hkl_dup_count = phys.grain_hkl_dup_count[keep_idx]
    phys_kept.grain_splits_emerged = phys.grain_splits_emerged[keep_idx]
    phys_kept.grain_n_expected_hkls = phys.grain_n_expected_hkls[keep_idx]
    phys_kept.grain_hkl_coverage = phys.grain_hkl_coverage[keep_idx]
    # Re-map per-candidate final_grain_id: candidates pointing to filtered
    # grains get -1; survivors get a contiguous re-numbering.
    if n_final > 0:
        old_to_new = np.full(n_final, -1, dtype=np.int64)
        old_to_new[keep_idx] = np.arange(n_kept, dtype=np.int64)
        phys_kept.final_grain_id = np.where(
            phys.final_grain_id >= 0,
            old_to_new[phys.final_grain_id.clip(min=0)],
            -1,
        )
    else:
        phys_kept.final_grain_id = phys.final_grain_id
    phys_kept.pass1_cluster_id = phys.pass1_cluster_id
    phys = phys_kept
    n_final = n_kept

    # -- Stage 4: trust tiers --
    t6 = time.time()
    tier_strict = assign_tiers(phys.grain_hkl_coverage, phys.grain_hkl_dup_count,
                                phys.grain_splits_emerged, scheme="strict")
    tier_loose  = assign_tiers(phys.grain_hkl_coverage, phys.grain_hkl_dup_count,
                                phys.grain_splits_emerged, scheme="loose")
    sum_strict = tier_summary(tier_strict)
    sum_loose  = tier_summary(tier_loose)
    log(f"[v4] Stage 4 (trust): strict {sum_strict['n_gold']:,} gold / "
        f"{sum_strict['n_silver']:,} silver / {sum_strict['n_bronze']:,} bronze;"
        f"  loose {sum_loose['n_gold']:,}/{sum_loose['n_silver']:,}/{sum_loose['n_bronze']:,}  "
        f"[{time.time()-t6:.1f}s]")

    # -- Per-final-grain representative arrays (one row per final grain) --
    # Physics-driven choice for X,Y,Z + OM: use the cluster-mean rather
    # than a single min-diff_pos candidate. The mean is the only choice
    # that satisfies the forward-model self-consistency test on
    # multi-cand grains (rep-OM-and-position re-predicts the cluster's
    # observed spots within strict pixel + OmegaStep tolerance). Using
    # a single rep introduces a >20% rate of physically-unreal grains
    # whose stored OM doesn't reproduce its own observed evidence
    # (see dev/paper/scripts/forward_predict_deep_audit.py).
    #
    # `rep_grain_cand` is kept as the min-diff_pos candidate ID for
    # backwards-compat reasons (RepSeed column, hkl_n_observed_seed,
    # legacy join keys); the geometric attributes are the mean.
    rep_grain_cand = np.zeros(n_final, dtype=np.int64)
    pos_per_g      = np.zeros((n_final, 3), dtype=np.float64)
    om_fz_per_g    = np.zeros((n_final, 3, 3), dtype=np.float64)
    conf_per_g     = np.zeros(n_final, dtype=np.float64)
    radius_naive_per_g = np.zeros(n_final, dtype=np.float64)
    qs_fz_all = orient_mat_to_quat(torch.from_numpy(OM_fz)).numpy()
    for g in range(n_final):
        members = np.flatnonzero(phys.final_grain_id == g)
        if len(members) == 0: continue
        rep_grain_cand[g] = members[np.argmin(diff_pos[members])]
        # Position: median of all member positions
        pos_per_g[g] = np.median(positions[members], axis=0)
        # OM: rotation-mean (hemisphere-align quats, average, renormalize)
        if len(members) == 1:
            om_fz_per_g[g] = OM_fz[members[0]]
        else:
            q_mem = qs_fz_all[members]
            ref = q_mem[0]
            signs = np.where((q_mem * ref).sum(axis=1) >= 0, 1.0, -1.0)
            q_mean = (signs[:, None] * q_mem).mean(axis=0)
            q_mean /= np.linalg.norm(q_mean) + 1e-12
            om_fz_per_g[g] = np.asarray(
                quat_to_orient_mat(torch.from_numpy(q_mean[None, :]))
            ).reshape(3, 3)
        # Confidence + naive radius: mean across members (representative
        # of the cluster, not biased by an outlier rep).
        conf_per_g[g] = float(np.mean(confidence[members]))
        radius_naive_per_g[g] = float(np.mean(rep_radius_naive[members]))
    rep_seed_idx = alive_idx[rep_grain_cand]
    quats_per_g  = orient_mat_to_quat(torch.from_numpy(om_fz_per_g)).numpy()

    # -- Stage 5: twin + sub-grain labeling --
    # Dispatcher chooses FCC/BCC/HCP/trigonal twin operators per space
    # group; HCP needs c/a from the unit cell. paramstest has
    # ``LatticeParameter a b c α β γ`` (Å + degrees).
    t7 = time.time()
    c_over_a = _grab_paramstest_lattice_c_over_a(ps)
    if 168 <= SG <= 194 and c_over_a is not None:
        log(f"[v4]   HCP c/a = {c_over_a:.4f}  (for twin operator construction)")
    twin_partner, twin_family, twin_type, n_twin = label_twins(
        grain_quats=quats_per_g,
        grain_positions=pos_per_g,
        space_group=SG, tol_deg=0.5,
        spatial_max_um=None,
        c_over_a=c_over_a,
    )
    grain_spot_sets = [
        set().union(*[spot_sets[c] for c in np.flatnonzero(phys.final_grain_id == g)])
        for g in range(n_final)
    ]
    subgrain_partner, n_sub = label_subgrains(
        grain_quats=quats_per_g, grain_positions=pos_per_g,
        grain_spot_sets=grain_spot_sets, space_group=SG,
    )
    log(f"[v4] Stage 5 (twins/subgrains): {n_twin:,} twin pairs, "
        f"{n_sub:,} sub-grain pairs  [{time.time()-t7:.1f}s]")

    # -- Stage 6: grain-size NNLS (only changes split clusters) --
    t8 = time.time()
    # Build per-spot (intensity, ring) from InputAll, tolerating header
    # variants: ``%YLab`` legacy vs joint-results plain, and
    # ``IntegratedIntensity(count)`` vs ``IntegratedIntensity``.
    inp_full = pd.read_csv(layer_dir / "InputAllExtraInfoFittingAll.csv",
                            sep=r"\s+", engine="c")
    inp_full.columns = [c.lstrip("%") for c in inp_full.columns]
    int_col = next((c for c in inp_full.columns
                    if c.startswith("IntegratedIntensity")), None)
    if int_col is None:
        raise RuntimeError(
            f"InputAllExtraInfoFittingAll.csv has no IntegratedIntensity column; "
            f"columns are: {list(inp_full.columns)}"
        )
    per_spot_int = dict(zip(inp_full["SpotID"].astype(int),
                             inp_full[int_col].astype(float)))
    per_spot_ring = dict(zip(inp_full["SpotID"].astype(int),
                             inp_full["RingNumber"].astype(int)))
    by_ring = inp_full.groupby("RingNumber")[int_col].median()
    ring_K = {int(r): float(v) if v > 0 else 1.0 for r, v in by_ring.items()}
    size_res = recompute_grain_sizes(
        final_grain_id_per_candidate=phys.final_grain_id,
        pass1_cluster_id=p1_cluster,
        spot_sets_per_candidate=spot_sets,
        per_spot_intensity=per_spot_int,
        per_spot_ring=per_spot_ring,
        ring_K=ring_K,
        rep_radius_naive_um_per_grain=radius_naive_per_g,
    )
    log(f"[v4] Stage 6 (grain-size NNLS): "
        f"{(size_res.deflation_factor < 0.9).sum():,} grains deflated >10%, "
        f"{(size_res.deflation_factor > 1.1).sum():,} inflated >10%  "
        f"[{time.time()-t8:.1f}s]")

    # Stage 6.5 drop policy is deferred to AFTER Stage 7/8 so the quality
    # score can use σ_Z + strain solver outcome. Initialize the masks here.
    drop_by_budget = np.zeros(n_final, dtype=bool)
    drop_by_budget_family = np.zeros(n_final, dtype=bool)
    budget_overcount_ratio = float("nan")
    budget_overcount_ratio_family = float("nan")
    family_quality_per_grain = np.full(n_final, np.nan, dtype=np.float64)
    family_V_per_grain = np.full(n_final, np.nan, dtype=np.float64)

    # -- Stage 7: per-grain position σ via midas_propagate (optional) --
    sigma_X_um = None
    sigma_Y_um = None
    sigma_Z_um = None
    n_spots_per_grain = None
    sigma_residual_rms_px = None
    if compute_position_sigma:
        from .compute.position_uncertainty import compute_per_grain_position_sigma
        from midas_diffract.forward import HEDMGeometry
        # Build a fresh HEDMGeometry with apply_tilts=True for the σ math
        span = geom.omega_max_deg - geom.omega_min_deg
        n_frames = max(int(round(span / max(geom.omega_step_deg, 1e-6))), 1)
        fm_geom = HEDMGeometry(
            Lsd=geom.lsd_um, y_BC=geom.y_BC, z_BC=geom.z_BC, px=geom.pixel_um,
            omega_start=geom.omega_min_deg, omega_step=geom.omega_step_deg,
            n_frames=n_frames,
            n_pixels_y=geom.n_pixels_y, n_pixels_z=geom.n_pixels_z,
            min_eta=geom.min_eta_deg, wavelength=geom.wavelength_a,
            tx=geom.tx_deg, ty=geom.ty_deg, tz=geom.tz_deg,
            wedge=geom.wedge_deg, flip_y=True, apply_tilts=True,
        )
        # Subsample if requested
        if position_sigma_max_grains is not None and n_final > position_sigma_max_grains:
            idx_sub = np.random.default_rng(0).choice(
                n_final, size=position_sigma_max_grains, replace=False,
            )
            grain_OM_sub = om_fz_per_g[idx_sub]
            grain_pos_sub = pos_per_g[idx_sub]
            rep_cand_sub = alive_idx[rep_grain_cand[idx_sub]]
            log(f"[v4] Stage 7 (per-grain σ): subsampling {position_sigma_max_grains:,}"
                f" of {n_final:,} grains (set position_sigma_max_grains=None to disable)")
        else:
            idx_sub = np.arange(n_final)
            grain_OM_sub = om_fz_per_g
            grain_pos_sub = pos_per_g
            rep_cand_sub = alive_idx[rep_grain_cand]

        # Lattice (a, b, c, α, β, γ) from paramstest
        latc_params = _grab_paramstest_lattice(ps)
        if latc_params is None:
            latc_params = np.array([3.6, 3.6, 3.6, 90, 90, 90], dtype=np.float64)
            log(f"[v4]   (no LatticeParameter in paramstest; defaulting cubic 3.6Å)")

        t_sigma = time.time()
        sigma_res = compute_per_grain_position_sigma(
            grain_OM=grain_OM_sub, grain_pos_um=grain_pos_sub,
            rep_cand_idx=rep_cand_sub,
            pk_path=layer_dir / "Results/ProcessKey.bin",
            inputall_df=inputall, hkls=hkls,
            geometry=fm_geom, latc=latc_params,
            omega_start_deg=geom.omega_min_deg,
            omega_step_deg=geom.omega_step_deg,
            log=log,
        )
        # Pad back to full n_final
        sigma_X_um = np.full(n_final, np.nan, dtype=np.float64)
        sigma_Y_um = np.full(n_final, np.nan, dtype=np.float64)
        sigma_Z_um = np.full(n_final, np.nan, dtype=np.float64)
        n_spots_per_grain = np.zeros(n_final, dtype=np.int32)
        sigma_residual_rms_px = np.full(n_final, np.nan, dtype=np.float64)
        sigma_X_um[idx_sub] = sigma_res.sigma_X_um
        sigma_Y_um[idx_sub] = sigma_res.sigma_Y_um
        sigma_Z_um[idx_sub] = sigma_res.sigma_Z_um
        n_spots_per_grain[idx_sub] = sigma_res.n_spots_matched
        sigma_residual_rms_px[idx_sub] = sigma_res.residual_rms_px
        n_sig_ok = int(sigma_res.ok.sum())
        log(f"[v4] Stage 7 (per-grain σ): {n_sig_ok:,} grains succeeded;  "
            f"σ_X med={np.nanmedian(sigma_X_um):.1f}  σ_Y med={np.nanmedian(sigma_Y_um):.1f}  "
            f"σ_Z med={np.nanmedian(sigma_Z_um):.1f} µm  [{time.time()-t_sigma:.1f}s]")
        # σ-aware re-tier (additional column in the leaf)
        tier_sigma_aware = assign_tiers(
            phys.grain_hkl_coverage, phys.grain_hkl_dup_count,
            phys.grain_splits_emerged, scheme="sigma_aware",
            sigma_X_um=sigma_X_um, sigma_Y_um=sigma_Y_um, sigma_Z_um=sigma_Z_um,
            n_spots_matched=n_spots_per_grain,
        )
        sum_sigma = tier_summary(tier_sigma_aware)
        log(f"[v4]   sigma-aware tier:  {sum_sigma['n_gold']:,} gold / "
            f"{sum_sigma['n_silver']:,} silver / {sum_sigma['n_bronze']:,} bronze")
    else:
        tier_sigma_aware = None

    # -- Stage 8: per-grain strain (optional) --
    strain_voigt = None
    if compute_strain:
        t_strain = time.time()
        strain_voigt = _compute_strain_per_grain(
            phys=phys, alive_idx=alive_idx,
            om_fz_per_g=om_fz_per_g, pos_per_g=pos_per_g,
            spot_sets=spot_sets, inputall_df=inputall,
            hkls=hkls, geometry=geom, latc=_grab_paramstest_lattice(ps),
            log=log,
        )
        n_ok = int(np.isfinite(strain_voigt).all(axis=1).sum())
        log(f"[v4] Stage 8 (strain): solved {n_ok:,} of {n_final:,} grains  "
            f"[{time.time()-t_strain:.1f}s]")

    # -- Stage 8.5: volume-budget drop policy (per-grain + family-aware) --
    # Runs only if user supplied Vsample in paramstest (= an independent
    # measurement of the illuminated sample volume).
    if v_truth > 0:
        from .compute.drop_policy import (
            compute_volume_budget_drops, compute_volume_budget_drops_family,
        )
        t85 = time.time()
        v_sample_true = v_truth
        # Quality score (used by BOTH per-grain v1 and family-aware v2).
        # σ_Z fallback: when σ_Z wasn't computed (e.g., sub-sampling), use the
        # MEDIAN measured σ_Z as a neutral fallback. This avoids artificially
        # boosting un-measured grains. If NO σ_Z was ever computed, use 50 µm.
        cov_arr = (phys.grain_hkl_coverage if hasattr(phys, "grain_hkl_coverage")
                   else np.ones(n_final))
        sigZ = sigma_Z_um if sigma_Z_um is not None else np.full(n_final, np.nan)
        sigZ_measured = sigZ[np.isfinite(sigZ)]
        sigZ_fallback = float(np.median(sigZ_measured)) if sigZ_measured.size else 50.0
        sigZ_eff = np.where(np.isfinite(sigZ),
                             np.maximum(sigZ, 5.0),
                             sigZ_fallback)
        Q = (np.asarray(conf_per_g, dtype=np.float64)
             * np.where(np.isfinite(cov_arr), cov_arr, 1.0)
             / sigZ_eff)
        log(f"[v4]   σ_Z fallback for un-measured grains: {sigZ_fallback:.1f} µm "
            f"(median of {sigZ_measured.size:,} measured)")
        # Per-grain (v1) — now QUALITY-ranked
        drop_res = compute_volume_budget_drops(
            volume_NNLS_um3=size_res.volume_nnls_um3,
            volume_naive_um3=size_res.volume_naive_um3,
            v_sample_true_um3=v_sample_true,
            recovery_floor=drop_recovery_floor,
            tolerance=drop_budget_tolerance,
            quality_score=Q,
        )
        drop_by_budget = drop_res.drop_by_budget
        budget_overcount_ratio = drop_res.overcounting_ratio
        # Family-aware (v2)
        fres = compute_volume_budget_drops_family(
            volume_NNLS_um3=size_res.volume_nnls_um3,
            twin_family_id=twin_family,
            quality_score=Q,
            v_sample_true_um3=v_sample_true,
            family_aggregation="max",
            tolerance=drop_budget_tolerance,
        )
        drop_by_budget_family = fres.drop_by_budget
        budget_overcount_ratio_family = fres.overcounting_ratio_family
        family_quality_per_grain = fres.family_quality[fres.family_id_per_grain]
        family_V_per_grain = fres.family_V_um3[fres.family_id_per_grain]
        log(f"[v4] Stage 8.5 (volume budget): V_sample_true = {v_sample_true:.3e} µm³")
        log(f"[v4]   per-grain:   ΣV = {drop_res.sum_V_NNLS_um3:.3e} µm³  "
            f"overcount = {budget_overcount_ratio:.2f}  "
            f"drop {drop_res.n_dropped:,}/{n_final:,} grains")
        log(f"[v4]   family(max): ΣV = {fres.sum_V_family_um3:.3e} µm³  "
            f"overcount = {budget_overcount_ratio_family:.2f}  "
            f"drop {fres.n_families_dropped:,}/{len(fres.family_V_um3):,} families "
            f"(= {fres.n_grains_dropped:,}/{n_final:,} grains)")
        log(f"[v4]   [{time.time()-t85:.1f}s]")

        # -- Stage 8.5b: force-keep "distinct" candidates (Path 2) --
        # The budget drop is conservative — it wrongly drops some real grains
        # that ARE distinct (different OM beyond peak resolution AND different
        # position beyond σ). This step recovers them.
        if force_keep_distinct_enabled and drop_by_budget.any():
            from .compute.drop_policy import compute_force_keep_distinct
            t85b = time.time()
            sigma_xyz = np.stack([
                sigma_X_um if sigma_X_um is not None else np.full(n_final, sigZ_fallback),
                sigma_Y_um if sigma_Y_um is not None else np.full(n_final, sigZ_fallback),
                sigma_Z_um if sigma_Z_um is not None else np.full(n_final, sigZ_fallback),
            ], axis=1)
            sigma_xyz = np.where(np.isfinite(sigma_xyz),
                                  np.maximum(sigma_xyz, 5.0),
                                  sigZ_fallback)
            fkd = compute_force_keep_distinct(
                grain_OMs=om_fz_per_g,
                grain_positions_um=pos_per_g,
                grain_sigma_xyz_um=sigma_xyz,
                drop_mask=drop_by_budget,
                space_group=SG,
                misori_deg_threshold=force_keep_distinct_misori_deg,
                sigma_distance_threshold=force_keep_distinct_sigma,
            )
            n_force_kept = fkd.n_force_kept
            drop_by_budget = fkd.new_drop_mask
            # Also apply force-keep to family variant
            drop_by_budget_family = drop_by_budget_family & ~fkd.force_kept_mask
            budget_overcount_ratio = (drop_res.sum_V_NNLS_um3
                                       / (v_sample_true * (1 - n_force_kept/n_final))) \
                                       if n_final else float("nan")
            log(f"[v4] Stage 8.5b (force-keep distinct, "
                f"misori≥{force_keep_distinct_misori_deg}°, σdist≥{force_keep_distinct_sigma}σ): "
                f"recovered {n_force_kept:,} grains from drop  "
                f"[{time.time()-t85b:.1f}s]")

        # -- Stage 8.5c: orphan-greedy reclaim (Path 3) --
        # Recover currently-dropped grains whose spot-sets uniquely cover
        # orphan spots (= spots not yet claimed by any kept grain). This
        # reduces orphan rate at the cost of keeping more lower-quality
        # candidates — but only those that contribute uniquely.
        if orphan_reclaim_enabled and drop_by_budget.any():
            from .compute.drop_policy import compute_orphan_greedy_reclaim
            t85c = time.time()
            orph_res = compute_orphan_greedy_reclaim(
                drop_mask=drop_by_budget,
                spot_sets=grain_spot_sets,
                quality_score=Q,
                min_unique_spots=orphan_reclaim_min_unique_spots,
                min_unique_fraction=orphan_reclaim_min_unique_fraction,
            )
            drop_by_budget = orph_res.new_drop_mask
            drop_by_budget_family = drop_by_budget_family & ~orph_res.reclaimed_mask
            log(f"[v4] Stage 8.5c (orphan-greedy reclaim, "
                f"min_unique_spots={orphan_reclaim_min_unique_spots}, "
                f"min_unique_fraction={orphan_reclaim_min_unique_fraction}): "
                f"reclaimed {orph_res.n_reclaimed:,} grains  "
                f"({orph_res.n_orphan_spots_before:,} orphan → "
                f"{orph_res.n_orphan_spots_after:,} orphan spots)  "
                f"[{time.time()-t85c:.1f}s]")

    # -- Stage 9: volume consistency check --
    from .compute.volume_consistency import (
        compute_volume_consistency, consistency_as_meta_dict,
    )
    v_sample_for_check = v_truth if v_truth > 0 else None
    vol_check = compute_volume_consistency(
        radius_um=size_res.radius_nnls_um,
        positions_um=pos_per_g,
        v_sample_um3=v_sample_for_check,
    )
    if vol_check.v_sample_um3:
        log(f"[v4] Stage 9 (volume consistency): "
            f"Σ V_grain = {vol_check.sum_v_grain_um3:.3e} µm³  "
            f"V_sample = {vol_check.v_sample_um3:.3e} µm³  "
            f"packing = {100*vol_check.packing_fraction:.3f}%  "
            f"in_box = {100*(vol_check.fraction_in_box or 0):.1f}%")
    else:
        log(f"[v4] Stage 9 (volume consistency): "
            f"Σ V_grain = {vol_check.sum_v_grain_um3:.3e} µm³  "
            f"(no Vsample in paramstest → no packing fraction)")

    # -- Emit hierarchical output --
    t9 = time.time()
    cfg = V4HierarchicalEmissionConfig(trust_scheme=trust_scheme)
    paths = emit_v4_grains(
        out_dir,
        rep_seed_per_grain=rep_seed_idx,
        positions_um=pos_per_g,
        orient_mats_3x3=om_fz_per_g,
        strain_voigt=strain_voigt,
        hkl_n_observed=phys.grain_n_unique_hkls,            # matched-based (primary)
        hkl_n_observed_seed=getattr(phys, "grain_n_unique_hkls_seed", None),
        hkl_n_expected=phys.grain_n_expected_hkls,
        hkl_coverage=phys.grain_hkl_coverage,
        hkl_dup_count=phys.grain_hkl_dup_count,
        splits_emerged_from=phys.grain_splits_emerged,
        trust_tier_strict=tier_strict,
        trust_tier_loose=tier_loose,
        trust_tier_sigma_aware=tier_sigma_aware,
        confidence=conf_per_g,
        twin_partner_id=twin_partner,
        twin_family_id=twin_family,
        twin_type=twin_type,
        subgrain_partner_id=subgrain_partner,
        radius_naive_um=size_res.radius_naive_um,
        radius_nnls_um=size_res.radius_nnls_um,
        radius_disc_um=np.sqrt(np.maximum(size_res.volume_nnls_um3, 0) / np.pi),
        volume_nnls_um3=size_res.volume_nnls_um3,
        sigma_R_nnls_um=getattr(size_res, "sigma_R_nnls_um", None),
        sigma_X_um=sigma_X_um,
        sigma_Y_um=sigma_Y_um,
        sigma_Z_um=sigma_Z_um,
        n_spots_matched=n_spots_per_grain,
        sigma_residual_rms_px=sigma_residual_rms_px,
        drop_by_budget=drop_by_budget,
        drop_by_budget_family=drop_by_budget_family,
        family_quality=family_quality_per_grain,
        family_V_um3=family_V_per_grain,
        volume_recovery=np.where(
            size_res.volume_naive_um3 > 0,
            size_res.volume_nnls_um3 / np.maximum(size_res.volume_naive_um3, 1e-30),
            np.nan,
        ),
        cfg=cfg,
        meta_extra={
            "layer_dir":          str(layer_dir),
            "space_group":        int(SG),
            "Lsd_um":             float(LSD),
            "n_alive_candidates": int(n_alive),
            "n_pass1_clusters":   int(n_p1),
            "n_final_grains":     int(n_final),
            "n_split_clusters":   int(n_split),
            "n_twin_pairs":       int(n_twin),
            "n_subgrain_pairs":   int(n_sub),
            "theta_star_deg":     float(theta_star) if theta_star is not None else None,
            "merge_primitive":    merge_primitive,
            "vsample_correction": float(radius_correction),
            "volume_consistency": consistency_as_meta_dict(vol_check),
            "drop_by_budget_n":            int(drop_by_budget.sum()),
            "drop_by_budget_family_n":     int(drop_by_budget_family.sum()),
            "drop_by_budget_overcount_ratio":         float(budget_overcount_ratio),
            "drop_by_budget_family_overcount_ratio":  float(budget_overcount_ratio_family),
            "indexing_rings":     indexing_rings,
            "trust_strict":       sum_strict,
            "trust_loose":        sum_loose,
            "total_wall_s":       float(time.time() - t_total),
        },
    )
    log(f"[v4] Emit: wrote {paths}  [{time.time()-t9:.1f}s]")

    # Also write the per-candidate audit (cluster IDs etc.) for debugging
    audit_path = out_dir / "v4_candidate_audit.csv"
    pd.DataFrame({
        "alive_idx": alive_idx,
        "pass1_cluster": p1_cluster,
        "final_grain_id": phys.final_grain_id,
        "seed_h": seed_h, "seed_k": seed_k, "seed_l": seed_l,
        "seed_alive": seed_ok,
    }).to_csv(audit_path, sep="\t", index=False)
    paths["audit"] = audit_path

    log(f"[v4] DONE — total {time.time()-t_total:.1f}s")
    return paths


def _compute_matched_hkls_per_grain(
    phys,                      # PhysicsClusterResult
    alive_idx: np.ndarray,     # (n_alive,) into OPF — original candidate indices
    spot_sets,                 # length n_alive — matched SpotID set per candidate
    *,
    om_fz: np.ndarray,         # (n_alive, 3, 3) — FZ-canonical OMs per candidate
    hkls,                      # HklTable
    inputall_df: pd.DataFrame, # SpotID-indexed: YLab, ZLab, Omega, RingNumber, Eta
    lsd_um: float,
    indexing_rings: set,
) -> np.ndarray:
    """Return (n_final_grains,) int32 — distinct (h,k,l) variants observed in
    the union of matched-spot lists per grain.

    Uses Stage 1's seed-(h,k,l) recovery (validated, works correctly on real
    data) on EVERY matched spot in each grain's union, with the grain's
    consensus FZ-canonical OM. The variant count is then the size of the
    distinct (h,k,l) set.

    Where each grain's consensus OM comes from:
        the lowest-DiffPos candidate within the grain — same convention as
        Stage 2's per-cluster consensus.
    """
    from midas_stress.orientation import (
        orient_mat_to_quat, quat_to_orient_mat, fundamental_zone,
    )
    from .compute.hkl_ingest import (
        _g_lab_from_observed, _rotate_lab_to_sample_by_omega,
    )

    n_final = phys.n_final_grains
    out = np.zeros(n_final, dtype=np.int32)

    # Restrict hkls to the indexing rings
    ring_arr = np.asarray(hkls.ring)
    in_ring = np.isin(ring_arr, list(indexing_rings))
    if not in_ring.any():
        return out
    h_v = hkls.h[in_ring]; k_v = hkls.k[in_ring]; l_v = hkls.l[in_ring]
    g_ref = hkls.g_crystal[in_ring]
    g_ref_norm = g_ref / np.linalg.norm(g_ref, axis=1, keepdims=True)
    hkl_arr = np.column_stack([h_v, k_v, l_v]).astype(np.int8)

    # Pre-filter inputall to indexing-ring spots
    ring_mask = inputall_df["RingNumber"].astype(int).isin(indexing_rings)
    ia_ring = inputall_df[ring_mask][["YLab", "ZLab", "Omega"]]
    valid_sids = set(ia_ring.index.values.tolist())

    # Pick representative per final grain (lowest DiffPos already used by Stage 2)
    # — but here we only need the OM, and all candidates in a grain share the
    # same FZ-canonical OM up to noise, so use the first candidate.
    for g in range(n_final):
        members = np.flatnonzero(phys.final_grain_id == g)
        if members.size == 0:
            continue
        rep_local = int(members[0])
        OM = om_fz[rep_local]
        # Union of matched SpotIDs across all candidates in this grain
        union_spots = set()
        for c in members:
            union_spots |= spot_sets[int(c)]
        union_spots &= valid_sids
        if not union_spots:
            continue
        # Variant recovery on each matched ring-spot
        sids = np.fromiter(union_spots, dtype=np.int64)
        rows = ia_ring.reindex(sids).dropna()
        if rows.empty:
            continue
        Y = rows["YLab"].to_numpy()
        Z = rows["ZLab"].to_numpy()
        ome = rows["Omega"].to_numpy()
        g_lab = _g_lab_from_observed(Y, Z, lsd_um)              # (S, 3)
        g_sample = _rotate_lab_to_sample_by_omega(g_lab, ome)   # (S, 3)
        g_crystal_obs = g_sample @ OM                            # (S, 3)
        g_crystal_obs /= (np.linalg.norm(g_crystal_obs, axis=1, keepdims=True) + 1e-12)
        cos = g_crystal_obs @ g_ref_norm.T                       # (S, V)
        best = np.argmax(cos, axis=1)
        variants = set(tuple(int(x) for x in hkl_arr[bi]) for bi in best)
        out[g] = len(variants)
    return out


# ---------------------------------------------------------------------------
# Pass-1 clustering primitives (selected via run_v4_pipeline.merge_primitive)
# ---------------------------------------------------------------------------


def _pass1_misori(
    *,
    OM_raw: np.ndarray,
    SG: int,
    n_alive: int,
    log,
):
    """Pass-1 misori clustering: 2-offset bucket prefilter + smart-antimode θ*.

    Equivalent to the in-line block that used to live in :func:`run_v4_pipeline`
    (extracted verbatim for readability and primitive selection).

    Returns
    -------
    (p1_cluster, twin_edges, attribution, theta_star) :
        - ``p1_cluster``: ``(n_alive,)`` int64 cluster label.
        - ``twin_edges``: empty (twin labelling is Stage-5's job).
        - ``attribution``: ``None`` (misori carries no per-spot evidence).
        - ``theta_star``: float — the selected misori threshold in degrees.
    """
    qs = orient_mat_to_quat(torch.from_numpy(OM_raw)).numpy()
    n_sym, sym = make_symmetries(SG)
    sym_q = np.asarray(sym, dtype=np.float64)
    reps = _qmul(qs[:, None, :], sym_q[None, :, :]).reshape(-1, 4)
    seed_local = np.repeat(np.arange(n_alive), n_sym)
    sgn = np.where(reps[:, 0] >= 0, 1.0, -1.0); reps *= sgn[:, None]
    PRE_DEG = 5.0
    cell = 2.0 * math.sin(math.radians(PRE_DEG) / 2.0)
    enc_chunks: list[np.ndarray] = []
    for offset in (0.0, 0.5 * cell):
        cell_idx = np.floor((reps + offset) / cell).astype(np.int64)
        order = np.lexsort((cell_idx[:, 3], cell_idx[:, 2],
                            cell_idx[:, 1], cell_idx[:, 0]))
        sc = cell_idx[order]; ss = seed_local[order]
        diff_b = np.any(np.diff(sc, axis=0) != 0, axis=1)
        breaks = np.concatenate([[0], np.flatnonzero(diff_b) + 1, [sc.shape[0]]])
        for k in np.flatnonzero(np.diff(breaks) >= 2):
            lo, hi = int(breaks[k]), int(breaks[k+1])
            members = np.unique(ss[lo:hi])
            if members.size < 2: continue
            ii, jj = np.triu_indices(members.size, k=1)
            a = members[ii].astype(np.int64); b = members[jj].astype(np.int64)
            enc_chunks.append(a * (1 << 31) + b)

    if enc_chunks:
        enc = np.unique(np.concatenate(enc_chunks))
        del enc_chunks
        A = (enc >> 31).astype(np.int64); B = (enc & ((1 << 31) - 1)).astype(np.int64)
        CH = 500_000
        miso = np.empty(len(A), dtype=np.float64)
        for s in range(0, len(A), CH):
            e = min(s + CH, len(A))
            qa = torch.from_numpy(np.ascontiguousarray(qs[A[s:e]]))
            qb = torch.from_numpy(np.ascontiguousarray(qs[B[s:e]]))
            miso[s:e] = np.rad2deg(misorientation_quat_batch(qa, qb, SG).numpy())
    else:
        A = np.array([], dtype=np.int64); B = np.array([], dtype=np.int64)
        miso = np.array([])

    # Smart antimode finder on log(misori) histogram.
    from scipy.ndimage import gaussian_filter1d
    if miso.size:
        log_m = np.log10(np.clip(miso, 1e-4, None))
        hist, edges = np.histogram(log_m, bins=120)
        mids = 0.5 * (edges[:-1] + edges[1:])
        smooth = gaussian_filter1d(hist.astype(float), sigma=2.0)
        is_max = np.zeros_like(smooth, dtype=bool)
        is_max[1:-1] = (smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:])
        is_min = np.zeros_like(smooth, dtype=bool)
        is_min[1:-1] = (smooth[1:-1] < smooth[:-2]) & (smooth[1:-1] < smooth[2:])
        max_idx = np.flatnonzero(is_max)
        min_idx = np.flatnonzero(is_min)
        if len(max_idx) >= 2:
            lo_i, hi_i = int(max_idx[0]), int(max_idx[-1])
            between = (min_idx > lo_i) & (min_idx < hi_i)
            if between.any():
                cand = min_idx[between]
                valley = int(cand[np.argmin(smooth[cand])])
                theta_star = float(10 ** mids[valley])
            else:
                theta_star = float(10 ** mids[(lo_i + hi_i) // 2])
        else:
            theta_star = 0.5
        theta_star = float(np.clip(theta_star, 0.1, 1.0))
    else:
        theta_star = 0.05
    log(f"[v4] Pass-1: θ* = {theta_star:.4f}°  (smart antimode from {len(A):,} pairs)")

    parent = np.arange(n_alive, dtype=np.int64)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    keep = miso < theta_star
    for ai, bi in zip(A[keep].tolist(), B[keep].tolist()):
        ra, rb = find(int(ai)), find(int(bi))
        if ra != rb: parent[ra] = rb
    roots = np.array([find(i) for i in range(n_alive)])
    _, p1_cluster = np.unique(roots, return_inverse=True)
    p1_cluster = p1_cluster.astype(np.int64)
    return p1_cluster, np.zeros((0, 2), dtype=np.int32), None, theta_star


def _pass1_forward_predict(
    *,
    OM_fz: np.ndarray,
    positions: np.ndarray,
    hkls,
    indexing_rings: list,
    inputall,
    geom,
    k_agree: Optional[int],
    y_tol_um: float,
    omega_tol_deg: float,
    om_split_tol_deg: Optional[float],
    qs_fz: np.ndarray,
    space_group: int,
    log,
):
    """Pass-1 forward-predict + variant-agreement clustering.

    Builds an HEDMGeometry from the GeometryConfig, predicts ring-spot
    positions for each FZ-canonical OM, snaps to the detected ring-spot
    cloud, and runs the agree/disagree graph from
    :mod:`compute.forward_predict_merge`. Auto-selects ``K_AGREE`` if the
    caller passes ``None``.

    Returns
    -------
    (p1_cluster, twin_edges, attribution, theta_star)
        - ``p1_cluster``: ``(n_alive,)`` int64 cluster label.
        - ``twin_edges``: ``(E, 2)`` int32 cluster-pair twin edges
          (consumed downstream by the twin-labeller).
        - ``attribution``: the
          :class:`compute.forward_predict_merge.ForwardPredictAttribution`
          object (kept for downstream variant-evidence audits).
        - ``theta_star``: ``None`` (the forward-predict primitive does
          not define a misori threshold; it uses variant agreement).
    """
    from midas_diffract.forward import HEDMGeometry
    from .compute.forward_predict_merge import (
        compute_forward_predict_attributions,
        build_forward_predict_graph,
        forward_predict_merge_components,
        select_k_agree_auto,
        select_om_spread_tol_auto,
        split_components_by_om_spread,
    )

    n_alive = int(OM_fz.shape[0])

    # Degenerate input: no rings or no candidates → every candidate is its
    # own cluster (no forward-predict evidence available).
    if not indexing_rings or n_alive == 0:
        log(f"[v4]   forward-predict: no indexing rings or no alive → "
            f"singleton clusters")
        return (
            np.arange(n_alive, dtype=np.int64),
            np.zeros((0, 2), dtype=np.int32),
            None,
            None,
        )

    # The forward-predict primitive runs on a SINGLE indexing ring (the
    # lowest-numbered = brightest, most-populated by the indexer). When
    # multiple indexing rings are listed in paramstest, we use only the
    # first one as the merge ring. Multi-ring evidence pooling is a
    # planned extension (see PROGRESS_v4.md "open gaps") that requires
    # cross-ring attribution-map union + deduplication — kept as a
    # follow-up to avoid disturbing the validated single-ring math.
    ring = int(min(indexing_rings))
    log(f"[v4] Pass-1 forward-predict on ring {ring}  "
        f"(of indexing rings {indexing_rings})")

    keep = hkls.ring == ring
    g_crystal_ring = hkls.g_crystal[keep].astype(np.float64)
    theta_deg_ring = hkls.theta_deg[keep].astype(np.float64)

    # Detected ring-spot cloud from InputAll. ``read_inputall_minimal``
    # returns a DataFrame indexed by SpotID, so SpotID lives on the
    # index — not in ``.columns``.
    ia = inputall
    ring_mask = ia["RingNumber"].astype(int).to_numpy() == ring
    det_y = ia.loc[ring_mask, "YLab"].to_numpy(np.float64)
    det_z = ia.loc[ring_mask, "ZLab"].to_numpy(np.float64)
    det_o = ia.loc[ring_mask, "Omega"].to_numpy(np.float64)
    det_sid = ia.index.to_numpy(np.int64)[ring_mask]
    log(f"[v4]   detected ring-{ring} spots: {len(det_y):,}  variants: {len(g_crystal_ring)}")

    # Materialise an HEDMGeometry from the GeometryConfig
    span = geom.omega_max_deg - geom.omega_min_deg
    n_frames = max(int(round(span / max(geom.omega_step_deg, 1e-6))), 1)
    fm_geom = HEDMGeometry(
        Lsd=geom.lsd_um, y_BC=geom.y_BC, z_BC=geom.z_BC, px=geom.pixel_um,
        omega_start=geom.omega_min_deg, omega_step=geom.omega_step_deg,
        n_frames=n_frames,
        n_pixels_y=geom.n_pixels_y, n_pixels_z=geom.n_pixels_z,
        min_eta=geom.min_eta_deg, wavelength=geom.wavelength_a,
        tx=geom.tx_deg, ty=geom.ty_deg, tz=geom.tz_deg,
        wedge=geom.wedge_deg, flip_y=True, apply_tilts=True,
    )

    t_pred = time.time()
    attrib = compute_forward_predict_attributions(
        OM_fz, positions,
        g_crystal_ring=g_crystal_ring, theta_deg_ring=theta_deg_ring,
        geometry=fm_geom,
        detected_y_um=det_y, detected_z_um=det_z,
        detected_omega_deg=det_o, detected_spot_id=det_sid,
        y_tol_um=y_tol_um, omega_tol_deg=omega_tol_deg,
    )
    log(f"[v4]   {attrib.cand_idx.size:,} attributions  "
        f"(snap rate {100*attrib.snap_rate:.1f}%)  [{time.time()-t_pred:.1f}s]")

    t_graph = time.time()
    graph = build_forward_predict_graph(attrib)
    log(f"[v4]   pair graph: {graph.pair_a.size:,} pairs  "
        f"[{time.time()-t_graph:.1f}s]")

    if k_agree is None:
        K = select_k_agree_auto(graph, n_alive)
        log(f"[v4]   K_AGREE = {K}  (auto-selected: smallest K with "
            f"max-comp ≤ max(100, n_alive/100))")
    else:
        K = int(k_agree)
        log(f"[v4]   K_AGREE = {K}  (user-specified)")

    p1_cluster, twin_edges = forward_predict_merge_components(graph, n_alive, k_agree=K)
    n_pre_split = int(p1_cluster.max() + 1) if len(p1_cluster) else 0

    # Post-split: catch chain-fusion via OM spread within each component.
    # The agree/disagree pair graph only sees pairs with at least one
    # shared snap; transitive merges across non-overlapping evidence
    # chains can fuse OM-divergent candidates into one component
    # (~20% of multi-cand grains on Indrajeet at K=4 spanned >5°).
    # ``om_split_tol_deg=None`` triggers data-driven auto-selection via
    # the misori-histogram antimode (same logic as Pass-1 misori θ*).
    if om_split_tol_deg is not None and om_split_tol_deg > 0:
        t_split = time.time()
        if om_split_tol_deg == "auto" or (isinstance(om_split_tol_deg, float)
                                          and om_split_tol_deg < 0):
            tol_used = select_om_spread_tol_auto(
                p1_cluster, om_fz_quat=qs_fz, space_group=space_group,
            )
            log(f"[v4]   OM-spread tol (auto) = {tol_used:.3f}°")
        else:
            tol_used = float(om_split_tol_deg)
        p1_cluster = split_components_by_om_spread(
            p1_cluster,
            om_fz_quat=qs_fz, space_group=space_group,
            om_tol_deg=tol_used,
        )
        n_post_split = int(p1_cluster.max() + 1) if len(p1_cluster) else 0
        log(f"[v4]   OM-spread split (>{tol_used:.3f}°): "
            f"{n_pre_split:,} → {n_post_split:,} clusters "
            f"(+{n_post_split - n_pre_split:,})  [{time.time()-t_split:.1f}s]")
        # Twin edges from the original components map onto the post-split
        # clusters via the relabel; recompute by taking the new labels.
        if len(twin_edges):
            # twin_edges holds OLD cluster ids; we need to remap to new
            # labels via any representative cand in the old cluster.
            # Note: split_components_by_om_spread reassigns IDs, so we
            # need the cand-level remap.
            # Simpler: drop twin_edges here — they will be reconstructed
            # at Stage 5 from the FZ-misori test on the post-split clusters.
            twin_edges = np.zeros((0, 2), dtype=np.int32)
    return p1_cluster, twin_edges, attrib, None


# ---------------------------------------------------------------------------
# Stage 8 helper — per-grain strain via Kenesei bounded lstsq
# ---------------------------------------------------------------------------


def _compute_strain_per_grain(
    *,
    phys,
    alive_idx: np.ndarray,
    om_fz_per_g: np.ndarray,
    pos_per_g: np.ndarray,
    spot_sets: list,
    inputall_df: pd.DataFrame,
    hkls,
    geometry,
    latc: Optional[np.ndarray],
    log,
) -> np.ndarray:
    """Solve per-grain strain via Kenesei bounded lstsq on matched spots.

    Returns ``(n_grains, 6)`` Voigt strain (ε_xx, ε_yy, ε_zz, ε_xy, ε_xz,
    ε_yz). NaN rows for grains with too few spots / failed solve.
    """
    from .compute.strain import solve_strain_kenesei_bounded
    from .compute.hkl_ingest import (
        _g_lab_from_observed, _rotate_lab_to_sample_by_omega,
    )

    n_g = int(om_fz_per_g.shape[0])
    eps = np.full((n_g, 6), np.nan, dtype=np.float64)

    # Reference d-spacings per ring (Bragg's law: d = λ/(2 sinθ))
    hkl_arr = np.stack([hkls.h, hkls.k, hkls.l], axis=1)
    d0_per_row = hkls.d_A
    g_cryst_per_row = hkls.g_crystal
    ring_per_row = hkls.ring

    # Group by ring for fast lookup of variant per spot
    ring_to_rows = {int(r): np.flatnonzero(ring_per_row == r) for r in np.unique(ring_per_row)}

    lsd_um = float(geometry.lsd_um)

    # InputAll columns needed
    ia_indexed = inputall_df.copy()
    if "Eta" in ia_indexed.columns:
        cols_needed = ["YLab", "ZLab", "Omega", "RingNumber", "Eta"]
    else:
        cols_needed = ["YLab", "ZLab", "Omega", "RingNumber"]

    n_ok = 0
    for g in range(n_g):
        members = np.flatnonzero(phys.final_grain_id == g)
        if len(members) == 0:
            continue
        # Union of matched SpotIDs across grain candidates
        union = set()
        for c in members:
            union |= spot_sets[int(c)]
        if not union:
            continue
        sids = np.fromiter(union, dtype=np.int64)
        try:
            rows = ia_indexed.loc[sids].dropna()
        except KeyError:
            continue
        if len(rows) < 6:
            continue

        Y = rows["YLab"].to_numpy(np.float64)
        Z = rows["ZLab"].to_numpy(np.float64)
        ome = rows["Omega"].to_numpy(np.float64)
        ring_s = rows["RingNumber"].astype(int).to_numpy()

        # Observed g_lab → g_sample → g_crystal (normalised)
        g_lab = _g_lab_from_observed(Y, Z, lsd_um)
        g_sample = _rotate_lab_to_sample_by_omega(g_lab, ome)
        OM = om_fz_per_g[g]
        g_crystal = g_sample @ OM
        g_crystal /= (np.linalg.norm(g_crystal, axis=1, keepdims=True) + 1e-12)

        # Observed d-spacing: λ/(2 sinθ_obs) where 2θ_obs is from the spot geometry
        # 2θ = arctan(sqrt(YLab² + ZLab²) / lsd)
        two_th = np.arctan2(np.sqrt(Y * Y + Z * Z), lsd_um)
        sin_th = np.sin(two_th / 2.0)
        wavelength_A = float(geometry.wavelength_a if hasattr(geometry, "wavelength_a")
                              else getattr(geometry, "wavelength", 0.0))
        if wavelength_A <= 0:
            continue
        d_obs = wavelength_A / (2.0 * np.clip(sin_th, 1e-30, None))

        # Reference d-spacing per spot: pick the hkl variant in the spot's ring
        # whose g_crystal is closest to the observed g_crystal direction.
        d_ref = np.empty(len(rows), dtype=np.float64)
        ok_mask = np.ones(len(rows), dtype=bool)
        for i_spot in range(len(rows)):
            r = int(ring_s[i_spot])
            if r not in ring_to_rows:
                ok_mask[i_spot] = False; continue
            cand_rows = ring_to_rows[r]
            cand_g = g_cryst_per_row[cand_rows]
            cand_g_n = cand_g / (np.linalg.norm(cand_g, axis=1, keepdims=True) + 1e-12)
            cos = cand_g_n @ g_crystal[i_spot]
            best = int(cand_rows[int(np.argmax(cos))])
            d_ref[i_spot] = d0_per_row[best]
        if not ok_mask.any():
            continue
        g_kept = g_crystal[ok_mask]
        d_obs_kept = d_obs[ok_mask]
        d_ref_kept = d_ref[ok_mask]
        if len(g_kept) < 6:
            continue
        try:
            res = solve_strain_kenesei_bounded(
                torch.from_numpy(g_kept),
                torch.from_numpy(d_obs_kept),
                torch.from_numpy(d_ref_kept),
            )
        except Exception:
            continue
        eps[g] = res.epsilon_voigt.detach().cpu().numpy()
        n_ok += 1

    log(f"[v4]   strain solver: {n_ok}/{n_g} grains solved")
    return eps
