"""C-parity output writers: Grains.csv, GrainIDsKey.csv, SpotMatrix.csv.

Format follows ``FF_HEDM/src/ProcessGrains.c`` line-for-line.

Grains.csv layout (47 columns)
------------------------------
  0       GrainID                 (= IDs[rep_pos], the SpotID at the rep seed)
  1..9    OM (3×3 row-major)      OPF[rep_pos][1..9]
  10..12  X, Y, Z                 OPF[rep_pos][11..13]
  13..18  a, b, c, α, β, γ        OPF[rep_pos][15..20]
  19      DiffPos                 OPF[rep_pos][22]
  20      DiffOme                 OPF[rep_pos][23]
  21      DiffAngle (= IA)        OPF[rep_pos][24]
  22      GrainRadius             OPF[rep_pos][25]
  23      Confidence              OPF[rep_pos][26]
  24..32  eFab[3][3] in microstrain (Fable–Beaudoin in sample frame)
  33..41  eKen[3][3] in microstrain (Kenesei in sample frame)
  42      RMSErrorStrain          (Kenesei RMSE in microstrain)
  43      PhaseNr
  44..46  Eul0, Eul1, Eul2        (degrees)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch

from .c_parity import OPF_OM, OPF_POS, OPF_LATTICE, OPF_DIFF_POS, OPF_DIFF_OME, OPF_IA, OPF_RADIUS, OPF_CONFIDENCE
from .c_parity_run import CParityKeptGrain, CParityResult
from .strain import (
    solve_strain_fable_beaudoin,
    solve_strain_kenesei_batched,
    solve_strain_kenesei_bounded,
)


# --------------------------------------------------------------------------
# Per-grain FitBest cache (shared by Grains.csv and SpotMatrix.csv writers)
# --------------------------------------------------------------------------


def gather_per_grain_spot_data(
    kept_grains: List["CParityKeptGrain"],
    fb,                                            # FitBest memmap or None
    *,
    distance_um: float,
    wavelength_a: float,
    ids_hash=None,
    progress: bool = True,
) -> List[Optional[dict]]:
    """Single pass over FitBest, returning one dict per kept grain (or None
    if the grain has no FitBest row).

    The dict carries everything BOTH writers need:
      - ``spot_ids`` (n,) int64           — used by SpotMatrix
      - ``y``, ``z`` (n,) float64         — used by SpotMatrix
      - ``g`` (n, 3) float64              — used by Kenesei (sample frame)
      - ``ds_obs`` (n,) float64           — used by Kenesei
      - ``ds_0``   (n,) float64           — used by Kenesei

    Eliminates the 22 k × 80 KB random-NFS-read round-trip that
    SpotMatrix.csv would otherwise pay a second time.
    """
    out: List[Optional[dict]] = []
    if fb is None:
        return [None] * len(kept_grains)
    fb_n = fb.shape[0]
    n = len(kept_grains)
    if progress:
        print(f"[c-parity emit] gather FitBest cache: {n:,} grains", flush=True)
    import time as _time
    t0 = _time.time()
    next_progress = 0
    for gi, g in enumerate(kept_grains):
        rep = g.rep_pos
        if rep >= fb_n:
            out.append(None)
            continue
        seed = np.array(fb[rep], copy=True)
        sid = seed[:, 0].astype(np.int64)
        valid = sid > 0
        if not valid.any():
            out.append(None)
            continue
        y = seed[valid, 1].astype(np.float64)
        z = seed[valid, 2].astype(np.float64)
        g_v = seed[valid, 4:7].astype(np.float64)
        rho = np.sqrt(y * y + z * z)
        sin_th = np.maximum(np.sin(np.arctan(rho / distance_um) / 2.0), 1e-30)
        ds_o = wavelength_a / (2.0 * sin_th)
        # d₀ per spot, from IDsHash.csv. There is NO safe fallback: Kenesei
        # strain is (d_obs − d₀)/d₀, so a substituted zero does not degrade
        # the answer, it destroys it — every grain pegs at the ±0.01 bound and
        # RMSErrorStrain comes out ~1e36. That is what `np.zeros_like(y)` here
        # produced on both datasetA and shade_LSHR, silently, in runs whose
        # positions and orientations were correct. Refuse instead: the caller
        # checks for IDsHash.csv and raises before reaching this point.
        ds_r = ids_hash.d_for_spot_ids(sid[valid])
        out.append({
            "spot_ids": sid[valid],
            "y": y, "z": z, "g": g_v,
            "ds_obs": ds_o, "ds_0": ds_r,
        })
        if progress and gi >= next_progress:
            print(f"[c-parity emit] gather {gi:,}/{n:,}  "
                  f"[{_time.time()-t0:.1f}s]", flush=True)
            next_progress += max(1, n // 50)
    if progress:
        print(f"[c-parity emit] gather done [{_time.time()-t0:.1f}s]",
              flush=True)
    return out


# --------------------------------------------------------------------------
# Euler angles from orientation matrix (matches C's OrientMat2Euler)
# --------------------------------------------------------------------------


def orient_mat_to_euler_rad(om: np.ndarray) -> np.ndarray:
    """Bit-exact replica of C ``OrientMat2Euler``
    (FF_HEDM/src/GetMisorientation.c:444-467).

    Uses C's ``sin_cos_to_angle(s, c) = acos(c) if s >= 0 else 2π - acos(c)``,
    which is NOT the same as ``atan2(s, c)`` — it is always in [0, 2π).
    Output is **radians** (C ProcessGrains writes radians to Grains.csv).
    """
    EPS = 1e-9
    if om.ndim == 1:
        om = om.reshape(3, 3)
    m22 = om[2, 2]

    def clamp_acos(v: float) -> float:
        return math.acos(max(-1.0, min(1.0, v)))

    def sin_cos_to_angle(s: float, c: float) -> float:
        if c > 1.0:
            c = 1.0
        if c < -1.0:
            c = -1.0
        return math.acos(c) if s >= 0.0 else (2.0 * math.pi - math.acos(c))

    if abs(m22 - 1.0) < EPS:
        phi = 0.0
    else:
        phi = clamp_acos(m22)
    sph = math.sin(phi)
    if abs(sph) < EPS:
        psi = 0.0
        if abs(m22 - 1.0) < EPS:
            theta = sin_cos_to_angle(om[1, 0], om[0, 0])
        else:
            theta = sin_cos_to_angle(-om[1, 0], om[0, 0])
    else:
        if abs(-om[1, 2] / sph) <= 1.0:
            psi = sin_cos_to_angle(om[0, 2] / sph, -om[1, 2] / sph)
        else:
            psi = sin_cos_to_angle(om[0, 2] / sph, 1.0)
        if abs(om[2, 1] / sph) <= 1.0:
            theta = sin_cos_to_angle(om[2, 0] / sph, om[2, 1] / sph)
        else:
            theta = sin_cos_to_angle(om[2, 0] / sph, 1.0)
    return np.array([psi, phi, theta])


# Backwards-compatibility alias removed deliberately — callers must use the
# RADIANS function name to avoid the previous degrees-vs-radians confusion.


# --------------------------------------------------------------------------
# Per-grain strain computation
# --------------------------------------------------------------------------


def compute_strain_for_grain(
    *,
    om_grain: np.ndarray,                      # (3,3) — sample-frame OM at rep
    lattice_strained: np.ndarray,              # (6,) a,b,c,α,β,γ from OPF
    lattice_reference: np.ndarray,             # (6,) reference (paramstest)
    spots_g: Optional[np.ndarray] = None,      # (n_spots, 3) — sample-frame g, FitBest[:, 4:7]
    spots_y: Optional[np.ndarray] = None,      # (n_spots,) — observed y-lab (µm)
    spots_z: Optional[np.ndarray] = None,      # (n_spots,) — observed z-lab (µm)
    spots_d_ref: Optional[np.ndarray] = None,  # (n_spots,) — reference d-spacing
    distance_um: Optional[float] = None,
    wavelength_a: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (eFab, eKen, ken_rmse) — Fable and Kenesei strain in the
    sample frame, in MICROSTRAIN units (×1e6).

    For C-parity Kenesei we follow the C reference exactly
    (FF_HEDM/src/CalcStrains.c:172-200):

      gobs[i] = SpotsInfo[i][0..2] / |SpotsInfo[i][0..2]|
                       (sample-frame g, FitBest cols 4-6, already
                        wedge- and ω-corrected by the indexer)
      dsObs[i] = λ / (2 sin( atan( √(y² + z²) / Lsd ) / 2 ))
                       (observed d-spacing from detector radius)
      B[i]     = (dsObs[i] - ds0[i]) / ds0[i]
      A[i]     = [gx², gy², gz², 2gxgy, 2gxgz, 2gygz]

    Then bounded LSQ for ε ∈ [-0.01, 0.01]^6.

    Critically: the g vector must be the SAMPLE-frame g (already accounts
    for the ω rotation at diffraction). C stores this in FitBest cols
    4-6; recomputing g from (y, z) alone gives the LAB-frame g at the
    diffraction event, which is a *different* vector.
    """
    # ── Fable: from lattice ratio ──────────────────────────────────────────
    lat_t = torch.from_numpy(np.asarray(lattice_strained, dtype=np.float64))
    lat_ref_t = torch.from_numpy(np.asarray(lattice_reference, dtype=np.float64))
    fable_grain = solve_strain_fable_beaudoin(lat_t, lat_ref_t).numpy()
    e_fab_sample = om_grain @ fable_grain @ om_grain.T

    # ── Kenesei: from per-spot residuals ───────────────────────────────────
    if (spots_g is None or spots_g.size == 0 or
            spots_y is None or distance_um is None):
        return e_fab_sample * 1e6, np.zeros((3, 3)), float("nan")

    # ds_obs from radial detector position, exactly as in C.
    rho = np.sqrt(spots_y * spots_y + spots_z * spots_z)
    two_theta = np.arctan(rho / distance_um)
    sin_theta = np.sin(two_theta / 2.0)
    sin_theta = np.maximum(sin_theta, 1e-30)
    ds_obs = wavelength_a / (2.0 * sin_theta)

    # spots_g is already sample-frame. Just re-normalise to be safe.
    g_norm = np.linalg.norm(spots_g, axis=1, keepdims=True)
    g_norm = np.maximum(g_norm, 1e-30)
    g_hat = spots_g / g_norm

    valid = (np.linalg.norm(spots_g, axis=1) > 0) & (spots_d_ref > 0)
    if not valid.any():
        return e_fab_sample * 1e6, np.zeros((3, 3)), float("nan")

    res = solve_strain_kenesei_bounded(
        g_obs=torch.from_numpy(g_hat[valid]),
        ds_obs=torch.from_numpy(ds_obs[valid]),
        ds_0=torch.from_numpy(spots_d_ref[valid]),
    )
    e_ken_sample = res.epsilon_tensor.numpy()
    n_used = max(1, int(res.n_spots))
    ken_rmse = float(res.residual_norm) / math.sqrt(n_used) * 1e6

    return e_fab_sample * 1e6, e_ken_sample * 1e6, ken_rmse


# --------------------------------------------------------------------------
# Grains.csv writer
# --------------------------------------------------------------------------


def write_grains_csv(
    *,
    out_path: Path,
    kept_grains: List[CParityKeptGrain],
    opf: np.ndarray,
    fb: Optional[np.ndarray],
    lattice_reference: np.ndarray,
    distance_um: float,
    wavelength_a: float,
    space_group: int,
    beam_thickness: float = 0.0,
    global_position: float = 0.0,
    num_phases: int = 1,
    phase_nr: int = 1,
    ids_hash=None,
    progress: bool = True,
    device: str = "cpu",
    spot_cache: Optional[List[Optional[dict]]] = None,
) -> dict:
    """Write Grains.csv in C ProcessGrains' 47-column layout.

    Returns a dict of summary stats (BeamCenter, NumGrains, etc.) for the
    caller's diagnostics.
    """
    n = len(kept_grains)
    if progress:
        print(f"[c-parity] writing Grains.csv: {n:,} grains → {out_path}  "
              f"(device={device})", flush=True)

    # Vol-weighted beam center, matches C ProcessGrains.c:1059-1071.
    beam_center_acc = 0.0
    full_vol_acc = 0.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Pass 1: gather per-grain (g, ds_obs, ds_0) for the BATCHED Kenesei.
    #            Compute Fable per-grain (cheap, closed form). Compute Euler.
    import time as _time
    t0 = _time.time()
    n_kept = len(kept_grains)

    # Use a pre-built FitBest cache if provided; else build it now.
    if spot_cache is None:
        spot_cache = gather_per_grain_spot_data(
            kept_grains, fb,
            distance_um=distance_um, wavelength_a=wavelength_a,
            ids_hash=ids_hash, progress=progress,
        )

    g_obs_list: List[np.ndarray] = []
    ds_obs_list: List[np.ndarray] = []
    ds_0_list: List[np.ndarray] = []

    e_fab_per_grain: List[np.ndarray] = []   # (3, 3) microstrain per grain
    eul_per_grain:   List[np.ndarray] = []   # (3,) per grain (radians)
    valid_strain:    List[bool] = []         # True iff Kenesei feasible

    for gi, g in enumerate(kept_grains):
        om = g.orient_mat

        # Fable (closed form, in microstrain).
        lat_t = torch.from_numpy(np.asarray(g.lattice, dtype=np.float64))
        lat_ref_t = torch.from_numpy(
            np.asarray(lattice_reference, dtype=np.float64)
        )
        fable_grain = solve_strain_fable_beaudoin(lat_t, lat_ref_t).numpy()
        e_fab = (om @ fable_grain @ om.T) * 1e6
        e_fab_per_grain.append(e_fab)
        eul_per_grain.append(orient_mat_to_euler_rad(om))

        # Pull per-grain Kenesei inputs from the cache.
        cache = spot_cache[gi]
        if cache is None:
            g_obs_list.append(np.empty((0, 3), dtype=np.float64))
            ds_obs_list.append(np.empty(0, dtype=np.float64))
            ds_0_list.append(np.empty(0, dtype=np.float64))
            valid_strain.append(False)
            continue
        g_obs_list.append(cache["g"])
        ds_obs_list.append(cache["ds_obs"])
        ds_0_list.append(cache["ds_0"])
        valid_strain.append(True)

    # ── Pass 2: BATCHED Kenesei solve (one tensor op for all B grains).
    print(f"[c-parity emit] batched Kenesei: {n_kept:,} grains on {device}",
          flush=True)
    t1 = _time.time()
    eps_voigt, rmse = solve_strain_kenesei_batched(
        g_obs_list, ds_obs_list, ds_0_list, device=device,
    )
    eps_voigt_np = eps_voigt.detach().cpu().numpy()
    rmse_np = rmse.detach().cpu().numpy()
    # Voigt → 3×3 symmetric tensor in sample frame, in microstrain.
    e_ken_per_grain = np.zeros((n_kept, 3, 3), dtype=np.float64)
    for gi, ev in enumerate(eps_voigt_np):
        if not valid_strain[gi]:
            continue
        e_ken_per_grain[gi, 0, 0] = ev[0]
        e_ken_per_grain[gi, 1, 1] = ev[1]
        e_ken_per_grain[gi, 2, 2] = ev[2]
        e_ken_per_grain[gi, 0, 1] = e_ken_per_grain[gi, 1, 0] = ev[3]
        e_ken_per_grain[gi, 0, 2] = e_ken_per_grain[gi, 2, 0] = ev[4]
        e_ken_per_grain[gi, 1, 2] = e_ken_per_grain[gi, 2, 1] = ev[5]
    e_ken_per_grain *= 1e6                                           # microstrain
    rmse_per_grain = rmse_np * 1e6                                   # microstrain
    rmse_per_grain[~np.asarray(valid_strain)] = float("nan")
    print(f"[c-parity emit] batched solve done [{_time.time()-t1:.1f}s]",
          flush=True)

    # ── Pass 3: build rows + accumulate beam-center moment.
    rows: List[List[float]] = []
    beam_center_acc = 0.0
    full_vol_acc = 0.0
    for gi, g in enumerate(kept_grains):
        rep = g.rep_pos
        om = g.orient_mat
        position = g.position
        lattice = g.lattice
        diff_pos = g.diff_pos
        diff_ome = g.diff_ome
        diff_angle = g.diff_angle
        radius = g.grain_radius
        confidence = g.confidence
        e_fab = e_fab_per_grain[gi]
        e_ken = e_ken_per_grain[gi]
        ken_rmse = rmse_per_grain[gi]
        eul_rad = eul_per_grain[gi]

        # Assemble 47-col row (matches ProcessGrains.c:1039-1058).
        row = [0.0] * 47
        row[0] = float(g.grain_id)
        # OPs[i][0..20] mapping (see c_parity.OPF_* constants):
        # OPs[0..8] = OPF[1..9] = OM
        # OPs[9..11] = OPF[11..13] = X, Y, Z
        # OPs[12..17] = OPF[15..20] = lattice
        # OPs[18..20] = OPF[22..24] = DiffPos, DiffOme, DiffAngle
        for k in range(9):
            row[1 + k] = float(om.flat[k])
        row[10] = float(position[0])
        row[11] = float(position[1])
        row[12] = float(position[2])
        for k in range(6):
            row[13 + k] = float(lattice[k])
        row[19] = float(diff_pos)
        row[20] = float(diff_ome)
        row[21] = float(diff_angle)
        row[22] = float(radius)
        row[23] = float(confidence)
        for r in range(3):
            for c in range(3):
                row[24 + 3 * r + c] = float(e_fab[r, c])
                row[33 + 3 * r + c] = float(e_ken[r, c])
        row[42] = float(ken_rmse)
        row[43] = float(phase_nr)
        row[44] = float(eul_rad[0])
        row[45] = float(eul_rad[1])
        row[46] = float(eul_rad[2])
        rows.append(row)

        v_norm = radius * radius * radius
        beam_center_acc += position[2] * v_norm     # row[12] = Z
        full_vol_acc += v_norm


    beam_center = (beam_center_acc / full_vol_acc) if full_vol_acc > 0 else 0.0

    # Now write the file.
    with open(out_path, "w") as f:
        f.write(f"%NumGrains {n}\n")
        f.write(f"%BeamCenter {beam_center:f}\n")
        f.write(f"%BeamThickness {beam_thickness:f}\n")
        f.write(f"%GlobalPosition {global_position:f}\n")
        f.write(f"%NumPhases {num_phases}\n")
        f.write(f"%PhaseInfo\n%\tSpaceGroup:{space_group}\n")
        f.write(f"%\tLattice Parameter: "
                f"{lattice_reference[0]:f} {lattice_reference[1]:f} "
                f"{lattice_reference[2]:f} {lattice_reference[3]:f} "
                f"{lattice_reference[4]:f} {lattice_reference[5]:f}\n")
        f.write(
            "%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\tX\tY\tZ\t"
            "a\tb\tc\talpha\tbeta\tgamma\tDiffPos\tDiffOme\tDiffAngle\t"
            "GrainRadius\tConfidence\t"
            "eFab11\teFab12\teFab13\teFab21\teFab22\teFab23\teFab31\teFab32\teFab33\t"
            "eKen11\teKen12\teKen13\teKen21\teKen22\teKen23\teKen31\teKen32\teKen33\t"
            "RMSErrorStrain\tPhaseNr\tEul0\tEul1\tEul2\n"
        )
        for row in rows:
            # Match C printf %d for GrainID/PhaseNr, %lf for the rest.
            line = (f"{int(row[0])}\t"
                    + "\t".join(f"{v:f}" for v in row[1:43])
                    + f"\t{int(row[43])}\t"
                    + "\t".join(f"{v:f}" for v in row[44:47])
                    + "\n")
            f.write(line)

    return {
        "n_grains": n,
        "beam_center": beam_center,
        "full_vol": full_vol_acc,
    }


# --------------------------------------------------------------------------
# SpotMatrix.csv writer (C ProcessGrains.c:1075-1092 layout)
# --------------------------------------------------------------------------


def load_input_extra_info_matrix(path: Path) -> np.ndarray:
    """Parse ``InputAllExtraInfoFittingAll.csv`` into ``(N, 10)`` float64.

    Mirrors C's sscanf at ProcessGrains.c:788-794 — only the columns the
    C code actually keeps are stored. The remaining columns (GrainRadius,
    OmegaIni, YOrig, ZOrig, intensity, mask, FitRMSE) are skipped.

    File format:
      0:YLab  1:ZLab  2:Omega  3:GrainRadius(skip)  4:SpotID  5:RingNumber
      6:Eta   7:2θ    8..10:skip                    11:YOrig(DetCor)
      12:ZOrig(DetCor)  13:OmegaOrig(DetCor)        14..17:skip

    Output column layout (matches C's ``InputMatrix[rowSpotID][...]``):
      [0]=Omega, [1]=SpotID, [2]=DetectorHor, [3]=DetectorVert,
      [4]=Eta,   [5]=RingNumber, [6]=YLab, [7]=ZLab,
      [8]=2*Theta, [9]=OmeRaw
    """
    import pandas as pd
    print(f"[c-parity emit] reading InputAllExtraInfoFittingAll.csv …", flush=True)
    df = pd.read_csv(
        path, sep=r"\s+", skiprows=1, header=None,
        usecols=[0, 1, 2, 4, 5, 6, 7, 11, 12, 13],
        names=["YLab", "ZLab", "Omega", "SpotID", "RingNr", "Eta", "TwoTheta",
               "DetH", "DetV", "OmeRaw"],
        dtype=np.float64, engine="c",
    )
    n = len(df)
    out = np.empty((n, 10), dtype=np.float64)
    out[:, 0] = df["Omega"].values
    out[:, 1] = df["SpotID"].values
    out[:, 2] = df["DetH"].values
    out[:, 3] = df["DetV"].values
    out[:, 4] = df["Eta"].values
    out[:, 5] = df["RingNr"].values
    out[:, 6] = df["YLab"].values
    out[:, 7] = df["ZLab"].values
    out[:, 8] = df["TwoTheta"].values
    out[:, 9] = df["OmeRaw"].values
    return out


def write_spot_matrix_csv(
    *,
    out_path: Path,
    kept_grains: List[CParityKeptGrain],
    fb,                                            # FitBest memmap (N, 5000, 22)
    input_matrix: np.ndarray,                      # (n_input, 10) from load_input_extra_info_matrix
    progress: bool = True,
    spot_cache: Optional[List[Optional[dict]]] = None,
) -> int:
    """Write SpotMatrix.csv in C ProcessGrains' 12-column layout.

    Per C ProcessGrains.c:1011-1037, one row per (kept_grain, matched_spot).
    Columns:
      0  GrainID (= grain.grain_id = SpotID at rep)
      1  SpotID
      2  Omega           IM[0]
      3  DetectorHor     IM[2]
      4  DetectorVert    IM[3]
      5  OmeRaw          IM[9]
      6  Eta             IM[4]
      7  RingNr (int)    IM[5]
      8  YLab            IM[6]
      9  ZLab            IM[7]
     10  Theta           IM[8] / 2
     11  StrainError     (per-spot Kenesei residual; 0 in this first pass)

    Returns number of rows written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_input = input_matrix.shape[0]
    n = len(kept_grains)
    if progress:
        print(f"[c-parity emit] writing SpotMatrix.csv: {n:,} grains → {out_path}",
              flush=True)

    # Prefer the shared cache; fall back to re-reading FitBest on-the-fly.
    if spot_cache is None and fb is not None:
        # FitBest cache not provided; do a single gather here. For SpotMatrix
        # we don't need ds_obs / ds_0, but the gather function fills them
        # cheaply. Pass any non-None ids_hash sentinel; ds_0 will be zeros.
        from .c_parity_emit import gather_per_grain_spot_data    # self-import
        spot_cache = gather_per_grain_spot_data(
            kept_grains, fb,
            distance_um=1.0, wavelength_a=1.0,
            ids_hash=None, progress=progress,
        )

    if spot_cache is None:
        if progress:
            print(f"[c-parity emit] no FitBest cache and no FitBest mmap "
                  f"— SpotMatrix.csv will be empty", flush=True)
        with open(out_path, "w") as out:
            out.write(
                "%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw"
                "\tEta\tRingNr\tYLab\tZLab\tTheta\tStrainError\n"
            )
        return 0

    import time as _time
    t0 = _time.time()

    # ── Vectorised gather: concatenate every grain's (grain_id, spot_id)
    #    list, then look up input_matrix once for all rows.
    grain_ids_chunks: List[np.ndarray] = []
    spot_ids_chunks: List[np.ndarray] = []
    for gi, g in enumerate(kept_grains):
        cache = spot_cache[gi]
        if cache is None:
            continue
        sids = cache["spot_ids"]
        n_g = sids.size
        if n_g == 0:
            continue
        grain_ids_chunks.append(np.full(n_g, g.grain_id, dtype=np.int64))
        spot_ids_chunks.append(sids.astype(np.int64))

    if not grain_ids_chunks:
        # No rows to write — emit just the header.
        with open(out_path, "w") as out:
            out.write(
                "%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw"
                "\tEta\tRingNr\tYLab\tZLab\tTheta\tStrainError\n"
            )
        return 0

    all_gid = np.concatenate(grain_ids_chunks)
    all_sid = np.concatenate(spot_ids_chunks)
    row_idx = all_sid - 1
    valid = (row_idx >= 0) & (row_idx < n_input)
    all_gid = all_gid[valid]
    all_sid = all_sid[valid]
    im_rows = input_matrix[row_idx[valid]]

    # Assemble (N, 12) output, then write in one np.savetxt call.
    N = all_gid.shape[0]
    out_arr = np.empty((N, 12), dtype=np.float64)
    out_arr[:, 0]  = all_gid                          # GrainID
    out_arr[:, 1]  = all_sid                          # SpotID
    out_arr[:, 2]  = im_rows[:, 0]                    # Omega
    out_arr[:, 3]  = im_rows[:, 2]                    # DetectorHor
    out_arr[:, 4]  = im_rows[:, 3]                    # DetectorVert
    out_arr[:, 5]  = im_rows[:, 9]                    # OmeRaw
    out_arr[:, 6]  = im_rows[:, 4]                    # Eta
    out_arr[:, 7]  = im_rows[:, 5]                    # RingNr (int via %d)
    out_arr[:, 8]  = im_rows[:, 6]                    # YLab
    out_arr[:, 9]  = im_rows[:, 7]                    # ZLab
    out_arr[:, 10] = im_rows[:, 8] / 2.0              # Theta
    out_arr[:, 11] = 0.0                              # StrainError (placeholder)

    # Match C printf at ProcessGrains.c:1021-1029. Note the trailing '\t'
    # before the newline — C's format string ends in "%lf\t\n", so we use
    # ``newline='\t\n'`` to recreate it.
    fmt = "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%d\t%f\t%f\t%f\t%f"
    with open(out_path, "w") as out:
        out.write(
            "%GrainID\tSpotID\tOmega\tDetectorHor\tDetectorVert\tOmeRaw"
            "\tEta\tRingNr\tYLab\tZLab\tTheta\tStrainError\n"
        )
        np.savetxt(out, out_arr, fmt=fmt, newline='\t\n')
    n_written = N

    if progress:
        print(f"[c-parity emit] SpotMatrix done: {n_written:,} rows  "
              f"in {_time.time()-t0:.1f}s", flush=True)
    return n_written


# --------------------------------------------------------------------------
# GrainIDsKey.csv writer (matches the in-Stage-1 format C uses)
# --------------------------------------------------------------------------


def write_grain_ids_key(
    *,
    out_path: Path,
    kept_grains: List[CParityKeptGrain],
) -> None:
    """One line per kept grain: ``rep_id rep_pos [other_id other_pos]+``.

    Matches the format C ProcessGrains writes from inside Stage 1's
    parallel-for emit (ProcessGrains.c:701-714), but only for *kept*
    grains (PassA + confidence filter survivors).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for g in kept_grains:
            tokens = [str(g.grain_id), str(g.rep_pos)]
            for mid, mpos in zip(g.member_ids, g.member_positions):
                if int(mpos) == g.rep_pos:
                    continue
                tokens.append(str(int(mid)))
                tokens.append(str(int(mpos)))
            f.write(" ".join(tokens) + " \n")
