"""Indexing-mode sinogram assembly.

Pure-Python port of :c:func:`generate_sinograms_from_indexing`
(findSingleSolutionPFRefactored.c:1882–2425). Differences from
tolerance-mode (:mod:`._sinogen`):

  1. **Source of spots**. The C code walks every voxel's indexer
     candidates (from ``IndexBest_all.bin``), and attributes each
     candidate's matched spot IDs to the unique grain whose orientation
     matches within ``max_ang_deg``.
  2. **Confidence filter** ``MIDAS_PF_SINO_CONF_MIN`` (default 0.5,
     env-overridable). Candidates below this are dropped — heuristic to
     trim air-region matches in polycrystalline data.
  3. **Scan-position consistency filter** ``MIDAS_PF_SINO_SCAN_TOL``
     (default 1.5 µm, env-overridable). For each candidate spot the
     code checks the spot's observed scan position against the voxel's
     ``s_V_at_ome = -x_V*cos(ome) + y_V*sin(ome)``; if neither the
     direct match nor the Friedel-pair sign flip passes
     ``|s_V_at_ome ± s_observed| <= s_scan_tol``, the spot is rejected
     as spurious. This is the key fix (~75% reduction in spurious cells
     in real polycrystalline data).

The remaining grouping → sino-fill → sort-by-omega → write logic is
identical to :mod:`._sinogen`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from midas_stress.orientation import (
    misorientation_om_batch,
    orient_mat_to_quat,
)

from ._consolidation_io import (
    CONSOLIDATED_VALS_COLS,
    ConsolidatedReader,
)
from ._sinogen import (
    _apply_variant,
    SPOTS_ARRAY_COLS,
    SinogenOutputs,
    write_clean_variant,
    write_occupancy,
)
from ._geom import ScanGrid

_DEG2RAD = np.pi / 180.0


@dataclass
class _CollectedSpot:
    spot_id: int
    scan_nr: int
    intensity: float
    omega: float
    eta: float
    y_cen: float
    z_cen: float
    theta: float
    ring_nr: int


def _read_env_filters(default_conf: float, default_scan_tol: float) -> tuple[float, float]:
    """Resolve ``MIDAS_PF_SINO_CONF_MIN`` and ``MIDAS_PF_SINO_SCAN_TOL`` envs.

    Mirrors the C ``s_filt_loaded`` block (lines 1971–1990). Bad values
    fall back to the defaults.
    """
    conf = default_conf
    e = os.environ.get("MIDAS_PF_SINO_CONF_MIN")
    if e:
        try:
            v = float(e)
            if 0.0 <= v <= 1.0:
                conf = v
        except ValueError:
            pass
    scan_tol = default_scan_tol
    s = os.environ.get("MIDAS_PF_SINO_SCAN_TOL")
    if s:
        try:
            scan_tol = float(s)
        except ValueError:
            pass
    return conf, scan_tol


def generate_sinograms_indexing(
    unique_key_arr: np.ndarray,
    unique_OM_arr: np.ndarray,
    all_spots: np.ndarray,
    *,
    n_scans: int,
    space_group: int,
    max_ang_deg: float,
    tol_ome: float,
    tol_eta: float,
    output_dir: str | Path,
    vals_reader: ConsolidatedReader,
    keys_reader: ConsolidatedReader,
    ids_reader: ConsolidatedReader,
    scan_grid: Optional[ScanGrid] = None,
    confidence_min: float = 0.5,
    scan_tolerance_um: float = 1.5,
    normalize_sino: bool = False,
    abs_transform: bool = False,
    conc_threshold: float = 0.0,
    conc_min_band_um: float = 4.0,
) -> SinogenOutputs:
    """Build sinograms in indexing mode + write all output files.

    Confidence threshold and scan tolerance default to the
    ``MIDAS_PF_SINO_CONF_MIN`` / ``MIDAS_PF_SINO_SCAN_TOL`` values
    (env-resolved at call time and overriding the keyword arguments).

    See module docstring for semantics.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_spots = np.ascontiguousarray(all_spots, dtype=np.float64)
    n_spots_all = int(all_spots.shape[0])
    n_grains = int(unique_key_arr.shape[0])
    total_vox = n_scans * n_scans

    if scan_grid is None:
        # Default = identity mapping; no scan-position filter possible.
        spatial_positions = None
        scan_to_spatial = np.arange(n_scans, dtype=np.int64)
    else:
        spatial_positions = scan_grid.spatial_positions
        scan_to_spatial = scan_grid.scan_to_spatial

    # Env overrides — caller-provided defaults give way to env if set.
    conf_min, scan_tol = _read_env_filters(confidence_min, scan_tolerance_um)

    # Precompute grain quaternions (one per unique grain). Used only for
    # legacy parity; the batch miso path uses OMs directly.
    if n_grains > 0:
        grain_OMs = np.ascontiguousarray(unique_OM_arr, dtype=np.float64)
    else:
        grain_OMs = np.empty((0, 9), dtype=np.float64)

    # Per-grain collected-spot buffers + dedup-by-spotID set.
    grain_spots: list[list[_CollectedSpot]] = [[] for _ in range(n_grains)]
    seen_spot_ids: list[set] = [set() for _ in range(n_grains)]

    for vox_nr in range(total_vox):
        n_cand = int(keys_reader.n_sol_arr[vox_nr])
        if n_cand <= 0:
            continue
        keys_data = keys_reader.get_keys(vox_nr)
        vals_data = vals_reader.get_vals(vox_nr)
        if keys_data is None or vals_data is None:
            continue
        all_ids = ids_reader.get_ids(vox_nr)
        total_ids_for_vox = int(ids_reader.n_sol_arr[vox_nr])
        if all_ids is None or total_ids_for_vox <= 0:
            continue

        # Voxel V's spatial position. If we don't have spatial_positions
        # we can't apply the scan-position filter; fall through with
        # x_V = y_V = 0 and the filter is effectively disabled.
        if spatial_positions is not None:
            v_row = vox_nr // n_scans
            v_col = vox_nr % n_scans
            x_V = float(spatial_positions[v_col])
            y_V = float(spatial_positions[v_row])
        else:
            x_V = 0.0
            y_V = 0.0

        # Confidence per candidate (col15 / col14).
        denom = vals_data[:, 14]
        with np.errstate(divide="ignore", invalid="ignore"):
            confs = np.where(denom > 0, vals_data[:, 15] / denom, 0.0)

        # Candidate OMs (cols 2..11).
        cand_OMs = vals_data[:, 2:11]  # (n_cand, 9)

        # Pre-filter by confidence.
        ok_conf = confs >= conf_min

        # Cumulative ID offset across solutions; we have to advance this
        # for *every* candidate regardless of conf so it stays aligned.
        id_offset = 0
        for ci in range(n_cand):
            n_ids_this = int(keys_data[ci, 2])
            if not ok_conf[ci]:
                id_offset += n_ids_this
                continue

            # Misorientation vs each grain. The C code breaks on the
            # first matching grain (first-match wins). Vectorize the
            # batch and pick first index <max_ang_rad.
            if n_grains == 0:
                id_offset += n_ids_this
                continue
            OMs1 = np.broadcast_to(cand_OMs[ci], (n_grains, 9))
            angles = misorientation_om_batch(OMs1, grain_OMs, space_group)
            max_ang_rad = max_ang_deg * _DEG2RAD
            matches = np.where(angles < max_ang_rad)[0]
            if matches.size == 0:
                id_offset += n_ids_this
                continue
            g = int(matches[0])  # first-match wins

            # Walk this candidate's spot IDs and apply per-spot filters.
            end = id_offset + n_ids_this
            if end > total_ids_for_vox:
                id_offset += n_ids_this
                continue
            ids_this = all_ids[id_offset:end]
            for sid in ids_this:
                sid_i = int(sid)
                if sid_i < 1 or sid_i > n_spots_all:
                    continue
                if sid_i in seen_spot_ids[g]:
                    continue
                idx = sid_i - 1
                if int(all_spots[idx, 4]) != sid_i:
                    continue

                # Scan-position consistency filter (the key fix).
                if scan_tol > 0.0 and spatial_positions is not None:
                    ome_s = float(all_spots[idx, 2])
                    cw = float(np.cos(ome_s * _DEG2RAD))
                    sw = float(np.sin(ome_s * _DEG2RAD))
                    s_V_at_ome = -x_V * cw + y_V * sw
                    spot_scan_nr = int(all_spots[idx, 9])
                    if spot_scan_nr < 0 or spot_scan_nr >= n_scans:
                        continue
                    s_observed = float(spatial_positions[int(scan_to_spatial[spot_scan_nr])])
                    if (
                        abs(s_V_at_ome - s_observed) > scan_tol
                        and abs(s_V_at_ome + s_observed) > scan_tol
                    ):
                        continue

                seen_spot_ids[g].add(sid_i)
                grain_spots[g].append(
                    _CollectedSpot(
                        spot_id=sid_i,
                        scan_nr=int(all_spots[idx, 9]),
                        intensity=float(all_spots[idx, 3]),
                        omega=float(all_spots[idx, 2]),
                        eta=float(all_spots[idx, 6]),
                        y_cen=float(all_spots[idx, 0]),
                        z_cen=float(all_spots[idx, 1]),
                        theta=float(all_spots[idx, 7]),
                        ring_nr=int(all_spots[idx, 5]),
                    )
                )
            id_offset += n_ids_this

    # --- Phase 2: group collected spots into HKL slots per grain.
    # spotSlot[g][i] = HKL slot (0..nextSlot-1).
    nr_hkls_per_grain = np.zeros(n_grains, dtype=np.int32)
    spot_slot: list[np.ndarray] = []
    max_n_hkls = 0
    for g in range(n_grains):
        ncs = len(grain_spots[g])
        slots = np.full(max(ncs, 1), -1, dtype=np.int64)
        next_slot = 0
        for i in range(ncs):
            if slots[i] >= 0:
                continue
            slots[i] = next_slot
            for j in range(i + 1, ncs):
                if slots[j] >= 0:
                    continue
                a = grain_spots[g][i]
                b = grain_spots[g][j]
                if (
                    a.ring_nr == b.ring_nr
                    and abs(b.omega - a.omega) < tol_ome
                    and abs(b.eta - a.eta) < tol_eta
                ):
                    slots[j] = next_slot
            next_slot += 1
        nr_hkls_per_grain[g] = next_slot
        if next_slot > max_n_hkls:
            max_n_hkls = next_slot
        spot_slot.append(slots[:ncs] if ncs > 0 else slots[:0])

    if max_n_hkls == 0:
        raise RuntimeError(
            "generate_sinograms_indexing: no spots collected for any of "
            f"{n_grains} grains."
        )

    # --- Phase 3: fill sino arrays.
    sz_shape = (n_grains, max_n_hkls, n_scans)
    sino = np.zeros(sz_shape, dtype=np.float64)
    spot_id_arr = np.full(sz_shape, -1, dtype=np.int32)
    spot_meta = np.full(sz_shape + (4,), np.nan, dtype=np.float64)
    max_int = np.zeros((n_grains, max_n_hkls), dtype=np.float64)
    sum_ome = np.zeros((n_grains, max_n_hkls), dtype=np.float64)
    count_ome = np.zeros((n_grains, max_n_hkls), dtype=np.int64)

    for g in range(n_grains):
        for si in range(len(grain_spots[g])):
            slot = int(spot_slot[g][si])
            cs = grain_spots[g][si]
            if slot < 0 or slot >= max_n_hkls:
                continue
            if cs.scan_nr < 0 or cs.scan_nr >= n_scans:
                continue
            spatial_col = int(scan_to_spatial[cs.scan_nr])
            if cs.intensity > sino[g, slot, spatial_col]:
                sino[g, slot, spatial_col] = cs.intensity
                spot_id_arr[g, slot, spatial_col] = cs.spot_id
                spot_meta[g, slot, spatial_col, 0] = cs.eta
                spot_meta[g, slot, spatial_col, 1] = cs.theta * 2.0
                spot_meta[g, slot, spatial_col, 2] = cs.y_cen
                spot_meta[g, slot, spatial_col, 3] = cs.z_cen
            if cs.intensity > max_int[g, slot]:
                max_int[g, slot] = cs.intensity
            if cs.intensity > 0:
                sum_ome[g, slot] += cs.omega
                count_ome[g, slot] += 1

    # --- Phase 4: average omegas.
    ome_arr = np.full((n_grains, max_n_hkls), -10000.0, dtype=np.float64)
    for g in range(n_grains):
        for s in range(max_n_hkls):
            if count_ome[g, s] > 0:
                ome_arr[g, s] = sum_ome[g, s] / count_ome[g, s]

    # --- Phase 5: sort within each grain by omega.
    for g in range(n_grains):
        valid = np.where(ome_arr[g] > -9999.0)[0]
        if valid.size == 0:
            continue
        order = valid[np.argsort(ome_arr[g, valid], kind="stable")]
        new_ome = np.full(max_n_hkls, -10000.0, dtype=np.float64)
        new_sino = np.zeros((max_n_hkls, n_scans), dtype=np.float64)
        new_sid = np.full((max_n_hkls, n_scans), -1, dtype=np.int32)
        new_meta = np.full((max_n_hkls, n_scans, 4), np.nan, dtype=np.float64)
        new_maxI = np.zeros(max_n_hkls, dtype=np.float64)
        for k_new, k_old in enumerate(order):
            new_ome[k_new] = ome_arr[g, k_old]
            new_sino[k_new] = sino[g, k_old]
            new_sid[k_new] = spot_id_arr[g, k_old]
            new_meta[k_new] = spot_meta[g, k_old]
            new_maxI[k_new] = max_int[g, k_old]
        max_int[g] = new_maxI
        ome_arr[g] = new_ome
        sino[g] = new_sino
        spot_id_arr[g] = new_sid
        spot_meta[g] = new_meta

    raw_sino = sino.copy()
    main_sino = _apply_variant(raw_sino, max_int, normalize=normalize_sino, abs_transform=abs_transform)

    nG = n_grains
    nH = max_n_hkls
    nS = n_scans
    main_name = f"sinos_{nG}_{nH}_{nS}.bin"
    omegas_name = f"omegas_{nG}_{nH}.bin"
    hkls_name = f"nrHKLs_{nG}.bin"
    spot_map_name = f"spotMapping_{nG}_{nH}_{nS}.bin"
    spot_meta_name = f"spotMeta_{nG}_{nH}_{nS}.bin"

    (out_dir / main_name).write_bytes(main_sino.astype(np.float64, copy=False).tobytes())
    (out_dir / omegas_name).write_bytes(ome_arr.astype(np.float64, copy=False).tobytes())
    (out_dir / hkls_name).write_bytes(nr_hkls_per_grain.astype(np.int32, copy=False).tobytes())
    (out_dir / spot_map_name).write_bytes(spot_id_arr.astype(np.int32, copy=False).tobytes())
    (out_dir / spot_meta_name).write_bytes(spot_meta.astype(np.float64, copy=False).tobytes())

    sino_paths: dict[str, str] = {"main": str(out_dir / main_name)}
    for label, do_norm, do_abs in [
        ("raw", False, False),
        ("norm", True, False),
        ("abs", False, True),
        ("normabs", True, True),
    ]:
        arr = _apply_variant(raw_sino, max_int, normalize=do_norm, abs_transform=do_abs)
        fn = f"sinos_{label}_{nG}_{nH}_{nS}.bin"
        (out_dir / fn).write_bytes(arr.astype(np.float64, copy=False).tobytes())
        sino_paths[label] = str(out_dir / fn)

    # Per-grain occupancy — always written (see write_occupancy).
    sino_paths["occupancy"] = write_occupancy(
        out_dir, raw_sino, nr_hkls_per_grain,
    )

    # Concentration-filtered variant (off unless conc_threshold > 0).
    if conc_threshold and conc_threshold > 0:
        clean_path, conc_path = write_clean_variant(
            out_dir, raw_sino, ome_arr, nr_hkls_per_grain,
            conc_threshold=conc_threshold,
            scan_positions=spatial_positions,
            min_band_um=conc_min_band_um,
        )
        sino_paths["clean"] = clean_path
        sino_paths["conc"] = conc_path

    return SinogenOutputs(
        n_grains=nG,
        max_n_hkls=nH,
        n_scans=nS,
        sino_paths=sino_paths,
        omegas_path=str(out_dir / omegas_name),
        nr_hkls_path=str(out_dir / hkls_name),
        spot_map_path=str(out_dir / spot_map_name),
        spot_meta_path=str(out_dir / spot_meta_name),
    )
