"""Seed-candidate enumeration.

Three strategies, selected by `IndexerParams.UseFriedelPairs`:
  0 -> generate_ideal_spots               (no Friedel pair)
  1 -> generate_ideal_spots_friedel       (true Friedel pair)
  2 -> generate_ideal_spots_friedel_mixed (mixed: ideal spots + Friedel filter)

Mirrors `GenerateIdealSpots*` from `FF_HEDM/src/IndexerOMP.c:867-1224`.

Each function returns a (n_seeds, 2) tensor of `(y0, z0)` seed positions.
The orientation grid (`orientation_grid.generate_candidate_orientations`) is
constructed per-seed downstream.
"""

from __future__ import annotations

import math

import torch

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def _calc_eta_angle(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Eta in degrees, atan2(-y, z) -> [-180, 180]."""
    return torch.atan2(-y, z) * RAD2DEG


def _calc_spot_position(
    ring_radius: float, eta_deg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = -sin(eta) * R; z = cos(eta) * R."""
    eta_rad = eta_deg * DEG2RAD
    return -torch.sin(eta_rad) * ring_radius, torch.cos(eta_rad) * ring_radius


def _eta_pole_equator(
    ring_rad: float, hbeam: float, rsample: float, ttheta_deg: float, eta_deg: float
) -> tuple[float, float, float]:
    """Returns (eta_pole, eta_equator, hbeam_eff). Mirrors the head of
    GenerateIdealSpots / FriedelEtaCalculation."""
    if eta_deg > 90:
        eta_hbeam = 180 - eta_deg
    elif eta_deg < -90:
        eta_hbeam = 180 - abs(eta_deg)
    else:
        eta_hbeam = 90 - abs(eta_deg)
    hbeam_eff = hbeam + 2.0 * (
        rsample * math.tan(ttheta_deg * DEG2RAD)
    ) * math.sin(eta_hbeam * DEG2RAD)
    eta_pole = 1.0 + RAD2DEG * math.acos(max(-1.0, min(1.0, 1.0 - (hbeam_eff / ring_rad))))
    eta_equator = 1.0 + RAD2DEG * math.acos(max(-1.0, min(1.0, 1.0 - (rsample / ring_rad))))
    return eta_pole, eta_equator, hbeam_eff


def _eta_quadrant(eta_deg: float, eta_pole: float, eta_equator: float):
    """Returns the (quadr_coeff, coeff_y0, coeff_z0) triple matching C's
    branching at IndexerOMP.c:887-947 / 770-797. Returns
    (0, None, None) if the eta falls in a wrap-around band; the caller
    handles those cases via quadr_coeff2 logic.
    """
    quadr_coeff = 0
    coeff_y0 = 0.0
    coeff_z0 = 0.0
    if eta_pole <= eta_deg <= (90 - eta_equator):
        quadr_coeff, coeff_y0, coeff_z0 = 1, -1.0, 1.0
    elif (90 + eta_equator) <= eta_deg <= (180 - eta_pole):
        quadr_coeff, coeff_y0, coeff_z0 = 2, -1.0, -1.0
    elif (-90 + eta_equator) <= eta_deg <= -eta_pole:
        quadr_coeff, coeff_y0, coeff_z0 = 2, 1.0, 1.0
    elif (-180 + eta_pole) <= eta_deg <= (-90 - eta_equator):
        quadr_coeff, coeff_y0, coeff_z0 = 1, 1.0, -1.0
    return quadr_coeff, coeff_y0, coeff_z0


# ---------------------------------------------------------------------------
# UseFriedelPairs == 0 — GenerateIdealSpots
# ---------------------------------------------------------------------------


def generate_ideal_spots(
    ys: float,
    zs: float,
    ttheta_deg: float,
    eta_deg: float,
    ring_rad: float,
    rsample: float,
    hbeam: float,
    step_size: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Discretize the seed locus on the ring into (y0, z0) candidates.

    Returns
    -------
    seeds : torch.Tensor (n_steps, 2) — (y0, z0) per candidate.
    """
    device = device or torch.device("cpu")
    eta_pole, eta_equator, hbeam_eff = _eta_pole_equator(
        ring_rad, hbeam, rsample, ttheta_deg, eta_deg
    )
    quadr_coeff, coeff_y0, coeff_z0 = _eta_quadrant(eta_deg, eta_pole, eta_equator)
    quadr_coeff2 = 0

    y0_max_rsample = ys + rsample
    y0_min_rsample = ys - rsample
    z0_max_hbeam = zs + 0.5 * hbeam_eff
    z0_min_hbeam = zs - 0.5 * hbeam_eff

    y0_max = 0.0
    y0_min = 0.0
    z0_max = 0.0
    z0_min = 0.0

    if quadr_coeff == 1:
        y0_max_z0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_max_hbeam ** 2))
        y0_min_z0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_min_hbeam ** 2))
        y0_max = min(y0_max_rsample, y0_max_z0)
        y0_min = max(y0_min_rsample, y0_min_z0)
    elif quadr_coeff == 2:
        y0_max_z0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_min_hbeam ** 2))
        y0_min_z0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_max_hbeam ** 2))
        y0_max = min(y0_max_rsample, y0_max_z0)
        y0_min = max(y0_min_rsample, y0_min_z0)
    else:
        if -eta_pole < eta_deg < eta_pole:
            y0_max, y0_min, coeff_z0 = y0_max_rsample, y0_min_rsample, 1.0
        elif eta_deg < (-180 + eta_pole) or eta_deg > (180 - eta_pole):
            y0_max, y0_min, coeff_z0 = y0_max_rsample, y0_min_rsample, -1.0
        elif (90 - eta_equator) < eta_deg < (90 + eta_equator):
            quadr_coeff2 = 1
            z0_max, z0_min, coeff_y0 = z0_max_hbeam, z0_min_hbeam, -1.0
        elif (-90 - eta_equator) < eta_deg < (-90 + eta_equator):
            quadr_coeff2 = 1
            z0_max, z0_min, coeff_y0 = z0_max_hbeam, z0_min_hbeam, 1.0
        else:
            return torch.empty((0, 2), device=device, dtype=dtype)

    if quadr_coeff2 == 0:
        y01, y02 = y0_min, y0_max
        z01 = coeff_z0 * math.sqrt(max(0.0, ring_rad ** 2 - y01 ** 2))
        z02 = coeff_z0 * math.sqrt(max(0.0, ring_rad ** 2 - y02 ** 2))
    else:
        z01, z02 = z0_min, z0_max
        y01 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z01 ** 2))
        y02 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z02 ** 2))
    length = math.hypot(y01 - y02, z01 - z02)
    n_steps = max(1, math.ceil(length / step_size))
    if n_steps % 2 == 0:
        n_steps += 1

    if n_steps == 1:
        if quadr_coeff2 == 0:
            y0 = (y0_max + y0_min) / 2.0
            z0 = coeff_z0 * math.sqrt(max(0.0, ring_rad ** 2 - y0 ** 2))
        else:
            z0 = (z0_max + z0_min) / 2.0
            y0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0 ** 2))
        return torch.tensor([[y0, z0]], device=device, dtype=dtype)

    if quadr_coeff2 == 0:
        ys_arr = torch.linspace(y0_min, y0_max, n_steps, device=device, dtype=dtype)
        zs_arr = coeff_z0 * torch.sqrt(
            torch.clamp(ring_rad ** 2 - ys_arr ** 2, min=0.0)
        )
    else:
        zs_arr = torch.linspace(z0_min, z0_max, n_steps, device=device, dtype=dtype)
        ys_arr = coeff_y0 * torch.sqrt(
            torch.clamp(ring_rad ** 2 - zs_arr ** 2, min=0.0)
        )
    return torch.stack([ys_arr, zs_arr], dim=-1)


# ---------------------------------------------------------------------------
# UseFriedelPairs == 1 — GenerateIdealSpotsFriedel
# ---------------------------------------------------------------------------


def _friedel_eta_calculation(
    ys: float, zs: float, ttheta_deg: float, eta_deg: float,
    ring_rad: float, rsample: float, hbeam: float,
) -> tuple[float, float]:
    """Returns (EtaMinFr, EtaMaxFr). Hoists the per-seed loop-invariants from
    `FriedelEtaCalculation` (IndexerOMP.c:762)."""
    eta_pole, eta_equator, hbeam_eff = _eta_pole_equator(
        ring_rad, hbeam, rsample, ttheta_deg, eta_deg
    )
    quadr_coeff, coeff_y0, coeff_z0 = _eta_quadrant(eta_deg, eta_pole, eta_equator)
    quadr_coeff2 = 0

    y0_max_rsample = ys + rsample
    y0_min_rsample = ys - rsample
    z0_max_hbeam = zs + 0.5 * hbeam_eff
    z0_min_hbeam = zs - 0.5 * hbeam_eff

    y0_max, y0_min, z0_min, z0_max = 0.0, 0.0, 0.0, 0.0

    if quadr_coeff == 1:
        y0_max_z0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_max_hbeam ** 2))
        y0_min_z0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_min_hbeam ** 2))
        y0_max = min(y0_max_rsample, y0_max_z0)
        y0_min = max(y0_min_rsample, y0_min_z0)
    elif quadr_coeff == 2:
        y0_max_z0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_min_hbeam ** 2))
        y0_min_z0 = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_max_hbeam ** 2))
        y0_max = min(y0_max_rsample, y0_max_z0)
        y0_min = max(y0_min_rsample, y0_min_z0)
    else:
        if -eta_pole < eta_deg < eta_pole:
            y0_max, y0_min, coeff_z0 = y0_max_rsample, y0_min_rsample, 1.0
        elif eta_deg < (-180 + eta_pole) or eta_deg > (180 - eta_pole):
            y0_max, y0_min, coeff_z0 = y0_max_rsample, y0_min_rsample, -1.0
        elif (90 - eta_equator) < eta_deg < (90 + eta_equator):
            quadr_coeff2 = 1
            z0_max, z0_min, coeff_y0 = z0_max_hbeam, z0_min_hbeam, -1.0
        elif (-90 - eta_equator) < eta_deg < (-90 + eta_equator):
            quadr_coeff2 = 1
            z0_max, z0_min, coeff_y0 = z0_max_hbeam, z0_min_hbeam, 1.0
        else:
            return -180.0, 180.0

    if quadr_coeff2 == 0:
        z0_min = coeff_z0 * math.sqrt(max(0.0, ring_rad ** 2 - y0_min ** 2))
        z0_max = coeff_z0 * math.sqrt(max(0.0, ring_rad ** 2 - y0_max ** 2))
    else:
        y0_min = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_min ** 2))
        y0_max = coeff_y0 * math.sqrt(max(0.0, ring_rad ** 2 - z0_max ** 2))

    dY_min = ys - y0_min
    dY_max = ys - y0_max
    dZ_min = zs - z0_min
    dZ_max = zs - z0_max
    Y_min_fr_ideal = y0_min
    Y_max_fr_ideal = y0_max
    Z_min_fr_ideal = -z0_min
    Z_max_fr_ideal = -z0_max
    Y_min_fr = Y_min_fr_ideal - dY_min
    Y_max_fr = Y_max_fr_ideal - dY_max
    Z_min_fr = Z_min_fr_ideal + dZ_min
    Z_max_fr = Z_max_fr_ideal + dZ_max
    eta1 = math.atan2(-(Y_min_fr + ys), (Z_min_fr - zs)) * RAD2DEG
    eta2 = math.atan2(-(Y_max_fr + ys), (Z_max_fr - zs)) * RAD2DEG
    return min(eta1, eta2), max(eta1, eta2)


def generate_ideal_spots_friedel(
    ys: float,
    zs: float,
    ttheta_deg: float,
    eta_deg: float,
    omega_deg: float,
    ring_nr: int,
    ring_rad: float,
    rsample: float,
    hbeam: float,
    ome_tol: float,
    radius_tol: float,
    obs_spots: torch.Tensor,            # (n_obs, 9) torch.float64
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Friedel-pair seed candidates: one (y0, z0) per matching observed spot.

    Mirrors `GenerateIdealSpotsFriedel` (IndexerOMP.c:1030).
    """
    device = device or obs_spots.device
    if omega_deg < 0:
        ome_f = omega_deg + 180.0
    else:
        ome_f = omega_deg - 180.0
    if eta_deg < 0:
        eta_f = -180.0 - eta_deg
    else:
        eta_f = 180.0 - eta_deg

    obs = obs_spots
    rno = obs[:, 5].round().to(torch.int64)
    ome_obs = obs[:, 2]
    yf = obs[:, 0]
    zf = obs[:, 1]

    mask_ring = rno == int(ring_nr)
    mask_ome = (ome_obs - ome_f).abs() <= ome_tol
    eta_transf = _calc_eta_angle(yf + ys, zf - zs)
    radius = torch.sqrt((yf + ys) ** 2 + (zf - zs) ** 2)
    mask_rad = (radius - 2.0 * ring_rad).abs() <= radius_tol

    eta_min_f, eta_max_f = _friedel_eta_calculation(
        ys, zs, ttheta_deg, eta_deg, ring_rad, rsample, hbeam
    )
    mask_eta = (eta_transf >= eta_min_f) & (eta_transf <= eta_max_f)

    keep = mask_ring & mask_ome & mask_rad & mask_eta
    if not bool(keep.any()):
        return torch.empty((0, 2), device=device, dtype=dtype)

    yf_keep = yf[keep]
    zf_keep = zf[keep]
    z_pos_acc_z = zs - ((zf_keep + zs) / 2.0)
    y_pos_acc_y = ys - ((-yf_keep + ys) / 2.0)
    eta_ideal_f = _calc_eta_angle(y_pos_acc_y, z_pos_acc_z)
    y0, z0 = _calc_spot_position(ring_rad, eta_ideal_f)
    return torch.stack([y0, z0], dim=-1).to(dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# UseFriedelPairs == 2 — GenerateIdealSpotsFriedelMixed
# ---------------------------------------------------------------------------


def generate_ideal_spots_friedel_mixed(
    ys: float,
    zs: float,
    ttheta_deg: float,
    eta_deg: float,
    omega_deg: float,
    ring_nr: int,
    ring_rad: float,
    lsd: float,
    rsample: float,
    hbeam: float,
    step_size_pos: float,
    ome_tol: float,
    radial_tol: float,
    eta_tol_um: float,
    obs_spots: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Mixed strategy: ideal spots + Friedel-pair filter against observed spots.

    Mirrors `GenerateIdealSpotsFriedelMixed` (IndexerOMP.c:1107). Implementation
    keeps the C structure (per-candidate inner loop) but vectorizes the
    observed-spot match per inner iteration.
    """
    device = device or obs_spots.device
    MIN_ETA_REJECT = 10.0
    if abs(math.sin(eta_deg * DEG2RAD)) < math.sin(MIN_ETA_REJECT * DEG2RAD):
        return torch.empty((0, 2), device=device, dtype=dtype)

    theta = ttheta_deg / 2.0
    eta_tol_deg = RAD2DEG * math.atan(eta_tol_um / ring_rad)

    candidates = generate_ideal_spots(
        ys, zs, ttheta_deg, eta_deg, ring_rad, rsample, hbeam, step_size_pos,
        device=device, dtype=dtype,
    )
    if candidates.shape[0] == 0:
        return candidates

    # CPU fast path: per-(sp_on_ring, n_pos) scalar inner loop. Was the
    # dominant remaining torch hot spot after Phase 2 (~3.8 s/call on
    # Wenxi). The torch fallback below stays for GPU and for compatibility
    # if numba isn't importable.
    if device.type != "cuda":
        try:
            from .seeds_numba import (
                generate_ideal_spots_friedel_mixed_numba,
                _NUMBA_AVAILABLE as _NUMBA_SEEDS_AVAILABLE,
            )
            if _NUMBA_SEEDS_AVAILABLE:
                result = generate_ideal_spots_friedel_mixed_numba(
                    ys, zs, ttheta_deg, eta_deg, omega_deg, ring_nr,
                    ring_rad, lsd, rsample, hbeam, step_size_pos,
                    ome_tol, radial_tol, eta_tol_um,
                    obs_spots, candidates,
                    device=device, dtype=dtype,
                )
                if result is not None:
                    return result
        except ImportError:
            pass

    # FPCandidates table: dict key = obs_spot_id, value = (sp_on_ring, diffPos2)
    fp_unique: list[int] = []
    fp_seen: dict[int, tuple[int, float]] = {}

    obs = obs_spots
    obs_id = obs[:, 4]
    obs_y = obs[:, 0]
    obs_z = obs[:, 1]
    obs_ome = obs[:, 2]
    obs_eta = obs[:, 6]
    obs_rad_diff = obs[:, 8]

    for sp_on_ring in range(candidates.shape[0]):
        y0_t = candidates[sp_on_ring, 0]
        z0_t = candidates[sp_on_ring, 1]
        y0 = float(y0_t.item())
        z0 = float(z0_t.item())
        # MakeUnitLength(Lsd, y0, z0)
        L = math.sqrt(lsd * lsd + y0 * y0 + z0 * z0)
        if L == 0:
            continue
        xi, yi, zi = lsd / L, y0 / L, z0 / L
        # spot_to_gv(Lsd, y0, z0, omega) -> g_vec
        g1, g2, g3 = _spot_to_gv(lsd, y0, z0, omega_deg)
        omegas_fp, etas_fp, n_sol = _calc_omega_solutions(-g1, -g2, -g3, theta)
        if n_sol <= 1:
            continue
        # pick the omega solution farthest from `omega` (180-difference, modular)
        diffs = []
        for k in range(n_sol):
            d = abs(omegas_fp[k] - omega_deg)
            if d > 180:
                d = 360 - d
            diffs.append(d)
        if diffs[0] < diffs[1]:
            omega_fp = omegas_fp[0]
            eta_fp = etas_fp[0]
        else:
            omega_fp = omegas_fp[1]
            eta_fp = etas_fp[1]
        yfp1, zfp1 = _calc_spot_position_scalar(ring_rad, eta_fp)
        n_min, n_max = _calc_n_max_min_scalar(xi, yi, ys, y0, rsample, step_size_pos)

        for n in range(n_min, n_max + 1):
            a, b, c = _spot_to_unrotated_scalar(
                xi, yi, zi, ys, zs, y0, z0, step_size_pos, n, omega_deg
            )
            if abs(c) > hbeam / 2:
                continue
            dy_disp, dz_disp = _displacement_spot_com(a, b, c, lsd, yfp1, zfp1, omega_fp)
            yfp = yfp1 + dy_disp
            zfp = zfp1 + dz_disp
            radial_pos_fp = math.sqrt(yfp * yfp + zfp * zfp) - ring_rad
            eta_fp_corr = math.atan2(-yfp, zfp) * RAD2DEG
            # Vectorized GetBin lookup against observed spots:
            # We don't reconstruct the full bin table here; instead we filter
            # observed spots directly by ring/eta/omega proximity (same mask
            # as the inner if-check at IndexerOMP.c:1183).
            mask = (
                (obs[:, 5].round().to(torch.int64) == int(ring_nr))
                & ((obs_rad_diff - radial_pos_fp).abs() < radial_tol)
                & ((obs_ome - omega_fp).abs() < ome_tol)
                & ((obs_eta - eta_fp_corr).abs() < eta_tol_deg)
            )
            if not bool(mask.any()):
                continue
            dy = yfp - obs_y[mask]
            dz = zfp - obs_z[mask]
            diff_pos2 = dy * dy + dz * dz
            obs_ids = obs_id[mask].to(torch.int64).tolist()
            for j, obs_int_id in enumerate(obs_ids):
                d2 = float(diff_pos2[j].item())
                if obs_int_id in fp_seen:
                    if d2 < fp_seen[obs_int_id][1]:
                        # Replace (improvement); but only if the new sp_on_ring
                        # isn't already represented elsewhere.
                        fp_seen[obs_int_id] = (sp_on_ring, d2)
                    # else skip
                else:
                    fp_seen[obs_int_id] = (sp_on_ring, d2)

    # Deduplicate sp_on_ring across the FPCandidates dict
    seen_sp = set()
    out_y0: list[float] = []
    out_z0: list[float] = []
    for _, (sp, _) in fp_seen.items():
        if sp in seen_sp:
            continue
        seen_sp.add(sp)
        out_y0.append(float(candidates[sp, 0].item()))
        out_z0.append(float(candidates[sp, 1].item()))

    if not out_y0:
        return torch.empty((0, 2), device=device, dtype=dtype)
    y_arr = torch.tensor(out_y0, device=device, dtype=dtype)
    z_arr = torch.tensor(out_z0, device=device, dtype=dtype)
    return torch.stack([y_arr, z_arr], dim=-1)


# ---------------------------------------------------------------------------
# Scalar helpers used by Friedel-mixed (single-call hot inner loop)
# ---------------------------------------------------------------------------


def _spot_to_gv(xi: float, yi: float, zi: float, omega_deg: float) -> tuple[float, float, float]:
    L = math.sqrt(xi * xi + yi * yi + zi * zi)
    if L == 0:
        return 0.0, 0.0, 0.0
    xn, yn, zn = xi / L, yi / L, zi / L
    g1r = -1 + xn
    g2r = yn
    cos_o = math.cos(-omega_deg * DEG2RAD)
    sin_o = math.sin(-omega_deg * DEG2RAD)
    g1 = g1r * cos_o - g2r * sin_o
    g2 = g1r * sin_o + g2r * cos_o
    return g1, g2, zn


def _calc_omega_solutions(
    x: float, y: float, z: float, theta: float,
) -> tuple[list[float], list[float], int]:
    """Returns (omegas_deg, etas_deg, n_sol). Mirrors `CalcOmega`."""
    n_sol = 0
    omegas: list[float] = []
    etas: list[float] = []
    L = math.sqrt(x * x + y * y + z * z)
    v = math.sin(theta * DEG2RAD) * L
    if abs(y) < 1e-12:
        if x != 0:
            cosome1 = -v / x
            if abs(cosome1) <= 1.0:
                ome = math.acos(cosome1) * RAD2DEG
                omegas.extend([ome, -ome])
                n_sol = 2
    else:
        y2 = y * y
        a = 1 + (x * x) / y2
        b = (2 * v * x) / y2
        c = (v * v) / y2 - 1
        discr = b * b - 4 * a * c
        if discr >= 0:
            for sign in (+1, -1):
                cosome = (-b + sign * math.sqrt(discr)) / (2 * a)
                if abs(cosome) <= 1.0:
                    ome_a = math.acos(cosome)
                    ome_b = -ome_a
                    eq_a = -x * math.cos(ome_a) + y * math.sin(ome_a)
                    eq_b = -x * math.cos(ome_b) + y * math.sin(ome_b)
                    if abs(eq_a - v) < abs(eq_b - v):
                        omegas.append(ome_a * RAD2DEG)
                    else:
                        omegas.append(ome_b * RAD2DEG)
                    n_sol += 1
    # Compute etas via RotateAroundZ then atan2(-y_rot, z). Mirrors C
    # `RotateAroundZ` + `CalcEtaAngle(gw[1], gw[2])` from CalcDiffractionSpots.c
    # — gw[1] is the **Y** component (sin*x + cos*y), NOT the X (cos*x - sin*y).
    # Earlier versions used gw1 (X), which produced wrong eta and silently
    # broke `generate_ideal_spots_friedel_mixed` for seeds whose obs friedel
    # partner was searched by (omega_fp, eta_fp).
    for ome in omegas:
        cz, sz = math.cos(ome * DEG2RAD), math.sin(ome * DEG2RAD)
        gw_x = cz * x - sz * y
        gw_y = sz * x + cz * y
        eta = math.atan2(-gw_y, z) * RAD2DEG
        etas.append(eta)
    return omegas, etas, n_sol


def _calc_spot_position_scalar(ring_radius: float, eta_deg: float) -> tuple[float, float]:
    er = eta_deg * DEG2RAD
    return -math.sin(er) * ring_radius, math.cos(er) * ring_radius


def _calc_n_max_min_scalar(
    xi: float, yi: float, ys: float, y0: float, r_sample: float, step_size: float,
) -> tuple[int, int]:
    dy = ys - y0
    a = xi * xi + yi * yi
    b = 2 * yi * dy
    c = dy * dy - r_sample * r_sample
    D = b * b - 4 * a * c
    P = math.sqrt(max(0.0, D))
    lambda_max = (-b + P) / (2 * a) + 20
    n_max = int(lambda_max * xi / step_size)
    return -n_max, n_max


def _spot_to_unrotated_scalar(
    xi: float, yi: float, zi: float, ys: float, zs: float, y0: float, z0: float,
    step_size_in_x: float, n: int, omega_deg: float,
) -> tuple[float, float, float]:
    lam = step_size_in_x * (n / xi)
    x1 = lam * xi
    y1 = ys - y0 + lam * yi
    z1 = zs - z0 + lam * zi
    co, so = math.cos(omega_deg * DEG2RAD), math.sin(omega_deg * DEG2RAD)
    return x1 * co + y1 * so, y1 * co - x1 * so, z1


def _displacement_spot_com(
    a: float, b: float, c: float, xi: float, yi: float, zi: float, omega_deg: float,
) -> tuple[float, float]:
    L = math.sqrt(xi * xi + yi * yi + zi * zi)
    xi /= L; yi /= L; zi /= L
    co, so = math.cos(omega_deg * DEG2RAD), math.sin(omega_deg * DEG2RAD)
    t = (a * co - b * so) / xi
    dy = (a * so + b * co) - t * yi
    dz = c - t * zi
    return dy, dz
