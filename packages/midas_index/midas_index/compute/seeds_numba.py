"""Numba CPU port of the indexer's per-seed candidate-locus generator.

Phase 3 of the numba pipeline (after Phase 1's bin lookup inline +
Phase 2's numba forward simulate). Targets the next hot spot:
``generate_ideal_spots_friedel_mixed`` was profiling at 3.8 s/call (8 calls
= 30 s of 120 s wall on Wenxi). Per-candidate, per-n_pos inner loop with
~50 scalar ops and an obs-table filter per inner iter — perfect numba fit.

The ``friedel`` (non-mixed) variant uses a vectorised obs-table mask in
torch and is fast (7.5 ms/call); we leave it on torch. The mixed variant
falls back here because its inner loop has Python-level dict accumulation
and a per-(sp_on_ring, n_pos) scalar inner that numba demolishes.

Serial outer loop (sp_on_ring) — friedel_mixed is called ~8 times per
voxel, so amortised parallelism here doesn't help. Inside the numba
kernel we use a per-obs-id ``best_diff2`` array (replaces the Python dict)
then collect unique sp_on_ring indices at the end.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    _NUMBA_AVAILABLE = False


DEG2RAD_ = math.pi / 180.0
RAD2DEG_ = 180.0 / math.pi


if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=False)
    def _calc_omega_solutions_inline(x, y, z, theta_deg):
        """Inline `CalcOmega`-equivalent. Returns up to 2 (omega_deg, eta_deg) solutions.

        Output: omega_arr, eta_arr (both float64[2]), n_sol (int).
        """
        omega_arr = np.zeros(2, dtype=np.float64)
        eta_arr = np.zeros(2, dtype=np.float64)
        n_sol = 0

        L = math.sqrt(x * x + y * y + z * z)
        v = math.sin(theta_deg * DEG2RAD_) * L

        if abs(y) < 1e-12:
            if x != 0.0:
                cosome1 = -v / x
                if abs(cosome1) <= 1.0:
                    ome = math.acos(cosome1) * RAD2DEG_
                    omega_arr[0] = ome
                    omega_arr[1] = -ome
                    n_sol = 2
        else:
            y2 = y * y
            a = 1.0 + (x * x) / y2
            b = (2.0 * v * x) / y2
            c = (v * v) / y2 - 1.0
            discr = b * b - 4.0 * a * c
            if discr >= 0.0:
                sqrt_d = math.sqrt(discr)
                for sign_i in range(2):
                    sign = 1.0 if sign_i == 0 else -1.0
                    cosome = (-b + sign * sqrt_d) / (2.0 * a)
                    if abs(cosome) <= 1.0:
                        ome_a = math.acos(cosome)
                        ome_b = -ome_a
                        eq_a = -x * math.cos(ome_a) + y * math.sin(ome_a)
                        eq_b = -x * math.cos(ome_b) + y * math.sin(ome_b)
                        if abs(eq_a - v) < abs(eq_b - v):
                            omega_arr[n_sol] = ome_a * RAD2DEG_
                        else:
                            omega_arr[n_sol] = ome_b * RAD2DEG_
                        n_sol += 1

        for k in range(n_sol):
            ome = omega_arr[k]
            cz = math.cos(ome * DEG2RAD_)
            sz = math.sin(ome * DEG2RAD_)
            gw_y = sz * x + cz * y
            eta_arr[k] = math.atan2(-gw_y, z) * RAD2DEG_

        return omega_arr, eta_arr, n_sol

    @njit(cache=True, fastmath=False)
    def _friedel_mixed_inner_numba(
        ys, zs, ttheta_deg, eta_deg, omega_deg, ring_nr_target,
        ring_rad, lsd, rsample, hbeam, step_size_pos,
        ome_tol, radial_tol, eta_tol_um,
        candidates_y, candidates_z,             # (n_cand,) each
        obs_y, obs_z, obs_ome, obs_eta,
        obs_ringrad, obs_rad, obs_id, obs_ring,
    ):
        """Numba port of ``generate_ideal_spots_friedel_mixed``.

        Per-(sp_on_ring, n_pos) scalar inner loop. Returns the unique set of
        ``sp_on_ring`` indices whose candidate ideal spot had a matching
        observed Friedel partner — same as the torch path's dedup logic.
        """
        n_cand = candidates_y.shape[0]
        n_obs = obs_y.shape[0]

        # Per-obs accumulator: best (sp_on_ring, diff_pos2) for each obs.
        # Initialised to (-1, +inf) — never matches → not selected.
        best_sp_on_ring = np.full(n_obs, -1, dtype=np.int64)
        best_diff2 = np.full(n_obs, np.inf, dtype=np.float64)

        eta_tol_deg = RAD2DEG_ * math.atan(eta_tol_um / ring_rad)
        theta = ttheta_deg / 2.0

        for sp_on_ring in range(n_cand):
            y0 = candidates_y[sp_on_ring]
            z0 = candidates_z[sp_on_ring]

            # MakeUnitLength(lsd, y0, z0)
            L = math.sqrt(lsd * lsd + y0 * y0 + z0 * z0)
            if L == 0.0:
                continue
            xi_n = lsd / L
            yi_n = y0 / L
            zi_n = z0 / L

            # spot_to_gv(lsd, y0, z0, omega_deg)
            cos_o = math.cos(-omega_deg * DEG2RAD_)
            sin_o = math.sin(-omega_deg * DEG2RAD_)
            g1r = -1.0 + xi_n
            g2r = yi_n
            g1 = g1r * cos_o - g2r * sin_o
            g2 = g1r * sin_o + g2r * cos_o
            g3 = zi_n

            # Omega solutions for (-g1, -g2, -g3) with theta
            omegas_fp, etas_fp, n_sol = _calc_omega_solutions_inline(-g1, -g2, -g3, theta)
            if n_sol <= 1:
                continue

            # Pick the omega farthest from `omega_deg` (180°-difference modular)
            d0 = abs(omegas_fp[0] - omega_deg)
            if d0 > 180.0:
                d0 = 360.0 - d0
            d1 = abs(omegas_fp[1] - omega_deg)
            if d1 > 180.0:
                d1 = 360.0 - d1
            if d0 < d1:
                omega_fp = omegas_fp[0]
                eta_fp = etas_fp[0]
            else:
                omega_fp = omegas_fp[1]
                eta_fp = etas_fp[1]

            # _calc_spot_position_scalar
            er = eta_fp * DEG2RAD_
            yfp1 = -math.sin(er) * ring_rad
            zfp1 = math.cos(er) * ring_rad

            # _calc_n_max_min_scalar
            dy_n = ys - y0
            a_q = xi_n * xi_n + yi_n * yi_n
            b_q = 2.0 * yi_n * dy_n
            c_q = dy_n * dy_n - rsample * rsample
            D_q = b_q * b_q - 4.0 * a_q * c_q
            if D_q < 0.0:
                D_q = 0.0
            P_q = math.sqrt(D_q)
            lambda_max = (-b_q + P_q) / (2.0 * a_q) + 20.0
            n_max = int(lambda_max * xi_n / step_size_pos)
            n_min = -n_max

            cos_om = math.cos(omega_deg * DEG2RAD_)
            sin_om = math.sin(omega_deg * DEG2RAD_)
            cos_om_fp = math.cos(omega_fp * DEG2RAD_)
            sin_om_fp = math.sin(omega_fp * DEG2RAD_)

            for n in range(n_min, n_max + 1):
                # _spot_to_unrotated_scalar
                lam = step_size_pos * (n / xi_n)
                x1 = lam * xi_n
                y1 = ys - y0 + lam * yi_n
                z1 = zs - z0 + lam * zi_n
                a_u = x1 * cos_om + y1 * sin_om
                b_u = y1 * cos_om - x1 * sin_om
                c_u = z1
                if abs(c_u) > hbeam / 2.0:
                    continue

                # _displacement_spot_com on (a_u, b_u, c_u) with (lsd, yfp1, zfp1, omega_fp)
                L_d = math.sqrt(lsd * lsd + yfp1 * yfp1 + zfp1 * zfp1)
                xi_d = lsd / L_d
                yi_d = yfp1 / L_d
                zi_d = zfp1 / L_d
                t = (a_u * cos_om_fp - b_u * sin_om_fp) / xi_d
                dy_disp = (a_u * sin_om_fp + b_u * cos_om_fp) - t * yi_d
                dz_disp = c_u - t * zi_d
                yfp = yfp1 + dy_disp
                zfp = zfp1 + dz_disp

                radial_pos_fp = math.sqrt(yfp * yfp + zfp * zfp) - ring_rad
                eta_fp_corr = math.atan2(-yfp, zfp) * RAD2DEG_

                # Per-obs filter (mirrors the torch mask)
                for k in range(n_obs):
                    if obs_ring[k] != ring_nr_target:
                        continue
                    if abs(obs_rad[k] - radial_pos_fp) >= radial_tol:
                        continue
                    if abs(obs_ome[k] - omega_fp) >= ome_tol:
                        continue
                    if abs(obs_eta[k] - eta_fp_corr) >= eta_tol_deg:
                        continue
                    dyy = yfp - obs_y[k]
                    dzz = zfp - obs_z[k]
                    diff2 = dyy * dyy + dzz * dzz
                    # Update per-obs best (smaller diff2 wins; matches dict semantics)
                    obs_int_id = obs_id[k]
                    if obs_int_id < 0 or obs_int_id >= n_obs:
                        # obs_id table is indexed by obs row index; use k as key
                        if diff2 < best_diff2[k]:
                            best_diff2[k] = diff2
                            best_sp_on_ring[k] = sp_on_ring
                    else:
                        if diff2 < best_diff2[k]:
                            best_diff2[k] = diff2
                            best_sp_on_ring[k] = sp_on_ring

        # Collect unique sp_on_ring indices from accumulator.
        seen_sp = np.zeros(n_cand, dtype=np.bool_)
        n_out = 0
        # First pass: count
        for k in range(n_obs):
            sp = best_sp_on_ring[k]
            if sp >= 0 and sp < n_cand and not seen_sp[sp]:
                seen_sp[sp] = True
                n_out += 1
        out_y0 = np.zeros(n_out, dtype=np.float64)
        out_z0 = np.zeros(n_out, dtype=np.float64)
        # Second pass: emit
        seen_sp = np.zeros(n_cand, dtype=np.bool_)
        idx_out = 0
        for k in range(n_obs):
            sp = best_sp_on_ring[k]
            if sp >= 0 and sp < n_cand and not seen_sp[sp]:
                seen_sp[sp] = True
                out_y0[idx_out] = candidates_y[sp]
                out_z0[idx_out] = candidates_z[sp]
                idx_out += 1
        return out_y0, out_z0

else:  # pragma: no cover
    def _friedel_mixed_inner_numba(*args, **kwargs):  # type: ignore
        raise ImportError("numba is required for seeds_numba")


def generate_ideal_spots_friedel_mixed_numba(
    ys, zs, ttheta_deg, eta_deg, omega_deg, ring_nr,
    ring_rad, lsd, rsample, hbeam, step_size_pos,
    ome_tol, radial_tol, eta_tol_um,
    obs_spots,         # torch.Tensor (n_obs, 9) — for fallback compatibility
    candidates,        # torch.Tensor (n_cand, 2) from generate_ideal_spots — pre-built
    *,
    device=None, dtype=None,
):
    """Numba-accelerated drop-in for ``generate_ideal_spots_friedel_mixed``.

    Mirrors the torch signature except ``candidates`` is required (caller
    already builds it via ``generate_ideal_spots``). Returns a torch (n_out, 2)
    tensor of (y0, z0) seed positions matching the original.
    """
    import torch
    if not _NUMBA_AVAILABLE:
        # Caller will fall back to torch via the original function.
        return None

    if candidates.shape[0] == 0:
        return candidates

    candidates_np = candidates.detach().cpu().numpy().astype(np.float64, copy=False)
    candidates_y = np.ascontiguousarray(candidates_np[:, 0])
    candidates_z = np.ascontiguousarray(candidates_np[:, 1])

    obs_np = obs_spots.detach().cpu().numpy()
    obs_y = np.ascontiguousarray(obs_np[:, 0].astype(np.float64, copy=False))
    obs_z = np.ascontiguousarray(obs_np[:, 1].astype(np.float64, copy=False))
    obs_ome = np.ascontiguousarray(obs_np[:, 2].astype(np.float64, copy=False))
    obs_eta = np.ascontiguousarray(obs_np[:, 6].astype(np.float64, copy=False))
    obs_ringrad = np.ascontiguousarray(obs_np[:, 3].astype(np.float64, copy=False))
    obs_rad = np.ascontiguousarray(obs_np[:, 8].astype(np.float64, copy=False))
    obs_id = np.ascontiguousarray(obs_np[:, 4].astype(np.int64, copy=False))
    obs_ring = np.ascontiguousarray(np.round(obs_np[:, 5]).astype(np.int64, copy=False))

    out_y0, out_z0 = _friedel_mixed_inner_numba(
        float(ys), float(zs), float(ttheta_deg), float(eta_deg),
        float(omega_deg), int(ring_nr),
        float(ring_rad), float(lsd), float(rsample), float(hbeam),
        float(step_size_pos),
        float(ome_tol), float(radial_tol), float(eta_tol_um),
        candidates_y, candidates_z,
        obs_y, obs_z, obs_ome, obs_eta,
        obs_ringrad, obs_rad, obs_id, obs_ring,
    )

    if out_y0.shape[0] == 0:
        return torch.empty((0, 2), device=obs_spots.device,
                           dtype=dtype if dtype is not None else torch.float64)
    target_device = device if device is not None else obs_spots.device
    target_dtype = dtype if dtype is not None else torch.float64
    return torch.stack([
        torch.from_numpy(out_y0).to(device=target_device, dtype=target_dtype),
        torch.from_numpy(out_z0).to(device=target_device, dtype=target_dtype),
    ], dim=-1)
