"""Phase 4 — fused numba kernel: simulate + compare_spots + avg_ia.

Eliminates the torch round-trip between simulate and compare. Profile of
Phase 3 showed ~50% of wall time was ``numpy.ascontiguousarray`` because
each call to ``_compare_spots_numba`` re-marshaled the theor columns from
the torch tensor returned by ``simulate_numba``. This kernel never builds
a torch theor at all — every per-(n, m, branch) row is computed and
consumed inline.

Inputs are pre-marshalled numpy arrays (cached across calls where
possible). Outputs are numpy too; the caller (_compute_group_gpu's CPU
path) re-wraps them into the SeedResult struct.

The kernel mirrors three previously separate kernels:
  * simulate (forward_numba._simulate_numba_inner)
  * compare_spots (matching._compare_spots_numba_inner)
  * avg_ia (matching._compute_avg_ia_numba_inner)

Fusing means each cell's theor row is in registers/cache when compare
reads it — no L2 round-trip via a (N, 2M, 14) intermediate.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit, prange  # type: ignore
    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    prange = None
    _NUMBA_AVAILABLE = False


if _NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, fastmath=False)
    def _simulate_and_compare_fused_numba(
        # Orientation candidates
        R,                              # (N, 3, 3) float64
        pos,                            # (N, 3) float64
        # HKL list
        hkls_cart,                      # (M, 3) float64
        thetas,                         # (M,) float64
        len_hkl,                        # (M,) float64
        ring_nr_per_hkl,                # (M,) int64
        ring_radius_lut,                # (max_ring+1,) float64
        ring_radius_lut_max_idx,        # int
        # Geometry params
        wedge_rad, Lsd, min_eta_rad, epsilon,
        # OmegaRange + BoxSize
        omega_ranges_deg,               # (n_ranges, 2) float64
        box_sizes,                      # (n_ranges, 4) float64
        has_omega_box,                  # bool
        # Bin tables
        bin_ndata,                      # (2*n_bins,) int32
        bin_data,                       # (n_bin_data,) int32
        eta_bin_size, ome_bin_size,     # float64
        n_eta_bins, n_ome_bins,         # int64
        # Obs columns
        obs_y, obs_z, obs_ome, obs_eta, obs_ringrad, obs_rad, obs_id,
        # Per-ring eta margin LUT
        eta_margins_lut,                # (max_n_rings,) float64
        eta_margins_max_idx,            # int
        # Tolerance scalars
        margin_radial, margin_rad,
        rings_to_reject,                # (n_reject,) int64
        ref_rad,                        # (N,) float64
        # Scan-aware
        scan_active, voxel_x, voxel_y,
        obs_scan_idx, scan_pos_arr,
        scan_pos_tol, friedel_sym,
    ):
        """Per-(n, m, branch) cell loop. Returns per-(N, K) match state +
        per-N reductions (n_matches, n_t_frac, avg_ia).

        Output arrays (all caller-allocated free shape):
          best_delta_ome   (N, K) float64
          best_matched_id  (N, K) int64
          best_matched_row (N, K) int64
          has_match        (N, K) bool
          theor_omega      (N, K) float64     # for downstream best_global gather
          theor_eta        (N, K) float64
          theor_ring_nr    (N, K) float64     # for skip_radial reconstruction
          n_matches        (N,) int64
          n_matches_frac   (N,) int64
          n_t_frac         (N,) int64
          avg_ia           (N,) float64
        """
        DEG2RAD_ = math.pi / 180.0
        RAD2DEG_ = 180.0 / math.pi
        N = R.shape[0]
        M = hkls_cart.shape[0]
        K = 2 * M
        almostzero = 1e-12
        cos_W = math.cos(wedge_rad)
        sin_W = math.sin(wedge_rad)
        PI = math.pi

        # Per-cell match state
        best_delta_ome = np.full((N, K), np.inf, dtype=np.float64)
        best_matched_id = np.full((N, K), -1, dtype=np.int64)
        best_matched_row = np.full((N, K), -1, dtype=np.int64)
        has_match = np.zeros((N, K), dtype=np.bool_)
        # Cached theor cols for downstream (avg_ia uses theor_y/z/omega; reduce.pack_score uses omega/eta/ring)
        theor_omega = np.zeros((N, K), dtype=np.float64)
        theor_eta = np.zeros((N, K), dtype=np.float64)
        theor_yl_disp = np.zeros((N, K), dtype=np.float64)
        theor_zl_disp = np.zeros((N, K), dtype=np.float64)
        theor_ring_nr = np.full((N, K), -1.0, dtype=np.float64)
        valid_mask = np.zeros((N, K), dtype=np.bool_)
        # Per-N reductions
        n_matches_out = np.zeros(N, dtype=np.int64)
        n_matches_frac_out = np.zeros(N, dtype=np.int64)
        n_t_frac_out = np.zeros(N, dtype=np.int64)
        avg_ia_out = np.zeros(N, dtype=np.float64)

        n_bin = bin_data.shape[0]
        n_obs = obs_ome.shape[0]
        n_scans_pos = scan_pos_arr.shape[0]
        n_bins_total = bin_ndata.shape[0] // 2
        bin_max_idx = n_bins_total - 1
        n_reject = rings_to_reject.shape[0]
        bins_per_ring = n_eta_bins * n_ome_bins

        for n in prange(N):
            R00 = R[n, 0, 0]; R01 = R[n, 0, 1]; R02 = R[n, 0, 2]
            R10 = R[n, 1, 0]; R11 = R[n, 1, 1]; R12 = R[n, 1, 2]
            R20 = R[n, 2, 0]; R21 = R[n, 2, 1]; R22 = R[n, 2, 2]
            ga = pos[n, 0]; gb = pos[n, 1]; gc = pos[n, 2]
            ref_rad_n = ref_rad[n]
            v_x = voxel_x[n] if scan_active else 0.0
            v_y = voxel_y[n] if scan_active else 0.0

            local_n_matches = 0
            local_n_matches_frac = 0
            local_n_t_frac = 0
            local_avg_ia_sum = 0.0
            local_avg_ia_count = 0

            for m in range(M):
                hx = hkls_cart[m, 0]; hy = hkls_cart[m, 1]; hz = hkls_cart[m, 2]
                theta_m = thetas[m]
                len_h = len_hkl[m]

                Gx_c = R00 * hx + R01 * hy + R02 * hz
                Gy_c = R10 * hx + R11 * hy + R12 * hz
                Gz_c = R20 * hx + R21 * hy + R22 * hz

                v_no_wedge = math.sin(theta_m) * len_h

                Gx_p = cos_W * Gx_c - sin_W * Gz_c
                Gy_p = Gy_c
                Gz_p = sin_W * Gx_c + cos_W * Gz_c
                Gx_eff = cos_W * Gx_p
                Gy_eff = cos_W * Gy_p
                v_eff = v_no_wedge + sin_W * Gz_p

                y2 = Gy_eff * Gy_eff + epsilon
                x2 = Gx_eff * Gx_eff
                a = 1.0 + x2 / y2
                b = 2.0 * v_eff * Gx_eff / y2
                c = v_eff * v_eff / y2 - 1.0
                discriminant = b * b - 4.0 * a * c

                gy_zero = abs(Gy_eff) < almostzero
                disc_geq_zero = discriminant >= 0.0
                disc_valid_quad = disc_geq_zero and not gy_zero

                sqrt_disc = math.sqrt(abs(discriminant))
                coswp = (-b + sqrt_disc) / (2.0 * a)
                coswn = (-b - sqrt_disc) / (2.0 * a)

                coswp_clamp = coswp
                if coswp_clamp > 1.0:
                    coswp_clamp = 1.0
                elif coswp_clamp < -1.0:
                    coswp_clamp = -1.0
                coswn_clamp = coswn
                if coswn_clamp > 1.0:
                    coswn_clamp = 1.0
                elif coswn_clamp < -1.0:
                    coswn_clamp = -1.0
                wap = math.acos(coswp_clamp); wan = math.acos(coswn_clamp)
                wbp = -wap; wbn = -wan
                eqap = -Gx_eff * math.cos(wap) + Gy_eff * math.sin(wap)
                eqbp = -Gx_eff * math.cos(wbp) + Gy_eff * math.sin(wbp)
                eqan = -Gx_eff * math.cos(wan) + Gy_eff * math.sin(wan)
                eqbn = -Gx_eff * math.cos(wbn) + Gy_eff * math.sin(wbn)
                all_wp = wap if abs(eqap - v_eff) < abs(eqbp - v_eff) else wbp
                all_wn = wan if abs(eqan - v_eff) < abs(eqbn - v_eff) else wbn

                cosome_special_valid = False
                special_w = 0.0
                if gy_zero and abs(Gx_eff) > epsilon:
                    cosome_special = -v_eff / Gx_eff
                    if abs(cosome_special) <= 1.0:
                        cos_clamp = cosome_special
                        if cos_clamp > 1.0:
                            cos_clamp = 1.0
                        elif cos_clamp < -1.0:
                            cos_clamp = -1.0
                        special_w = math.acos(cos_clamp)
                        cosome_special_valid = True

                coswp_in_range = (coswp >= -1.0) and (coswp <= 1.0)
                coswn_in_range = (coswn >= -1.0) and (coswn <= 1.0)
                if cosome_special_valid:
                    omega_p_rad = special_w
                    omega_n_rad = -special_w
                    valid_p_quad = True
                    valid_n_quad = True
                else:
                    omega_p_rad = all_wp if (disc_valid_quad and coswp_in_range) else 0.0
                    omega_n_rad = all_wn if (disc_valid_quad and coswn_in_range) else 0.0
                    valid_p_quad = disc_valid_quad and coswp_in_range
                    valid_n_quad = disc_valid_quad and coswn_in_range

                rn = ring_nr_per_hkl[m]
                if rn < 0:
                    ring_idx = 0
                elif rn > ring_radius_lut_max_idx:
                    ring_idx = ring_radius_lut_max_idx
                else:
                    ring_idx = rn
                ring_radius = ring_radius_lut[ring_idx]

                # skip_radial for this ring
                skip_radial_m = False
                for kk in range(n_reject):
                    if rn == rings_to_reject[kk]:
                        skip_radial_m = True
                        break

                # eta margin for this ring
                ring_clamped = rn
                if ring_clamped < 0:
                    ring_clamped = 0
                elif ring_clamped > eta_margins_max_idx:
                    ring_clamped = eta_margins_max_idx
                eta_marg = eta_margins_lut[ring_clamped]

                for branch in range(2):
                    if branch == 0:
                        omega_rad = omega_p_rad
                        valid_branch = valid_p_quad
                        out_k = m
                    else:
                        omega_rad = omega_n_rad
                        valid_branch = valid_n_quad
                        out_k = M + m

                    cos_w = math.cos(omega_rad)
                    sin_w = math.sin(omega_rad)
                    mx = cos_w * Gx_p - sin_w * Gy_p
                    my = sin_w * Gx_p + cos_w * Gy_p
                    mz = Gz_p
                    Gy_lab = my
                    Gz_lab = -sin_W * mx + cos_W * mz

                    r_yz = math.sqrt(Gy_lab * Gy_lab + Gz_lab * Gz_lab)
                    if r_yz < epsilon:
                        r_yz = epsilon
                    ratio = Gz_lab / r_yz
                    if ratio > 1.0:
                        ratio = 1.0
                    elif ratio < -1.0:
                        ratio = -1.0
                    eta_unsigned = math.acos(ratio)
                    if Gy_lab > 0:
                        eta_rad = -eta_unsigned
                    elif Gy_lab < 0:
                        eta_rad = eta_unsigned
                    else:
                        eta_rad = eta_unsigned

                    omega_deg = omega_rad * RAD2DEG_
                    eta_deg = eta_rad * RAD2DEG_
                    yl_no_disp = -math.sin(eta_rad) * ring_radius
                    zl_no_disp = math.cos(eta_rad) * ring_radius

                    abs_eta = abs(eta_rad)
                    eta_ok = (abs_eta >= min_eta_rad) and ((PI - abs_eta) >= min_eta_rad)
                    is_valid = valid_branch and eta_ok

                    if has_omega_box and is_valid:
                        any_range_ok = False
                        for r_idx in range(omega_ranges_deg.shape[0]):
                            if (omega_deg > omega_ranges_deg[r_idx, 0]
                                    and omega_deg < omega_ranges_deg[r_idx, 1]
                                    and yl_no_disp > box_sizes[r_idx, 0]
                                    and yl_no_disp < box_sizes[r_idx, 1]
                                    and zl_no_disp > box_sizes[r_idx, 2]
                                    and zl_no_disp < box_sizes[r_idx, 3]):
                                any_range_ok = True
                                break
                        if not any_range_ok:
                            is_valid = False

                    # COM displacement
                    L = math.sqrt(Lsd * Lsd + yl_no_disp * yl_no_disp + zl_no_disp * zl_no_disp)
                    xi_n = Lsd / L
                    yi_n = yl_no_disp / L
                    zi_n = zl_no_disp / L
                    t = (ga * cos_w - gb * sin_w) / xi_n
                    dy = (ga * sin_w + gb * cos_w) - t * yi_n
                    dz = gc - t * zi_n
                    yl_disp = yl_no_disp + dy
                    zl_disp = zl_no_disp + dz

                    eta_deg_post = math.atan2(-yl_disp, zl_disp) * RAD2DEG_
                    rad_diff = math.sqrt(yl_disp * yl_disp + zl_disp * zl_disp) - ring_radius

                    # Stash theor cols for downstream best_global gather + skip_radial reconstruction
                    theor_omega[n, out_k] = omega_deg
                    theor_eta[n, out_k] = eta_deg
                    theor_yl_disp[n, out_k] = yl_disp
                    theor_zl_disp[n, out_k] = zl_disp
                    theor_ring_nr[n, out_k] = float(rn)
                    valid_mask[n, out_k] = is_valid

                    if not is_valid:
                        continue

                    # n_t_frac (excludes rings_to_reject)
                    if not skip_radial_m:
                        local_n_t_frac += 1

                    # ── Bin lookup + match
                    if rn < 0:
                        continue
                    i_eta = int((180.0 + eta_deg_post) / eta_bin_size)
                    i_ome = int((180.0 + omega_deg) / ome_bin_size)
                    bin_pos = (rn - 1) * bins_per_ring + i_eta * n_ome_bins + i_ome
                    if bin_pos < 0:
                        bin_pos = 0
                    elif bin_pos > bin_max_idx:
                        bin_pos = bin_max_idx
                    n_p = np.int64(bin_ndata[bin_pos * 2])
                    if n_p == 0:
                        continue
                    offset = np.int64(bin_ndata[bin_pos * 2 + 1])

                    s_proj_nt = 0.0
                    if scan_active:
                        s_proj_nt = v_x * math.sin(omega_deg * DEG2RAD_) + v_y * math.cos(omega_deg * DEG2RAD_)

                    local_best_dome = np.inf
                    local_best_id = np.int64(-1)
                    local_best_row = np.int64(-1)
                    local_has_match = False

                    for m_idx in range(n_p):
                        row_idx = offset + m_idx
                        if row_idx < 0 or row_idx >= n_bin:
                            continue
                        row = np.int64(bin_data[row_idx])
                        if row < 0 or row >= n_obs:
                            continue
                        if abs(rad_diff - obs_rad[row]) >= margin_radial:
                            continue
                        if abs(eta_deg_post - obs_eta[row]) >= eta_marg:
                            continue
                        if not skip_radial_m:
                            if abs(ref_rad_n - obs_ringrad[row]) >= margin_rad:
                                continue
                        if scan_active:
                            scan_idx = obs_scan_idx[row]
                            if scan_idx < 0:
                                scan_idx = 0
                            elif scan_idx >= n_scans_pos:
                                scan_idx = n_scans_pos - 1
                            scan_pos = scan_pos_arr[scan_idx]
                            diff = abs(s_proj_nt - scan_pos)
                            if diff >= scan_pos_tol:
                                if friedel_sym:
                                    diff_f = abs(s_proj_nt + scan_pos)
                                    if diff_f >= scan_pos_tol:
                                        continue
                                else:
                                    continue
                        d_ome = abs(omega_deg - obs_ome[row])
                        if d_ome < local_best_dome:
                            local_best_dome = d_ome
                            local_best_id = obs_id[row]
                            local_best_row = row
                        local_has_match = True

                    if local_has_match:
                        best_delta_ome[n, out_k] = local_best_dome
                        best_matched_id[n, out_k] = local_best_id
                        best_matched_row[n, out_k] = local_best_row
                        has_match[n, out_k] = True
                        local_n_matches += 1
                        if not skip_radial_m:
                            local_n_matches_frac += 1

                        # ── Inline avg_ia for this matched cell
                        om_t = omega_deg * DEG2RAD_
                        co_t = math.cos(om_t)
                        so_t = math.sin(om_t)
                        vr_x = co_t * ga - so_t * gb
                        vr_y = so_t * ga + co_t * gb
                        xi = Lsd - vr_x
                        yi = yl_disp - vr_y
                        zi = zl_disp - gc
                        L2 = math.sqrt(xi * xi + yi * yi + zi * zi + 1e-60)
                        Linv = 1.0 / L2
                        xn = xi * Linv; yn = yi * Linv; zn = zi * Linv
                        g1r = -1.0 + xn
                        g2r = yn
                        g1_t = g1r * co_t + g2r * so_t
                        g2_t = -g1r * so_t + g2r * co_t
                        g3_t = zn
                        # Obs side
                        om_o = obs_ome[local_best_row] * DEG2RAD_
                        co_o = math.cos(om_o)
                        so_o = math.sin(om_o)
                        vr_xo = co_o * ga - so_o * gb
                        vr_yo = so_o * ga + co_o * gb
                        xi2 = Lsd - vr_xo
                        yi2 = obs_y[local_best_row] - vr_yo
                        zi2 = obs_z[local_best_row] - gc
                        L3 = math.sqrt(xi2 * xi2 + yi2 * yi2 + zi2 * zi2 + 1e-60)
                        Linv2 = 1.0 / L3
                        xn2 = xi2 * Linv2; yn2 = yi2 * Linv2; zn2 = zi2 * Linv2
                        g1r2 = -1.0 + xn2
                        g2r2 = yn2
                        g1_o = g1r2 * co_o + g2r2 * so_o
                        g2_o = -g1r2 * so_o + g2r2 * co_o
                        g3_o = zn2
                        n1 = math.sqrt(g1_t * g1_t + g2_t * g2_t + g3_t * g3_t)
                        n2 = math.sqrt(g1_o * g1_o + g2_o * g2_o + g3_o * g3_o)
                        if n1 >= 1e-30 and n2 >= 1e-30:
                            cos_ia = (g1_t * g1_o + g2_t * g2_o + g3_t * g3_o) / (n1 * n2)
                            if cos_ia > 1.0:
                                cos_ia = 1.0
                            elif cos_ia < -1.0:
                                cos_ia = -1.0
                            ia = math.acos(cos_ia) * RAD2DEG_
                            local_avg_ia_sum += abs(ia)
                            local_avg_ia_count += 1

            n_matches_out[n] = local_n_matches
            n_matches_frac_out[n] = local_n_matches_frac
            if local_n_t_frac < 1:
                local_n_t_frac = 1
            n_t_frac_out[n] = local_n_t_frac
            if local_avg_ia_count > 0:
                avg_ia_out[n] = local_avg_ia_sum / local_avg_ia_count

        return (
            best_delta_ome, best_matched_id, best_matched_row, has_match,
            theor_omega, theor_eta, theor_yl_disp, theor_zl_disp,
            theor_ring_nr, valid_mask,
            n_matches_out, n_matches_frac_out, n_t_frac_out, avg_ia_out,
        )

else:  # pragma: no cover
    def _simulate_and_compare_fused_numba(*args, **kwargs):  # type: ignore
        raise ImportError("numba is required for the fused kernel")
