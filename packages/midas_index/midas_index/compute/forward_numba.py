"""Numba CPU port of the indexer's forward simulation.

Mirrors ``IndexerForwardAdapter.simulate`` (which calls ``HEDMForwardModel.
calc_bragg_geometry`` then applies wedge, ring-radius lookup, OmegaRange +
BoxSize gating, COM displacement, and assembles the 14-column TheorSpots
layout). The torch version is preserved for GPU; this numba version runs
when the indexer is on CPU and replaces the per-seed simulate call that
profiled at ~5 s/call on Wenxi-class PF data — biggest hot spot after
``compare_spots`` + ``avg_ia`` were ported in earlier phases.

Per-(n, m, ±) cell loop:
    1. G_C = R[n] @ hkls_cart[m]                (3-vec)
    2. Apply wedge tilt → effective (Gx, Gy, v_eff)
    3. Solve quadratic for omega → two branches (omega_p, omega_n)
    4. For each branch:
       a. Compute Bragg geometry → eta, two_theta
       b. Apply ring-radius lookup → yl_no_disp, zl_no_disp
       c. Validity: discriminant, cos in [-1, 1], min_eta, OmegaRange + BoxSize
       d. COM displacement → yl_disp, zl_disp
       e. Post-disp eta_deg_post, rad_diff
       f. Write 14-col theor row + valid bit

Numba parallelises over the N (orientation candidate) axis via ``prange``.
Total work scales O(N · M · 2) — same as the torch path but with no tensor
allocations and tight per-cell arithmetic.

Currently lacks: per-panel η coverage mask (multi-detector FF). Falls back
to the torch path when ``IndexerForwardAdapter._has_panel_coverage`` is True.
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


# ---------------------------------------------------------------------------
# Numba kernel — full simulate per cell
# ---------------------------------------------------------------------------


if _NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True, fastmath=False)
    def _simulate_numba_inner(
        R,                        # (N, 3, 3) float64
        pos,                      # (N, 3) float64
        hkls_cart,                # (M, 3) float64
        thetas,                   # (M,) float64 — Bragg angles (rad)
        len_hkl,                  # (M,) float64 — |hkls_cart[m]|
        ring_nr_per_hkl,          # (M,) int64
        ring_radius_lut,          # (max_ring + 1,) float64
        ring_radius_lut_max_idx,  # scalar int64 — max index in lut
        wedge_rad,                # scalar
        Lsd,                      # scalar
        min_eta_rad,              # scalar
        omega_ranges_deg,         # (n_ranges, 2) float64 — (omega_min, omega_max)
        box_sizes,                # (n_ranges, 4) float64 — (y_min, y_max, z_min, z_max)
        has_omega_box,            # bool
        epsilon,                  # scalar (typically 1e-12 or so)
    ):
        """Per-(n, m, ±) numba forward simulate.

        Returns ``(theor, valid)`` with shapes ``(N, 2*M, 14)`` and ``(N, 2*M)``,
        layout matching the torch version's 14-col schema:
            0: zero placeholder
            1: spotnr
            2: hkl_idx
            3: distance
            4: yl_no_disp
            5: zl_no_disp
            6: omega_deg
            7: eta_deg
            8: theta_deg
            9: ring_nr (as float)
            10: yl_disp
            11: zl_disp
            12: eta_deg_post
            13: rad_diff
        """
        DEG2RAD_ = math.pi / 180.0
        RAD2DEG_ = 180.0 / math.pi
        N = R.shape[0]
        M = hkls_cart.shape[0]
        K = 2 * M  # two omega branches per HKL
        almostzero = 1e-12

        cos_W = math.cos(wedge_rad)
        sin_W = math.sin(wedge_rad)

        theor = np.zeros((N, K, 14), dtype=np.float64)
        valid = np.zeros((N, K), dtype=np.bool_)

        # Pi cached
        PI = math.pi

        for n in prange(N):
            # Pre-extract R rows for this n.
            R00 = R[n, 0, 0]; R01 = R[n, 0, 1]; R02 = R[n, 0, 2]
            R10 = R[n, 1, 0]; R11 = R[n, 1, 1]; R12 = R[n, 1, 2]
            R20 = R[n, 2, 0]; R21 = R[n, 2, 1]; R22 = R[n, 2, 2]
            ga = pos[n, 0]
            gb = pos[n, 1]
            gc = pos[n, 2]

            for m in range(M):
                hx = hkls_cart[m, 0]
                hy = hkls_cart[m, 1]
                hz = hkls_cart[m, 2]
                theta_m = thetas[m]
                len_h = len_hkl[m]

                # G_C = R @ hkls_cart[m]  (3-vec)
                Gx_c = R00 * hx + R01 * hy + R02 * hz
                Gy_c = R10 * hx + R11 * hy + R12 * hz
                Gz_c = R20 * hx + R21 * hy + R22 * hz

                v_no_wedge = math.sin(theta_m) * len_h

                # Wedge: G' = R_y(-W) @ G_C
                Gx_p = cos_W * Gx_c - sin_W * Gz_c
                Gy_p = Gy_c
                Gz_p = sin_W * Gx_c + cos_W * Gz_c
                # Effective for quadratic solver
                Gx_eff = cos_W * Gx_p
                Gy_eff = cos_W * Gy_p
                v_eff = v_no_wedge + sin_W * Gz_p

                # Quadratic: a*cos²(ω) + b*cos(ω) + c = 0
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

                # safe_arccos: clamp to [-1, 1] then acos
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
                wap = math.acos(coswp_clamp)
                wan = math.acos(coswn_clamp)
                wbp = -wap
                wbn = -wan

                # Pick branch that best satisfies -Gx*cos(w)+Gy*sin(w)=v
                eqap = -Gx_eff * math.cos(wap) + Gy_eff * math.sin(wap)
                eqbp = -Gx_eff * math.cos(wbp) + Gy_eff * math.sin(wbp)
                eqan = -Gx_eff * math.cos(wan) + Gy_eff * math.sin(wan)
                eqbn = -Gx_eff * math.cos(wbn) + Gy_eff * math.sin(wbn)
                all_wp = wap if abs(eqap - v_eff) < abs(eqbp - v_eff) else wbp
                all_wn = wan if abs(eqan - v_eff) < abs(eqbn - v_eff) else wbn

                # Gy ~ 0 special case
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
                # Final omega values for both branches
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

                # Ring radius lookup
                rn = ring_nr_per_hkl[m]
                if rn < 0:
                    ring_idx = 0
                elif rn > ring_radius_lut_max_idx:
                    ring_idx = ring_radius_lut_max_idx
                else:
                    ring_idx = rn
                ring_radius = ring_radius_lut[ring_idx]

                # Process both branches: K-axis index 0..M-1 is +, M..2M-1 is -
                # (matches the torch path's reshape view(2, N, m).permute → cat).
                for branch in range(2):
                    if branch == 0:
                        omega_rad = omega_p_rad
                        valid_branch = valid_p_quad
                        out_k = m              # branch 0 (+) at columns [0, M)
                    else:
                        omega_rad = omega_n_rad
                        valid_branch = valid_n_quad
                        out_k = M + m          # branch 1 (-) at columns [M, 2M)

                    cos_w = math.cos(omega_rad)
                    sin_w = math.sin(omega_rad)
                    # m_lab = R_z(omega) @ G'  (sample-frame)
                    mx = cos_w * Gx_p - sin_w * Gy_p
                    my = sin_w * Gx_p + cos_w * Gy_p
                    mz = Gz_p
                    # G_lab = R_y(W) @ m
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
                    # eta = -sign(Gy_lab) * eta_unsigned
                    if Gy_lab > 0:
                        eta_rad = -eta_unsigned
                    elif Gy_lab < 0:
                        eta_rad = eta_unsigned
                    else:
                        eta_rad = eta_unsigned  # sign(0) = 0 → +eta

                    omega_deg = omega_rad * RAD2DEG_
                    eta_deg = eta_rad * RAD2DEG_
                    theta_deg = theta_m * RAD2DEG_   # (was theta_deg = (two_theta_rad * 0.5) * RAD2DEG; equivalent)
                    yl_no_disp = -math.sin(eta_rad) * ring_radius
                    zl_no_disp = math.cos(eta_rad) * ring_radius

                    # Eta bounds: |eta| >= min_eta AND (pi - |eta|) >= min_eta
                    abs_eta = abs(eta_rad)
                    eta_ok = (abs_eta >= min_eta_rad) and ((PI - abs_eta) >= min_eta_rad)
                    is_valid = valid_branch and eta_ok

                    # OmegaRange + BoxSize gating
                    if has_omega_box and is_valid:
                        any_range_ok = False
                        for r_idx in range(omega_ranges_deg.shape[0]):
                            o_min = omega_ranges_deg[r_idx, 0]
                            o_max = omega_ranges_deg[r_idx, 1]
                            y_min = box_sizes[r_idx, 0]
                            y_max = box_sizes[r_idx, 1]
                            z_min = box_sizes[r_idx, 2]
                            z_max = box_sizes[r_idx, 3]
                            if (omega_deg > o_min and omega_deg < o_max
                                    and yl_no_disp > y_min and yl_no_disp < y_max
                                    and zl_no_disp > z_min and zl_no_disp < z_max):
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

                    distance = L  # sqrt(Lsd² + yl_nd² + zl_nd²)
                    # Post-disp eta and rad_diff
                    eta_deg_post = math.atan2(-yl_disp, zl_disp) * RAD2DEG_
                    rad_diff = math.sqrt(yl_disp * yl_disp + zl_disp * zl_disp) - ring_radius

                    # Assemble 14-col row
                    theor[n, out_k, 0] = 0.0
                    theor[n, out_k, 1] = float(n * K + out_k)    # spotnr
                    theor[n, out_k, 2] = float(m)                 # hkl_idx
                    theor[n, out_k, 3] = distance
                    theor[n, out_k, 4] = yl_no_disp
                    theor[n, out_k, 5] = zl_no_disp
                    theor[n, out_k, 6] = omega_deg
                    theor[n, out_k, 7] = eta_deg
                    theor[n, out_k, 8] = theta_deg
                    theor[n, out_k, 9] = float(rn)
                    theor[n, out_k, 10] = yl_disp
                    theor[n, out_k, 11] = zl_disp
                    theor[n, out_k, 12] = eta_deg_post
                    theor[n, out_k, 13] = rad_diff
                    valid[n, out_k] = is_valid

        return theor, valid

else:  # pragma: no cover
    def _simulate_numba_inner(*args, **kwargs):  # type: ignore
        raise ImportError("numba is required for forward_numba")


# ---------------------------------------------------------------------------
# Python wrapper — invoked by IndexerForwardAdapter.simulate on CPU
# ---------------------------------------------------------------------------


def simulate_numba(adapter, R, pos, lattice=None):
    """Drop-in CPU replacement for ``IndexerForwardAdapter.simulate``.

    Computes the forward Bragg geometry + ring radius lookup + OmegaRange
    gating + COM displacement entirely in a per-(n, m, ±) numba kernel.
    Outputs match the torch path's 14-col theor shape ``(N, 2M, 14)`` and
    ``valid (N, 2M) bool``.

    Falls back to the torch ``adapter.simulate`` if numba isn't available
    or if the adapter has a multi-detector panel-coverage mask (which
    the numba path doesn't yet implement).
    """
    import torch
    if not _NUMBA_AVAILABLE or adapter._has_panel_coverage:
        return adapter.simulate(R, pos, lattice=lattice)

    if lattice is not None:
        # Per-call strained HKLs: re-derive on CPU.
        hkls_cart_t, thetas_t = adapter._model.correct_hkls_latc(lattice)
    else:
        hkls_cart_t = adapter.hkls_real[:, :3]
        thetas_t = adapter.hkls_real[:, 5]

    device = R.device
    dtype = R.dtype

    R_np = np.ascontiguousarray(R.detach().cpu().numpy().astype(np.float64, copy=False))
    pos_np = np.ascontiguousarray(pos.detach().cpu().numpy().astype(np.float64, copy=False))
    hkls_cart_np = np.ascontiguousarray(hkls_cart_t.detach().cpu().numpy().astype(np.float64, copy=False))
    thetas_np = np.ascontiguousarray(thetas_t.detach().cpu().numpy().astype(np.float64, copy=False))

    len_hkl = np.linalg.norm(hkls_cart_np, axis=-1)

    ring_nr_per_hkl_np = np.ascontiguousarray(
        adapter.ring_nr_per_hkl.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    ring_radius_lut_np = np.ascontiguousarray(
        adapter.ring_radius_lut.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    wedge_rad = float(adapter.geom.wedge * math.pi / 180.0) if hasattr(adapter.geom, "wedge") else 0.0
    # Indexer params usually carry wedge directly:
    if hasattr(adapter.params, "Wedge"):
        wedge_rad = float(adapter.params.Wedge) * math.pi / 180.0
    Lsd = float(adapter.params.Distance)
    min_eta_rad = float(adapter.params.ExcludePoleAngle) * math.pi / 180.0

    if adapter.omega_ranges is not None and adapter.omega_ranges.numel() > 0:
        omega_ranges_np = np.ascontiguousarray(
            adapter.omega_ranges.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        box_sizes_np = np.ascontiguousarray(
            adapter.box_sizes.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        has_omega_box = True
    else:
        omega_ranges_np = np.zeros((0, 2), dtype=np.float64)
        box_sizes_np = np.zeros((0, 4), dtype=np.float64)
        has_omega_box = False

    theor_np, valid_np = _simulate_numba_inner(
        R_np, pos_np,
        hkls_cart_np, thetas_np, len_hkl,
        ring_nr_per_hkl_np, ring_radius_lut_np, int(ring_radius_lut_np.shape[0] - 1),
        wedge_rad, Lsd, min_eta_rad,
        omega_ranges_np, box_sizes_np, has_omega_box,
        1e-12,
    )

    theor = torch.from_numpy(theor_np).to(device=device, dtype=dtype)
    valid = torch.from_numpy(valid_np).to(device=device)
    return theor, valid
