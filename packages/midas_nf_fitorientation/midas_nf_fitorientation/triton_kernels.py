"""Triton kernels for the NF-HEDM fit hot path.

The eager PyTorch forward (Bragg geometry → projection → tilts → obs
lookup) launches ~60 small kernels per ``fn`` call. With ~50 NM
iterations × 4 candidate evaluations, that's ~12 K kernel launches per
voxel — and on H100 each launch costs ~5 µs of CPU, so the workload
becomes launch-overhead bound.

This module replaces the whole pipeline with **one** Triton kernel
that:

1. computes ``R = euler2mat(eulers)`` per grain;
2. solves the wedge-aware Bragg quadratic for both omega solutions
   per HKL;
3. projects the predicted spot through the voxel position at every
   detector distance;
4. applies the per-distance NF tilt ray-plane correction;
5. extracts the bit at ``(d, frame, y, z)`` from the packed obs
   volume;
6. AND-products across distances and accumulates per-grain ``hit`` and
   ``total`` counts via ``tl.sum``.

The caller divides ``hit / total`` to get the hard FracOverlap. Same
math as :func:`midas_nf_fitorientation.obs_volume.hard_fraction`
composed with :func:`midas_nf_fitorientation.soft_overlap.forward_batched_grains`,
but in a single launch.

Currently covers the NF orientation-fit path: ``flip_y=False``,
``multi_mode="layered"``. Wedge and tilts are toggled at compile time
via ``HAS_WEDGE`` and ``HAS_TILTS`` so the kernel specialises for the
common ``wedge=0``, ``tilts=0`` case at no runtime cost.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False
    triton = None
    tl = None


_PI = math.pi
_DEG2RAD = _PI / 180.0
_RAD2DEG = 180.0 / _PI

# Placeholder passed as ``ome_box_ptr`` when the gate is off (NBOX=0). The
# kernel never dereferences it, but Triton still needs a real pointer. Cached
# per device: ``fused_hard_frac`` is called once per NM iteration per chunk,
# so allocating a throwaway tensor here would be thousands of device allocs.
_DUMMY_OME_BOX: "dict[torch.device, torch.Tensor]" = {}


def _dummy_ome_box(device: "torch.device") -> "torch.Tensor":
    t = _DUMMY_OME_BOX.get(device)
    if t is None:
        t = torch.zeros(1, 6, dtype=torch.float32, device=device)
        _DUMMY_OME_BOX[device] = t
    return t


if HAS_TRITON:

    @triton.jit
    def fused_hard_frac_kernel(
        # ---- input pointers ----
        eul_ptr,           # (B, 3) fp32  (Bunge ZXZ in radians)
        pos_ptr,           # (B, 3) fp32  (voxel positions in µm)
        hkls_ptr,          # (M, 3) fp32  (Cartesian G in 1/Å)
        thetas_ptr,        # (M,)   fp32  (Bragg angles in radians)
        Lsd_ptr,           # (D,)   fp32
        ybc_ptr,           # (D,)   fp32
        zbc_ptr,           # (D,)   fp32
        R_tilt_ptr,        # (D, 9) fp32  flattened tilt rotation per d
        obs_packed_ptr,    # (total_bytes,) uint8
        ome_box_ptr,       # (NBOX, 6) fp32  [ome_lo, ome_hi, ymin, ymax, zmin, zmax]
        # ---- output pointers ----
        hit_ptr,           # (B,) int32
        total_ptr,         # (B,) int32
        # ---- scalar geometry ----
        px,
        wedge_rad,
        omega_start_deg,
        omega_step_deg,
        min_eta_rad,
        n_frames,
        n_y,
        n_z,
        # ---- sizes / flags ----
        M,
        D: tl.constexpr,
        HAS_TILTS: tl.constexpr,
        HAS_WEDGE: tl.constexpr,
        NBOX: tl.constexpr,      # 0 ⇒ gate disabled (default, bit-identical to before)
        BLOCK_M: tl.constexpr,
    ):
        """One thread block per grain; threads cooperate over BLOCK_M
        HKLs as a single SIMT vector.
        """
        b = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_M)
        m_mask = offsets < M

        EPS = 1e-7
        ALMOST_ZERO = 1e-12
        TWO_PI = 2.0 * 3.14159265358979323846
        DEG2RAD = 3.14159265358979323846 / 180.0
        RAD2DEG = 180.0 / 3.14159265358979323846

        # ---- per-grain scalars (broadcast to all HKLs) ----
        e0 = tl.load(eul_ptr + b * 3 + 0)
        e1 = tl.load(eul_ptr + b * 3 + 1)
        e2 = tl.load(eul_ptr + b * 3 + 2)
        pos_x = tl.load(pos_ptr + b * 3 + 0)
        pos_y = tl.load(pos_ptr + b * 3 + 1)
        # pos_z always 0 in our pipeline; ignore.

        # Rotation matrix: Bunge ZXZ → Rz(e0)·Rx(e1)·Rz(e2)
        c0 = tl.cos(e0); s0 = tl.sin(e0)
        c1 = tl.cos(e1); s1 = tl.sin(e1)
        c2 = tl.cos(e2); s2 = tl.sin(e2)
        R00 = c0 * c2 - s0 * c1 * s2
        R01 = -s0 * c1 * c2 - c0 * s2
        R02 = s0 * s1
        R10 = s0 * c2 + c0 * c1 * s2
        R11 = c0 * c1 * c2 - s0 * s2
        R12 = -c0 * s1
        R20 = s1 * s2
        R21 = s1 * c2
        R22 = c1

        # Wedge (compile-time skip when zero).
        if HAS_WEDGE:
            cos_W = tl.cos(wedge_rad)
            sin_W = tl.sin(wedge_rad)
        else:
            cos_W = 1.0
            sin_W = 0.0

        # ---- vectorised over HKL ----
        h0 = tl.load(hkls_ptr + offsets * 3 + 0, mask=m_mask, other=0.0)
        h1 = tl.load(hkls_ptr + offsets * 3 + 1, mask=m_mask, other=0.0)
        h2 = tl.load(hkls_ptr + offsets * 3 + 2, mask=m_mask, other=0.0)
        theta = tl.load(thetas_ptr + offsets, mask=m_mask, other=0.0)

        # G_C = R @ H
        G_x = R00 * h0 + R01 * h1 + R02 * h2
        G_y = R10 * h0 + R11 * h1 + R12 * h2
        G_z = R20 * h0 + R21 * h1 + R22 * h2

        G_mag = tl.sqrt(G_x * G_x + G_y * G_y + G_z * G_z)
        sin_theta = tl.sin(theta)

        # Wedge transform: G' = R_y(-W) @ G; (Gx_eff, Gy_eff, v_eff)
        Gx_p = cos_W * G_x - sin_W * G_z
        Gy_p = G_y
        Gz_p = sin_W * G_x + cos_W * G_z
        Gx = cos_W * Gx_p
        Gy = cos_W * Gy_p
        v_eff = sin_theta * G_mag + sin_W * Gz_p

        # ---- Omega quadratic solver ----
        x2 = Gx * Gx
        y2 = Gy * Gy
        a_coef = 1.0 + x2 / (y2 + EPS)
        b_coef = 2.0 * v_eff * Gx / (y2 + EPS)
        c_coef = v_eff * v_eff / (y2 + EPS) - 1.0
        disc = b_coef * b_coef - 4.0 * a_coef * c_coef
        sqrt_disc = tl.sqrt(tl.abs(disc))

        coswp = (-b_coef + sqrt_disc) / (2.0 * a_coef)
        coswn = (-b_coef - sqrt_disc) / (2.0 * a_coef)
        coswp_clamp = tl.maximum(-1.0 + EPS, tl.minimum(1.0 - EPS, coswp))
        coswn_clamp = tl.maximum(-1.0 + EPS, tl.minimum(1.0 - EPS, coswn))
        wap = tl.extra.cuda.libdevice.acos(coswp_clamp)
        wan = tl.extra.cuda.libdevice.acos(coswn_clamp)
        wbp = -wap
        wbn = -wan

        # Branch select: pick the omega satisfying -Gx·cos(ω)+Gy·sin(ω)=v.
        eqap = -Gx * tl.cos(wap) + Gy * tl.sin(wap)
        eqbp = -Gx * tl.cos(wbp) + Gy * tl.sin(wbp)
        eqan = -Gx * tl.cos(wan) + Gy * tl.sin(wan)
        eqbn = -Gx * tl.cos(wbn) + Gy * tl.sin(wbn)

        all_wp = tl.where(tl.abs(eqap - v_eff) < tl.abs(eqbp - v_eff), wap, wbp)
        all_wn = tl.where(tl.abs(eqan - v_eff) < tl.abs(eqbn - v_eff), wan, wbn)

        # Gy ≈ 0 special case.
        gy_zero = tl.abs(Gy) < ALMOST_ZERO
        cosome_special = -v_eff / (Gx + EPS)
        cosome_valid = (
            (tl.abs(cosome_special) <= 1.0) & gy_zero & (tl.abs(Gx) > EPS)
        )
        special_clamp = tl.maximum(-1.0 + EPS, tl.minimum(1.0 - EPS, cosome_special))
        special_w = tl.extra.cuda.libdevice.acos(special_clamp)

        omega_p = tl.where(cosome_valid, special_w, all_wp)
        omega_n = tl.where(cosome_valid, -special_w, all_wn)

        disc_valid = (disc >= 0.0) & (~gy_zero)
        coswp_valid = (coswp >= -1.0) & (coswp <= 1.0)
        coswn_valid = (coswn >= -1.0) & (coswn <= 1.0)
        valid_p = (disc_valid & coswp_valid) | cosome_valid
        valid_n = (disc_valid & coswn_valid) | cosome_valid

        two_theta = 2.0 * theta

        # Loop over both omega solutions (K = 2). Triton doesn't have
        # a native scalar "K" loop, so we just inline both branches —
        # each ~70 lines of math, but cheap because we already have all
        # the per-(b, m) state in registers.
        hit_total_per_block = tl.zeros((BLOCK_M,), dtype=tl.int32)
        total_per_block = tl.zeros((BLOCK_M,), dtype=tl.int32)

        # ---- K = 0 branch ----
        omega = omega_p
        valid_k = valid_p

        # Eta from rotated lab-frame G.
        cos_w_om = tl.cos(omega); sin_w_om = tl.sin(omega)
        m_x = cos_w_om * Gx_p - sin_w_om * Gy_p
        m_y = sin_w_om * Gx_p + cos_w_om * Gy_p
        m_z = Gz_p
        Gy_lab = m_y
        Gz_lab = -sin_W * m_x + cos_W * m_z
        r_yz = tl.sqrt(Gy_lab * Gy_lab + Gz_lab * Gz_lab)
        r_yz_safe = tl.maximum(r_yz, EPS)
        eta_arg = tl.maximum(-1.0 + EPS, tl.minimum(1.0 - EPS, Gz_lab / r_yz_safe))
        eta = tl.extra.cuda.libdevice.acos(eta_arg)
        # Match the eager path exactly: forward.py:973 is
        # ``eta = -sign(Gy_lab) * eta`` and torch.sign(0) == 0, so
        # Gy_lab == 0 must give eta = 0, not acos(...).  Measure-zero,
        # but the OmegaRange/BoxSize gate makes eta decide ACCEPTANCE.
        eta = tl.where(Gy_lab > 0.0, -eta, tl.where(Gy_lab < 0.0, eta, 0.0))

        eta_ok = (tl.abs(eta) >= min_eta_rad) & ((3.14159265358979323846 - tl.abs(eta)) >= min_eta_rad)
        valid_k = valid_k & eta_ok

        # Frame number.
        frame_nr_f = (omega * RAD2DEG - omega_start_deg) / omega_step_deg
        frame_in_range = (frame_nr_f >= 0.0) & (frame_nr_f < n_frames)
        valid_k = valid_k & frame_in_range
        f_idx = tl.minimum(n_frames - 1, tl.maximum(0, frame_nr_f.to(tl.int32)))

        # Project at voxel position.
        cw = tl.cos(omega); sw = tl.sin(omega)
        x_grain = pos_x * cw - pos_y * sw
        y_grain = pos_x * sw + pos_y * cw
        # z_grain = 0 (pos_z = 0).

        tan_2th = tl.extra.cuda.libdevice.tan(two_theta)
        sin_eta = tl.sin(eta); cos_eta = tl.cos(eta)

        # ---- paired OmegaRange / BoxSize gate ----
        # Port of the KeepSpot loop in CalcDiffrSpots_Furnace
        # (NF_HEDM/src/CalcDiffractionSpots.c:225-243). Gates the NOMINAL
        # ring position at Lsd[0] -- before grain displacement, tilts and
        # BC conversion -- in MICROMETRES, with strict < / > at every edge,
        # paired by index with any-pair-accepts. A rejected spot leaves the
        # denominator entirely (it never enters TheorSpots in the C).
        if NBOX > 0:
            RingRad0 = tl.load(Lsd_ptr + 0) * tan_2th
            yl_nom = -sin_eta * RingRad0
            zl_nom = cos_eta * RingRad0
            ome_deg_k = omega * RAD2DEG
            keep_box = tl.zeros((BLOCK_M,), dtype=tl.int32)
            for ib in tl.static_range(NBOX):
                o_lo = tl.load(ome_box_ptr + ib * 6 + 0)
                o_hi = tl.load(ome_box_ptr + ib * 6 + 1)
                y_lo = tl.load(ome_box_ptr + ib * 6 + 2)
                y_hi = tl.load(ome_box_ptr + ib * 6 + 3)
                z_lo = tl.load(ome_box_ptr + ib * 6 + 4)
                z_hi = tl.load(ome_box_ptr + ib * 6 + 5)
                acc = ((ome_deg_k > o_lo) & (ome_deg_k < o_hi)
                       & (yl_nom > y_lo) & (yl_nom < y_hi)
                       & (zl_nom > z_lo) & (zl_nom < z_hi))
                keep_box = keep_box | acc.to(tl.int32)
            valid_k = valid_k & (keep_box > 0)

        # Per-distance projection + bit lookup, AND across distances.
        all_d_hit = tl.full((BLOCK_M,), 1, dtype=tl.int32)
        all_d_in_bounds = tl.full((BLOCK_M,), 1, dtype=tl.int32)

        for d in tl.static_range(D):
            Lsd_d = tl.load(Lsd_ptr + d)
            ybc_d = tl.load(ybc_ptr + d)
            zbc_d = tl.load(zbc_ptr + d)
            dist = Lsd_d - x_grain
            ydet = y_grain - dist * tan_2th * sin_eta
            zdet = dist * tan_2th * cos_eta

            if HAS_TILTS:
                # NF ray-plane intersection through the tilted detector
                # plane. P0 = -Lsd_d · R[:, 0]; P1 = ydet·R[:, 1] + zdet·R[:, 2];
                # solve x_out=0 line.
                R00d = tl.load(R_tilt_ptr + d * 9 + 0)
                R10d = tl.load(R_tilt_ptr + d * 9 + 3)
                R20d = tl.load(R_tilt_ptr + d * 9 + 6)
                R01d = tl.load(R_tilt_ptr + d * 9 + 1)
                R11d = tl.load(R_tilt_ptr + d * 9 + 4)
                R21d = tl.load(R_tilt_ptr + d * 9 + 7)
                R02d = tl.load(R_tilt_ptr + d * 9 + 2)
                R12d = tl.load(R_tilt_ptr + d * 9 + 5)
                R22d = tl.load(R_tilt_ptr + d * 9 + 8)
                p0x = -Lsd_d * R00d
                p0y = -Lsd_d * R10d
                p0z = -Lsd_d * R20d
                P1x = ydet * R01d + zdet * R02d
                P1y = ydet * R11d + zdet * R12d
                P1z = ydet * R21d + zdet * R22d
                ABCx = P1x - p0x
                ABCy = P1y - p0y
                ABCz = P1z - p0z
                safe = tl.where(tl.abs(ABCx) < EPS, EPS, ABCx)
                ydet = p0y - ABCy * p0x / safe
                zdet = p0z - ABCz * p0x / safe

            y_pix_f = ydet / px + ybc_d
            z_pix_f = zdet / px + zbc_d
            in_bd = (y_pix_f >= 0.0) & (y_pix_f < n_y) & (z_pix_f >= 0.0) & (z_pix_f < n_z)
            all_d_in_bounds = all_d_in_bounds & in_bd.to(tl.int32)

            # Floor to int (matches C ``(int)floor``); clamp for safe load.
            y_idx = tl.maximum(0, tl.minimum(n_y - 1, tl.floor(y_pix_f).to(tl.int32)))
            z_idx = tl.maximum(0, tl.minimum(n_z - 1, tl.floor(z_pix_f).to(tl.int32)))
            bit_idx = (
                (((d * n_frames + f_idx).to(tl.int64) * n_y + y_idx) * n_z + z_idx)
            )
            byte_idx = bit_idx >> 3
            bit_pos = (bit_idx & 7).to(tl.uint8)
            byte_val = tl.load(obs_packed_ptr + byte_idx)
            bit_val = ((byte_val >> bit_pos) & 1).to(tl.int32)
            all_d_hit = all_d_hit & bit_val

        weight = (valid_k.to(tl.int32)) & all_d_in_bounds
        hit_total_per_block += weight & all_d_hit
        total_per_block += weight

        # ---- K = 1 branch (mirror of K=0 with omega_n / valid_n) ----
        omega = omega_n
        valid_k = valid_n

        cos_w_om = tl.cos(omega); sin_w_om = tl.sin(omega)
        m_x = cos_w_om * Gx_p - sin_w_om * Gy_p
        m_y = sin_w_om * Gx_p + cos_w_om * Gy_p
        m_z = Gz_p
        Gy_lab = m_y
        Gz_lab = -sin_W * m_x + cos_W * m_z
        r_yz = tl.sqrt(Gy_lab * Gy_lab + Gz_lab * Gz_lab)
        r_yz_safe = tl.maximum(r_yz, EPS)
        eta_arg = tl.maximum(-1.0 + EPS, tl.minimum(1.0 - EPS, Gz_lab / r_yz_safe))
        eta = tl.extra.cuda.libdevice.acos(eta_arg)
        # Match the eager path exactly: forward.py:973 is
        # ``eta = -sign(Gy_lab) * eta`` and torch.sign(0) == 0, so
        # Gy_lab == 0 must give eta = 0, not acos(...).  Measure-zero,
        # but the OmegaRange/BoxSize gate makes eta decide ACCEPTANCE.
        eta = tl.where(Gy_lab > 0.0, -eta, tl.where(Gy_lab < 0.0, eta, 0.0))

        eta_ok = (tl.abs(eta) >= min_eta_rad) & ((3.14159265358979323846 - tl.abs(eta)) >= min_eta_rad)
        valid_k = valid_k & eta_ok

        frame_nr_f = (omega * RAD2DEG - omega_start_deg) / omega_step_deg
        frame_in_range = (frame_nr_f >= 0.0) & (frame_nr_f < n_frames)
        valid_k = valid_k & frame_in_range
        f_idx = tl.minimum(n_frames - 1, tl.maximum(0, frame_nr_f.to(tl.int32)))

        cw = tl.cos(omega); sw = tl.sin(omega)
        x_grain = pos_x * cw - pos_y * sw
        y_grain = pos_x * sw + pos_y * cw

        tan_2th = tl.extra.cuda.libdevice.tan(two_theta)
        sin_eta = tl.sin(eta); cos_eta = tl.cos(eta)

        # Same OmegaRange/BoxSize gate as the K=0 branch.
        if NBOX > 0:
            RingRad0 = tl.load(Lsd_ptr + 0) * tan_2th
            yl_nom = -sin_eta * RingRad0
            zl_nom = cos_eta * RingRad0
            ome_deg_k = omega * RAD2DEG
            keep_box = tl.zeros((BLOCK_M,), dtype=tl.int32)
            for ib in tl.static_range(NBOX):
                o_lo = tl.load(ome_box_ptr + ib * 6 + 0)
                o_hi = tl.load(ome_box_ptr + ib * 6 + 1)
                y_lo = tl.load(ome_box_ptr + ib * 6 + 2)
                y_hi = tl.load(ome_box_ptr + ib * 6 + 3)
                z_lo = tl.load(ome_box_ptr + ib * 6 + 4)
                z_hi = tl.load(ome_box_ptr + ib * 6 + 5)
                acc = ((ome_deg_k > o_lo) & (ome_deg_k < o_hi)
                       & (yl_nom > y_lo) & (yl_nom < y_hi)
                       & (zl_nom > z_lo) & (zl_nom < z_hi))
                keep_box = keep_box | acc.to(tl.int32)
            valid_k = valid_k & (keep_box > 0)

        all_d_hit = tl.full((BLOCK_M,), 1, dtype=tl.int32)
        all_d_in_bounds = tl.full((BLOCK_M,), 1, dtype=tl.int32)

        for d in tl.static_range(D):
            Lsd_d = tl.load(Lsd_ptr + d)
            ybc_d = tl.load(ybc_ptr + d)
            zbc_d = tl.load(zbc_ptr + d)
            dist = Lsd_d - x_grain
            ydet = y_grain - dist * tan_2th * sin_eta
            zdet = dist * tan_2th * cos_eta

            if HAS_TILTS:
                R00d = tl.load(R_tilt_ptr + d * 9 + 0)
                R10d = tl.load(R_tilt_ptr + d * 9 + 3)
                R20d = tl.load(R_tilt_ptr + d * 9 + 6)
                R01d = tl.load(R_tilt_ptr + d * 9 + 1)
                R11d = tl.load(R_tilt_ptr + d * 9 + 4)
                R21d = tl.load(R_tilt_ptr + d * 9 + 7)
                R02d = tl.load(R_tilt_ptr + d * 9 + 2)
                R12d = tl.load(R_tilt_ptr + d * 9 + 5)
                R22d = tl.load(R_tilt_ptr + d * 9 + 8)
                p0x = -Lsd_d * R00d
                p0y = -Lsd_d * R10d
                p0z = -Lsd_d * R20d
                P1x = ydet * R01d + zdet * R02d
                P1y = ydet * R11d + zdet * R12d
                P1z = ydet * R21d + zdet * R22d
                ABCx = P1x - p0x
                ABCy = P1y - p0y
                ABCz = P1z - p0z
                safe = tl.where(tl.abs(ABCx) < EPS, EPS, ABCx)
                ydet = p0y - ABCy * p0x / safe
                zdet = p0z - ABCz * p0x / safe

            y_pix_f = ydet / px + ybc_d
            z_pix_f = zdet / px + zbc_d
            in_bd = (y_pix_f >= 0.0) & (y_pix_f < n_y) & (z_pix_f >= 0.0) & (z_pix_f < n_z)
            all_d_in_bounds = all_d_in_bounds & in_bd.to(tl.int32)

            y_idx = tl.maximum(0, tl.minimum(n_y - 1, tl.floor(y_pix_f).to(tl.int32)))
            z_idx = tl.maximum(0, tl.minimum(n_z - 1, tl.floor(z_pix_f).to(tl.int32)))
            bit_idx = (
                (((d * n_frames + f_idx).to(tl.int64) * n_y + y_idx) * n_z + z_idx)
            )
            byte_idx = bit_idx >> 3
            bit_pos = (bit_idx & 7).to(tl.uint8)
            byte_val = tl.load(obs_packed_ptr + byte_idx)
            bit_val = ((byte_val >> bit_pos) & 1).to(tl.int32)
            all_d_hit = all_d_hit & bit_val

        weight = (valid_k.to(tl.int32)) & all_d_in_bounds
        hit_total_per_block += weight & all_d_hit
        total_per_block += weight

        # Mask out off-tail HKLs and reduce to per-grain scalars.
        hit_total_per_block = tl.where(m_mask, hit_total_per_block, 0)
        total_per_block = tl.where(m_mask, total_per_block, 0)

        hit_count = tl.sum(hit_total_per_block, axis=0)
        total_count = tl.sum(total_per_block, axis=0)

        tl.store(hit_ptr + b, hit_count)
        tl.store(total_ptr + b, total_count)


# ---------------------------------------------------------------------------
#  Python wrapper
# ---------------------------------------------------------------------------

def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def fused_hard_frac(
    eul: torch.Tensor,
    pos: torch.Tensor,
    hkls_cart: torch.Tensor,
    thetas_rad: torch.Tensor,
    Lsd: torch.Tensor,
    ybc: torch.Tensor,
    zbc: torch.Tensor,
    R_tilt: torch.Tensor,
    obs_packed: torch.Tensor,
    *,
    px: float,
    wedge_rad: float,
    omega_start_deg: float,
    omega_step_deg: float,
    min_eta_rad: float,
    n_frames: int,
    n_y: int,
    n_z: int,
    has_tilts: bool,
    has_wedge: bool,
    ome_box: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """One-launch fused forward + obs lookup. Returns ``(B,) float32``
    of hard FracOverlap per grain.

    All input tensors must be on the same CUDA device. ``eul``, ``pos``,
    ``hkls_cart``, ``thetas_rad``, ``Lsd``, ``ybc``, ``zbc``, ``R_tilt``
    must be ``float32``; ``obs_packed`` is ``uint8``.
    """
    if not HAS_TRITON:
        raise RuntimeError("triton not installed")
    if not eul.is_cuda:
        raise RuntimeError("fused_hard_frac requires a CUDA tensor")
    B = eul.shape[0]
    M = hkls_cart.shape[0]
    D = int(Lsd.numel())

    BLOCK_M = _next_pow2(M)
    if BLOCK_M < 32:
        BLOCK_M = 32

    # Paired OmegaRange/BoxSize gate. ``ome_box`` is (NBOX, 6) fp32:
    # [ome_lo, ome_hi, ymin, ymax, zmin, zmax], omega in DEGREES and the box
    # in MICROMETRES, matching CalcDiffractionSpots.c:225-243. None ⇒ gate off,
    # and the kernel is then bit-identical to the pre-gate version.
    if ome_box is None:
        NBOX = 0
        ome_box_t = _dummy_ome_box(eul.device)
    else:
        ome_box_t = ome_box.to(device=eul.device, dtype=torch.float32).contiguous()
        if ome_box_t.ndim != 2 or ome_box_t.shape[1] != 6:
            raise ValueError(
                f"ome_box must be (NBOX, 6), got {tuple(ome_box_t.shape)}"
            )
        if ome_box_t.shape[0] == 0:
            # NBOX=0 means "gate off, keep everything" here, which is the
            # OPPOSITE of the C: CalcDiffractionSpots.c:192 initialises
            # KeepSpot=0 at function scope and only resets it inside the
            # range loop, so NOmegaRanges==0 there drops every spot. Pass
            # ome_box=None to disable the gate; an empty table is a bug.
            raise ValueError(
                "ome_box has 0 rows; pass ome_box=None to disable the gate "
                "(an empty table would silently keep every spot, the opposite "
                "of the C behaviour for NOmegaRanges==0)"
            )
        NBOX = int(ome_box_t.shape[0])

    hit = torch.empty(B, dtype=torch.int32, device=eul.device)
    total = torch.empty(B, dtype=torch.int32, device=eul.device)

    grid = (B,)
    fused_hard_frac_kernel[grid](
        eul, pos, hkls_cart, thetas_rad,
        Lsd, ybc, zbc, R_tilt,
        obs_packed,
        ome_box_t,
        hit, total,
        float(px),
        float(wedge_rad),
        float(omega_start_deg),
        float(omega_step_deg),
        float(min_eta_rad),
        int(n_frames),
        int(n_y),
        int(n_z),
        M,
        D=D,
        HAS_TILTS=has_tilts,
        HAS_WEDGE=has_wedge,
        NBOX=NBOX,
        BLOCK_M=BLOCK_M,
    )

    # Caller handles 0/0 → 0 explicitly.
    total_clamp = total.clamp(min=1).to(torch.float32)
    return hit.to(torch.float32) / total_clamp
