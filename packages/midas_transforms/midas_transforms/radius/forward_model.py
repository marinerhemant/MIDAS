"""Forward model for the Sharma-Offerman V-map.

Predicts observed per-spot intensity from per-voxel volume ``V[v]``,
per-ring scale ``K[r]``, per-ring theoretical intensity ``I_th[r]``, and
a beam profile.  Absorption (P3) is wired behind a ``use_absorption``
flag — see :mod:`midas_transforms.geometry.absorption`.

Formula (per spot ``s`` belonging to grain ``g`` at omega ``ω`` and scan
position ``p``):

.. math::

    I_{\\rm pred}(s) = K[r(s)] \\, I_{\\rm th}[r(s)] \\, \\sum_{v \\in g}
                       V[v] \\, w(p,\\, {\\rm proj}_\\omega(v)) \\, A_{\\rm absorp}(s, v)

The projected voxel position depends on ``scan_axis``:

* ``"pf"`` — pf-HEDM xy scan: ``proj = v_x · sin(ω) + v_y · cos(ω)``
  (the MIDAS convention used by
  ``midas_index.compute.matching.compare_spots``).
* ``"z"``  — FF-HEDM height (Z) scan, for samples taller than the beam:
  ``proj = v_z`` (no ω dependence).
* ``"none"`` — no beam profile attenuation; every voxel in the grain
  contributes fully.  Used for compact-grain FF, where the whole grain
  sits within the beam at every ω.

The summation is over voxels in the same grain that fall inside the
sample mask.

Vectorization
-------------
We loop over **grains** (a small Python loop, typically 5–100 grains for
pf-HEDM, 100–10 000 for FF), and inside each grain the per-(spot, voxel)
weight matrix is computed fully vectorized.  This avoids padded jagged
tensors while still keeping the differentiable inner kernel a single
broadcasted call.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from ..geometry.beam import BeamProfile
    from ..geometry.sample import SampleGrid


__all__ = ["predicted_spot_intensities"]


def _trapezoid_cdf(x, A, B, H):
    """CDF (integral from -inf) of the symmetric trapezoid c(u):

        c(u) = 0                       |u| >= A
             = H*(A-|u|)/(A-B)         B <= |u| < A   (ramp)
             = H                       |u| < B        (plateau)

    Analytic, differentiable, vectorized. ``A >= B >= 0`` (A==B => box).
    Returns ``∫_{-A}^{x} c(u) du`` clamped to [0, total].
    """
    import torch

    # Guard the ramp denominator when A==B (box: ω∈{0,90°}); the ramp branch
    # is masked out there so the value substituted is irrelevant but must be
    # finite for autograd.
    AmB = (A - B).clamp_min(1e-12)
    left_ramp_area = 0.5 * H * (A - B)            # area of one ramp

    xc = x.clamp(min=-A, max=A)
    # Region masks on the clamped coordinate.
    in_left = xc < -B                              # -A..-B  (left ramp)
    in_plat = (xc >= -B) & (xc <= B)               # -B..B   (plateau)
    # right ramp: xc > B

    left_val = 0.5 * H * (xc + A) ** 2 / AmB
    plat_val = left_ramp_area + H * (xc + B)
    right_val = (
        left_ramp_area + 2.0 * H * B
        + H * (A * (xc - B) - 0.5 * (xc * xc - B * B)) / AmB
    )
    out = torch.where(in_left, left_val,
                      torch.where(in_plat, plat_val, right_val))
    # Below -A => 0; above A => total (handled by the clamp on xc giving the
    # right-ramp value at A == total).
    return out


def _tophat_trapezoid_overlap(scan_pos, proj, voxel_size, omega, width):
    """Exact overlap of a TopHat beam window with the trapezoidal projection
    of a square voxel rotated by ``omega`` (Siddon-style, finite beam).

    Reduces to ``TopHat.fraction_over_voxel`` at ω∈{0,90°}. Differentiable in
    all tensor args. Shapes broadcast: ``scan_pos``/``omega`` (ng_s,1),
    ``proj`` (ng_s,ng_v), ``voxel_size``/``width`` 0-d.
    """
    import torch

    cw = (voxel_size * torch.cos(omega).abs())     # (ng_s,1)
    sw = (voxel_size * torch.sin(omega).abs())
    A = 0.5 * (cw + sw)                            # outer half-width
    B = 0.5 * (cw - sw).abs()                      # plateau half-width
    mx = torch.maximum(torch.cos(omega).abs(), torch.sin(omega).abs())
    H = voxel_size / mx.clamp_min(1e-12)           # plateau height (∫c = Δ²)

    delta = scan_pos - proj                        # (ng_s,ng_v) beam-centre offset
    hi = delta + 0.5 * width
    lo = delta - 0.5 * width
    area = _trapezoid_cdf(hi, A, B, H) - _trapezoid_cdf(lo, A, B, H)
    return area / (voxel_size * voxel_size)


def predicted_spot_intensities(
    V_voxel: "torch.Tensor",                       # (Nv,) — per-voxel V (µm³ in arbitrary K-units)
    K_ring: "torch.Tensor",                         # (R,)  — per-ring scale
    theoretical_intensity_per_ring: "torch.Tensor", # (R,)  — from theoretical.py
    spot_ring_idx: "torch.Tensor",                  # (Ns,) int — ring index per spot
    spot_grain_idx: "torch.Tensor",                 # (Ns,) int — grain index per spot
    spot_scan_pos_um: "torch.Tensor",               # (Ns,) float — scan position (µm)
    spot_omega_rad: "torch.Tensor",                 # (Ns,) float — omega (radians)
    sample_grid: "SampleGrid",
    beam_profile: "BeamProfile",
    *,
    scan_axis: str = "pf",                                      # "pf" | "z" | "none"
    beam_geometry: str = "center",                              # "center" | "siddon"
    spot_weight: Optional["torch.Tensor"] = None,               # (Ns,) soft-attr weight
    spot_out_index: Optional["torch.Tensor"] = None,            # (Ns,) int -> output spot
    n_out_spots: Optional[int] = None,
    use_absorption: bool = False,
    incident_dirs_per_spot: Optional["torch.Tensor"] = None,    # (Ns, 3) unit vec, lab frame
    diffracted_dirs_per_spot: Optional["torch.Tensor"] = None,  # (Ns, 3) unit vec, lab frame
    mu_per_cm: Optional["torch.Tensor"] = None,                 # 0-d or (Ns,) — μ in cm⁻¹
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
) -> "torch.Tensor":
    """Predict I_observed for every spot.

    Parameters
    ----------
    scan_axis : ``"pf"`` | ``"z"`` | ``"none"``
        Geometry of the beam-voxel intersection:

        * ``"pf"``  — pf-HEDM xy scan; ``proj = v_x sin ω + v_y cos ω``.
        * ``"z"``   — FF height scan; ``proj = v_z``.
        * ``"none"`` — no beam attenuation (compact-grain FF); every voxel
          in the grain contributes its full ``V[v]``.  ``spot_scan_pos_um``
          and ``beam_profile`` are then ignored.

    Other parameters and the absorption path are unchanged from the PF
    description.  When ``use_absorption=True``, the ``sample_grid`` must
    have been built via :meth:`SampleGrid.from_regular_grid`.

    Returns
    -------
    I_pred : (Ns,) torch tensor
        Differentiable in ``V_voxel``, ``K_ring``,
        ``theoretical_intensity_per_ring``, ``spot_scan_pos_um``,
        ``spot_omega_rad``, ``sample_grid.voxel_positions``,
        ``sample_grid.voxel_size_um``, ``beam_profile`` parameters, and
        (when absorption is on) ``mu_per_cm`` + directions.
        Spots whose grain index is < 0, or whose grain contains no
        in-sample voxels, get ``I_pred = 0``.
    """
    import torch

    if scan_axis not in {"pf", "z", "none"}:
        raise ValueError(
            f"scan_axis must be one of {{'pf', 'z', 'none'}}; got {scan_axis!r}"
        )

    if use_absorption:
        if (incident_dirs_per_spot is None
                or diffracted_dirs_per_spot is None
                or mu_per_cm is None):
            raise ValueError(
                "use_absorption=True requires incident_dirs_per_spot, "
                "diffracted_dirs_per_spot, and mu_per_cm"
            )
        from ..geometry.absorption import absorption_factor

    dt = dtype or V_voxel.dtype
    dev = device or V_voxel.device

    n_spots = spot_ring_idx.shape[0]
    # Soft attribution: the per-spot arrays may be an *expanded* list with one
    # row per (observed-spot, grain) membership. ``spot_out_index`` maps each
    # row back to its observed spot and ``spot_weight`` is the attribution
    # weight; the per-row contributions are scatter-summed into the observed
    # spots. When both are None this is the hard 1-spot→1-grain path and the
    # output is identical to the previous index_copy behaviour.
    n_out = int(n_out_spots) if n_out_spots is not None else n_spots
    I_pred = torch.zeros(n_out, dtype=dt, device=dev)

    grain_map = sample_grid.grain_map        # (Nv,) int
    sample_mask = sample_grid.sample_mask    # (Nv,) bool
    voxel_pos = sample_grid.voxel_positions  # (Nv, 3)
    voxel_size = sample_grid.voxel_size_um   # 0-d

    valid_spot_mask = spot_grain_idx >= 0
    if not bool(valid_spot_mask.any().item()):
        return I_pred
    grain_ids = torch.unique(spot_grain_idx[valid_spot_mask])

    for g_t in grain_ids:
        g = int(g_t.item())
        spot_sel = (spot_grain_idx == g)
        vox_sel = (grain_map == g) & sample_mask
        if not bool(vox_sel.any().item()):
            continue
        spot_idx = spot_sel.nonzero(as_tuple=False).squeeze(-1)    # (ng_s,)
        vox_idx = vox_sel.nonzero(as_tuple=False).squeeze(-1)      # (ng_v,)
        ng_s = int(spot_idx.shape[0])
        ng_v = int(vox_idx.shape[0])

        omega = spot_omega_rad[spot_idx]               # (ng_s,)
        scan_p = spot_scan_pos_um[spot_idx]            # (ng_s,)
        ring = spot_ring_idx[spot_idx]                 # (ng_s,)

        v_pos = voxel_pos[vox_idx]                      # (ng_v, 3)

        if scan_axis == "pf":
            sin_w = torch.sin(omega).unsqueeze(1)       # (ng_s, 1)
            cos_w = torch.cos(omega).unsqueeze(1)       # (ng_s, 1)
            v_x = v_pos[:, 0].unsqueeze(0)              # (1, ng_v)
            v_y = v_pos[:, 1].unsqueeze(0)              # (1, ng_v)
            proj = v_x * sin_w + v_y * cos_w            # (ng_s, ng_v)
            if beam_geometry == "siddon" and hasattr(beam_profile, "width_um"):
                # Exact overlap with the rotated-square (trapezoidal) voxel
                # footprint instead of the fixed-width box. ω-aware.
                weights = _tophat_trapezoid_overlap(
                    scan_p.unsqueeze(1), proj, voxel_size,
                    omega.unsqueeze(1), beam_profile.width_um,
                )                                       # (ng_s, ng_v)
            else:
                weights = beam_profile.fraction_over_voxel(
                    scan_p.unsqueeze(1), proj, voxel_size,
                )                                       # (ng_s, ng_v)
        elif scan_axis == "z":
            v_z = v_pos[:, 2].unsqueeze(0).expand(ng_s, -1)   # (ng_s, ng_v)
            weights = beam_profile.fraction_over_voxel(
                scan_p.unsqueeze(1), v_z, voxel_size,
            )                                                  # (ng_s, ng_v)
        else:  # scan_axis == "none" — compact FF, full illumination
            weights = torch.ones((ng_s, ng_v), dtype=dt, device=dev)

        if use_absorption:
            inc_g = incident_dirs_per_spot[spot_idx]       # (ng_s, 3)
            dif_g = diffracted_dirs_per_spot[spot_idx]     # (ng_s, 3)
            mu_g = (
                mu_per_cm[spot_idx]
                if mu_per_cm.ndim > 0 else mu_per_cm
            )
            inc_bv = inc_g.unsqueeze(1).expand(-1, ng_v, -1).reshape(-1, 3)
            dif_bv = dif_g.unsqueeze(1).expand(-1, ng_v, -1).reshape(-1, 3)
            vox_bv = vox_idx.unsqueeze(0).expand(ng_s, -1).reshape(-1)
            if mu_g.ndim > 0:
                mu_bv = mu_g.unsqueeze(1).expand(-1, ng_v).reshape(-1)
            else:
                mu_bv = mu_g
            absorp_flat = absorption_factor(
                sample_grid, vox_bv, inc_bv, dif_bv, mu_bv,
            )
            absorp = absorp_flat.reshape(ng_s, ng_v)
            weights = weights * absorp

        V_g = V_voxel[vox_idx].unsqueeze(0)             # (1, ng_v)
        contrib = (V_g * weights).sum(dim=1)            # (ng_s,)

        I_pred_g = (
            K_ring[ring]
            * theoretical_intensity_per_ring[ring]
            * contrib
        )                                                # (ng_s,)
        if spot_weight is not None:
            I_pred_g = I_pred_g * spot_weight[spot_idx]
        out_idx = spot_out_index[spot_idx] if spot_out_index is not None else spot_idx
        # index_add (not copy): a single observed spot may receive
        # contributions from several grains (soft attribution).
        I_pred = I_pred.index_add(0, out_idx, I_pred_g)

    return I_pred
