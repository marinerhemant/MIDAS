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
    I_pred = torch.zeros(n_spots, dtype=dt, device=dev)

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
            weights = beam_profile.fraction_over_voxel(
                scan_p.unsqueeze(1), proj, voxel_size,
            )                                           # (ng_s, ng_v)
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
        I_pred = I_pred.index_copy(0, spot_idx, I_pred_g)

    return I_pred
