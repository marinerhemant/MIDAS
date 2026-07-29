"""Multi-grain ODF inversion: amortize GPU launch overhead across grains.

The single-grain ``fit_grain_odf`` driver in :mod:`inversion` runs one
forward + one splat + one Adam step per grain per inner iteration. On
GPU, a single forward pass is dominated by kernel-launch overhead --
benchmarks show that K=1 and K=256 orientations take essentially the
same wall time at our problem scales -- so per-grain inversion leaves
the GPU mostly idle.

This module batches G grains' worth of forward + splat into a single
launch. All G grains share the same ``HEDMForwardModel`` and the same
patch dimensions; per-grain quantities (R_avg, position, ODF
parameters, measured patches, spot indexer, Delta table) are held in
flat tensors with G as the leading dimension.

API
---

    fit_many_grains(grains, model, ...) -> List[GrainFitResult]

where ``grains`` is a list of :class:`MultiGrainSpec`. Returns the
result for each grain in the order given.

Constraints (current MVP)
-------------------------
- All grains use the same K (ODF particle count). Trivial to lift but
  simplifies the batch shapes.
- All grains use the same patch dimensions (F, P, sigma_yz, sigma_f).
- All grains share the same upstream HKL list (same crystal). Trivial
  to lift.
- ``measured_patches`` and ``spot_indexer`` may have different lengths
  per grain. The driver pads to the per-batch maximum and masks padded
  spots out of the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch

from midas_grain_odf.forward_helpers import forward_pairs
from midas_grain_odf.inversion import GrainFitResult
from midas_grain_odf.losses import (
    compute_delta_table,
    odf_weighted_predicted_centroid,
)
from midas_grain_odf.odf import GrainODF
from midas_grain_odf.spot_extract import (
    SpotPatchSpec, splat_spots_to_patches,
)


@dataclass
class MultiGrainSpec:
    """One grain's inputs to the multi-grain inversion."""
    odf: GrainODF
    position: torch.Tensor              # (3,)
    measured_y: torch.Tensor            # (S_g,)
    measured_z: torch.Tensor            # (S_g,)
    measured_f: torch.Tensor            # (S_g,)
    measured_patches: torch.Tensor      # (S_g, F, P, P)
    spot_indexer: torch.Tensor          # (S_g,) long; flat into 2*M
    lattice_params: Optional[torch.Tensor] = None


def fit_many_grains(
    grains: Sequence[MultiGrainSpec],
    model,
    *,
    patch_F: int = 7,
    patch_P: int = 31,
    sigma_yz: float = 1.0,
    sigma_f: float = 0.6,
    delta_iters: int = 2,
    delta_tol: float = 0.05,
    inner_steps: int = 200,
    optimizer_name: Optional[str] = None,
    use_pixel_mask: bool = False,
    chunk_size_g: Optional[int] = None,
    lr_axis_angle: float = 1e-4,
    lr_logits: float = 0.1,
    outlier_alpha: float = 5.0,
    verbose: bool = False,
) -> List[GrainFitResult]:
    """Batched single-step Adam over G grains' ODFs.

    The forward pass for all G grains is a single ``forward_pairs`` call
    (G * K orientations); the splatter operates on a (G, S_max, F, P, P)
    tensor with per-grain spot masking; the inner Adam step updates all
    G ODFs simultaneously.
    """
    G = len(grains)
    if G == 0:
        return []

    # Validate uniformity assumptions.
    K = grains[0].odf.K
    for g in grains:
        assert g.odf.K == K, "fit_many_grains currently requires identical K"

    device = grains[0].position.device
    # Use the dtype of the first grain's measured patches (allows fp32 vs fp64
    # without rebuilding the multigrain driver).
    dtype = grains[0].measured_patches.dtype

    M = int(model.hkls.shape[0])  # number of HKLs
    M2 = 2 * M                     # 2 omega solutions per HKL

    # Pad spot lists to a common length S_max per batch.
    s_lens = [int(g.spot_indexer.numel()) for g in grains]
    S_max = max(s_lens)

    spot_idx = torch.zeros(G, S_max, dtype=torch.long, device=device)
    spot_mask = torch.zeros(G, S_max, dtype=dtype, device=device)
    meas_y = torch.zeros(G, S_max, dtype=dtype, device=device)
    meas_z = torch.zeros(G, S_max, dtype=dtype, device=device)
    meas_f = torch.zeros(G, S_max, dtype=dtype, device=device)
    # When chunked, keep the (G, S_max, F, P, P) measured-patch buffer on
    # CPU and stage chunk slices to GPU on demand. This is the dominant
    # contributor to peak GPU memory at large G or large patch.
    _patch_device = (device if (chunk_size_g is None or chunk_size_g >= G)
                      else torch.device("cpu"))
    meas_patches = torch.zeros(
        G, S_max, patch_F, patch_P, patch_P,
        dtype=dtype, device=_patch_device,
    )
    for g_idx, gs in enumerate(grains):
        s_g = s_lens[g_idx]
        spot_idx[g_idx, :s_g] = gs.spot_indexer.to(device)
        spot_mask[g_idx, :s_g] = 1.0
        meas_y[g_idx, :s_g] = gs.measured_y.to(device).to(dtype)
        meas_z[g_idx, :s_g] = gs.measured_z.to(device).to(dtype)
        meas_f[g_idx, :s_g] = gs.measured_f.to(device).to(dtype)
        meas_patches[g_idx, :s_g] = gs.measured_patches.to(_patch_device).to(dtype)

    # Stage 1: per-grain Delta table from single-orientation prediction.
    with torch.no_grad():
        R_avgs = torch.stack([g.odf.R_avg.to(device) for g in grains], dim=0)  # (G, 3, 3)
        positions = torch.stack([g.position.to(device) for g in grains], dim=0)  # (G, 3)
        spots_avg = forward_pairs(model, R_avgs, positions)
        # spots_avg.y_pixel shape: (1, 2*N, M) with N = G; the 2*N axis is
        # laid out as [omega_p_block, omega_n_block] (2 blocks of N), NOT
        # interleaved.  Recover the (G, 2, M) per-grain tensor via:
        py_full = spots_avg.y_pixel.reshape(2, G, M).permute(1, 0, 2).reshape(G, M2)
        pz_full = spots_avg.z_pixel.reshape(2, G, M).permute(1, 0, 2).reshape(G, M2)
        pf_full = spots_avg.frame_nr.reshape(2, G, M).permute(1, 0, 2).reshape(G, M2)
        pv_full = spots_avg.valid.reshape(2, G, M).permute(1, 0, 2).reshape(G, M2)
        # gather single-orient predictions onto the per-grain spot list
        py = torch.gather(py_full, 1, spot_idx)
        pz = torch.gather(pz_full, 1, spot_idx)
        pf = torch.gather(pf_full, 1, spot_idx)
        pv = torch.gather(pv_full, 1, spot_idx)

    delta_y = torch.zeros_like(py)
    delta_z = torch.zeros_like(pz)
    delta_f = torch.zeros_like(pf)
    keep = torch.zeros_like(pv, dtype=torch.bool)
    for g_idx in range(G):
        s_g = s_lens[g_idx]
        dy, dz, df, k_mask = compute_delta_table(
            py[g_idx, :s_g], pz[g_idx, :s_g], pf[g_idx, :s_g],
            meas_y[g_idx, :s_g], meas_z[g_idx, :s_g], meas_f[g_idx, :s_g],
            pv[g_idx, :s_g], outlier_alpha,
        )
        delta_y[g_idx, :s_g] = dy
        delta_z[g_idx, :s_g] = dz
        delta_f[g_idx, :s_g] = df
        keep[g_idx, :s_g] = k_mask

    spot_keep_mask = keep.to(dtype) * spot_mask  # (G, S_max)

    # Build a single SpotPatchSpec for the batched splatter.
    #
    # Anchor for grain g, slot s is py[g, s] (single-orient pred).
    # The splatter sees a flat S = G * S_max spots; we shift the spot
    # coords by Delta inline and add per-pair indices.
    flat_spec = SpotPatchSpec(
        n_spots=G * S_max,
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=py.reshape(-1).detach().clone(),
        anchor_z=pz.reshape(-1).detach().clone(),
        anchor_f=pf.reshape(-1).detach().clone(),
    )

    # Optimizer over all G grains' ODF parameters jointly.
    aa_params, other_params = [], []
    for gs in grains:
        for name, p in gs.odf.named_parameters():
            if "axis_angle" in name or "mode_axis_angle" in name:
                aa_params.append(p)
            else:
                other_params.append(p)
    from midas_grain_odf.inversion import _resolve_optimizer
    optimizer_name = _resolve_optimizer(optimizer_name)
    if optimizer_name == "lbfgs":
        all_params = aa_params + other_params
        optimizer = torch.optim.LBFGS(
            all_params,
            lr=lr_axis_angle,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe",
        )
        # Cap inner_steps for L-BFGS (Adam-tuned 200-800 wastes compute).
        import os as _os
        inner_steps = min(inner_steps, int(_os.environ.get("MIDAS_LBFGS_STEPS", "50")))
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": aa_params, "lr": lr_axis_angle},
                {"params": other_params, "lr": lr_logits},
            ]
        )

    # Pre-flatten patches for the loss.
    meas_patches_flat = meas_patches.reshape(
        G * S_max, patch_F, patch_P, patch_P,
    )

    losses_per_grain = [[] for _ in grains]

    # Cluster-wide per-spot pixel mask (Voronoi partition), recomputed
    # at each delta refresh. None when use_pixel_mask=False so the fast
    # mean-MSE path is preserved.
    cluster_pixel_mask = [None]

    def _refresh_cluster_pixel_mask():
        if not use_pixel_mask:
            cluster_pixel_mask[0] = None
            return
        from midas_grain_odf.spot_extract import compute_per_spot_pixel_mask
        anchor_y_flat = (py + delta_y).reshape(-1)
        anchor_z_flat = (pz + delta_z).reshape(-1)
        anchor_f_flat = (pf + delta_f).reshape(-1)
        flat_spec = SpotPatchSpec(
            n_spots=G * S_max, patch_F=patch_F, patch_P=patch_P,
            sigma_yz=sigma_yz, sigma_f=sigma_f,
            anchor_y=anchor_y_flat,
            anchor_z=anchor_z_flat,
            anchor_f=anchor_f_flat,
        )
        # Every flat-slot anchor is a competitor for every other slot.
        # Self-id = its own flat index, so self-exclusion works.
        slot_ids = torch.arange(G * S_max, dtype=torch.long, device=device)
        with torch.no_grad():
            mask = compute_per_spot_pixel_mask(
                flat_spec,
                other_anchor_y=anchor_y_flat,
                other_anchor_z=anchor_z_flat,
                other_anchor_f=anchor_f_flat,
                other_spot_id=slot_ids,
            )
        # Store as bool to cut mask memory by ~8x (a (G*S_max, F, P, P)
        # fp64 mask at G=20, P=327 is 46 GB; bool is 5.8 GB).
        # When chunked, also stage the mask buffer on CPU so the per-chunk
        # transfer mirrors the meas-patches handling.
        mask_bool = (mask > 0.5)
        if chunk_size_g is not None and chunk_size_g < G:
            mask_bool = mask_bool.to("cpu")
        del mask
        cluster_pixel_mask[0] = mask_bool

    def _forward_loss():
        # Sample K orientations from each grain's ODF and concatenate.
        all_R = []
        all_w = []
        for gs in grains:
            R, w = gs.odf.sample()
            all_R.append(R)
            all_w.append(w)
        R_cat = torch.stack(all_R, dim=0)        # (G, K, 3, 3)
        w_cat = torch.stack(all_w, dim=0)        # (G, K)

        # Forward all (grain, orientation) pairs in one call. We treat
        # each (g, k) as a (position, orientation) pair: G*K positions,
        # one orientation per position. Each position is the grain's
        # centroid, repeated K times.
        R_pairs = R_cat.reshape(G * K, 3, 3)
        pos_repeat = positions.unsqueeze(1).expand(G, K, 3).reshape(G * K, 3)
        spots = forward_pairs(model, R_pairs, pos_repeat)
        # Output: y_pixel shape (1, 2*N, M) with N = G*K. Layout is
        # [omega_p_block_of_N, omega_n_block_of_N] (2 blocks of N), NOT
        # interleaved.  Decompose: 2*N = 2 (omega) * G * K (positions
        # in flat (g, k) order). Permute to (G, K, 2, M) and flatten the
        # last two so the M2 axis matches spot_indexer = om_branch*M + hkl.
        sy = spots.y_pixel.reshape(2, G, K, M).permute(1, 2, 0, 3).reshape(G, K, M2)
        sz = spots.z_pixel.reshape(2, G, K, M).permute(1, 2, 0, 3).reshape(G, K, M2)
        sf = spots.frame_nr.reshape(2, G, K, M).permute(1, 2, 0, 3).reshape(G, K, M2)
        sv = spots.valid.reshape(2, G, K, M).permute(1, 2, 0, 3).reshape(G, K, M2)

        # Gather only the spots in spot_idx for each grain.
        # spot_idx has shape (G, S_max); expand to (G, K, S_max).
        spot_idx_g = spot_idx.unsqueeze(1).expand(G, K, S_max)
        sy_g = torch.gather(sy, 2, spot_idx_g)   # (G, K, S_max)
        sz_g = torch.gather(sz, 2, spot_idx_g)
        sf_g = torch.gather(sf, 2, spot_idx_g)
        sv_g = torch.gather(sv, 2, spot_idx_g)

        # Apply Delta. delta_* shape (G, S_max). Broadcast K.
        shifted_y = sy_g + delta_y.unsqueeze(1)
        shifted_z = sz_g + delta_z.unsqueeze(1)
        shifted_f = sf_g + delta_f.unsqueeze(1)

        # Flatten the per-grain (S_max,) into a single (G*S_max,) spot
        # list for the splatter. Flatten over (G, K) -> K and
        # (G, S_max) -> G*S_max so that w_cat[g] applies to all S_max
        # slots of grain g.
        # The splatter expects positions of shape (K, S_total) and weights
        # of shape (K,). Multi-grain: use w_cat[g] for grain g's S slots.
        # Approach: per-grain-flat splatter run, then concatenate.
        # To keep one launch, we flatten (G, S_max) -> S_total = G * S_max
        # as patches; the K-axis stays as ODF. The complication is the
        # weights w_cat differ across grains, so we tile them as a
        # (K, G * S_max) effective weight matrix where w_eff[k, slot] =
        # w_cat[grain(slot), k].
        #
        # The current splatter takes weights of shape (K,) and applies
        # them uniformly across all spots. To support per-spot K-weights
        # we set valid -> valid * w_per_grain on a per-spot basis and
        # leave weights == 1.
        weights_one = torch.ones(K, dtype=dtype, device=device)
        # w_per_pair[k, g*S_max + s] = w_cat[g, k] * sv_g[g, k, s] * spot_keep_mask[g, s]
        # But we also want spot_mask (drop padded entries).
        wt = w_cat.unsqueeze(2).expand(G, K, S_max)              # (G, K, S_max)
        keep_b = spot_keep_mask.unsqueeze(1).expand(G, K, S_max) # (G, K, S_max)
        valid_eff = (sv_g * wt * keep_b).reshape(K, G * S_max)
        # shifted_* reshape to (K, G*S_max)
        shifted_y_flat = shifted_y.permute(1, 0, 2).reshape(K, G * S_max)
        shifted_z_flat = shifted_z.permute(1, 0, 2).reshape(K, G * S_max)
        shifted_f_flat = shifted_f.permute(1, 0, 2).reshape(K, G * S_max)

        # The flat spec has Delta-shifted anchors:
        anchor_y_flat = (py + delta_y).reshape(-1)
        anchor_z_flat = (pz + delta_z).reshape(-1)
        anchor_f_flat = (pf + delta_f).reshape(-1)
        flat_spec_step = SpotPatchSpec(
            n_spots=G * S_max,
            patch_F=patch_F, patch_P=patch_P,
            sigma_yz=sigma_yz, sigma_f=sigma_f,
            anchor_y=anchor_y_flat, anchor_z=anchor_z_flat,
            anchor_f=anchor_f_flat,
        )

        # Memory-frugal sparse-tile path: when chunk_size_g < G, splat one
        # chunk of grains at a time and immediately reduce to per-spot loss,
        # using torch.utils.checkpoint to drop the (chunk_size_g*S_max, F, P, P)
        # intermediate before the next chunk. Peak memory drops from
        # O(G * S_max * F * P^2) to O(chunk_size_g * S_max * F * P^2).
        eff_chunk = chunk_size_g if (chunk_size_g is not None
                                      and chunk_size_g < G) else None
        if eff_chunk is None:
            pred_patches_flat = splat_spots_to_patches(
                flat_spec_step,
                shifted_y_flat, shifted_z_flat, shifted_f_flat,
                weights_one, valid_eff,
            )  # (G * S_max, F, P, P)
            diff = (pred_patches_flat - meas_patches_flat) ** 2
            if cluster_pixel_mask[0] is not None:
                mask_flat = cluster_pixel_mask[0]
                diff = diff * mask_flat   # bool broadcasts to dtype
                per_spot_norm = mask_flat.flatten(1).to(dtype).sum(
                    dim=1).clamp(min=1.0)
                per_spot = diff.flatten(1).sum(dim=1) / per_spot_norm
            else:
                per_spot = diff.flatten(1).mean(dim=1)        # (G*S_max,)
        else:
            from torch.utils.checkpoint import checkpoint as _ckpt

            # The chunked closure transfers mp from CPU to GPU INSIDE the
            # checkpoint, so the saved inputs are the CPU slices (cheap)
            # rather than 1/chunk_size_g duplicate GPU buffers (expensive).
            def _chunk_loss(sy_c, sz_c, sf_c, ve_c,
                             ay_c, az_c, af_c, mp_in, mk_in):
                if mp_in.device != device:
                    mp_c = mp_in.to(device, non_blocking=True)
                else:
                    mp_c = mp_in
                if mk_in is not None and mk_in.device != device:
                    mk_c = mk_in.to(device, non_blocking=True)
                else:
                    mk_c = mk_in
                spec_c = SpotPatchSpec(
                    n_spots=int(ay_c.shape[0]),
                    patch_F=patch_F, patch_P=patch_P,
                    sigma_yz=sigma_yz, sigma_f=sigma_f,
                    anchor_y=ay_c, anchor_z=az_c, anchor_f=af_c,
                )
                pred_c = splat_spots_to_patches(
                    spec_c, sy_c, sz_c, sf_c, weights_one, ve_c,
                )
                diff_c = (pred_c - mp_c) ** 2
                if mk_c is not None:
                    diff_c = diff_c * mk_c
                    norm_c = mk_c.flatten(1).to(diff_c.dtype).sum(
                        dim=1).clamp(min=1.0)
                    return diff_c.flatten(1).sum(dim=1) / norm_c
                return diff_c.flatten(1).mean(dim=1)

            per_spot_chunks = []
            for g0 in range(0, G, eff_chunk):
                g1 = min(g0 + eff_chunk, G)
                s0, s1 = g0 * S_max, g1 * S_max
                mk_c = (cluster_pixel_mask[0][s0:s1]
                        if cluster_pixel_mask[0] is not None else None)
                # Pass the CPU (or device) slice through; transfer to GPU
                # happens inside _chunk_loss so it isn't held across chunks.
                mp_in = meas_patches_flat[s0:s1]
                per_spot_chunks.append(_ckpt(
                    _chunk_loss,
                    shifted_y_flat[:, s0:s1].contiguous(),
                    shifted_z_flat[:, s0:s1].contiguous(),
                    shifted_f_flat[:, s0:s1].contiguous(),
                    valid_eff[:, s0:s1].contiguous(),
                    anchor_y_flat[s0:s1].contiguous(),
                    anchor_z_flat[s0:s1].contiguous(),
                    anchor_f_flat[s0:s1].contiguous(),
                    mp_in,
                    mk_c,
                    use_reentrant=False,
                ))
            per_spot = torch.cat(per_spot_chunks, dim=0)
        per_spot = per_spot.reshape(G, S_max)
        per_grain_n = spot_keep_mask.sum(dim=1).clamp(min=1.0)
        per_grain_loss = (per_spot * spot_keep_mask).sum(dim=1) / per_grain_n
        return per_grain_loss

    converged = [False] * G

    for outer in range(delta_iters):
        if verbose:
            kept = int(spot_keep_mask.sum())
            total = int(spot_mask.sum())
            print(f"  multigrain iter {outer}: keep {kept} / {total}")
        _refresh_cluster_pixel_mask()
        if optimizer_name == "lbfgs":
            last_per_grain = [None]

            def closure():
                optimizer.zero_grad()
                per_grain_loss = _forward_loss()
                loss = per_grain_loss.mean()
                loss.backward()
                last_per_grain[0] = per_grain_loss.detach()
                return loss

            for step in range(inner_steps):
                optimizer.step(closure)
                pgl = last_per_grain[0]
                if verbose and (step % max(1, inner_steps // 10) == 0
                                or step == inner_steps - 1):
                    print(f"    step {step:4d}  mean loss = {float(pgl.mean()):.4e}")
                for g_idx in range(G):
                    losses_per_grain[g_idx].append(float(pgl[g_idx].item()))
        else:
            for step in range(inner_steps):
                optimizer.zero_grad()
                per_grain_loss = _forward_loss()
                loss = per_grain_loss.mean()
                loss.backward()
                optimizer.step()
                if verbose and (step % max(1, inner_steps // 10) == 0
                                or step == inner_steps - 1):
                    print(f"    step {step:4d}  mean loss = {float(loss):.4e}")
                for g_idx in range(G):
                    losses_per_grain[g_idx].append(float(per_grain_loss[g_idx].item()))

        # Stage 3: refresh Delta from ODF-weighted predicted centroid.
        with torch.no_grad():
            all_R = torch.stack([gs.odf.sample()[0] for gs in grains], dim=0)
            all_w = torch.stack([gs.odf.sample()[1] for gs in grains], dim=0)
            R_pairs = all_R.reshape(G * K, 3, 3)
            pos_repeat = positions.unsqueeze(1).expand(G, K, 3).reshape(G * K, 3)
            spots = forward_pairs(model, R_pairs, pos_repeat)
            # Same layout fix as the inner-loop forward.
            sy = spots.y_pixel.reshape(2, G, K, M).permute(1, 2, 0, 3).reshape(G, K, M2)
            sz = spots.z_pixel.reshape(2, G, K, M).permute(1, 2, 0, 3).reshape(G, K, M2)
            sf = spots.frame_nr.reshape(2, G, K, M).permute(1, 2, 0, 3).reshape(G, K, M2)
            sv = spots.valid.reshape(2, G, K, M).permute(1, 2, 0, 3).reshape(G, K, M2)
            spot_idx_g = spot_idx.unsqueeze(1).expand(G, K, S_max)
            sy_g = torch.gather(sy, 2, spot_idx_g)
            sz_g = torch.gather(sz, 2, spot_idx_g)
            sf_g = torch.gather(sf, 2, spot_idx_g)
            sv_g = torch.gather(sv, 2, spot_idx_g)
            for g_idx in range(G):
                s_g = s_lens[g_idx]
                cy, cz, cf = odf_weighted_predicted_centroid(
                    sy_g[g_idx, :, :s_g], sz_g[g_idx, :, :s_g],
                    sf_g[g_idx, :, :s_g], all_w[g_idx], sv_g[g_idx, :, :s_g],
                )
                dy, dz, df, k_mask = compute_delta_table(
                    cy, cz, cf,
                    meas_y[g_idx, :s_g], meas_z[g_idx, :s_g],
                    meas_f[g_idx, :s_g],
                    (sv_g[g_idx, :, :s_g].sum(dim=0) > 0).to(dtype),
                    outlier_alpha,
                )
                delta_y[g_idx, :s_g] = dy
                delta_z[g_idx, :s_g] = dz
                delta_f[g_idx, :s_g] = df
                keep[g_idx, :s_g] = k_mask
            spot_keep_mask = keep.to(dtype) * spot_mask

    # Build per-grain results
    results: List[GrainFitResult] = []
    for g_idx, gs in enumerate(grains):
        s_g = s_lens[g_idx]
        results.append(GrainFitResult(
            odf=gs.odf,
            delta_y=delta_y[g_idx, :s_g].detach().clone(),
            delta_z=delta_z[g_idx, :s_g].detach().clone(),
            delta_f=delta_f[g_idx, :s_g].detach().clone(),
            keep=keep[g_idx, :s_g].clone(),
            losses=losses_per_grain[g_idx],
            delta_iters_run=delta_iters,
            converged=converged[g_idx],
        ))
    return results
