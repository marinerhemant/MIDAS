"""End-to-end integration test: synthesize data -> CSV/JSON/NPZ -> CLI -> H5.

Generates a 2-grain synthetic FF-HEDM dataset, writes the geometry / grains
/ spots / frames to disk in the formats consumed by the package, runs the
CLI's ``fit`` command, and verifies the output HDF5.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from conftest import random_orientation  # noqa: E402

from midas_grain_odf.cli import main as cli_main  # noqa: E402
from midas_grain_odf.forward_helpers import forward_orientations  # noqa: E402
from midas_grain_odf.io import (  # noqa: E402
    GrainOdfConfig, GrainRow, enumerate_hkls,
    write_grains_csv, write_spots_csv, SpotObservation,
)
from midas_grain_odf.odf import axis_angle_to_matrix  # noqa: E402
from midas_grain_odf.spot_extract import (  # noqa: E402
    SpotPatchSpec, splat_spots_to_patches,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _make_config():
    # Shortened Lsd so FCC rings land on a 400-px detector. Reduces frames
    # file size; for production use the standard FF geometry (Lsd~884 mm,
    # detector ~3000+ px).
    return GrainOdfConfig(
        Lsd=150_000.0,                # 150 mm
        y_BC=200.0, z_BC=200.0,
        px=74.8,
        omega_start=-180.0, omega_step=2.0,
        n_frames=180,
        n_pixels_y=400, n_pixels_z=400,
        min_eta=6.0,
        wavelength=0.1729,
        lattice=(3.61, 3.61, 3.61, 90.0, 90.0, 90.0),
        crystal_system="cubic_fcc",
        d_min_A=1.0,
        h_max=2,
    )


def _build_model(cfg: GrainOdfConfig):
    from midas_diffract.forward import HEDMForwardModel, HEDMGeometry
    geom = HEDMGeometry(
        Lsd=cfg.Lsd, y_BC=cfg.y_BC, z_BC=cfg.z_BC, px=cfg.px,
        omega_start=cfg.omega_start, omega_step=cfg.omega_step,
        n_frames=cfg.n_frames,
        n_pixels_y=cfg.n_pixels_y, n_pixels_z=cfg.n_pixels_z,
        min_eta=cfg.min_eta, wavelength=cfg.wavelength,
        flip_y=True,
    )
    G, th, intt = enumerate_hkls(cfg)
    model = HEDMForwardModel(
        hkls=G, thetas=th, geometry=geom, hkls_int=intt,
    )
    return model.to(torch.float64), G, th, intt


def _synthesize_grain_data(model, intt, R_avg, position, *,
                           aa_planted, w_planted,
                           sigma_yz, sigma_f, patch_F, patch_P):
    """Run forward sim for one grain; return measured spot rows + frames patch."""
    delta_R = axis_angle_to_matrix(aa_planted)
    R_planted = R_avg.unsqueeze(0) @ delta_R

    spots_p = forward_orientations(model, R_planted, position)
    sy_p = spots_p.y_pixel.reshape(R_planted.shape[0], -1)
    sz_p = spots_p.z_pixel.reshape(R_planted.shape[0], -1)
    sf_p = spots_p.frame_nr.reshape(R_planted.shape[0], -1)
    sv_p = spots_p.valid.reshape(R_planted.shape[0], -1)

    spots_avg = forward_orientations(model, R_avg.unsqueeze(0), position)
    sy_a = spots_avg.y_pixel.reshape(1, -1).squeeze(0)
    sz_a = spots_avg.z_pixel.reshape(1, -1).squeeze(0)
    sf_a = spots_avg.frame_nr.reshape(1, -1).squeeze(0)
    sv_a = spots_avg.valid.reshape(1, -1).squeeze(0)

    valid_global = (sv_a > 0.5) & (sv_p.sum(dim=0) > 0)
    spot_indexer = torch.nonzero(valid_global, as_tuple=False).squeeze(-1)

    sy_sel = sy_p[:, spot_indexer]
    sz_sel = sz_p[:, spot_indexer]
    sf_sel = sf_p[:, spot_indexer]
    sv_sel = sv_p[:, spot_indexer]

    # Measured centroids (ODF-weighted means).
    w_norm = (w_planted.reshape(-1, 1) * sv_sel).sum(dim=0).clamp(min=1e-12)
    meas_y = (w_planted.reshape(-1, 1) * sv_sel * sy_sel).sum(dim=0) / w_norm
    meas_z = (w_planted.reshape(-1, 1) * sv_sel * sz_sel).sum(dim=0) / w_norm
    meas_f = (w_planted.reshape(-1, 1) * sv_sel * sf_sel).sum(dim=0) / w_norm

    # Splat the planted ODF into per-spot patches anchored at the *measured*
    # centroid (so the synthetic frames will contain real intensity at the
    # measured spot locations).
    spec = SpotPatchSpec(
        n_spots=int(spot_indexer.numel()),
        patch_F=patch_F, patch_P=patch_P,
        sigma_yz=sigma_yz, sigma_f=sigma_f,
        anchor_y=meas_y.detach().clone(),
        anchor_z=meas_z.detach().clone(),
        anchor_f=meas_f.detach().clone(),
    )
    patches = splat_spots_to_patches(
        spec, sy_sel, sz_sel, sf_sel, w_planted, sv_sel,
    )

    # Build the spot list. We need (h, k, l, omega_branch, y, z, f) per
    # measured spot. spot_indexer values are flat indices in (om_branch * M
    # + hkl) -- decode them.
    M = int(intt.shape[0])
    spot_rows = []
    for s_local, flat in enumerate(spot_indexer.tolist()):
        om_branch = flat // M
        hkl_idx = flat % M
        h, k, l = (int(intt[hkl_idx, i].item()) for i in range(3))
        spot_rows.append(SpotObservation(
            grain_id=-1,    # patched up by caller
            hkl_h=h, hkl_k=k, hkl_l=l,
            omega_branch=om_branch,
            y_pixel=float(meas_y[s_local].item()),
            z_pixel=float(meas_z[s_local].item()),
            frame=float(meas_f[s_local].item()),
        ))
    return spot_rows, patches, meas_y, meas_z, meas_f, spot_indexer


def _stamp_patches_into_frames(
    frames: np.ndarray,
    patches: torch.Tensor,
    anchor_y: torch.Tensor,
    anchor_z: torch.Tensor,
    anchor_f: torch.Tensor,
    patch_F: int,
    patch_P: int,
):
    """Add per-spot patches into the (n_frames, ny, nz) array (in-place)."""
    n_frames, n_y, n_z = frames.shape
    half_F = patch_F // 2
    half_P = patch_P // 2
    cy = anchor_y.detach().round().long().cpu().numpy()
    cz = anchor_z.detach().round().long().cpu().numpy()
    cf = anchor_f.detach().round().long().cpu().numpy()
    P = patches.detach().cpu().numpy()
    for s in range(P.shape[0]):
        f0 = int(cf[s]) - half_F
        y0 = int(cy[s]) - half_P
        z0 = int(cz[s]) - half_P
        for df in range(patch_F):
            f = f0 + df
            if not (0 <= f < n_frames):
                continue
            for dy in range(patch_P):
                y = y0 + dy
                if not (0 <= y < n_y):
                    continue
                for dz in range(patch_P):
                    z = z0 + dz
                    if not (0 <= z < n_z):
                        continue
                    frames[f, y, z] += P[s, df, dy, dz]


# ---------------------------------------------------------------------------
#  Test
# ---------------------------------------------------------------------------


def test_end_to_end_cli_2grains():
    deg = math.pi / 180.0
    cfg = _make_config()
    model, G, th, intt = _build_model(cfg)
    M = int(intt.shape[0])

    # Place grain centers in the box-beam volume; small offsets keep them
    # from coincidentally colliding spots.
    grain_specs = [
        {
            "grain_id": 0,
            "R_avg": random_orientation(seed=11).to(torch.float64),
            "position": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            "aa": torch.tensor([
                [0.00 * deg, 0.00 * deg, 0.00 * deg],
                [0.05 * deg, 0.00 * deg, 0.00 * deg],
                [0.00 * deg, 0.04 * deg, 0.03 * deg],
            ], dtype=torch.float64),
            "w": torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64),
        },
        {
            "grain_id": 1,
            "R_avg": random_orientation(seed=23).to(torch.float64),
            "position": torch.tensor([20.0, -15.0, 0.0], dtype=torch.float64),
            "aa": torch.tensor([
                [0.00 * deg, 0.00 * deg, 0.00 * deg],
                [-0.04 * deg, 0.03 * deg, 0.00 * deg],
            ], dtype=torch.float64),
            "w": torch.tensor([0.7, 0.3], dtype=torch.float64),
        },
    ]

    sigma_yz, sigma_f = 1.0, 0.6
    patch_F, patch_P = 7, 31

    frames = np.zeros(
        (cfg.n_frames, cfg.n_pixels_y, cfg.n_pixels_z), dtype=np.float64,
    )
    grain_rows = []
    spot_rows_all = []

    for g in grain_specs:
        spot_rows, patches, my, mz, mf, _ = _synthesize_grain_data(
            model, intt, g["R_avg"], g["position"],
            aa_planted=g["aa"], w_planted=g["w"],
            sigma_yz=sigma_yz, sigma_f=sigma_f,
            patch_F=patch_F, patch_P=patch_P,
        )
        for r in spot_rows:
            r.grain_id = g["grain_id"]
        spot_rows_all.extend(spot_rows)
        _stamp_patches_into_frames(
            frames, patches, my, mz, mf, patch_F, patch_P,
        )
        grain_rows.append(GrainRow(
            grain_id=g["grain_id"],
            R_avg=g["R_avg"],
            position=g["position"],
            lattice=None,
        ))

    # Write artifacts to a temp dir.
    workdir = Path(tempfile.mkdtemp(prefix="midas_grain_odf_test_"))
    try:
        cfg_path = workdir / "geometry.json"
        cfg.to_json(cfg_path)
        grains_path = workdir / "grains.csv"
        write_grains_csv(grain_rows, grains_path)
        spots_path = workdir / "spots.csv"
        write_spots_csv(spot_rows_all, spots_path)
        frames_path = workdir / "frames.npy"
        np.save(frames_path, frames.astype(np.float32))
        out_path = workdir / "results.h5"

        # Drive the CLI for grain 0 only (faster).
        rc = cli_main([
            "fit",
            "--geometry", str(cfg_path),
            "--grains", str(grains_path),
            "--spots", str(spots_path),
            "--frames", str(frames_path),
            "--output", str(out_path),
            "--odf-type", "particle",
            "--K", "32",
            "--theta-max-deg", "0.15",
            "--patch-F", str(patch_F),
            "--patch-P", str(patch_P),
            "--delta-iters", "2",
            "--inner-steps", "200",
            "--lr-axis-angle", "1e-4",
            "--lr-logits", "0.1",
            "--grain-id", "0",
        ])
        assert rc == 0

        with h5py.File(out_path, "r") as h5:
            assert "grain_0" in h5
            g0 = h5["grain_0"]
            losses = np.asarray(g0["loss_history"])
            assert losses.size > 0
            print(f"  grain 0: initial loss = {losses[0]:.3e}, "
                  f"final = {losses[-1]:.3e}, "
                  f"ratio = {losses[-1]/losses[0]:.3e}")
            # Reduction should be at least 10x; real-world variation comes
            # from the random init relative to spot count.
            assert losses[-1] < 0.2 * losses[0]

            R_samples = np.asarray(g0["R_samples"])
            weights = np.asarray(g0["weights"])
            assert R_samples.shape == (32, 3, 3)
            assert weights.shape == (32,)
            assert np.allclose(weights.sum(), 1.0, atol=1e-6)
            assert g0.attrs["odf_type"] == "particle"
            print(f"  weights sum = {weights.sum():.6f}, "
                  f"odf_type = {g0.attrs['odf_type']}, "
                  f"converged = {bool(g0.attrs['converged'])}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
