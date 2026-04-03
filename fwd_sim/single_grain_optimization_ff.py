#!/usr/bin/env python
"""FF-HEDM single-grain optimization demo.

Demonstrates end-to-end differentiable optimization of crystallographic
orientation and lattice parameters (strain) for a single grain using
the HEDMForwardModel + SpotMatchingLoss pipeline.

Workflow:
    1. Generate synthetic "observed" spots from a known ground-truth grain
    2. Optionally add realistic Gaussian noise at the measurement resolution
    3. Perturb the orientation and lattice parameters
    4. Use L-BFGS to recover the ground truth via 3-phase optimization:
       Phase 1: orientation only (eta/omega sensitive)
       Phase 2: strain only (2theta sensitive)
       Phase 3: joint refinement
    5. Compare recovered parameters to ground truth

Weights are derived from measurement resolution:
    sigma_2th = 0.5 * px / Lsd           (radial, ~1e-4 rad)
    sigma_eta = 0.5 / R_ring_pixels      (tangential, ~1e-3 rad)
    sigma_ome = 0.25 * omega_step * d2r  (omega, ~1.1e-3 rad)
Using weight = sigma normalizes each coordinate by its uncertainty.

Usage:
    python single_grain_optimization_ff.py           # both modes
    python single_grain_optimization_ff.py --clean   # noise-free only
    python single_grain_optimization_ff.py --noisy   # noisy only
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hedm_forward import HEDMForwardModel, HEDMGeometry
from hedm_losses import SpotMatchingLoss

MIDAS_HOME = Path("/Users/hsharma/opt/MIDAS")
BUILD_BIN = MIDAS_HOME / "build" / "bin"
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi


def generate_hkls(work_dir, latc, wl, sg=225):
    """Run GetHKLList to generate hkls.csv and parse it."""
    param_file = work_dir / "params.txt"
    with open(param_file, "w") as f:
        f.write(f"LatticeParameter {' '.join(str(v) for v in latc)}\n")
        f.write(f"Wavelength {wl}\n")
        f.write(f"SpaceGroup {sg}\n")
        f.write(f"Lsd 1000000\nMaxRingRad 500000\n")
    result = subprocess.run(
        [str(BUILD_BIN / "GetHKLList"), str(param_file)],
        cwd=str(work_dir), capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, f"GetHKLList failed:\n{result.stderr}"
    data = np.loadtxt(work_dir / "hkls.csv", skiprows=1)
    return (
        torch.tensor(data[:, 5:8], dtype=torch.float64),
        torch.tensor(data[:, 8] * DEG2RAD, dtype=torch.float64),
        torch.tensor(data[:, 0:3], dtype=torch.float64),
    )


def run_optimization(model, obs_coords, gt_euler_rad, gt_latc, gt_pos,
                     init_euler_rad, init_latc, loss_fn, label=""):
    """Run the 3-phase optimization and return results."""
    pos = gt_pos.unsqueeze(0)
    R_gt = HEDMForwardModel.euler2mat(gt_euler_rad)

    def make_closure(opt_params):
        def closure():
            for p in opt_params:
                if p.grad is not None:
                    p.grad.zero_()
            spots = model(
                opt_euler.unsqueeze(0), pos, lattice_params=opt_latc,
            )
            coords, valid = HEDMForwardModel.predict_spot_coords(
                spots, space="angular"
            )
            pred_flat = coords.squeeze().reshape(-1, 3)
            valid_flat = valid.squeeze().reshape(-1)
            pred_valid = pred_flat[valid_flat > 0.5]
            if pred_valid.shape[0] == 0:
                return torch.tensor(1e6, dtype=torch.float64, requires_grad=True)
            dists = torch.cdist(obs_coords, pred_valid)
            min_dists, nn_idx = dists.min(dim=1)
            keep = min_dists < 0.5
            if keep.sum() < 5:
                return torch.tensor(1e6, dtype=torch.float64, requires_grad=True)
            loss = loss_fn(pred_valid[nn_idx[keep]], obs_coords[keep])
            loss.backward()
            return loss
        return closure

    def log_step(step, loss_val):
        with torch.no_grad():
            R_cur = HEDMForwardModel.euler2mat(opt_euler)
            trace = torch.trace(R_gt.T @ R_cur)
            misori = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * RAD2DEG
            lat_err = abs(opt_latc[0].item() - gt_latc[0].item())
        print(f"{step:5d}  {loss_val.item():12.6e}  {misori.item():12.6f}  {lat_err:10.6f}")
        return misori.item(), lat_err

    print(f"\n{'Step':>5}  {'Loss':>12}  {'Misori(deg)':>12}  {'Lat_a err':>10}")
    print("-" * 55)

    # Phase 1: Orientation only
    print("--- Phase 1: Orientation (euler angles) ---")
    opt_euler = init_euler_rad.clone().requires_grad_(True)
    opt_latc = init_latc.clone()
    opt_latc.requires_grad_(False)
    optimizer = torch.optim.LBFGS(
        [opt_euler], lr=1.0, max_iter=20, line_search_fn="strong_wolfe",
    )
    for step in range(15):
        loss_val = optimizer.step(make_closure([opt_euler]))
        misori, _ = log_step(step, loss_val)
        if misori < 0.001:
            break

    # Phase 2: Strain only
    print("--- Phase 2: Strain (lattice parameters) ---")
    opt_euler.requires_grad_(False)
    opt_latc.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [opt_latc], lr=1.0, max_iter=20, line_search_fn="strong_wolfe",
    )
    for step in range(15):
        loss_val = optimizer.step(make_closure([opt_latc]))
        _, lat_err = log_step(step + 15, loss_val)
        if lat_err < 1e-5:
            break

    # Phase 3: Joint refinement
    print("--- Phase 3: Joint refinement ---")
    opt_euler.requires_grad_(True)
    opt_latc.requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [opt_euler, opt_latc], lr=0.5, max_iter=20, line_search_fn="strong_wolfe",
    )
    for step in range(10):
        loss_val = optimizer.step(make_closure([opt_euler, opt_latc]))
        misori, lat_err = log_step(step + 30, loss_val)
        if misori < 0.001 and lat_err < 1e-5:
            print("  Converged!")
            break

    # Results
    with torch.no_grad():
        R_final = HEDMForwardModel.euler2mat(opt_euler)
        trace = torch.trace(R_gt.T @ R_final)
        final_misori = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * RAD2DEG
        lat_errors = (opt_latc - gt_latc).abs()

    return {
        "euler_recovered": opt_euler.detach() * RAD2DEG,
        "latc_recovered": opt_latc.detach(),
        "misori_deg": final_misori.item(),
        "lat_errors": lat_errors,
    }


def main():
    print("=" * 70)
    print("  FF-HEDM Single-Grain Orientation + Strain Recovery Demo")
    print("=" * 70)

    # Parse args
    modes = ["clean", "noisy"]
    if "--clean" in sys.argv:
        modes = ["clean"]
    elif "--noisy" in sys.argv:
        modes = ["noisy"]

    # ── Material & geometry ──
    latc_nominal = [4.08, 4.08, 4.08, 90.0, 90.0, 90.0]
    wavelength = 0.172979
    Lsd = 1_000_000.0;  px = 200.0
    y_BC, z_BC = 1024.0, 1024.0;  n_pix = 2048
    omega_start, omega_step = 0.0, 0.25;  n_frames = 1440

    geometry = HEDMGeometry(
        Lsd=Lsd, y_BC=y_BC, z_BC=z_BC, px=px,
        omega_start=omega_start, omega_step=omega_step,
        n_frames=n_frames, n_pixels_y=n_pix, n_pixels_z=n_pix,
        min_eta=6.0, wavelength=wavelength,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        hkls_cart, thetas, hkls_int = generate_hkls(
            Path(tmpdir), latc_nominal, wavelength
        )
    print(f"HKLs: {hkls_cart.shape[0]} reflections")

    model = HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geometry, hkls_int=hkls_int,
    )

    # ── Ground truth ──
    gt_euler_deg = torch.tensor([45.0, 30.0, 60.0], dtype=torch.float64)
    gt_euler_rad = gt_euler_deg * DEG2RAD
    gt_latc = torch.tensor([4.082, 4.079, 4.081, 90.01, 89.99, 90.02],
                           dtype=torch.float64)
    gt_pos = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

    # Forward simulate ground truth
    spots_gt = model(gt_euler_rad.unsqueeze(0), gt_pos.unsqueeze(0),
                     lattice_params=gt_latc)
    coords_gt, valid_gt = HEDMForwardModel.predict_spot_coords(
        spots_gt, space="angular"
    )
    mask = valid_gt.squeeze() > 0.5
    obs_clean = coords_gt.squeeze()[mask].detach().clone()
    n_obs = obs_clean.shape[0]

    # Measurement resolutions
    typical_ring_px = 500.0
    sigma_2th = 0.5 * px / Lsd
    sigma_eta = 0.5 / typical_ring_px
    sigma_ome = 0.25 * abs(omega_step) * DEG2RAD

    # Resolution-based weights (weight = sigma, normalized)
    weights = torch.tensor([sigma_2th, sigma_eta, sigma_ome], dtype=torch.float64)
    weights = weights / weights.mean()
    loss_fn = SpotMatchingLoss(metric="l2", weights=weights)

    print(f"\nGround truth: Euler={gt_euler_deg.tolist()} deg, "
          f"Lattice={gt_latc.tolist()}")
    print(f"Valid spots: {n_obs}")
    print(f"Weights [2th, eta, ome]: [{weights[0]:.4f}, {weights[1]:.4f}, "
          f"{weights[2]:.4f}]")
    print(f"Resolutions: sigma_2th={sigma_2th:.2e}, sigma_eta={sigma_eta:.2e}, "
          f"sigma_ome={sigma_ome:.2e} rad")

    # Perturbed starting point (same for both runs)
    torch.manual_seed(42)
    init_euler_rad = (
        gt_euler_rad
        + torch.randn(3, dtype=torch.float64) * 3.0 * DEG2RAD
    )
    init_latc = gt_latc + torch.randn(6, dtype=torch.float64) * torch.tensor(
        [0.005, 0.005, 0.005, 0.05, 0.05, 0.05], dtype=torch.float64,
    )

    R_gt = HEDMForwardModel.euler2mat(gt_euler_rad)
    R_init = HEDMForwardModel.euler2mat(init_euler_rad)
    trace = torch.trace(R_gt.T @ R_init)
    init_misori = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * RAD2DEG
    print(f"\nStarting perturbation: {init_misori.item():.4f} deg misorientation")

    all_success = True

    for mode in modes:
        print(f"\n{'=' * 70}")
        if mode == "clean":
            print("  MODE: Noise-free (ideal spots)")
            obs = obs_clean.clone()
        else:
            print("  MODE: Noisy (Gaussian noise at measurement resolution)")
            torch.manual_seed(123)
            noise = torch.randn_like(obs_clean) * torch.tensor(
                [sigma_2th, sigma_eta, sigma_ome], dtype=torch.float64
            )
            obs = obs_clean + noise
        print(f"{'=' * 70}")

        results = run_optimization(
            model, obs, gt_euler_rad, gt_latc, gt_pos,
            init_euler_rad, init_latc, loss_fn, label=mode,
        )

        print(f"\n  Ground truth Euler (deg): {gt_euler_deg.tolist()}")
        print(f"  Recovered    Euler (deg): "
              f"{[f'{v:.4f}' for v in results['euler_recovered'].tolist()]}")
        print(f"  Final misorientation: {results['misori_deg']:.6f} deg")
        print(f"\n  Ground truth Lattice: {gt_latc.tolist()}")
        print(f"  Recovered    Lattice: "
              f"{[f'{v:.6f}' for v in results['latc_recovered'].tolist()]}")
        print(f"  Lattice errors: "
              f"{[f'{v:.2e}' for v in results['lat_errors'].tolist()]}")

        if mode == "clean":
            ok = results["misori_deg"] < 0.1 and results["lat_errors"][:3].max().item() < 2e-3
            label = "misori < 0.1 deg, lat err < 2e-3 A"
        else:
            ok = results["misori_deg"] < 0.05 and results["lat_errors"][:3].max().item() < 5e-3
            label = "misori < 0.05 deg, lat err < 5e-3 A (noise-limited)"

        print(f"\n  {'PASS' if ok else 'FAIL'} ({label})")
        all_success = all_success and ok

    return all_success


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
