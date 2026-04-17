#!/usr/bin/env python
"""Verify C vs Python forward model agreement in pf-HEDM mode.

Generates a synthetic pf-HEDM microstructure, runs ForwardSimulationCompressed
with WriteSpots=1, then runs the Python HEDMForwardModel on the same
orientations and geometry. Compares predicted spots in angular space
(omega, eta, 2theta).

Usage:
    cd /Users/hsharma/opt/MIDAS/fwd_sim
    python verify_pf_fwdsim.py

Requires:
    - Built C executables in FF_HEDM/bin/
    - midas_env conda environment
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MIDAS_HOME = Path(__file__).resolve().parent.parent
BUILD_BIN = MIDAS_HOME / "FF_HEDM" / "bin"
sys.path.insert(0, str(MIDAS_HOME / "fwd_sim"))
sys.path.insert(0, str(MIDAS_HOME / "utils"))

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

from calcMiso import Euler2OrientMat


def parse_hkls_csv(path):
    """Parse hkls.csv -> (hkls_cart, thetas, ring_nrs)."""
    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    import torch
    return (
        torch.tensor(data[:, 5:8], dtype=torch.float64),   # g1,g2,g3
        torch.tensor(data[:, 8] * DEG2RAD, dtype=torch.float64),  # theta -> rad
        data[:, 4].astype(int),  # ring numbers
    )


def parse_spot_matrix_gen(path):
    """Parse SpotMatrixGen.csv from ForwardSimulationCompressed."""
    data = np.loadtxt(path, skiprows=1, delimiter="\t")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        "grain_id": data[:, 0].astype(int),
        "omega_deg": data[:, 2],
        "det_hor": data[:, 3],
        "det_vert": data[:, 4],
        "omega_raw_deg": data[:, 5],
        "eta_deg": data[:, 6],
        "ring_nr": data[:, 7].astype(int),
        "theta_deg": data[:, 10],
        "scan_nr": data[:, 12].astype(int) if data.shape[1] > 12 else np.zeros(len(data), dtype=int),
    }


def main():
    import torch
    from hedm_forward import HEDMForwardModel, HEDMGeometry

    # ---------------------------------------------------------------
    # Configuration: match pf-HEDM test geometry
    # ---------------------------------------------------------------
    NGRAINS = 5
    SEED = 42
    latc = [3.5950, 3.5950, 3.5950, 90.0, 90.0, 90.0]
    wl = 0.172979
    Lsd = 1_000_000.0   # 1 m
    px = 200.0
    y_bc = 1024.0
    z_bc = 1024.0
    n_pix = 2048
    omega_start = -180.0
    omega_step = 0.25
    n_frames = 1440
    sgnum = 225
    min_eta = 5.0

    # Generate random orientations (same seed as test)
    rng = np.random.RandomState(SEED)
    eulers_deg = []
    orient_mats_flat = []
    for _ in range(NGRAINS):
        e = rng.uniform([0, 0, 0], [360, 180, 360])
        eulers_deg.append(e)
        om = Euler2OrientMat(e * DEG2RAD)
        orient_mats_flat.append(om)
    eulers_deg = np.array(eulers_deg)
    orient_mats_flat = np.array(orient_mats_flat)  # (N, 9)

    print(f"Ground truth orientations ({NGRAINS} grains):")
    for i, e in enumerate(eulers_deg):
        print(f"  Grain {i+1}: Euler=({e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}) deg")

    # ---------------------------------------------------------------
    # Step 1: Run C forward simulation with WriteSpots=1
    # ---------------------------------------------------------------
    work = Path(tempfile.mkdtemp(prefix="pf_verify_"))
    print(f"\nWork dir: {work}")

    # Write Grains.csv
    grains_csv = work / "Grains.csv"
    with open(grains_csv, "w") as f:
        f.write(f"%NumGrains {NGRAINS}\n")
        f.write("%BeamCenter 0.000000\n")
        f.write("%BeamThickness 10000.000000\n")
        f.write("%GlobalPosition 0.000000\n")
        f.write("%NumPhases 1\n")
        f.write("%PhaseInfo\n")
        f.write(f"%\tSpaceGroup:{sgnum}\n")
        lstr = " ".join(f"{v:.6f}" for v in latc)
        f.write(f"%\tLattice Parameter: {lstr}\n")
        f.write("%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33"
                "\tX\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\n")
        for i in range(NGRAINS):
            om = orient_mats_flat[i]
            vals = list(om) + [0.0, 0.0, 0.0] + latc
            line = f"{i+1}\t" + "\t".join(f"{v:.6f}" for v in vals)
            f.write(line + "\n")

    # Write positions (single scan at 0)
    pos_file = work / "positions.csv"
    pos_file.write_text("0.0\n")

    # Write parameter file
    param_file = work / "params_pf.txt"
    with open(param_file, "w") as f:
        f.write(f"LatticeConstant {' '.join(str(v) for v in latc)}\n")
        f.write(f"LatticeParameter {' '.join(str(v) for v in latc)}\n")
        f.write(f"Wavelength {wl}\n")
        f.write(f"SpaceGroup {sgnum}\n")
        f.write(f"Lsd {Lsd}\n")
        f.write(f"BC {y_bc} {z_bc}\n")
        f.write(f"px {px}\n")
        f.write(f"NrPixels {n_pix}\n")
        f.write(f"tx 0\nty 0\ntz 0\n")
        f.write(f"Wedge 0\n")
        f.write(f"OmegaStart {omega_start}\n")
        f.write(f"OmegaEnd {omega_start + omega_step * n_frames}\n")
        f.write(f"OmegaStep {omega_step}\n")
        f.write(f"MaxRingRad {Lsd * 0.3}\n")
        f.write(f"ExcludePoleAngle {min_eta}\n")
        f.write(f"InFileName {grains_csv}\n")
        f.write(f"OutFileName {work / 'sim_out'}\n")
        f.write(f"PositionsFile {pos_file}\n")
        f.write(f"WriteSpots 1\n")
        f.write(f"WriteImage 0\n")
        f.write(f"nScans 1\n")
        f.write(f"PeakIntensity 2000\n")
        f.write(f"GaussWidth 2.0\n")

    # Run GetHKLList
    print("\n--- Running C: GetHKLList ---")
    result = subprocess.run(
        [str(BUILD_BIN / "GetHKLList"), str(param_file)],
        cwd=str(work), capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"GetHKLList FAILED:\n{result.stderr}")
        sys.exit(1)

    hkls_file = work / "hkls.csv"
    if not hkls_file.exists():
        print("ERROR: hkls.csv not generated")
        sys.exit(1)

    hkls_cart, thetas, ring_nrs = parse_hkls_csv(hkls_file)
    print(f"  {hkls_cart.shape[0]} HKL reflections, "
          f"{len(np.unique(ring_nrs))} unique rings")

    # Run ForwardSimulationCompressed
    print("\n--- Running C: ForwardSimulationCompressed ---")
    result = subprocess.run(
        [str(BUILD_BIN / "ForwardSimulationCompressed"),
         str(param_file), "1"],
        cwd=str(work), capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"ForwardSim FAILED:\n{result.stderr}")
        sys.exit(1)

    spot_file = work / "SpotMatrixGen.csv"
    if not spot_file.exists():
        print("ERROR: SpotMatrixGen.csv not generated")
        sys.exit(1)

    c_spots = parse_spot_matrix_gen(spot_file)
    n_c_total = len(c_spots["omega_deg"])
    print(f"  C produced {n_c_total} total spots")

    # ---------------------------------------------------------------
    # Step 2: Run Python forward model
    # ---------------------------------------------------------------
    print("\n--- Running Python: HEDMForwardModel ---")

    geometry = HEDMGeometry(
        Lsd=Lsd, y_BC=y_bc, z_BC=z_bc, px=px,
        omega_start=omega_start, omega_step=omega_step,
        n_frames=n_frames,
        n_pixels_y=n_pix, n_pixels_z=n_pix,
        min_eta=min_eta, wavelength=wl,
    )
    model = HEDMForwardModel(
        hkls=hkls_cart, thetas=thetas, geometry=geometry,
        device=torch.device("cpu"),
    )
    model.ring_indices = torch.tensor(ring_nrs, dtype=torch.long)

    # ---------------------------------------------------------------
    # Step 3: Compare per grain
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Per-grain comparison: C SpotMatrixGen vs Python HEDMForwardModel")
    print("=" * 70)

    total_c = 0
    total_py = 0
    total_matched = 0

    for g in range(NGRAINS):
        grain_id = g + 1
        mask = c_spots["grain_id"] == grain_id
        c_ome = c_spots["omega_deg"][mask]
        c_eta = c_spots["eta_deg"][mask]
        c_theta = c_spots["theta_deg"][mask]
        c_ring = c_spots["ring_nr"][mask]
        c_y = c_spots["det_hor"][mask]
        c_z = c_spots["det_vert"][mask]
        n_c = len(c_ome)

        # Python: use orientation matrix directly
        om_tensor = torch.tensor(
            orient_mats_flat[g].reshape(3, 3), dtype=torch.float64
        ).unsqueeze(0)  # (1, 3, 3)

        pos = torch.zeros(1, 3, dtype=torch.float64)

        omega, eta, two_theta, valid = model.calc_bragg_geometry(
            om_tensor, hkls_cart, thetas
        )
        spots = model.project_to_detector(omega, eta, two_theta, pos, valid)
        py_coords, py_valid = HEDMForwardModel.predict_spot_coords(
            spots, space="angular"
        )
        # py_coords: (1, 2, M, 3) -> (2M, 3), py_valid: (1, 2, M) -> (2M,)
        pc = py_coords.squeeze(0).reshape(-1, 3).numpy()
        pv = py_valid.squeeze(0).reshape(-1).numpy()

        # Also get detector coords for pixel comparison
        py_det, _ = HEDMForwardModel.predict_spot_coords(
            spots, space="detector"
        )
        pd = py_det.squeeze(0).reshape(-1, 3).numpy()

        valid_mask = pv > 0.5
        py_2theta = pc[valid_mask, 0] * RAD2DEG  # rad -> deg
        py_eta = pc[valid_mask, 1] * RAD2DEG
        py_omega = pc[valid_mask, 2] * RAD2DEG
        py_ypx = pd[valid_mask, 0]
        py_zpx = pd[valid_mask, 1]

        # Get ring indices for valid Python spots
        ri = model.ring_indices.numpy()
        ri_doubled = np.tile(ri, 2)  # (2M,)
        py_ring = ri_doubled[valid_mask]

        n_py = len(py_omega)

        # Match C spots to Python spots
        matched = 0
        ome_errors = []
        eta_errors = []
        y_errors = []
        z_errors = []
        unmatched_c_indices = []

        for i in range(n_c):
            # Filter by ring first
            ring_match = py_ring == c_ring[i]
            if not np.any(ring_match):
                unmatched_c_indices.append(i)
                continue

            # Within same ring, find closest in (omega, eta)
            d_ome = np.abs(py_omega[ring_match] - c_ome[i])
            d_eta = np.abs(py_eta[ring_match] - c_eta[i])
            d_eta = np.minimum(d_eta, 360.0 - d_eta)
            d_ome = np.minimum(d_ome, 360.0 - d_ome)
            dist = np.sqrt(d_ome**2 + d_eta**2)

            best = np.argmin(dist)
            if dist[best] < 0.5:  # within 0.5 degree
                matched += 1
                ome_errors.append(d_ome[best])
                eta_errors.append(d_eta[best])
                # Find pixel errors (need index into full valid array)
                idx_in_ring = np.where(ring_match)[0][best]
                y_errors.append(abs(py_ypx[idx_in_ring] - c_y[i]))
                z_errors.append(abs(py_zpx[idx_in_ring] - c_z[i]))
            else:
                unmatched_c_indices.append(i)

        match_pct = 100 * matched / n_c if n_c else 0
        print(f"\n  Grain {grain_id}: C={n_c} spots, Python={n_py} spots, "
              f"matched={matched} ({match_pct:.1f}%)")

        if ome_errors:
            print(f"    Omega err: mean={np.mean(ome_errors):.4f} deg, "
                  f"max={np.max(ome_errors):.4f} deg")
            print(f"    Eta err:   mean={np.mean(eta_errors):.4f} deg, "
                  f"max={np.max(eta_errors):.4f} deg")
            print(f"    Y pixel:   mean={np.mean(y_errors):.3f} px, "
                  f"max={np.max(y_errors):.3f} px")
            print(f"    Z pixel:   mean={np.mean(z_errors):.3f} px, "
                  f"max={np.max(z_errors):.3f} px")

        if unmatched_c_indices and len(unmatched_c_indices) <= 10:
            print(f"    Unmatched C spots ({len(unmatched_c_indices)}):")
            for idx in unmatched_c_indices[:5]:
                print(f"      omega={c_ome[idx]:.3f} eta={c_eta[idx]:.3f} "
                      f"ring={c_ring[idx]} theta={c_theta[idx]:.4f}")

        total_c += n_c
        total_py += n_py
        total_matched += matched

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    total_pct = 100 * total_matched / total_c if total_c else 0
    print(f"  TOTAL: C={total_c}, Python={total_py}, "
          f"matched={total_matched} ({total_pct:.1f}%)")
    print("=" * 70)

    if total_pct < 95:
        print("\n  *** WARNING: <95% match — possible convention issue ***")
    else:
        print("\n  OK: C and Python forward models agree in pf-HEDM mode.")

    # Cleanup
    shutil.rmtree(str(work), ignore_errors=True)
    print(f"\nCleaned up {work}")


if __name__ == "__main__":
    main()
