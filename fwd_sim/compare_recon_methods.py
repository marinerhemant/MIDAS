#!/usr/bin/env python
"""Compare FBP vs MLEM reconstruction on pf-HEDM sinograms.

Requires: test_pf_hedm.py run with --no-cleanup so that the
intermediate sinogram files are preserved.

Usage:
    python compare_recon_methods.py /path/to/pfhedm_test/

Reads the per-grain sinograms and thetas from the test output,
reconstructs with both FBP and MLEM, and compares grain maps.
"""

import sys
import os
import numpy as np
from pathlib import Path
from PIL import Image

# Add MIDAS paths
MIDAS_HOME = Path("/Users/hsharma/opt/MIDAS")
sys.path.insert(0, str(MIDAS_HOME / "TOMO"))
sys.path.insert(0, str(MIDAS_HOME / "utils"))

from midas_tomo_python import run_tomo_from_sinos
from mlem_recon import mlem, osem


def load_sinograms(work_dir):
    """Load per-grain sinograms and thetas from test output."""
    work = Path(work_dir)

    # Find sinogram files
    sino_dir = work / "Sinos"
    theta_dir = work / "Thetas"

    if not sino_dir.exists():
        print(f"ERROR: {sino_dir} not found. Run test_pf_hedm.py --no-cleanup first.")
        sys.exit(1)

    # Count grains
    sino_files = sorted(sino_dir.glob("sino_raw_grNr_*.tif"))
    n_grains = len(sino_files)
    if n_grains == 0:
        # Try alternate naming
        sino_files = sorted(sino_dir.glob("sino_grNr_*.tif"))
        n_grains = len(sino_files)

    print(f"Found {n_grains} grain sinograms")

    sinos = []
    thetas_list = []
    for g in range(n_grains):
        # Try different naming conventions
        for pattern in [f"sino_raw_grNr_{g:04d}.tif", f"sino_grNr_{g:04d}.tif"]:
            sino_path = sino_dir / pattern
            if sino_path.exists():
                break
        else:
            print(f"  Grain {g}: sinogram not found, skipping")
            sinos.append(None)
            thetas_list.append(None)
            continue

        sino = np.array(Image.open(sino_path), dtype=np.float64)
        sinos.append(sino)

        theta_path = theta_dir / f"thetas_grNr_{g:04d}.txt"
        if theta_path.exists():
            thetas = np.loadtxt(theta_path)
        else:
            thetas = np.linspace(0, 180, sino.shape[0], endpoint=False)
        thetas_list.append(thetas)

        print(f"  Grain {g}: sino shape {sino.shape}, "
              f"{(sino > 0).sum()}/{sino.size} non-zero")

    return sinos, thetas_list


def reconstruct_fbp(sino, thetas, nScans):
    """Reconstruct using FBP (current method)."""
    sino_for_tomo = sino.T  # (nThetas, nScans)
    recon_arr = run_tomo_from_sinos(
        sino_for_tomo, '/tmp/tomo_compare', thetas,
        shifts=0.0, filterNr=2, doLog=0,
        extraPad=0, autoCentering=1, numCPUs=1, doCleanup=1)
    recon_full = recon_arr[0, 0, :, :]
    # Crop to nScans x nScans
    center = recon_full.shape[0] // 2
    half = nScans // 2
    return recon_full[center - half:center - half + nScans,
                      center - half:center - half + nScans]


def reconstruct_mlem(sino, thetas, nScans, n_iter=50):
    """Reconstruct using MLEM."""
    sino_for_tomo = sino.T  # (nThetas, nScans)
    recon = mlem(sino_for_tomo, thetas, n_iter=n_iter)
    return recon[:nScans, :nScans]


def reconstruct_osem(sino, thetas, nScans, n_iter=15, n_subsets=4):
    """Reconstruct using OS-EM."""
    sino_for_tomo = sino.T
    recon = osem(sino_for_tomo, thetas, n_iter=n_iter, n_subsets=n_subsets)
    return recon[:nScans, :nScans]


def build_grain_map(all_recons):
    """Build grain map from per-grain reconstructions (argmax)."""
    stack = np.array(all_recons)  # (nGrains, nScans, nScans)
    max_val = np.max(stack, axis=0)
    grain_map = np.argmax(stack, axis=0)
    grain_map[max_val <= 0] = -1
    return grain_map


def load_ground_truth(work_dir):
    """Load ground truth grain map from microstructure file."""
    work = Path(work_dir)
    micro_file = work / "microstructure.ebsd"
    if not micro_file.exists():
        return None

    data = np.loadtxt(micro_file, skiprows=1)
    # Columns: x, y, euler1, euler2, euler3, grainID
    n = int(np.sqrt(data.shape[0]))
    grain_ids = data[:, -1].reshape(n, n).astype(int) if data.shape[1] > 5 else None
    return grain_ids


def main():
    if len(sys.argv) < 2:
        work_dir = MIDAS_HOME / "FF_HEDM" / "Example" / "pfhedm_test"
    else:
        work_dir = Path(sys.argv[1])

    print("=" * 60)
    print("  FBP vs MLEM Reconstruction Comparison")
    print("=" * 60)

    sinos, thetas_list = load_sinograms(work_dir)
    n_grains = len(sinos)

    # Determine nScans from sinogram shape
    for s in sinos:
        if s is not None:
            nScans = s.shape[0]  # sino shape is (nScans, nSpots)
            break
    print(f"\nnScans = {nScans}")

    # Reconstruct all grains with each method
    methods = {
        "FBP": lambda s, t: reconstruct_fbp(s, t, nScans),
        "MLEM-50": lambda s, t: reconstruct_mlem(s, t, nScans, n_iter=50),
        "OSEM-15x4": lambda s, t: reconstruct_osem(s, t, nScans, n_iter=15, n_subsets=4),
    }

    results = {}
    for name, recon_fn in methods.items():
        print(f"\n--- Reconstructing with {name} ---")
        all_recons = []
        for g in range(n_grains):
            if sinos[g] is None:
                all_recons.append(np.zeros((nScans, nScans)))
                continue
            recon = recon_fn(sinos[g], thetas_list[g])
            all_recons.append(recon)
            print(f"  Grain {g}: max={recon.max():.1f}, "
                  f"nonzero={np.sum(recon > 0.01 * recon.max())}")

        grain_map = build_grain_map(all_recons)
        results[name] = {"recons": all_recons, "grain_map": grain_map}

    # Compare grain maps
    print("\n" + "=" * 60)
    print("  GRAIN MAP COMPARISON")
    print("=" * 60)

    for name, res in results.items():
        gm = res["grain_map"]
        n_assigned = np.sum(gm >= 0)
        print(f"\n  {name}:")
        print(f"    Assigned: {n_assigned}/{gm.size} pixels")
        print(f"    Grain distribution: ", end="")
        for g in range(n_grains):
            count = np.sum(gm == g)
            print(f"G{g}={count} ", end="")
        print()

    # Cross-compare
    ref_name = "FBP"
    ref_map = results[ref_name]["grain_map"]
    for name, res in results.items():
        if name == ref_name:
            continue
        gm = res["grain_map"]
        agree = np.sum((gm == ref_map) & (ref_map >= 0))
        total = np.sum(ref_map >= 0)
        disagree = total - agree
        print(f"\n  {name} vs {ref_name}: {agree}/{total} agree "
              f"({100 * agree / total:.1f}%), {disagree} differ")

        # Show where they differ
        diff_mask = (gm != ref_map) & (ref_map >= 0)
        if diff_mask.any():
            ys, xs = np.where(diff_mask)
            print(f"    Differing pixels (first 10):")
            for i in range(min(10, len(ys))):
                print(f"      ({ys[i]},{xs[i]}): {ref_name}=G{ref_map[ys[i],xs[i]]}"
                      f" -> {name}=G{gm[ys[i],xs[i]]}")

    # Save results
    out_dir = work_dir / "recon_comparison"
    os.makedirs(out_dir, exist_ok=True)
    for name, res in results.items():
        safe_name = name.replace("-", "_").replace("x", "x")
        np.save(out_dir / f"grain_map_{safe_name}.npy", res["grain_map"])
        for g, recon in enumerate(res["recons"]):
            np.save(out_dir / f"recon_{safe_name}_grain{g}.npy", recon)
    print(f"\n  Results saved to {out_dir}")


if __name__ == "__main__":
    main()
