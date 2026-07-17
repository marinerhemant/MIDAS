"""Build a self-contained reference dataset and compare midas-index vs C IndexerOMP.

Pipeline:
  1. Generate random grain orientations (Cu, FCC, sg=225).
  2. Forward-simulate spots via midas_diffract -> 9-col Spots.bin layout.
  3. Generate hkls.csv via the MIDAS C tool GetHKLList.
  4. Write paramstest.txt + SpotsToIndex.csv.
  5. Build Data.bin/nData.bin via midas_index.io.build_bin_index
     (mirrors C SaveBinData.c::SpreadSpotsAcrossBins).
  6. Run BOTH:
       - C IndexerOMP                   -> golden/IndexBest.bin
       - python -m midas_index          -> midas/IndexBest.bin
  7. Compare records, report mismatches.

Usage:
    cd packages/midas_index/tests/data
    python build_reference.py --n-grains 5 --seed 42

The 5-grain default runs in seconds; --n-grains 500 is the full reference.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from midas_diffract import HEDMForwardModel, HEDMGeometry
from midas_index.io import build_bin_index
from midas_stress.orientation import (
    axis_angle_to_orient_mat,
    misorientation_om,
)


def _resolve_midas_bin_dir() -> Path:
    """Locate FF_HEDM/bin. Override via the MIDAS_BIN_DIR env var."""
    env = os.environ.get("MIDAS_BIN_DIR")
    if env:
        return Path(env)
    # Walk up from this file (tests/data/build_reference.py):
    #   parents[0] = tests/data
    #   parents[1] = tests
    #   parents[2] = midas_index
    #   parents[3] = packages
    #   parents[4] = MIDAS root
    here = Path(__file__).resolve()
    midas_root = here.parents[4]
    return midas_root / "FF_HEDM" / "bin"


MIDAS_BIN_DIR = _resolve_midas_bin_dir()
INDEXER_OMP_BIN = MIDAS_BIN_DIR / "IndexerOMP"
GETHKLLIST_BIN = MIDAS_BIN_DIR / "GetHKLList"

# --- crystal & geometry ---
SPACE_GROUP = 225
LATTICE = (4.08, 4.08, 4.08, 90.0, 90.0, 90.0)         # Cu
WAVELENGTH = 0.172979
DISTANCE = 1_000_000.0                                   # Lsd in um
PIXEL_SIZE = 200.0
N_PIXELS = 2048
RSAMPLE = 250.0
HBEAM = 200.0
RING_NUMBERS = [1, 2, 3, 4]                              # FCC: 111, 200, 220, 311
MAX_RING_RADIUS_UM = 250_000.0
ETA_BIN_SIZE = 0.1
OME_BIN_SIZE = 0.1
MARGIN_OME = 0.5
MARGIN_RAD = 500.0
MARGIN_RADIAL = 500.0
MARGIN_ETA = 500.0                                       # um (yes, that's the C convention)
EXCLUDE_POLE_ANGLE = 6.0
STEPSIZE_POS = 100.0
STEPSIZE_ORIENT = 0.5
MIN_MATCHES_TO_ACCEPT_FRAC = 0.1   # synthetic data: only seed-ring theor can match
                                    # against same-ring obs (RefRad filter), so the
                                    # achievable max is ~1/n_rings. The C indexer's
                                    # default 0.6 in production assumes RingsToReject
                                    # configured for non-seed rings.
USE_FRIEDEL_PAIRS = 0   # 0 = generate_ideal_spots; 1 = Friedel


# ---------------------------------------------------------------------------
# Step 1: random grain orientations
# ---------------------------------------------------------------------------


def identity_orientations(n: int) -> np.ndarray:
    """Return n copies of the identity rotation — useful for sanity checks."""
    return np.tile(np.eye(3), (n, 1, 1))


def random_orientations(n: int, seed: int) -> np.ndarray:
    """Return (n, 3, 3) random rotation matrices via uniform quaternion sampling."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# ---------------------------------------------------------------------------
# Step 2: hkls.csv via the MIDAS C tool
# ---------------------------------------------------------------------------


def run_get_hkl_list(work_dir: Path) -> Path:
    """Invoke C GetHKLList in `work_dir`. Writes hkls.csv there."""
    cmd = [
        str(GETHKLLIST_BIN),
        "--sg", str(SPACE_GROUP),
        "--lp", *[str(v) for v in LATTICE],
        "--wl", str(WAVELENGTH),
        "--lsd", str(DISTANCE),
        "--maxR", str(MAX_RING_RADIUS_UM),
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, check=True)
    return work_dir / "hkls.csv"


# ---------------------------------------------------------------------------
# Step 3: forward-simulate spots
# ---------------------------------------------------------------------------


def load_hkls_for_forward(hkls_csv: Path, ring_filter: list[int]):
    """Read hkls.csv and return (hkls_cart, thetas_rad, hkls_int_with_ring) tensors
    with rows filtered to the requested rings.

    `hkls_int_with_ring` is (n_hkls, 4) layout `[h, k, l, ring_nr]`. The
    forward model takes `(n_hkls, 3)` — caller slices accordingly.
    """
    rows_cart, rows_int, thetas = [], [], []
    with open(hkls_csv) as f:
        f.readline()  # header
        for line in f:
            t = line.split()
            if len(t) < 11:
                continue
            h, k, l = int(t[0]), int(t[1]), int(t[2])
            ring_nr = int(t[4])
            if ring_nr not in ring_filter:
                continue
            g1, g2, g3 = float(t[5]), float(t[6]), float(t[7])
            # hkls.csv stores Theta in DEGREES; HEDMForwardModel wants radians.
            theta_rad = float(t[8]) * math.pi / 180.0
            rows_cart.append([g1, g2, g3])
            rows_int.append([h, k, l, ring_nr])
            thetas.append(theta_rad)
    if not rows_cart:
        raise RuntimeError(f"No hkls in rings {ring_filter}")
    return (
        torch.tensor(rows_cart, dtype=torch.float64),
        torch.tensor(thetas, dtype=torch.float64),
        torch.tensor(rows_int, dtype=torch.long),
    )


def forward_via_adapter(R: np.ndarray, params, hkls_real_np, hkls_int_np):
    """Run the indexer's own forward adapter and pull the raw 14-col TheorSpots.

    This guarantees the obs layout matches what midas-index expects internally.
    """
    from midas_index.compute.forward_adapter import IndexerForwardAdapter
    adapter = IndexerForwardAdapter(
        params=params,
        hkls_real=torch.as_tensor(hkls_real_np, dtype=torch.float64),
        hkls_int=torch.as_tensor(hkls_int_np, dtype=torch.long),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    R_t = torch.as_tensor(R, dtype=torch.float64)
    pos_t = torch.zeros(R_t.shape[0], 3, dtype=torch.float64)
    theor, valid = adapter.simulate(R_t, pos_t)
    # theor has shape (n_grains, K, 14); columns documented in §1.5.1.
    return theor, valid


def assemble_obs_from_theor(theor: torch.Tensor, valid: torch.Tensor) -> np.ndarray:
    """TheorSpots [N, K, 14] -> 9-column Spots.bin layout."""
    rows = []
    spot_id = 1
    for g in range(theor.shape[0]):
        for k in range(theor.shape[1]):
            if not bool(valid[g, k]):
                continue
            y = float(theor[g, k, 10].item())   # col 10 = yl_disp
            z = float(theor[g, k, 11].item())   # col 11 = zl_disp
            omega_deg = float(theor[g, k, 6].item())
            eta_deg = float(theor[g, k, 7].item())
            theta_deg = float(theor[g, k, 8].item())
            ring_nr = int(theor[g, k, 9].item())
            radial = math.sqrt(y * y + z * z)
            rad_diff = float(theor[g, k, 13].item())
            rows.append([y, z, omega_deg, radial, float(spot_id),
                         float(ring_nr), eta_deg, theta_deg * 2.0, rad_diff])
            spot_id += 1
    return np.asarray(rows, dtype=np.float64)


def ring_radius_from_2theta(two_theta_deg: float) -> float:
    return DISTANCE * math.tan(two_theta_deg * math.pi / 180.0)


# ---------------------------------------------------------------------------
# Step 4: writers for Spots.bin, paramstest.txt, SpotsToIndex.csv
# ---------------------------------------------------------------------------


def write_spots_bin(work_dir: Path, obs: np.ndarray) -> None:
    obs = np.ascontiguousarray(obs, dtype=np.float64)
    (work_dir / "Spots.bin").write_bytes(obs.tobytes())


def write_data_bins(work_dir: Path, obs: np.ndarray) -> None:
    n_rings = max(RING_NUMBERS)
    ring_radii = {r: ring_radius_for_ring_nr(r) for r in RING_NUMBERS}
    data, ndata = build_bin_index(
        obs,
        eta_bin_size=ETA_BIN_SIZE, ome_bin_size=OME_BIN_SIZE,
        n_rings=n_rings,
        margin_eta=MARGIN_ETA, margin_ome=MARGIN_OME,
        stepsize_orient=STEPSIZE_ORIENT,
        ring_radii=ring_radii,
    )
    (work_dir / "Data.bin").write_bytes(data.tobytes())
    (work_dir / "nData.bin").write_bytes(ndata.tobytes())


def ring_radius_for_ring_nr(ring_nr: int) -> float:
    """Compute ideal ring radius for ring_nr using lattice + wavelength."""
    a = LATTICE[0]
    if ring_nr == 1: hkl_sq = 3      # 111
    elif ring_nr == 2: hkl_sq = 4    # 200
    elif ring_nr == 3: hkl_sq = 8    # 220
    elif ring_nr == 4: hkl_sq = 11   # 311
    elif ring_nr == 5: hkl_sq = 12   # 222
    else:
        raise ValueError(f"unsupported ring_nr {ring_nr}")
    d = a / math.sqrt(hkl_sq)
    sin_th = WAVELENGTH / (2.0 * d)
    if sin_th >= 1.0:
        return 0.0
    two_theta = 2.0 * math.asin(sin_th)
    return DISTANCE * math.tan(two_theta)


def write_paramstest(work_dir: Path, output_subdir: Path) -> Path:
    p = work_dir / "paramstest.txt"
    lines = [
        f"Wavelength {WAVELENGTH}",
        f"Distance {DISTANCE}",
        f"px {PIXEL_SIZE}",
        f"SpaceGroup {SPACE_GROUP}",
        f"LatticeConstant " + " ".join(str(v) for v in LATTICE),
        f"Rsample {RSAMPLE}",
        f"Hbeam {HBEAM}",
        f"StepsizePos {STEPSIZE_POS}",
        f"StepsizeOrient {STEPSIZE_ORIENT}",
        f"MarginOme {MARGIN_OME}",
        f"MarginRadius {MARGIN_RAD}",
        f"MarginRadial {MARGIN_RADIAL}",
        f"MarginEta {MARGIN_ETA}",
        f"EtaBinSize {ETA_BIN_SIZE}",
        f"OmeBinSize {OME_BIN_SIZE}",
        f"ExcludePoleAngle {EXCLUDE_POLE_ANGLE}",
        f"MinMatchesToAcceptFrac {MIN_MATCHES_TO_ACCEPT_FRAC}",
        f"OmegaRange -180 180",
        f"BoxSize -2000000 2000000 -2000000 2000000",
        f"UseFriedelPairs {USE_FRIEDEL_PAIRS}",
        f"OutputFolder {output_subdir}",
    ]
    for r in RING_NUMBERS:
        lines.append(f"RingNumbers {r}")
    for r in RING_NUMBERS:
        lines.append(f"RingRadii {ring_radius_for_ring_nr(r)}")
    p.write_text("\n".join(lines) + "\n")
    return p


def write_spots_to_index(work_dir: Path, obs: np.ndarray, n_seeds: int) -> Path:
    """Pick n_seeds spot ids — prefer ring 1 spots, but allow others."""
    sti = work_dir / "SpotsToIndex.csv"
    spot_ids = obs[:, 4].astype(int)
    ring_nrs = obs[:, 5].astype(int)
    ring1 = spot_ids[ring_nrs == 1]
    if len(ring1) >= n_seeds:
        chosen = ring1[:n_seeds]
    else:
        chosen = spot_ids[:n_seeds]
    with open(sti, "w") as f:
        for sid in chosen:
            f.write(f"{int(sid)}\n")
    return sti


# ---------------------------------------------------------------------------
# Step 5: run both indexers and compare
# ---------------------------------------------------------------------------


def run_c_indexer(work_dir: Path, n_seeds: int, num_procs: int = 4) -> Path:
    out = work_dir / "golden"
    out.mkdir(exist_ok=True)
    # Re-write paramstest.txt with golden output folder; IndexerOMP reads
    # the file from cwd.
    write_paramstest(work_dir, out)
    cmd = [str(INDEXER_OMP_BIN), "paramstest.txt", "0", "1", str(n_seeds), str(num_procs)]
    print("$", " ".join(cmd), "  (cwd:", work_dir, ")")
    subprocess.run(cmd, cwd=work_dir, check=True)
    return out / "IndexBest.bin"


def run_torch_indexer(work_dir: Path, n_seeds: int, num_procs: int = 4) -> Path:
    out = work_dir / "midas"
    out.mkdir(exist_ok=True)
    # midas-index resolves its input binaries from OutputFolder (the layer
    # directory) — see midas-index 53e3d8ad. In a real pipeline the layer
    # dir holds Spots.bin/Data.bin/nData.bin/hkls.csv/SpotsToIndex.csv AND is
    # the OutputFolder; this synthetic harness writes them to the workdir
    # root, so mirror them into the OutputFolder to match production layout.
    # (The C indexer reads them from cwd instead, so its path is unaffected.)
    for fn in ("Spots.bin", "Data.bin", "nData.bin", "hkls.csv",
               "SpotsToIndex.csv"):
        shutil.copy2(work_dir / fn, out / fn)
    write_paramstest(work_dir, out)
    # Honor MIDAS_INDEX_DEVICE / MIDAS_INDEX_DTYPE if set; otherwise default
    # to cpu/float64 for byte-identical comparison vs the C reference.
    device = os.environ.get("MIDAS_INDEX_DEVICE", "cpu")
    dtype = os.environ.get("MIDAS_INDEX_DTYPE", "float64")
    cmd = [
        sys.executable, "-m", "midas_index",
        "paramstest.txt", "0", "1", str(n_seeds), str(num_procs),
        "--device", device, "--dtype", dtype,
    ]
    print("$", " ".join(cmd), "  (cwd:", work_dir, ")")
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    subprocess.run(cmd, cwd=work_dir, check=True, env=env)
    return out / "IndexBest.bin"


def compare_index_best(golden: Path, ours: Path, n_seeds: int) -> dict:
    """Decode 15-double records, compare per-seed."""
    g = np.fromfile(golden, dtype=np.float64).reshape(n_seeds, 15)
    o = np.fromfile(ours, dtype=np.float64).reshape(n_seeds, 15)

    report = {"n_seeds": n_seeds, "n_compared": 0, "per_seed": []}
    for i in range(n_seeds):
        gr = g[i]
        orow = o[i]
        # Skip empty slots (both zero -> no result on either side)
        if (gr == 0).all() and (orow == 0).all():
            continue
        report["n_compared"] += 1

        # IndexBest.bin layout (matches WriteBestMatchBin in IndexerOMP.c:1620):
        #   [0]      avg_ia
        #   [1..9]   orientation matrix flat
        #   [10..12] position
        #   [13]     n_t_spots (TOTAL theor spots predicted)
        #   [14]     n_matches (matched against obs)
        c_orient = gr[1:10].reshape(3, 3)
        m_orient = orow[1:10].reshape(3, 3)

        try:
            ang_rad, _ = misorientation_om(
                c_orient.flatten().tolist(),
                m_orient.flatten().tolist(),
                SPACE_GROUP,
            )
            miso_deg = math.degrees(float(ang_rad))
        except Exception:
            miso_deg = float("nan")

        report["per_seed"].append({
            "seed_idx": i,
            "miso_deg": miso_deg,
            "n_matches_c": int(gr[14]),
            "n_matches_m": int(orow[14]),
            "n_t_c": int(gr[13]),
            "n_t_m": int(orow[13]),
            "ia_c": float(gr[0]),
            "ia_m": float(orow[0]),
            "pos_diff": float(np.linalg.norm(gr[10:13] - orow[10:13])),
        })
    return report


def print_report(rep: dict) -> None:
    print()
    print("=" * 78)
    print(f"Compared {rep['n_compared']}/{rep['n_seeds']} non-empty seed slots")
    print("=" * 78)
    if rep["n_compared"] == 0:
        print("  No populated slots in either output.")
        return
    misos = [s["miso_deg"] for s in rep["per_seed"]]
    n_match_c = [s["n_matches_c"] for s in rep["per_seed"]]
    n_match_m = [s["n_matches_m"] for s in rep["per_seed"]]
    print(f"  miso_deg     mean={np.mean(misos):8.4f}  median={np.median(misos):8.4f}  max={np.max(misos):8.4f}")
    print(f"  n_matches    C: mean={np.mean(n_match_c):.2f}, ours: mean={np.mean(n_match_m):.2f}")
    n_within_1deg = sum(1 for m in misos if m < 1.0)
    print(f"  seeds within 1° of C orientation: {n_within_1deg}/{rep['n_compared']}")
    print()
    print("  Per-seed (first 10):")
    print(f"  {'idx':>4} {'miso_deg':>10} {'nM_C':>5} {'nM_m':>5} {'nT_C':>5} {'nT_m':>5} "
          f"{'IA_C':>9} {'IA_m':>9} {'posΔ':>10}")
    for s in rep["per_seed"][:10]:
        print(f"  {s['seed_idx']:>4} {s['miso_deg']:>10.4f} {s['n_matches_c']:>5} "
              f"{s['n_matches_m']:>5} {s['n_t_c']:>5} {s['n_t_m']:>5} "
              f"{s['ia_c']:>9.4f} {s['ia_m']:>9.4f} {s['pos_diff']:>10.3f}")


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-grains", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--identity", action="store_true",
                        help="Use n_grains copies of the identity rotation (sanity check)")
    parser.add_argument("--n-procs", type=int, default=4)
    parser.add_argument("--workdir", type=Path,
                        default=Path(__file__).parent / "ref_dataset")
    parser.add_argument("--keep-existing", action="store_true",
                        help="Don't wipe the workdir at start")
    args = parser.parse_args()

    if not INDEXER_OMP_BIN.exists():
        print(f"FATAL: IndexerOMP not found at {INDEXER_OMP_BIN}")
        return 2
    if not GETHKLLIST_BIN.exists():
        print(f"FATAL: GetHKLList not found at {GETHKLLIST_BIN}")
        return 2

    if args.workdir.exists() and not args.keep_existing:
        shutil.rmtree(args.workdir)
    args.workdir.mkdir(parents=True, exist_ok=True)
    work = args.workdir
    print(f"Working in {work}")

    # 1. hkls.csv
    print("\n[1/6] Generating hkls.csv via GetHKLList...")
    run_get_hkl_list(work)

    # 2. Random orientations
    if args.identity:
        print(f"\n[2/6] Using {args.n_grains} identity rotations (sanity check)...")
        R = identity_orientations(args.n_grains)
    else:
        print(f"\n[2/6] Sampling {args.n_grains} random orientations (seed={args.seed})...")
        R = random_orientations(args.n_grains, args.seed)

    # 3. Forward sim — use the indexer's own forward adapter so obs format
    # matches exactly what midas-index expects.
    print("\n[3/6] Forward-simulating spots via IndexerForwardAdapter...")
    hkls_cart, thetas, hkls_int = load_hkls_for_forward(work / "hkls.csv", RING_NUMBERS)
    # Re-pack to the (n_hkls, 7) and (n_hkls, 4) layout expected by the adapter.
    n_h = hkls_cart.shape[0]
    hkls_real_np = np.zeros((n_h, 7), dtype=np.float64)
    hkls_real_np[:, 0:3] = hkls_cart.numpy()
    hkls_real_np[:, 3] = hkls_int[:, 3].numpy().astype(np.float64)         # ring nr
    hkls_real_np[:, 4] = 0.0                                                # d_spacing (unused by adapter)
    hkls_real_np[:, 5] = thetas.numpy()                                     # theta in radians
    hkls_real_np[:, 6] = np.array([
        ring_radius_for_ring_nr(int(r)) for r in hkls_int[:, 3].numpy()
    ])
    hkls_int_np = hkls_int.numpy().astype(np.int64)

    # Build a temporary IndexerParams that mirrors paramstest.txt
    from midas_index import IndexerParams
    params = IndexerParams(
        Wavelength=WAVELENGTH, Distance=DISTANCE, px=PIXEL_SIZE,
        SpaceGroup=SPACE_GROUP, LatticeConstant=tuple(LATTICE),
        Rsample=RSAMPLE, Hbeam=HBEAM,
        StepsizePos=STEPSIZE_POS, StepsizeOrient=STEPSIZE_ORIENT,
        MarginOme=MARGIN_OME, MarginRad=MARGIN_RAD,
        MarginRadial=MARGIN_RADIAL, MarginEta=MARGIN_ETA,
        EtaBinSize=ETA_BIN_SIZE, OmeBinSize=OME_BIN_SIZE,
        ExcludePoleAngle=EXCLUDE_POLE_ANGLE,
        MinMatchesToAcceptFrac=MIN_MATCHES_TO_ACCEPT_FRAC,
        RingNumbers=list(RING_NUMBERS),
        RingRadii={r: ring_radius_for_ring_nr(r) for r in RING_NUMBERS},
        OmegaRanges=[(-180.0, 180.0)],
        BoxSizes=[(-2_000_000.0, 2_000_000.0, -2_000_000.0, 2_000_000.0)],
        UseFriedelPairs=USE_FRIEDEL_PAIRS,
        OutputFolder=str(work),
    )
    theor, valid = forward_via_adapter(R, params, hkls_real_np, hkls_int_np)
    obs = assemble_obs_from_theor(theor, valid)
    print(f"   {obs.shape[0]} synthetic spots from {args.n_grains} grains")

    # 4. Spots.bin + Data.bin/nData.bin + paramstest.txt + SpotsToIndex.csv
    print("\n[4/6] Writing inputs...")
    write_spots_bin(work, obs)
    write_data_bins(work, obs)
    sti = write_spots_to_index(work, obs, n_seeds=args.n_grains)
    n_seeds = sum(1 for _ in open(sti))
    print(f"   {n_seeds} seed spots -> SpotsToIndex.csv")

    # 5. Run both indexers
    print("\n[5/6] Running C IndexerOMP...")
    golden_bin = run_c_indexer(work, n_seeds, args.n_procs)

    print("\n[5/6] Running midas-index...")
    ours_bin = run_torch_indexer(work, n_seeds, args.n_procs)

    # 6. Compare
    print("\n[6/6] Comparing IndexBest.bin (golden vs midas-index)...")
    report = compare_index_best(golden_bin, ours_bin, n_seeds)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
