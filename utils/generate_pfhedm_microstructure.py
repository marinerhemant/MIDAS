#!/usr/bin/env python3
"""Generate a synthetic 2D polycrystalline microstructure in EBSD format
for pf-HEDM forward simulation with ForwardSimulationCompressed.

Creates:
  - microstructure.ebsd   : EBSD-format voxel file (x, y, z, Euler1, Euler2, Euler3)
  - positions.csv         : Scan Y-positions for multi-scan simulation
  - Parameters_pfhedm.txt : Parameter file for ForwardSimulationCompressed

Usage:
    python generate_pfhedm_microstructure.py [--outdir DIR] [--ngrains N]
                                              [--size S] [--step D]
                                              [--nscans NS] [--beamsize BS]
"""

import argparse
import os
import numpy as np
from scipy.spatial import KDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def random_euler_angles(rng):
    """Generate uniformly random Euler angles (Bunge convention) in degrees.

    Returns (phi1, Phi, phi2) in degrees.
    """
    phi1 = rng.uniform(0, 360)
    Phi = np.degrees(np.arccos(rng.uniform(-1, 1)))
    phi2 = rng.uniform(0, 360)
    return phi1, Phi, phi2


def generate_voronoi_microstructure(size_um, step_um, n_grains, seed=42):
    """Generate a 2D Voronoi-tessellated polycrystalline microstructure.

    Args:
        size_um: Side length of the square domain in micrometers.
        step_um: Grid spacing (mesh size) in micrometers.
        n_grains: Number of grains (Voronoi seeds).
        seed: Random seed for reproducibility.

    Returns:
        voxels: ndarray of shape (N, 6) with columns [x, y, z, e1, e2, e3]
                positions in µm, Euler angles in degrees.
        grain_ids: ndarray of shape (N,) with grain ID per voxel.
        grain_orientations: dict mapping grain_id -> (e1, e2, e3).
    """
    rng = np.random.default_rng(seed)
    half = size_um / 2.0

    # Grid points centered at (0, 0)
    coords_1d = np.arange(-half + step_um / 2.0, half, step_um)
    nx = len(coords_1d)
    print(f"Grid: {nx} x {nx} = {nx * nx} voxels")
    print(f"Domain: [{-half}, {half}] x [{-half}, {half}] µm")

    xx, yy = np.meshgrid(coords_1d, coords_1d, indexing='ij')
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Random grain seeds
    grain_seeds = rng.uniform(-half, half, size=(n_grains, 2))

    # Assign each grid point to nearest seed (Voronoi tessellation)
    tree = KDTree(grain_seeds)
    _, grain_ids = tree.query(grid_points)

    # Random orientation per grain
    grain_orientations = {}
    for gid in range(n_grains):
        grain_orientations[gid] = random_euler_angles(rng)

    # Build voxel array: x, y, z, euler1, euler2, euler3
    n_voxels = len(grid_points)
    voxels = np.zeros((n_voxels, 6))
    voxels[:, 0] = grid_points[:, 0]  # x
    voxels[:, 1] = grid_points[:, 1]  # y
    voxels[:, 2] = 0.0                # z (single 2D layer)

    for gid, (e1, e2, e3) in grain_orientations.items():
        mask = grain_ids == gid
        voxels[mask, 3] = e1
        voxels[mask, 4] = e2
        voxels[mask, 5] = e3

    return voxels, grain_ids, grain_orientations, nx


def write_ebsd_file(voxels, filepath):
    """Write microstructure in EBSD format for ForwardSimulationCompressed.

    Format:
        #EBSD
        x,y,z,euler1,euler2,euler3
        <data lines: x y z e1 e2 e3>
    """
    with open(filepath, 'w') as f:
        f.write("#EBSD\n")
        f.write("x,y,z,euler1,euler2,euler3\n")
        for row in voxels:
            f.write(f"{row[0]:.3f} {row[1]:.3f} {row[2]:.3f} "
                    f"{row[3]:.6f} {row[4]:.6f} {row[5]:.6f}\n")
    print(f"Written {len(voxels)} voxels to {filepath}")


def euler_to_ipf_color(phi1_deg, Phi_deg, phi2_deg):
    """Convert Euler angles to an IPF-Z color (cubic symmetry approximation).

    Uses a simplified mapping: orientation matrix applied to [001],
    then the resulting crystal direction is mapped to an RGB color
    in the standard stereographic triangle.
    """
    p1 = np.radians(phi1_deg)
    P = np.radians(Phi_deg)
    p2 = np.radians(phi2_deg)
    # Bunge convention orientation matrix
    c1, s1 = np.cos(p1), np.sin(p1)
    cP, sP = np.cos(P), np.sin(P)
    c2, s2 = np.cos(p2), np.sin(p2)
    # g matrix (crystal -> sample)
    g = np.array([
        [c1*c2 - s1*cP*s2, -c1*s2 - s1*cP*c2,  s1*sP],
        [s1*c2 + c1*cP*s2, -s1*s2 + c1*cP*c2, -c1*sP],
        [sP*s2,             sP*c2,              cP   ]
    ])
    # Sample Z direction [0,0,1] in crystal frame
    hkl = g.T @ np.array([0, 0, 1.0])
    # Force into fundamental zone (cubic): all positive, sort descending
    hkl = np.abs(hkl)
    hkl.sort()
    hkl = hkl[::-1]  # descending: h >= k >= l
    h, k, l = hkl
    norm = np.sqrt(h*h + k*k + l*l)
    if norm < 1e-10:
        return np.array([0.5, 0.5, 0.5])
    h, k, l = h/norm, k/norm, l/norm
    # Map to RGB: [001]=red, [011]=green, [111]=blue
    r = h - k
    g_val = k - l
    b = l * np.sqrt(3)
    total = r + g_val + b
    if total < 1e-10:
        return np.array([0.5, 0.5, 0.5])
    return np.array([r/total, g_val/total, b/total])


def save_orientation_map(voxels, grain_ids, grain_orientations, nx, filepath):
    """Save an IPF-Z orientation map as a PNG image."""
    print(f"Generating orientation map ({nx}×{nx})...")
    rgb = np.zeros((nx, nx, 3))
    # Precompute color per grain
    grain_colors = {}
    for gid, (e1, e2, e3) in grain_orientations.items():
        grain_colors[gid] = euler_to_ipf_color(e1, e2, e3)

    grain_ids_2d = grain_ids.reshape(nx, nx)
    for gid, color in grain_colors.items():
        mask = grain_ids_2d == gid
        rgb[mask] = color

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    half = voxels[:, 0].max() + 0.5  # approximate half-domain
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin='lower',
              extent=[-half, half, -half, half])
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(f'Synthetic Microstructure — {len(grain_orientations)} grains '
                 f'(IPF-Z coloring)')
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved orientation map: {filepath}")


def write_positions_csv(n_scans, size_um, filepath):
    """Write scan Y-positions equally spaced across the domain.

    Args:
        n_scans: Number of scan positions.
        size_um: Total scan range in µm (centered at 0).
        filepath: Output path.
    """
    half = size_um / 2.0
    positions = np.linspace(-half, half, n_scans)
    with open(filepath, 'w') as f:
        for pos in positions:
            f.write(f"{pos:.6f}\n")
    print(f"Written {n_scans} positions to {filepath}")
    print(f"  Range: [{positions[0]:.1f}, {positions[-1]:.1f}] µm, "
          f"step: {positions[1] - positions[0]:.2f} µm")


def write_parameter_file(filepath, outdir, n_scans, beam_size, fstem='pfhedm'):
    """Write a parameter file for pf-HEDM simulation and reconstruction.

    Based on FF_HEDM/Example/Parameters.txt with pf-HEDM modifications.
    Includes parameters needed by both ForwardSimulationCompressed and pf_MIDAS.py.
    """
    ebsd_path = os.path.join(outdir, "microstructure.ebsd")
    out_stem = os.path.join(outdir, f"{fstem}_sim")

    content = f"""###############################################################################
#                   MIDAS pf-HEDM Simulation Parameter File
#
#  Generated by generate_pfhedm_microstructure.py
#  Based on FF_HEDM/Example/Parameters.txt
###############################################################################

# =============================================================================
#  CALIBRANT / MATERIAL (Au, SG 225)
# =============================================================================
LatticeConstant 4.08 4.08 4.08 90 90 90
SpaceGroup 225

# =============================================================================
#  FORWARD SIMULATION (pf-HEDM mode)
# =============================================================================
InFileName {ebsd_path}
OutFileName {out_stem}
nScans {n_scans}
BeamSize {beam_size}
PeakIntensity 5000
GaussWidth 1
WriteSpots 0

# =============================================================================
#  DETECTOR GEOMETRY (same as FF test)
# =============================================================================
NrPixels 2048
px 200
Lsd 1000000.0000
BC 1022 1022
tx 0
ty 0
tz 0
p0 0
p1 0
p2 0
p3 0
p4 0
Wedge 0
RhoD 204800

# =============================================================================
#  BEAM / SAMPLE
# =============================================================================
Wavelength 0.22291
OmegaStart 180
OmegaEnd -180
OmegaStep -0.25
OmegaRange -180 180

# =============================================================================
#  RING SELECTION
# =============================================================================
RingThresh 1 10
RingThresh 2 10
RingThresh 3 10
RingThresh 4 10
RingThresh 5 10

# =============================================================================
#  SAMPLE GEOMETRY
# =============================================================================
Vsample 10000000
BeamThickness {beam_size}
GlobalPosition 100
Rsample 300
Hbeam 300

# =============================================================================
#  PHASE & TIMING
# =============================================================================
NumPhases 1
PhaseNr 1
tInt 0.3
tGap 0.15

# =============================================================================
#  IMAGE / MISC
# =============================================================================
ImTransOpt 0
UpperBoundThreshold 70000
NrFilesPerSweep 1
SimulationBatches 10

# =============================================================================
#  pf-HEDM RECONSTRUCTION PIPELINE (pf_MIDAS.py)
# =============================================================================
FileStem {fstem}
Ext .zip
StartNr 1
EndNr 1440
StartFileNrFirstLayer 1
OmegaFirstFile 180
OverAllRingToIndex 1
Padding 6
MaxAng 1
TolEta 1
TolOme 1
Width 2000
MinOmeSpotIDsToIndex -180
MaxOmeSpotIDsToIndex 180
BoxSize -1000000 1000000 -1000000 1000000
RingToIndex 1
RingToIndex 2
RingToIndex 3
RingToIndex 4
RingToIndex 5
PositionsFile {os.path.join(outdir, 'positions.csv')}
"""
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"Written parameter file: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outdir', default=None,
                        help='Output directory (default: FF_HEDM/Example/pfhedm_sim)')
    parser.add_argument('--ngrains', type=int, default=100,
                        help='Number of grains (Voronoi seeds)')
    parser.add_argument('--size', type=float, default=500.0,
                        help='Domain size in µm (square)')
    parser.add_argument('--step', type=float, default=1.0,
                        help='Grid step / mesh size in µm')
    parser.add_argument('--nscans', type=int, default=51,
                        help='Number of scan positions')
    parser.add_argument('--beamsize', type=float, default=15.0,
                        help='Beam size in µm')
    parser.add_argument('--scan_size', type=float, default=None,
                        help='Total scan range in µm (default: same as --size). '
                             'Set larger than --size to add empty padding at the edges.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Default output directory
    if args.outdir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        midas_home = os.path.dirname(script_dir)
        args.outdir = os.path.join(midas_home, 'FF_HEDM', 'Example', 'pfhedm_sim')

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Output directory: {args.outdir}")
    scan_size = args.scan_size if args.scan_size is not None else args.size
    print(f"Configuration: {args.ngrains} grains, {args.size}×{args.size} µm domain, "
          f"{args.step} µm step, {args.nscans} scans over {scan_size} µm, "
          f"{args.beamsize} µm beam")
    print()

    # 1. Generate microstructure
    voxels, grain_ids, grain_orientations, nx = generate_voronoi_microstructure(
        args.size, args.step, args.ngrains, seed=args.seed)

    # 2. Write EBSD file
    ebsd_path = os.path.join(args.outdir, 'microstructure.ebsd')
    write_ebsd_file(voxels, ebsd_path)

    # 3. Write positions.csv
    scan_size = args.scan_size if args.scan_size is not None else args.size
    positions_path = os.path.join(args.outdir, 'positions.csv')
    write_positions_csv(args.nscans, scan_size, positions_path)

    # 4. Write parameter file
    param_path = os.path.join(args.outdir, 'Parameters_pfhedm.txt')
    write_parameter_file(param_path, args.outdir, args.nscans, args.beamsize)

    # 5. Save orientation map
    map_path = os.path.join(args.outdir, 'orientation_map.png')
    save_orientation_map(voxels, grain_ids, grain_orientations, nx, map_path)

    # Summary
    n_voxels = len(voxels)
    unique_grains = len(set(grain_ids))
    print(f"\n{'='*60}")
    print(f"  Microstructure Summary")
    print(f"{'='*60}")
    print(f"  Voxels:        {n_voxels:,}")
    print(f"  Grains:        {unique_grains}")
    print(f"  Domain:        {args.size} × {args.size} µm")
    print(f"  Step:          {args.step} µm")
    print(f"  Scans:         {args.nscans} over {scan_size} µm")
    print(f"  Beam size:     {args.beamsize} µm")
    print(f"{'='*60}")
    print(f"\nTo run the simulation:")
    print(f"  cd {args.outdir}")
    print(f"  $MIDAS_HOME/FF_HEDM/bin/ForwardSimulationCompressed "
          f"Parameters_pfhedm.txt 8")


if __name__ == '__main__':
    main()
