"""Generate random GrainsSim.csv files for FF-HEDM testing.

Creates N grains with random orientations, positions within the sample
volume, and lattice parameters with small random deviations from nominal.
"""
import numpy as np
from pathlib import Path


def random_orientation_matrices(n, rng=None):
    """Generate n uniformly random rotation matrices on SO(3).

    Uses random quaternions from isotropic Gaussian, normalized to unit length.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Random quaternions: 4 Gaussian components, normalized
    q = rng.standard_normal((n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Ensure positive scalar part for uniqueness
    q[q[:, 0] < 0] *= -1

    # Quaternion (w, x, y, z) to rotation matrix
    mats = np.zeros((n, 3, 3))
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    mats[:, 0, 0] = 1 - 2*(y*y + z*z)
    mats[:, 0, 1] = 2*(x*y - w*z)
    mats[:, 0, 2] = 2*(x*z + w*y)
    mats[:, 1, 0] = 2*(x*y + w*z)
    mats[:, 1, 1] = 1 - 2*(x*x + z*z)
    mats[:, 1, 2] = 2*(y*z - w*x)
    mats[:, 2, 0] = 2*(x*z - w*y)
    mats[:, 2, 1] = 2*(y*z + w*x)
    mats[:, 2, 2] = 1 - 2*(x*x + y*y)
    return mats


def generate_grains_csv(output_path, n_grains, lattice_params, rsample, hbeam,
                        beam_thickness, space_group=225, max_strain_pct=0.1,
                        seed=None):
    """Generate a GrainsSim.csv with random grains.

    Args:
        output_path: Path to write the CSV
        n_grains: Number of grains to generate
        lattice_params: Nominal [a, b, c, alpha, beta, gamma]
        rsample: Sample radius in um (positions within +-rsample)
        hbeam: Beam height in um (Z positions within +-hbeam/2)
        beam_thickness: Beam thickness for header
        space_group: Crystal space group number
        max_strain_pct: Maximum lattice parameter deviation (percent)
        seed: Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    a0, b0, c0, al0, be0, ga0 = lattice_params

    # Random orientations
    orient_mats = random_orientation_matrices(n_grains, rng)

    # Random positions within cylinder: X,Y within rsample disc, Z within hbeam
    positions = np.zeros((n_grains, 3))
    count = 0
    while count < n_grains:
        x = rng.uniform(-rsample, rsample)
        y = rng.uniform(-rsample, rsample)
        if x*x + y*y <= rsample*rsample:
            positions[count, 0] = x
            positions[count, 1] = y
            positions[count, 2] = rng.uniform(-hbeam/2, hbeam/2)
            count += 1

    # Lattice parameters with small random deviations
    frac = max_strain_pct / 100.0
    lattice = np.zeros((n_grains, 6))
    lattice[:, 0] = a0 * (1 + rng.uniform(-frac, frac, n_grains))
    lattice[:, 1] = b0 * (1 + rng.uniform(-frac, frac, n_grains))
    lattice[:, 2] = c0 * (1 + rng.uniform(-frac, frac, n_grains))
    lattice[:, 3] = al0 * (1 + rng.uniform(-frac, frac, n_grains))
    lattice[:, 4] = be0 * (1 + rng.uniform(-frac, frac, n_grains))
    lattice[:, 5] = ga0 * (1 + rng.uniform(-frac, frac, n_grains))

    # Compute Euler angles from orientation matrices
    # Using atan2 decomposition of ZXZ convention (MIDAS convention)
    eulers = np.zeros((n_grains, 3))
    for i in range(n_grains):
        m = orient_mats[i]
        if abs(m[2, 2]) < 1.0 - 1e-10:
            eulers[i, 1] = np.arccos(np.clip(m[2, 2], -1, 1))
            eulers[i, 0] = np.arctan2(m[2, 0], -m[2, 1])
            eulers[i, 2] = np.arctan2(m[0, 2], m[1, 2])
        else:
            eulers[i, 1] = 0.0 if m[2, 2] > 0 else np.pi
            eulers[i, 0] = np.arctan2(-m[0, 1], m[0, 0])
            eulers[i, 2] = 0.0
    eulers_deg = np.degrees(eulers)

    # Write GrainsSim.csv
    beam_center = 0.0  # Will be overwritten by ForwardSim
    with open(output_path, 'w') as f:
        # Header (8 lines)
        f.write(f"%NumGrains {n_grains}\n")
        f.write(f"%BeamCenter {beam_center:.6f}\n")
        f.write(f"%BeamThickness {beam_thickness:.6f}\n")
        f.write(f"%GlobalPosition 0.000000\n")
        f.write(f"%NumPhases 1\n")
        f.write(f"%PhaseInfo\n")
        f.write(f"%\tSpaceGroup:{space_group}\n")
        f.write(f"%\tLattice Parameter: {a0:.6f} {b0:.6f} {c0:.6f} "
                f"{al0:.6f} {be0:.6f} {ga0:.6f}\n")
        # Column header
        f.write("%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t"
                "X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\t"
                "DiffPos\tDiffOme\tDiffAngle\tGrainRadius\tConfidence\t"
                "eFab11\teFab12\teFab13\teFab21\teFab22\teFab23\t"
                "eFab31\teFab32\teFab33\t"
                "eKen11\teKen12\teKen13\teKen21\teKen22\teKen23\t"
                "eKen31\teKen32\teKen33\t"
                "RMSErrorStrain\tPhaseNr\tEul0\tEul1\tEul2\n")

        for i in range(n_grains):
            gid = i + 1
            om = orient_mats[i].flatten()
            pos = positions[i]
            lat = lattice[i]
            eul = eulers_deg[i]

            # Strain tensors (zeros for simulation)
            zeros9 = "\t".join(["0.000000"] * 9)

            f.write(f"{gid}\t")
            f.write("\t".join(f"{v:.6f}" for v in om))
            f.write(f"\t{pos[0]:.6f}\t{pos[1]:.6f}\t{pos[2]:.6f}\t")
            f.write(f"{lat[0]:.6f}\t{lat[1]:.6f}\t{lat[2]:.6f}\t")
            f.write(f"{lat[3]:.6f}\t{lat[4]:.6f}\t{lat[5]:.6f}\t")
            # DiffPos, DiffOme, DiffAngle, GrainRadius, Confidence
            f.write("0.000000\t0.000000\t0.000000\t300.000000\t1.000000\t")
            # eFab (9), eKen (9)
            f.write(f"{zeros9}\t{zeros9}\t")
            # RMSErrorStrain, PhaseNr, Euler angles
            f.write(f"0.000000\t1.000000\t{eul[0]:.6f}\t{eul[1]:.6f}\t{eul[2]:.6f}\t\n")

    print(f"Generated {n_grains} random grains -> {output_path}")
    return n_grains
