"""CPFEM -> DFXM black-box coupling on a REAL JAX-CPFEM field.

Loads the elastic deformation-gradient field Fe(r) from a JAX-CPFEM single-crystal
copper tensile solve (produced on sentosa by dev/paper/runs/.../export_Fe_sentosa.py),
runs the DFXM strain forward + inverse, and scores the recovery against the CP ground
truth. Demonstrates the DFXM stack on a physically-realistic crystal-plasticity field
(not a synthetic plant-and-recover) — the credibility upgrade for a methods paper.

Needs a developer clone: the field lives in the gitignored dev/ tree, so an installed
copy will report the missing file and exit.

Run:  export KMP_DUPLICATE_LIB_OK=TRUE; python -m midas_dfxm.examples.cpfem_coupling
"""
from __future__ import annotations

import os

import torch

from midas_dfxm.cpfem import cpfem_true_strain, load_cpfem_field, validate_dfxm_on_cpfem
from midas_dfxm.examples._figures import dev_path

FULL_SET = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 0), (0, 2, 2), (2, 0, 2), (2, 2, 2)]


def main():
    npz = dev_path("paper", "runs", "cpfem_singlecrystal_copper", "Fe_field.npz")
    if not os.path.exists(npz):
        print(f"CPFEM field not found: {npz}\n"
              "Regenerate on sentosa via export_Fe_sentosa.py and copy back.")
        return

    field = load_cpfem_field(npz)
    eps = cpfem_true_strain(field)
    mean = eps.mean(0)
    print(f"Real JAX-CPFEM Cu tensile field: {field.n_voxels} voxels")
    print(f"  elastic strain (Voigt) mean = {[round(float(x), 5) for x in mean]}")
    nu = -0.5 * (float(mean[0]) + float(mean[1])) / float(mean[2])
    print(f"  axial eps_zz = {float(mean[2]):+.4f}, transverse = {float(mean[0]):+.4f}"
          f"  -> apparent Poisson ratio ~ {nu:.2f}")

    print("DFXM strain inverse vs CP ground truth:")
    for noise in (0.0, 1e-4, 3e-4):
        out = validate_dfxm_on_cpfem(field, FULL_SET, noise_std=noise, seed=0)
        print(f"  noise={noise:.0e}: rms={out['rms_error']:.2e}  max={out['max_error']:.2e}")


if __name__ == "__main__":
    main()
