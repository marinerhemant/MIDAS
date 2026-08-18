"""Regenerate the pymicro reference table used by ``tests/test_pymicro_crosscheck.py``.

``pymicro.crystal.microstructure.Orientation.dct_omega_angles`` solves the same
problem as :func:`midas_dct_tt.scan.bragg_flashes`, independently. We freeze its
output rather than calling it from the test suite because pymicro pins
``numpy<2`` and cannot be imported alongside this project's torch build -- so it
gets its own interpreter and its own installed tree.

Usage
-----
Install pymicro somewhere isolated, then run this file with that tree on the
path (NOT installed into the project environment)::

    python -m pip install --target /tmp/pymicro_env pymicro
    PYTHONPATH=/tmp/pymicro_env python examples/pymicro_reference.py

It prints a Python literal; paste it over ``REFERENCE`` in the test. Record the
pymicro version it came from in the test's docstring.

Conventions, for whoever regenerates this
-----------------------------------------
* pymicro's ``orientation_matrix()`` is Poulsen's ``g``: **sample -> crystal**.
  Our ``reference_orientation`` is the transpose, so the table stores ``g.T``.
* pymicro's rotation matrix is CCW about ``+z``. Compare against
  ``bragg_flashes(..., omega_sign=DCT_OMEGA_SIGN_CCW)``, not the 1-ID aero
  default.
* pymicro works in nm and keV; we work in Angstrom. ``lambda[nm] = 1.2398 /
  E[keV]``.

Last run: pymicro 0.6.1, agreement 1.1e-13 deg over all nine cases.
"""
import json

import numpy as np
from pymicro.crystal.lattice import HklPlane, Lattice
from pymicro.crystal.microstructure import Orientation

A_ANGSTROM = 3.6356          # fcc reference cell, matches midas_dfxm
LAMBDA_A = 0.172979          # ~71.7 keV, 1-ID / HEXM
EULERS = ([0.0, 0.0, 0.0], [35.0, 27.0, 14.0], [123.0, 61.0, 250.0])
HKLS = ((1, 1, 1), (2, 0, 0), (2, 2, 0))


def main():
    e_keV = 1.2398 / (LAMBDA_A / 10.0)
    lattice = Lattice.cubic(A_ANGSTROM / 10.0)      # pymicro works in nm

    print(f"# pymicro {getattr(__import__('pymicro'), '__version__', '?')}, "
          f"a = {A_ANGSTROM} A, lambda = {LAMBDA_A} A ({e_keV:.3f} keV)")
    print("REFERENCE = [")
    for euler in EULERS:
        o = Orientation.from_euler(euler)
        g_crystal_to_sample = o.orientation_matrix().T
        for hkl in HKLS:
            plane = HklPlane(*hkl, lattice=lattice)
            omegas = sorted(float(w) % 360.0 for w in o.dct_omega_angles(plane, e_keV))
            g = [[round(float(v), 12) for v in row] for row in g_crystal_to_sample]
            print(f"    dict(euler={euler}, hkl={hkl},")
            print(f"         g_crystal_to_sample={g},")
            print(f"         omegas={[repr(w) for w in omegas]}),".replace("'", ""))
    print("]")


if __name__ == "__main__":
    main()
