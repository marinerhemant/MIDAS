"""Regression tests for the two robustness fixes in 0.8.1.

1. midas-stress imports and its NumPy API works with PyTorch absent (the
   package promises a torch-free NumPy path; it must not import torch eagerly).
2. Symmetry operators from make_symmetries are EXACT: unit quaternions giving
   orthonormal, det=+1 matrices whose set is closed under composition. Before
   the fix the 5-decimal tables gave ~3e-5 errors.
"""
import builtins
import importlib
import math
import subprocess
import sys

import numpy as np
import pytest

import midas_stress
from midas_stress.orientation import (
    make_symmetries, quat_to_orient_mat, misorientation_om, axis_angle_to_orient_mat,
)

_ALL_SG = [1, 2, 15, 74, 88, 142, 148, 149, 167, 176, 194, 206, 230]


@pytest.mark.parametrize("sg", _ALL_SG)
def test_symmetry_operators_are_exact(sg):
    n, sym = make_symmetries(sg)
    assert len(sym) == n
    oms = np.array([np.asarray(quat_to_orient_mat(q)).reshape(3, 3) for q in sym])
    for q, M in zip(sym, oms):
        assert abs(np.linalg.norm(q) - 1) < 1e-12          # unit quaternion
        assert abs(np.linalg.det(M) - 1) < 1e-12           # proper rotation
        assert np.abs(M @ M.T - np.eye(3)).max() < 1e-12   # orthonormal
    # no duplicate operators
    for i in range(n):
        for j in range(i + 1, n):
            assert np.abs(oms[i] - oms[j]).max() > 1e-6
    # closed under composition (a group)
    for A in oms:
        for B in oms:
            gap = np.abs(oms - (A @ B)).reshape(n, -1).max(axis=1).min()
            assert gap < 1e-12


def test_hex_and_cubic_counts():
    assert make_symmetries(194)[0] == 12   # HCP
    assert make_symmetries(225)[0] == 24   # FCC
    assert make_symmetries(229)[0] == 24   # BCC


def test_sigma3_twin_angle_exact():
    # 60 deg about <111> in cubic is the Sigma-3 twin; its (symmetry-reduced)
    # misorientation is exactly 60 deg. With rounded operators this drifted;
    # with exact operators it lands on 60 to machine precision.
    om = axis_angle_to_orient_mat([1, 1, 1], 60.0)
    ang, _ = misorientation_om(np.eye(3).ravel(), np.asarray(om).reshape(9), 225)
    assert abs(math.degrees(ang) - 60.0) < 1e-4


def test_imports_and_numpy_path_without_torch():
    """In a fresh interpreter with torch importing blocked, midas_stress must
    import and its NumPy misorientation path must run."""
    code = r"""
import builtins, sys
_real = builtins.__import__
def _blocked(name, *a, **k):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError("No module named 'torch'")
    return _real(name, *a, **k)
builtins.__import__ = _blocked
import numpy as np, math
import midas_stress
from midas_stress._optional import HAS_TORCH
assert HAS_TORCH is False
from midas_stress import misorientation_om, make_symmetries, axis_angle_to_orient_mat
om = axis_angle_to_orient_mat([1, 1, 1], 60.0)
ang, _ = misorientation_om(np.eye(3).ravel(), np.asarray(om).reshape(9), 225)
assert abs(math.degrees(ang) - 60.0) < 1e-4
n, sym = make_symmetries(194)
assert n == 12 and abs(np.linalg.norm(sym[1]) - 1) < 1e-12
# a torch-only entry point must raise a clear ModuleNotFoundError, not crash on None
try:
    midas_stress.fit_joint_d0_stiffness
    raise SystemExit('torch-only symbol did not raise')
except ModuleNotFoundError:
    pass
print('OK')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, f"torch-free import failed:\n{r.stdout}\n{r.stderr}"
    assert r.stdout.strip().endswith("OK")


def test_is_torch_helper_survives_missing_torch(monkeypatch):
    """_is_torch must return False (not raise) when torch is the stand-in."""
    from midas_stress import _optional
    if _optional.HAS_TORCH:
        pytest.skip("torch is installed; stand-in path exercised in the subprocess test")
    from midas_stress import orientation
    assert orientation._is_torch(np.zeros(3)) is False
