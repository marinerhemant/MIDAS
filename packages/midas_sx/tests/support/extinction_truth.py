"""Independent primary-extinction truth for the simulator (PREREGISTER A21).

Per-reflection factor y_true(hkl) = [ int dy  I_TT(y; A) ] / (pi A),  A = pi t / Lambda_hkl,
with I_TT from midas_dfxm.takagi_taupin.solve_tt_laue (perfect crystal, plane wave, REAL susceptibilities, symmetric Laue),
|F| from gemmi (simulate.f2_truth), Lambda from the TT module's own extinction_length. It shares with the refiner only the physics
model (perfect block, symmetric Laue); no code from midas_hkls.extinction or the refiner.

The y-integral uses the closed form that solve_tt_laue reproduces to machine precision (checked in the selftest; the solver
itself costs ~17 ms per call, 2e6 calls per table). Tabulated on a grid of A and interpolated by a cubic spline (cached in
results/extinction_truth_table.npz).

  python extinction_truth.py selftest      # solver vs closed form, table vs independent Bessel form, interpolation error
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R_E_A = 2.8179403262e-5
TABLE = Path(tempfile.gettempdir()) / "midas_sx_extinction_truth_table.npz"          # cache of the truth table (regenerated if missing)
A_GRID = np.concatenate([np.linspace(0.0, 2.0, 41)[1:], np.linspace(2.0, 8.0, 61)[1:]])


def _solver_intensity(A: float, y: np.ndarray) -> np.ndarray:
    """|Dh|^2(y) from solve_tt_laue for a crystal of t/Lambda = A/pi (one exact depth step)."""
    import torch
    from midas_dfxm.takagi_taupin import solve_tt_laue
    lam_A, theta = 0.5, 20.0                                            # any values: only A and y matter for a real, lossless crystal
    chih = 1.0e-5 + 0j
    g = math.cos(math.radians(theta))
    Lam_um = lam_A * g / abs(chih) * 1e-4                               # TT module's Lambda = lambda cos(theta) / (C |chih|)
    t = A / math.pi * Lam_um
    out = np.empty(len(y))
    for i, yy in enumerate(y):
        _, Dh = solve_tt_laue(0j, chih, chih, wavelength_A=lam_A, theta_B_deg=theta, thickness_um=t, y=float(yy), n_depth=1)
        out[i] = float(Dh.abs() ** 2)
    return out


def _R_of_A(A: float) -> float:
    """Integral over the deviation y of the TT intensity, over its kinematic limit pi A.

    Uses the closed form sin^2(A sqrt(1+y^2))/(1+y^2) that ``solve_tt_laue`` reproduces (selftest 1: ~1e-15), because calling the
    solver ~2e6 times costs ~17 ms each. y = sinh(th) removes the infinite range and the slow 1/y^2 tail; Simpson on 1e6 nodes to
    th = 18 (the integrand ~ e^-th beyond), converged to <1e-7 for A <= 8 (selftest 2 checks it against (1/2A) int J0)."""
    from scipy.integrate import simpson
    th = np.linspace(0.0, 18.0, 1_000_001)
    return 2.0 * simpson(np.sin(A * np.cosh(th)) ** 2 / np.cosh(th), x=th) / (math.pi * A)


def build_table(force: bool = False):
    if TABLE.exists() and not force:
        d = np.load(TABLE)
        return d["A"], d["R"]
    R = np.array([_R_of_A(A) for A in A_GRID])
    TABLE.parent.mkdir(exist_ok=True)
    np.savez(TABLE, A=A_GRID, R=R)
    return A_GRID, R


def _spline():
    from scipy.interpolate import CubicSpline
    A, R = build_table()
    return CubicSpline(np.concatenate([[0.0], A]), np.concatenate([[1.0], R]))


def lambda_um(F2: np.ndarray, d_A: np.ndarray, V_A3: float, wavelength_A: float) -> np.ndarray:
    """Pendelloesung length from the TT module's convention, chih = r_e lambda^2 |F| / (pi V)."""
    from midas_dfxm.takagi_taupin import bragg_angle_deg, extinction_length
    out = np.empty(len(F2))
    for i, (f2, d) in enumerate(zip(F2, d_A)):
        th = bragg_angle_deg(float(d), wavelength_A)
        chih = R_E_A * wavelength_A ** 2 * math.sqrt(f2) / (math.pi * V_A3)
        out[i] = extinction_length(chih + 0j, chih + 0j, wavelength_A=wavelength_A, theta_B_deg=th)
    return out


def make_factor(structure, wavelength_A: float, t_um: float, sigma_ln: float = 0.0, n_quad: int = 41):
    """f2_factor(hkl, F2) for simulate.simulate. ``sigma_ln > 0``: log-normal thickness distribution with median ``t_um`` (T2)."""
    import gemmi
    cell = gemmi.UnitCell(*structure.cell)
    V = cell.volume
    spl = _spline()

    def factor(hkl, F2):
        d = np.array([cell.calculate_d(list(map(int, h))) for h in hkl])
        L = lambda_um(F2, d, V, wavelength_A)
        if t_um <= 0.0:
            return np.ones(len(F2))
        if sigma_ln <= 0.0:
            return spl(np.clip(math.pi * t_um / L, 0.0, A_GRID[-1]))
        z, w = np.polynomial.hermite_e.hermegauss(n_quad)               # E over ln t ~ N(ln t_um, sigma^2)
        w = w / w.sum()
        tt = t_um * np.exp(sigma_ln * z)
        return sum(wi * spl(np.clip(math.pi * ti / L, 0.0, A_GRID[-1])) for ti, wi in zip(tt, w))
    return factor


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        import simulate as sim
        from scipy.integrate import quad
        from scipy.special import j0
        # 1. solver vs the closed form it is supposed to reproduce, at random (A, y)
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(20):
            A, y = rng.uniform(0.1, 6.0), rng.uniform(0.0, 30.0)
            cf = math.sin(A * math.sqrt(1 + y * y)) ** 2 / (1 + y * y)
            worst = max(worst, abs(_solver_intensity(A, np.array([y]))[0] - cf))
        print(f"solver vs closed form, 20 random (A, y): max abs diff {worst:.2e}")
        # 2. table vs the independent Bessel identity, at the grid points
        A, R = build_table(force=True)
        ref = np.array([quad(j0, 0, 2 * a, limit=500)[0] / (2 * a) for a in A])
        print(f"table vs (1/2A) int J0 on {len(A)} grid points: max abs diff {np.abs(R - ref).max():.2e}")
        # 3. spline between grid points
        spl = _spline()
        aa = rng.uniform(0.0, 8.0, 200)
        ref2 = np.array([quad(j0, 0, 2 * a, limit=500)[0] / (2 * a) if a > 0 else 1.0 for a in aa])
        print(f"spline at 200 random A in [0,8]: max abs diff {np.abs(spl(aa) - ref2).max():.2e}")
        # 4. what the factors look like for S2 at 30 keV
        st = sim.structure_s2_bilayer()
        hkl = sim.enumerate_hkl(st, 0.8)
        F2 = sim.f2_truth(st, hkl)
        for t in (5.0, 10.0):
            y = make_factor(st, 0.4133, t)(hkl, F2)
            o = np.argsort(F2)[::-1][:5]
            print(f"t = {t} um: y(strongest 5) = {np.round(y[o], 3)}, F2-weighted mean(1-y) = {np.sum(F2 * (1 - y)) / F2.sum():.4f}")
