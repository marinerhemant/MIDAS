"""Does midas_hkls.primary_extinction_factor(t, Lambda) = tanh(t/Lambda)/(t/Lambda) match the angle-integrated perfect-crystal
symmetric-Laue reflectivity of midas_dfxm.takagi_taupin (closed form sin^2(A sqrt(1+y^2))/(1+y^2), A = pi t / Lambda)?
Ratio to kinematic: R(A) = int I dy / (pi A)  [= (1/2A) int_0^{2A} J0].  python extinction_convention_check.py"""
import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.integrate import quad
from scipy.special import j0
from midas_hkls.extinction import primary_extinction_factor

def R_numeric(A):
    """Direct y-integral of the TT closed form via y = sinh(th): int_0^inf sin^2(A cosh th)/cosh(th) dth, sin^2 = (1-cos)/2,
    the constant part integrated by Simpson to th = 18 (tail ~ e^-18); converged to <1e-7 at these A."""
    th = np.linspace(0, 18, 2_000_001)
    w = (np.sin(A*np.cosh(th))**2)/np.cosh(th)
    from scipy.integrate import simpson
    return 2*simpson(w, x=th)/(np.pi*A)

def R_bessel(A):
    return quad(j0, 0, 2*A, limit=500)[0]/(2*A)

lines = ["A=pi t/Lambda | R numeric (TT closed form) | R Bessel (1/2A int J0) | tanh(x)/x with x=t/Lambda (midas_hkls) | tanh(A)/A | 1-R numeric | 1-y_hkls"]
for A in (0.1, 0.3, 0.5, 1.0, 2.0, 3.0):
    t_over_L = A/np.pi
    yh = primary_extinction_factor(t_over_L, 1.0)
    lines.append(f"{A:4.2f} | {R_numeric(A):.5f} | {R_bessel(A):.5f} | {yh:.5f} | {np.tanh(A)/A:.5f} | {1-R_bessel(A):.5f} | {1-yh:.5f}")
print("\n".join(lines))
open("results/extinction_convention_check.txt","w").write("\n".join(lines)+"\n")
