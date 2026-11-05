"""FF Grains.csv health check (ff-hedm manual §8 style) + residual-by-confidence and duplicate check.
Usage: python ff_check.py Grains.csv [Rsample Hbeam [space-group=225]]
(Generalised from an earlier single-experiment kit: the space group for the duplicate test was hard-coded to 225.)"""
import sys, numpy as np
from scipy.spatial import cKDTree
from midas_stress.orientation import misorientation_om_batch
fn = sys.argv[1]; RS = float(sys.argv[2]) if len(sys.argv) > 2 else 1000; HB = float(sys.argv[3]) if len(sys.argv) > 3 else 1000
SG = int(sys.argv[4]) if len(sys.argv) > 4 else 225
g = np.genfromtxt(fn, comments="%")
OM, X, Y, Z = g[:, 1:10], g[:, 10], g[:, 11], g[:, 12]
a = g[:, 13]; dpos, dome, dang, rad, conf = g[:, 19], g[:, 20], g[:, 21], g[:, 22], g[:, 23]
e = g[:, 24:33]; rmse = g[:, 42]
r = np.hypot(X, Y)
print(f"{fn}: {len(g)} grains")
print(f"  position: r max {r.max():.0f} (Rsample {RS:.0f}), frac r>0.95*Rsample {np.mean(r>0.95*RS):.3f}; |Z| max {np.abs(Z).max():.0f} (Hbeam/2 {HB/2:.0f}), frac |Z|>0.95*Hbeam/2 {np.mean(np.abs(Z)>0.475*HB):.3f}")
print(f"  confidence median {np.median(conf):.3f}, frac >=0.9 {np.mean(conf>=0.9):.3f}, frac <0.6 {np.mean(conf<0.6):.3f}")
print(f"  lattice a median {np.median(a):.5f} p5-p95 {np.percentile(a,5):.5f}-{np.percentile(a,95):.5f}; max|eFab| {np.abs(e).max():.0f} ue, frac any |eFab|>=9999 {np.mean(np.any(np.abs(e)>=9999,1)):.3f}")
print(f"  GrainRadius median {np.median(rad):.1f} p5-p95 {np.percentile(rad,5):.1f}-{np.percentile(rad,95):.1f}; RMSErrorStrain median {np.median(rmse):.0f}")
print("  DiffPos / DiffAngle by confidence bin:")
for lo, hi in [(0,.6),(.6,.7),(.7,.8),(.8,.9),(.9,.95),(.95,1.01)]:
    m = (conf>=lo)&(conf<hi)
    if m.sum(): print(f"    C [{lo:.2f},{hi:.2f}): n={m.sum():5d}  DiffPos med {np.median(dpos[m]):7.1f} um  DiffAngle med {np.median(dang[m]):.3f} deg  radius med {np.median(rad[m]):.1f}")
# duplicates: pairs closer than 50 um AND misoriented < 1 deg
t = cKDTree(np.c_[X, Y, Z]); P = np.array(sorted(t.query_pairs(50.0)))
if len(P):
    m = np.degrees(misorientation_om_batch(OM[P[:,0]], OM[P[:,1]], SG))
    d = P[m < 1.0]
    print(f"  duplicate pairs (<50 um apart, <1 deg misoriented): {len(d)}; grains involved {len(np.unique(d))}")
