"""Step 1c positive control on the independent simulator: predicted-position integration vs blob-driven. Compared on the crossings the simulator actually RENDERED (res.truth), stratified by true intensity.
python -I p_sim_control.py [alpha list, e.g. 1.0 0.2 0.08]"""
import sys, warnings
from dataclasses import replace
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.reduce import DacModel, PolarizationModel
from midas_sx.refine import EIGHT_PI2
from midas_sx.workflow import ScanSetup, reduce_scan
NAME = sys.argv[1]; alphas = [float(x) for x in sys.argv[2:]] or [1.0, 0.3]
exp = sim.make_experiment(det_n=512, centre_rc=(255.5, 255.5), omega_start=-38.0, n_frames=304, dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bg0=40.0, bg1=0.0)
st = sim.structure_cu() if NAME == "cu" else sim.structure_s2_bilayer(); res = sim.simulate(st, exp, orientation_seed=3, n_int_max=2e5, target_hits=40_000, max_pairs=8_000_000)
cr = Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
setup = ScanSetup(exp.wavelength_A, exp.dist_mm, exp.pixel_mm, exp.centre_rc, exp.omega_start, exp.omega_step, dac=dac, pol=PolarizationModel(exp.pol_fraction))
T = res.truth.copy(); print("truth rows (rendered crossings):", len(T), "columns", list(T.columns))
def truth_of(o):
    """truth F2 of the rendered crossing nearest in (row, col) with the same hkl; NaN when the simulator rendered none within 3 px."""
    f = np.full(len(o), np.nan); w = np.full(len(o), np.nan)
    for i, r in enumerate(o.itertuples()):
        t = T[(T.h == r.h) & (T.k == r.k) & (T.l == r.l)]
        if len(t):
            d = np.hypot(t.row.to_numpy() - r.row, t.col.to_numpy() - r.col); j = int(np.argmin(d))
            if d[j] < 3.0: f[i] = t.F2.iloc[j]; w[i] = t.w_total.iloc[j]
    return f, w
for alpha in alphas:
    raw = replace(res, signal=res.signal * alpha).frames(seed=5)
    out = {}
    for mode in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[mode] = reduce_scan(raw, cr, res.U, setup, refine_orientation=False, integrate_predicted=mode)
    ob, op = out[False].obs.copy(), out[True].obs.copy()
    for o in (ob, op): o["ft"], o["wt"] = truth_of(o)
    ob, op = ob[np.isfinite(ob.ft)], op[np.isfinite(op.ft)]
    strong = ob[ob.F2 / ob.sigF2 > 20]; k = (strong.F2 * strong.ft).sum() / (strong.ft ** 2).sum()
    print(f"\n=== alpha {alpha}: blob-driven {len(ob)} rendered obs; predicted {len(op)} rendered obs ({int((~op.blob_found).sum())} without a blob); k={k:.4g}")
    m = op.merge(ob, on=["h", "k", "l"], suffixes=("_p", "_b")); m = m[(m.frame_p - m.frame_b).abs() < 1.5]; s = m[m.F2_b / m.sigF2_b > 20]
    r = s.F2_p / s.F2_b; print(f"strong common (blob I/sig>20): n={len(s)}  median F2_p/F2_b {np.median(r):.4f}  robust sd {1.4826 * np.median(np.abs(r - np.median(r))):.4f}")
    edges = np.quantile(k * op.ft, [0, 0.1, 0.25, 0.5, 0.75, 1.0]); edges[-1] *= 1.01
    z = lambda d: (d.F2 - k * d.ft) / d.sigF2
    print("z = (F2 - k F2true)/sigF2: mean +- se (n) per true-F2 bin; sum-ratio = sum F2 / sum (k F2true)")
    print(f"{'bin (k F2true)':>22s} | {'blob-driven':>20s} | {'pred, blob found':>20s} | {'pred, no blob':>20s} | {'pred, all':>20s} | sum-ratio pred all")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = lambda d: d[(k * d.ft >= lo) & (k * d.ft < hi)]
        cells = []
        for d in (sel(ob), sel(op)[sel(op).blob_found], sel(op)[~sel(op).blob_found], sel(op)):
            zz = z(d); cells.append(f"{zz.mean():+6.2f} +- {zz.std() / np.sqrt(max(len(zz), 1)):4.2f} ({len(zz):3d})" if len(zz) > 1 else "        -")
        sr = sel(op).F2.sum() / (k * sel(op).ft).sum()
        print(f"[{lo:9.3g},{hi:9.3g}) | " + " | ".join(f"{c:>20s}" for c in cells) + f" | {sr:6.3f}")
