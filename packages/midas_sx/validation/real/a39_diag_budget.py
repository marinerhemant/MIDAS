"""A39 error budget: scatter of symmetry-equivalent observations about their merged mean, split into the counting-statistics part (sigI/I) and a systematic floor f (relative), by intensity, resolution, run, omega and detector region.
model per observation: var(log I_i - log <I>) = (sigI/I)^2 + f^2 ; f^2 estimated per bin as max(0, var_obs - mean (sigI/I)^2). Also saturation / count-rate screen from the blob peak counts.
python -I a39_diag_budget.py <out dir> <cif> <runs>"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from midas_hkls import Lattice, SpaceGroup
from midas_sx.merge import scale_groups
out, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
sg = SpaceGroup.from_hm("P 21 21 21"); ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
o = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
o, tab = scale_groups(o, sg)
key = np.array([sg.asu_representative(int(a), int(b), int(c)) for a, b, c in o[["h", "k", "l"]].to_numpy()]); o["u"] = [tuple(k) for k in key]
g = o.groupby("u"); o["n"] = g.F2.transform("size"); o = o[(o.n >= 2) & (o.F2 > 0)].copy()
# leave-one-out mean so a pair gives an unbiased deviation: dev_i = log F2_i - log mean(others)
s = g.F2.transform("sum").loc[o.index]; o["loo"] = (s - o.F2) / (o.n - 1); o = o[o.loo > 0].copy(); o["dev"] = np.log(o.F2 / o.loo); 
o["rel_sig"] = o.sigF2 / o.F2
o["d"] = [lat.d_spacing(*map(int, h)) for h in o[["h", "k", "l"]].to_numpy()]; o["om"] = ((o.omega + 180) % 360) - 180
# var(dev) = rel_sig_i^2 + rel_sig_others^2 (approx. rel_sig^2/(n-1)) + 2 f^2  -> estimate f
o["rel_sig"] = o.sigF2 / o.F2; o["pred_stat"] = o.rel_sig ** 2 * (1 + 1.0 / (o.n - 1))
print(f"{len(o)} observations of {o.u.nunique()} unique reflections with n>=2 (inter-run scales applied: {dict(zip(tab.group, tab.scale.round(3)))})")
def floor(sub):
    v = 1.4826 ** 2 * np.median(np.abs(sub.dev - np.median(sub.dev))) ** 2
    return pd.Series(dict(n=len(sub), robust_sd_dev=np.sqrt(v), stat_pred=np.sqrt(np.median(sub.pred_stat)), floor_f=np.sqrt(max(0.0, v - np.median(sub.pred_stat)) / 2)))
def by(col, bins, label):
    print(f"\n-- floor f (relative F2) by {label}"); print(o.groupby(pd.cut(o[col], bins), observed=True).apply(floor).round(3).to_string())
o["Isig"] = 1 / o.rel_sig
by("Isig", [0, 5, 10, 20, 50, 100, 1e9], "I/sigma"); by("d", [0.6, 0.75, 0.9, 1.1, 1.5, 2.5, 20], "d (A)"); by("om", np.arange(-160, 20, 30), "omega (deg)")
by("row", np.linspace(0, 1679, 6), "detector row"); by("col", np.linspace(0, 1475, 6), "detector col"); by("L", [0, 1.5, 2, 3, 5, 1e3], "Lorentz")
print("\n-- floor f by run"); print(o.groupby("run").apply(floor).round(3).to_string())
print("\noverall:", floor(o).round(3).to_dict(), " strong (I/sigma>50):", floor(o[o.Isig > 50]).round(3).to_dict())
print("\nmultiplicity of unique reflections (all runs, n>=1):"); oo = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")) for r in runs]); oo["u"] = [tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in oo[["h", "k", "l"]].to_numpy()]
print(oo.groupby("u").size().value_counts().sort_index().to_string())
# saturation / count rate screen from blob peak counts (full-run blobs reduced)
for r in runs:
    sp = pd.read_csv(os.path.join(out, r, "spots_blind.csv"), usecols=["peak_counts", "integrated"]); pc = sp.peak_counts
    print(f"{r}: blob peak counts max {pc.max():.0f}; >2e4: {(pc > 2e4).sum()}; >1e5: {(pc > 1e5).sum()}; >5e5: {(pc > 5e5).sum()}  (Pilatus cutoff 791927; 0.2 s exposure => 1e5 counts = 5e5 c/s/px)")
