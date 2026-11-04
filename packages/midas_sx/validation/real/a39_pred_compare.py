"""A39-FLOOR step 1c readout, reference-free. Blob-driven (<base>/rNN) vs predicted-position (<pred>/rNN) reductions of the same runs.
 (1) counts, blob_found split, reject reasons; (2) common reflections: F2 ratio; (3) recovery of the strongest predicted-but-unmatched reflections of <base>/rNN/diag_unindexed_top2000.csv;
 (4) the no-blob observations: I/sigma distribution; (5) R_int, R_split (merge of runs {02,03} vs run 04), completeness (unique, n>=2), per-L floor via a39_diag_budget.
python -I a39_pred_compare.py <base dir> <pred dir> <runs, e.g. r02,r03,r04>"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from midas_hkls import SpaceGroup
from midas_sx.merge import merge_observations, r_int, scale_groups
base, pred, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
sg = SpaceGroup.from_hm("P 21 21 21")
rob = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
for r in runs:
    b = pd.read_pickle(os.path.join(base, r, "obs.pkl")); p = pd.read_pickle(os.path.join(pred, r, "obs.pkl")); t = pd.read_pickle(os.path.join(pred, r, "targets.pkl")); rj = pd.read_pickle(os.path.join(pred, r, "rejected.pkl"))
    print(f"\n===== {r}: blob-driven {len(b)} obs | predicted: {len(t)} targets ({int(t.blob_found.sum())} with a blob), {len(p)} kept ({int(p.blob_found.sum())} with a blob, {int((~p.blob_found).sum())} without)")
    print("   predicted rejects:", rj.reason.value_counts().to_dict())
    m = p.merge(b, on=["h", "k", "l"], suffixes=("_p", "_b")); m = m[(m.frame_p - m.frame_b).abs() < 2.0]
    ratio = m.F2_p / m.F2_b; s = m[m.F2_b / m.sigF2_b > 20]; rs = s.F2_p / s.F2_b
    print(f"   common obs {len(m)} of {len(b)} blob-driven; strong (I/sig>20) n={len(s)}: median F2_p/F2_b {np.median(rs):.4f}, robust sd {rob(rs):.4f}, |ratio-1|>5 %: {int((np.abs(rs - 1) > 0.05).sum())}")
    miss = b.merge(p[["h", "k", "l", "frame"]], on=["h", "k", "l"], how="left", suffixes=("", "_p")); lost = b[~b.index.isin(miss[(miss.frame - miss.frame_p).abs() < 2.0].index)] if False else None
    nb = p[~p.blob_found]; z = nb.I / nb.sigI
    print(f"   no-blob obs: n={len(nb)}, I/sigma median {z.median():+.2f}, fraction I<0 {np.mean(nb.I < 0):.2f}, fraction I/sig>3 {np.mean(z > 3):.3f}, >10 {np.mean(z > 10):.3f}; median L {nb.L.median():.2f} (blob-found median L {p[p.blob_found].L.median():.2f})")
    dpath = os.path.join(base, r, "diag_unindexed_top2000.csv")
    if os.path.exists(dpath):
        d = pd.read_csv(dpath).head(200); u = d[d.dist >= 4]
        rows = []
        for q in u.itertuples():
            c = p[(p.h == q.h) & (p.k == q.k) & (p.l == q.l) & ((p.frame - q.frame).abs() < 12)]
            if len(c):
                c = c.iloc[int(np.argmin((c.frame - q.frame).abs()))]; rows.append((q.h, q.k, q.l, round(q.integrated), round(c.I), round(c.sigI), round(c.I / c.sigI, 1), bool(c.blob_found), round(c.frame - q.frame, 1)))
            else:
                rj_ = rj[(rj.h == q.h) & (rj.k == q.k) & (rj.l == q.l) & ((rj.frame - q.frame).abs() < 12)]
                rows.append((q.h, q.k, q.l, round(q.integrated), None, None, None, None, rj_.reason.iloc[0] if len(rj_) else "not predicted"))
        df = pd.DataFrame(rows, columns=["h", "k", "l", "blob_integrated", "I_pred", "sigI_pred", "I/sig", "blob_found", "dframe|reason"])
        print(f"   strongest 200 blobs, {len(u)} unmatched at dist>=4 in the base diagnostic; the first 12 with their predicted-position observation:"); print(df.head(12).to_string(index=False))
        print(f"   of those {len(u)}: observed in predicted mode {int(df['I_pred'].notna().sum())}, with I/sig>10 {int((df['I/sig'].astype(float) > 10).sum())}; rejected/not predicted: {df[df['I_pred'].isna()]['dframe|reason'].value_counts().to_dict()}")
        s102 = p[(p.h.abs() == 1) & (p.k == 0) & (p.l.abs() == 2)][["h", "k", "l", "frame", "row", "col", "I", "sigI", "blob_found"]]
        print("   (+-1 0 +-2) in predicted mode:"); print(s102.round(1).to_string(index=False))
# merges
def load(d): 
    o = pd.concat([pd.read_pickle(os.path.join(d, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True); return o
for name, d in (("blob-driven", base), ("predicted", pred)):
    o = load(d); o, tab = scale_groups(o, sg); mg = merge_observations(o, sg)
    A = merge_observations(o[o.run.isin(runs[:-1])], sg).set_index(["h", "k", "l"]); B = merge_observations(o[o.run == runs[-1]], sg).set_index(["h", "k", "l"])
    com = A.index.intersection(B.index); fa, fb = A.loc[com].F2.to_numpy(), B.loc[com].F2.to_numpy()
    rsplit = np.abs(fa - fb).sum() / (0.5 * (fa + fb)).sum()
    print(f"\n[{name}] obs {len(o)}; unique {len(mg)}, n>=2 {int((mg.n >= 2).sum())}; R_int(scaled) {r_int(o, sg):.4f}; R_split({'+'.join(runs[:-1])} vs {runs[-1]}) {rsplit:.4f} on {len(com)} common reflections; run scales {dict(zip(tab.group, tab.scale.round(3)))}")
