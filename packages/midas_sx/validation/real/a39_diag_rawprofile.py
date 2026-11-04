"""Fork A / PREREG_lowangle_dframe.md raw check: for 12 offending strong spots (tth<20 or eta<0, dframe < -0.8 frames) and 8 controls (dframe ~ 0), read the raw CBF frames of run 02 around the spot, form the box-sum omega profile
(box = blob extent + 2 px, ring-median background), and compare the profile peak / centroid / half-max midpoint with the blob frame and the calculated crossing frame.
python -I a39_diag_rawprofile.py <out_p0 dir> <frames dir> <prefix> <run, e.g. r02>"""
import glob, os, sys
import numpy as np, pandas as pd
import fabio
d, fdir, prefix, run = sys.argv[1:5]
t = pd.read_pickle(os.path.join(d, "dframe_table.pkl")); t = t[t.run == run].copy()
t["dfr"] = (((t.omega_meas - t.omega + 180) % 360) - 180) / -0.2; t["frame_calc"] = t.frame - t.dfr
sp = pd.read_csv(os.path.join(d, run, "spots_blind.csv"))
t["key"] = list(zip(t.frame.round(4), t.row.round(3), t.col.round(3))); sp["key"] = list(zip(sp.frame.round(4), sp.row.round(3), sp.col.round(3)))
t = t.merge(sp.drop_duplicates("key")[["key", "extent_row", "extent_col", "extent_frame"]], on="key", how="left", suffixes=("", "_s"))
c = t[(t.integrated > 2e4) & (t.full_n_sub_peaks == 1) & (t.dfr.abs() < 4) & (t.n_frames < 60) & (t.row > 40) & (t.row < 1640) & (t.col > 40) & (t.col < 1430)]
bad = c[((c.tth < 20) | (c.eta < 0)) & (c.dfr < -0.8)].sort_values("integrated", ascending=False).head(12)
ctl = c[(c.dfr.abs() < 0.15) & (c.eta > 0) & (c.tth > 30)].sort_values("integrated", ascending=False).head(8)
sel = pd.concat([bad.assign(kind="offending"), ctl.assign(kind="control")]).reset_index(drop=True)
print(f"{len(bad)} offending + {len(ctl)} control spots; run {run}")
files = sorted(glob.glob(os.path.join(fdir, f"{prefix}*.cbf")))
need = set()
for r in sel.itertuples():
    f0 = int(r.frame - r.n_frames - 12); f1 = int(r.frame + r.n_frames + 12); need.update(range(max(f0, 0), min(f1, len(files) - 1) + 1))
cache = {}
HW = 14
for f in sorted(need):
    cache[f] = np.asarray(fabio.open(files[f]).data, float)
rows_out = []
for r in sel.itertuples():
    er = max(int(round(r.extent_row)) if r.extent_row == r.extent_row else 3, 2) + 2; ec = max(int(round(r.extent_col)) if r.extent_col == r.extent_col else 3, 2) + 2
    er, ec = min(er, 10), min(ec, 10)
    r0, c0 = int(round(r.row)), int(round(r.col))
    fl = np.arange(max(int(r.frame - r.n_frames - 12), 0), min(int(r.frame + r.n_frames + 12), len(files) - 1) + 1)
    P = []
    for f in fl:
        im = cache[f]; sub = im[r0 - er - 6:r0 + er + 7, c0 - ec - 6:c0 + ec + 7]
        yy, xx = np.mgrid[-er - 6:er + 7, -ec - 6:ec + 7]
        box = (np.abs(yy) <= er) & (np.abs(xx) <= ec); ring = ((np.abs(yy) > er + 1) | (np.abs(xx) > ec + 1)) & ~box
        good = sub >= 0
        bg = np.median(sub[ring & good]) if (ring & good).sum() > 10 else 0.0
        P.append(((sub[box & good] - bg).sum()))
    P = np.array(P); pk = int(np.argmax(P)); thr = 0.5 * P[pk]
    lo = pk
    while lo > 0 and P[lo - 1] > thr: lo -= 1
    hi = pk
    while hi < len(P) - 1 and P[hi + 1] > thr: hi += 1
    # sub-frame half-max edges by linear interpolation
    def edge(i, j):
        return i + (thr - P[i]) / (P[j] - P[i]) if P[j] != P[i] else i
    le = edge(lo - 1, lo) if lo > 0 else lo; he = edge(hi + 1, hi) if hi < len(P) - 1 else hi
    mid = 0.5 * (le + he) + fl[0]
    w = np.clip(P, 0, None); w[w < 0.05 * P[pk]] = 0; cen = (w * fl).sum() / w.sum()
    rows_out.append(dict(kind=r.kind, hkl=f"{int(r.h)} {int(r.k)} {int(r.l)}", tth=round(r.tth, 1), eta=round(r.eta), L=round(r.L, 1), blob=round(r.frame, 2), calc=round(r.frame_calc, 2), peak=int(fl[pk]), halfmax_mid=round(mid, 2), fwhm=round(he - le, 1),
                         centroid5=round(cen, 2), blob_minus_calc=round(r.frame - r.frame_calc, 2), prof_cen_minus_blob=round(cen - r.frame, 2), halfmax_minus_blob=round(mid - r.frame, 2), halfmax_minus_calc=round(mid - r.frame_calc, 2), skew_prof=round(((w * (fl - cen) ** 3).sum() / w.sum()) / (((w * (fl - cen) ** 2).sum() / w.sum()) ** 1.5), 2)))
o = pd.DataFrame(rows_out); pd.set_option("display.width", 250); print(o.to_string(index=False))
for k, g in o.groupby("kind"):
    print(f"{k}: n {len(g)}; |profile centroid - blob| < 0.3 frame: {(g.prof_cen_minus_blob.abs() < 0.3).sum()}; |half-max midpoint - blob| < 0.3: {(g.halfmax_minus_blob.abs() < 0.3).sum()}; median blob-calc {g.blob_minus_calc.median():+.2f}, median profile centroid - calc {(g.centroid5 - g.calc).median():+.2f}, median half-max mid - calc {g.halfmax_minus_calc.median():+.2f}")
o.to_csv(os.path.join(d, run, "rawprofile_check.csv"), index=False)
