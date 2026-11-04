"""PREREGISTER A31 analysis: python analyse_fdsg.py > results/fdsg_analysis.txt"""
import glob, json
import numpy as np, pandas as pd
import sys
EXT = len(sys.argv) > 1 and sys.argv[1] == "ext"
rows = [json.loads(l) for f in sorted(glob.glob("results/fdsg_ext_*.jsonl" if EXT else "results/fdsg_2*.jsonl")) for l in open(f)]
d = pd.DataFrame(rows)
print("# A31 analysis (find_domains with a wrong declared space group); seeds", d.seed.min(), "..", d.seed.max(), "n seeds", d.seed.nunique())
if "error" in d:
    print("errors:", int(d.error.notna().sum()))
conds = ["SG139", "SG87", "SG107", "SG119", "SG140", "SG71"] if EXT else ["SG139", "SG123", "SG129", "SG1"]
res = {}
for c in conds:
    g = d[d.cond == c]
    ok = int(g.ok.sum()); n = len(g)
    third = int((g.n_found >= 3).sum())
    nf = g.n_found.value_counts().sort_index().to_dict()
    sec = [sorted(r, key=lambda z: -z["n"])[1]["n"] for r in g.domains if len(r) > 1] if "domains" in g else []
    res[c] = dict(ok=ok, n=n, third=third)
    print(f"{c:6s} sg {int(g.sg.iloc[0]):3d}: success {ok}/{n} ({100*ok/n:.0f} %), domains found per seed {nf}, seeds with >= 3 domains {third}, "
          f"median reflections largest/second {np.median([r[0]['n'] for r in g.domains if len(r)]):.0f}/{np.median(sec) if sec else float('nan'):.0f}, median seconds {g.seconds.median():.1f}")
p = res["SG139"]["ok"] / res["SG139"]["n"]
print("\nReadings against A31:")
conf, refute = True, False
for c in conds[1:]:
    r = res[c]
    diff = r["ok"] - res["SG139"]["ok"]
    noise = 2 * np.sqrt(max(p * (1 - p), 1e-9) / r["n"] * 2)
    meets = (r["ok"] >= 23) and (r["third"] <= res["SG139"]["third"] + 2)
    fragile = r["ok"] < 15
    conf &= meets; refute |= fragile
    print(f"  {c}: success {r['ok']}/{r['n']}, difference to SG139 {diff:+d} seeds (null band +-{noise*r['n']:.1f} seeds), third-domain seeds {r['third']} vs {res['SG139']['third']}: "
          f"{'CONFIRM criteria met' if meets else 'not met'}{'; REFUTE criterion (< 15/25) met' if fragile else ''}")
if EXT:
    oks = [res[c]["ok"] for c in conds[1:]]
    pred = all(o <= 15 for o in oks) and max(oks) - min(oks) <= 2
    null = all(o >= 23 for o in oks)
    print("\nA31b: prediction (each <= 15/25 and equal within 2 seeds):", "HOLDS" if pred else "does not hold", "| null (each >= 23/25):", "HOLDS" if null else "does not hold", "| successes", dict(zip(conds[1:], oks)))
else:
    print("\nREGISTERED READING:", "REFUTED (fragile to a wrong space group)" if refute else "CONFIRMED (robust)" if conf else "INCONCLUSIVE")
print("Control SG139 must be >= 23/25 (A25 found 50/50 on seeds 0..49); if it is not, the chain of this script is suspect and the reading is void:", "OK" if res["SG139"]["ok"] >= 23 else "FAILS")
