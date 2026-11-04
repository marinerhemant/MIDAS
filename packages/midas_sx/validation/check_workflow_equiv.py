"""One-off: midas_sx.workflow.reduce_twin90_scan vs the validated A27 chain (run_pm2.py) on P11 seed 3: same frames, same wrong start."""
import sys, json, time
V = "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"
sys.path.insert(0, V + "/validation"); sys.path.insert(0, V)
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.reduce import DacModel, PolarizationModel
from midas_sx.refine import EIGHT_PI2
from midas_sx.workflow import ScanSetup, reduce_twin90_scan, refine_twin90_structure
from midas_sx.predict import enumerate_hkl

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
st = sim.structure_s2p(1.011)
f1, f2 = CACHE / f"{st.name}_dom1.npz", CACHE / f"{st.name}_dom2.npz"
z1 = np.load(f1)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(f1) + ".exp.json")).items()})
U1 = z1["U"]; sig = z1["signal"]; sig += np.load(f2)["signal"]
res = sim.SimResult(exp=exp, structure=st, U=U1, signal=sig, truth=pd.read_pickle(str(f1) + ".truth.pkl"), scale=float(z1["scale"]))
crystal = Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
raw = res.frames(seed=1000 + seed)
rng_a = np.random.default_rng(777 + seed)
ax = rng_a.normal(size=3); ax /= np.linalg.norm(ax)
U1s = Rot.from_rotvec(np.radians(0.3) * ax).as_matrix() @ U1
phi = rng_a.uniform(0, 2 * np.pi)
sign = rng_a.choice([-1.0, 1.0])
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
setup = ScanSetup(exp.wavelength_A, exp.dist_mm * (1 + 0.01 * sign), exp.pixel_mm, (exp.centre_rc[0] + 0.5 * np.cos(phi), exp.centre_rc[1] + 0.5 * np.sin(phi)),
                  exp.omega_start, exp.omega_step, dac=dac, pol=PolarizationModel(fraction=exp.pol_fraction))
d_min = max(exp.d_min_A, exp.wavelength_A / (2.0 * np.sin(np.radians(exp.tth_max_deg() / 2.0))))
t0 = time.time()
tred = reduce_twin90_scan(raw, crystal, U1s, setup, d_min=d_min, m0=0.3)
print("reduce_twin90_scan", round(time.time() - t0), "s; pair orientation used:", tred.pair_orientation_used, "rows", len(tred.obs), flush=True)
ref = np.load(V + f"/validation/results/pm2_merged/P11_seed{seed}.npz", allow_pickle=True)
O = tred.obs
print("saved rows", len(ref["h"]))
same_n = len(O) == len(ref["h"])
if same_n:
    for c in ("h", "k", "l", "p0", "p1", "p2"):
        print(c, "identical:", bool(np.array_equal(O[c].to_numpy(), ref[c])))
    for c in ("F2", "sigF2", "sep_px"):
        a, b = O[c].to_numpy(float), ref[c].astype(float)
        print(c, "max rel diff:", float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))))
    print("kind identical:", bool((O.kind.to_numpy() == ref["kind"]).all()))
fit = refine_twin90_structure(crystal, tred, gate_free=("O1a.U", "O1b.U", "O2.U", "O3.U"))
print("refine_twin90_structure: converged", fit.converged, "GooF", round(fit.goof, 3), "m", round(fit.twin_fraction, 4), "+-", round(fit.sigma_twin_fraction, 4), "rejected", len(fit.rejected))
