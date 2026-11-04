"""PREREGISTER A37 step 2b (diagnostic of the step-2 pick): the ab-initio cell for the winning convention class, on all spots and on the strongest 2500, with the
conventional cell, the correct volume and the chance level. Reads out/step1 and out/step2's pick; prints, writes cell_ab_initio.json."""
import json, os, sys
CODE = os.environ["CODE"]
sys.path[:0] = [os.path.join(CODE, p) for p in ("midas_sx", "midas_hkls", "midas_defect", "midas_stress")]
import numpy as np, pandas as pd, torch
from midas_hkls.ab_initio import index_ab_initio
from midas_hkls.conventional import to_conventional
from midas_hkls import Lattice
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import Convention, transform_rc
from midas_sx.geometry import Scan, make_geometry

out = sys.argv[1]
spots = pd.read_csv(os.path.join(out, "spots.csv"))
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]
pick = json.load(open(os.path.join(out, "convention_pick.json")))
conv = Convention(pick["dihedral"], pick["omega_sign"])
CENTRE_RAW = (1021.1, 1025.4)


def g_of(sp):
    r, c = transform_rc(sp.row.to_numpy(), sp.col.to_numpy(), nr, nc, conv.dihedral)
    cr, cc = transform_rc(np.array([CENTRE_RAW[0]]), np.array([CENTRE_RAW[1]]), nr, nc, conv.dihedral)
    shape = (nr, nc) if conv.dihedral < 4 else (nc, nr)
    scan = Scan(conv.omega_sign * H["angle_start"], conv.omega_sign * H["angle_step"], H["n_frames"])
    geom = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=H["distance_mm"], pixel_mm=H["pixel_mm"], centre_rc=(float(cr[0]), float(cc[0])), shape=shape, scan=scan)
    qlab = pixel_to_qlab(r, c, geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_start + (sp.frame.to_numpy() + 0.5) * scan.omega_step), dtype=torch.float64)
    return qlab_to_qsample(qlab, om, 0.0).numpy()


def vol(cell):
    a, b, c, al, be, ga = cell
    ca, cb, cg = (np.cos(np.radians(x)) for x in (al, be, ga))
    return a * b * c * np.sqrt(max(1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg, 0))


print("convention:", conv.name)
rec = {}
for label, sp in (("strongest2500", spots.sort_values("integrated", ascending=False).head(2500)), ("all", spots)):
    g = g_of(sp)
    for tol in (0.10, 0.15):
        res = index_ab_initio(g, two_pi=True, tol=tol, min_reflections=40)
        print(f"\n[{label} n={len(g)} tol={tol}] {res}")
        if res.success:
            print(f"   volume {vol(res.cell):.1f} A^3; indexed {res.n_indexed}/{res.n_reflections}; diagnostics keys {list(res.diagnostics)[:8]}")
            try:
                cv = to_conventional(res.cell)
                print("   conventional:", cv)
            except Exception as e:
                print("   to_conventional failed:", e)
            rec[f"{label}_{tol}"] = dict(cell=[float(x) for x in res.cell], n_indexed=int(res.n_indexed), n=int(res.n_reflections), volume=float(vol(res.cell)))
json.dump(rec, open(os.path.join(out, "cell_ab_initio.json"), "w"), indent=1)
