"""A39 check: with refine_wavelength OFF the refined geometry and orientation of run NN equal (exactly) the numbers an earlier run of a39_run.py stored (reduction.json, U.npy), and with it ON the
orientation-only fit gives eps +- sigma. Same cached spots, same start (U0.npy), same settings as a39_run.py.

    python -I a39_check_default_identical.py <dir with rNN/ (spots_blind.csv, U0.npy, U.npy, reduction.json)> <step1 dir with runNN/info.json> <rNN>"""
import json, os, sys, time
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from midas_hkls import Lattice
from midas_sx.conventions import Convention
from midas_sx.orient import index_and_refine
from a39_geom import swing_geometry
d, step1, run = sys.argv[1:4]
H = json.load(open(os.path.join(step1, "run" + run[1:], "info.json")))["header"]
old = json.load(open(os.path.join(d, run, "reduction.json"))); U_old = np.load(os.path.join(d, run, "U.npy"))
ss = gemmi.read_small_structure(os.path.join(os.path.dirname(os.path.abspath(d)), "1574527.cif")); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
g00, scan0 = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
# the SAME path as a39_run.py: ScanSetup -> setup.geometry (Lsd goes through mm and back, which is not always the identical float)
from midas_sx.workflow import ScanSetup
from midas_sx.reduce import DacModel, PolarizationModel
setup = ScanSetup(H["wavelength_A"], g00.lsd_um / 1000.0, H["pixel_mm"], (g00.bcz_px, g00.bcy_px), scan0.omega_start, scan0.omega_step, dac=DacModel(),
                  pol=PolarizationModel(fraction=H["extra"]["polarization"], plane_eta_deg=float(os.environ.get("POL_PLANE", 0.0))), tilts_deg=(g00.tx_deg, g00.ty_deg, g00.tz_deg))
g0, scan = setup.geometry((H["shape"][0], H["shape"][1]), H["n_frames"]), setup.scan(H["n_frames"])
sp = pd.read_csv(os.environ.get("SPOTS_CSV", os.path.join(d, run, "spots_blind.csv"))); U0 = np.load(os.path.join(d, run, "U0.npy"))
for flag in ((False,) if os.environ.get("ONLY_OFF") else (False, True)):
    t0 = time.time()
    o = index_and_refine(sp, lat, U0, g0, scan, refine_tilts=True, refine_wedge=True, refine_wavelength=flag)
    new = dict(tilts_deg=[o.geom.tx_deg, o.geom.ty_deg, o.geom.tz_deg], wedge_deg=o.geom.wedge_deg, centre_rc=[o.geom.bcz_px, o.geom.bcy_px], lsd_mm=o.geom.lsd_um / 1000, rms_px=o.rms_px, rms_frame=o.rms_frame, n_used=o.n_used)
    if not flag:
        same = {k: bool(np.array_equal(np.asarray(new[k], float), np.asarray(old[k], float))) for k in new}
        print(f"{run} refine_wavelength=False: identical to the stored run? {same}; U identical {np.array_equal(o.U, U_old)}; max |dU| {np.abs(o.U - U_old).max():.2e}; wavelength {o.geom.wavelength_A!r} == header {g0.wavelength_A!r}: {o.geom.wavelength_A == g0.wavelength_A} [{time.time()-t0:.0f} s]", flush=True)
    else:
        print(f"{run} refine_wavelength=True: eps {o.lambda_cell_scale:+.3e} +- {o.sigma_lambda_cell_scale:.2e}; converged {o.converged} ({o.reason}); rms {o.rms_px:.3f} px {o.rms_frame:.3f} fr (stored {old['rms_px']:.3f}, {old['rms_frame']:.3f}); Lsd {o.geom.lsd_um/1000:.3f} (stored {old['lsd_mm']:.3f}); "
              f"tilts {np.round([o.geom.tx_deg, o.geom.ty_deg, o.geom.tz_deg], 3)} wedge {o.geom.wedge_deg:+.3f} centre {o.geom.bcz_px:.2f},{o.geom.bcy_px:.2f}; n_used {o.n_used} [{time.time()-t0:.0f} s]", flush=True)
