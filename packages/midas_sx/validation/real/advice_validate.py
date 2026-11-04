"""Advice validation runner (PREREG_advice.md). Stages:

    python -I advice_validate.py sim <kind: base|coarse|wedge> <seed>          # simulate once, cache the signal volume
    python -I advice_validate.py scenario <C|Cw|C6x|C6|D1|D2|D3|D4|D5|D6|D7|D8>  # reduce + assess, write report
    python -I advice_validate.py eval                                          # apply the registered conditions to all reports

Everything lives under $ADVICE_WORK (default ./advice_work): sim_<kind>_<seed>/ (signal.npy + meta.pkl), red/<scenario>_<seed>.pkl, reports/<scenario>.{txt,json}.
Injections go into the SIGNAL (before Poisson noise): frames = Poisson(signal * factor + background).
"""
import json, os, pickle, sys, time, warnings
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.join(_sx, "validation"), os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from dataclasses import replace
from scipy.spatial.transform import Rotation as R

import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.advice import Acquisition, assess
from midas_sx.refine import EIGHT_PI2
from midas_sx.reduce import DacModel, PolarizationModel
from midas_sx.workflow import ScanSetup, reduce_scan

WORK = os.environ.get("ADVICE_WORK", os.path.join(os.getcwd(), "advice_work"))
for sub in ("red", "reports"):
    os.makedirs(os.path.join(WORK, sub), exist_ok=True)
SEEDS = (3, 4, 5)
SEEDS6 = (6, 7, 8)
ST = sim.structure_s2_bilayer()


def crystal_of(st):
    return Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)


CRYSTAL = crystal_of(ST)


HI = bool(os.environ.get("ADVICE_HI"))                  # Amendment 6: high-count bases (n_int_max = 2e7), report/reduction names prefixed H_
HI_KINDS = {"base": "hi", "coarse": "coarsehi", "wedge": "wedgehi"}
PHYS = {"hi": "base", "coarsehi": "coarse", "wedgehi": "wedge"}


def K(kind):
    return HI_KINDS.get(kind, kind) if HI else kind


def NAME(n):
    return ("H_" + n) if HI else n


def make_exp(kind):
    kind = PHYS.get(kind, kind)
    kw = dict(det_n=512, centre_rc=(255.5, 255.5), omega_start=-38.0, n_frames=304, dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bg0=40.0, bg1=0.0)
    if kind == "coarse":
        kw.update(omega_step=1.0, n_frames=76)
    return sim.make_experiment(**kw)


def sim_cached(kind, seed):
    d = os.path.join(WORK, f"sim_{kind}_{seed}")
    if os.path.exists(os.path.join(d, "meta.pkl")):
        meta = pickle.load(open(os.path.join(d, "meta.pkl"), "rb"))
        return sim.SimResult(meta["exp"], ST, meta["U"], np.load(os.path.join(d, "signal.npy")), meta["truth"], meta["scale"])
    os.makedirs(d, exist_ok=True)
    exp = make_exp(kind)
    t0 = time.time()
    if kind in ("wedge", "wedgehi"):
        W = np.radians(0.4)
        sim.AXIS = np.array([-np.sin(W), 0.0, np.cos(W)])        # rotation axis R_y(-W) e_z (the package convention)
    try:
        res = sim.simulate(ST, exp, orientation_seed=seed, n_int_max=2e7 if kind in PHYS else 2e5, target_hits=40_000, max_pairs=8_000_000)
    finally:
        sim.AXIS = np.array([0.0, 0.0, 1.0])
    np.save(os.path.join(d, "signal.npy"), res.signal.astype(np.float32))
    pickle.dump(dict(exp=res.exp, U=res.U, truth=res.truth, scale=res.scale), open(os.path.join(d, "meta.pkl"), "wb"))
    print(f"simulated {kind} seed {seed} in {time.time() - t0:.0f} s; {len(res.truth)} truth rows", flush=True)
    return res


def render(res, factor=None, extra=None, seed=5, cutoff=None):
    sig = res.signal.astype(np.float32)
    if factor is not None:
        sig = sig * np.asarray(factor, np.float32)[:, None, None]
    if extra is not None:
        sig = sig + extra
    fr = replace(res, signal=sig).frames(seed=seed)
    if cutoff is not None:
        fr = np.minimum(fr, np.float32(cutoff))
    return fr


def setup_of(exp):
    dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
    return ScanSetup(exp.wavelength_A, exp.dist_mm * 1.01, exp.pixel_mm, (exp.centre_rc[0] + 0.5, exp.centre_rc[1] - 0.3), exp.omega_start, exp.omega_step, dac=dac, pol=PolarizationModel(exp.pol_fraction))


def reduce_run(name, seed, frames, res, wedge=False, n_frames=None):
    path = os.path.join(WORK, "red", f"{name}_{seed}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    exp = res.exp
    setup = setup_of(exp)
    U0 = R.from_rotvec(np.radians(0.25) * np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)).as_matrix() @ res.U
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        red = reduce_scan(frames, CRYSTAL, U0, setup, allow_unconverged=True, refine_tilts=wedge, refine_wedge=wedge)
    print(f"reduced {name} seed {seed}: kept {red.counts['kept']} of {red.counts['input']}, orientation converged {red.orientation.converged if red.orientation else None}, "
          f"wedge {red.geom.wedge_deg:+.3f}, {time.time() - t0:.0f} s", flush=True)
    red.__dict__["header_geometry"] = setup.geometry(frames.shape[1:], frames.shape[0])
    pickle.dump(red, open(path, "wb"))
    return red


def run_assess(name, reds, acq, checks=None, **opts):
    hg = {k: getattr(v, "header_geometry", None) for k, v in reds.items()}
    if os.environ.get("ADVICE_DMIN") and "d_min" not in opts:
        opts["d_min"] = float(os.environ["ADVICE_DMIN"])          # Amendment 5: the simulator stops at 0.648 A in every scene
    name = NAME(name)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = assess(reds, CRYSTAL, acquisition=acq, header_geometry=hg, checks=checks, **opts)
    txt = rep.text()
    open(os.path.join(WORK, "reports", f"{name}.txt"), "w").write(txt)
    json.dump(rep.to_dict(), open(os.path.join(WORK, "reports", f"{name}.json"), "w"), indent=1, default=str)
    print(f"assess {name}: {len(rep.findings)} findings ({time.time() - t0:.0f} s)")
    for f in sorted(rep.findings, key=lambda f: f.severity):
        print(f"   [{f.severity}] {f.id}")
    return rep


ACQ = Acquisition(exposure_s=1.0, has_i0=False, has_calibrant=False, max_range_deg=76.0, dac=setup_of(make_exp('base')).dac)     # the cell aperture is declared (Amendment 2)


def get_reds(name, seeds, kind="base", frames_fn=None, wedge=False):
    reds = {}
    for s in seeds:
        res = sim_cached(K(kind), s)
        fr = frames_fn(res, s) if frames_fn else render(res)
        reds[f"r{s}"] = reduce_run(NAME(name), s, fr, res, wedge=wedge)
    return reds


def ring_extra(res, seed, red_clean):
    """200 extra spots on the circle 2theta = 12 deg (PREREG_advice D7)."""
    exp = res.exp
    rng = np.random.default_rng(1000 + seed)
    n_fr = res.signal.shape[0]
    r_px = exp.dist_mm * np.tan(np.radians(12.0)) / exp.pixel_mm
    pk = red_clean.spots["peak_counts"].to_numpy(float)
    amp = rng.choice(pk[np.isfinite(pk) & (pk > 0)], 200)
    extra = np.zeros(res.signal.shape, np.float32)
    sp, sf = 0.7, 2.0
    for a in amp:
        az = rng.uniform(0, 2 * np.pi)
        r0 = exp.centre_rc[0] + r_px * np.cos(az)
        c0 = exp.centre_rc[1] + r_px * np.sin(az)
        f0 = rng.uniform(6, n_fr - 6)
        fi = np.arange(int(f0 - 8), int(f0 + 9)); fi = fi[(fi >= 0) & (fi < n_fr)]
        ri = np.arange(int(r0 - 4), int(r0 + 5)); ri = ri[(ri >= 0) & (ri < exp.det_n)]
        ci = np.arange(int(c0 - 4), int(c0 + 5)); ci = ci[(ci >= 0) & (ci < exp.det_n)]
        g = (np.exp(-0.5 * ((fi - f0) / sf) ** 2)[:, None, None] * np.exp(-0.5 * ((ri - r0) / sp) ** 2)[None, :, None] * np.exp(-0.5 * ((ci - c0) / sp) ** 2)[None, None, :])
        extra[np.ix_(fi, ri, ci)] += (a * g).astype(np.float32)
    return extra


def main():
    cmd = sys.argv[1]
    if cmd == "sim":
        sim_cached(sys.argv[2], int(sys.argv[3]))
        return
    if cmd == "eval":
        from advice_validate_eval import evaluate
        evaluate(os.path.join(WORK, "reports"))
        return
    scen = sys.argv[2]
    N = 304
    if scen == "C":
        reds = get_reds("C", SEEDS)
        run_assess("C", reds, ACQ)
    elif scen == "Cw":
        reds = get_reds("Cw", SEEDS, wedge=True)
        run_assess("Cw", reds, ACQ, checks=["geometry_consistency"])
    elif scen == "C6x":
        get_reds("C", SEEDS6)
    elif scen == "C6hi":                                                  # Amendment 3: 6 runs with 100 x the counts
        reds = get_reds("C6hi", SEEDS + SEEDS6, kind="hi")
        run_assess("C6hi", reds, ACQ)
        run_assess("C6hi_dmin", reds, ACQ, d_min=0.66)                         # Amendment 4: the simulator stops at 0.648 A
    elif scen == "C6":
        reds = get_reds("C", SEEDS + SEEDS6)
        run_assess("C6", reds, ACQ)
        run_assess("C6_dmin", reds, ACQ, d_min=0.66)
    elif scen == "D1":
        ph = {3: 0.0, 4: 1.3, 5: 2.1}
        reds = get_reds("D1", SEEDS, frames_fn=lambda res, s: render(res, factor=1 + 0.25 * np.sin(2 * np.pi * 1.5 * np.arange(N) / N + ph[s])))
        run_assess("D1", reds, ACQ)
    elif scen == "D2":
        base = get_reds("C", (3,))["r3"]
        cut = 4.0 * float(np.nanmedian(base.spots["peak_counts"].to_numpy(float)))
        print(f"count cutoff = 4 x median blob peak = {cut:.0f}", flush=True)
        reds = get_reds("D2", SEEDS, frames_fn=lambda res, s: render(res, cutoff=cut))
        run_assess("D2", reds, replace(ACQ, count_cutoff=cut))
        json.dump(dict(cutoff=cut), open(os.path.join(WORK, "reports", "D2_cutoff.json"), "w"))
    elif scen == "D3":
        reds = get_reds("D3", SEEDS, kind="coarse")
        run_assess("D3", reds, ACQ)
    elif scen == "D4":
        reds = {}
        for s in SEEDS:
            res = sim_cached(K("base"), s)
            fr = render(res)[:160]
            reds[f"r{s}"] = reduce_run(NAME("D4"), s, fr, res)
        run_assess("D4_unset", reds, replace(ACQ, max_range_deg=None))
        run_assess("D4_set", reds, ACQ)
        run_assess("D4_set40", reds, replace(ACQ, max_range_deg=40.0))        # Amendment 2: the scan equals the environment limit
    elif scen == "D5":
        base = get_reds("C", (3,))
        run_assess("D5", base, ACQ)
    elif scen == "D6":
        reds = get_reds("D6", SEEDS, kind="wedge", wedge=True)
        run_assess("D6", reds, ACQ)
    elif scen == "D7":
        clean = get_reds("C", SEEDS)
        reds = get_reds("D7", SEEDS, frames_fn=lambda res, s: render(res, extra=ring_extra(res, s, clean[f"r{s}"])))
        run_assess("D7", reds, ACQ)
    elif scen == "D8":
        def fn(res, s):
            rng = np.random.default_rng(2000 + s)
            return render(res, factor=np.exp(0.10 * rng.standard_normal(N)))
        reds = get_reds("D8", SEEDS, frames_fn=fn)
        run_assess("D8", reds, ACQ)
    elif scen == "D8b":                                                   # Amendment 2: sigma 0.25 per frame
        def fn(res, s):
            rng = np.random.default_rng(3000 + s)
            return render(res, factor=np.exp(0.25 * rng.standard_normal(N)))
        reds = get_reds("D8b", SEEDS, frames_fn=fn)
        run_assess("D8b", reds, ACQ)
    else:
        raise SystemExit(f"unknown scenario {scen}")


if __name__ == "__main__":
    main()
