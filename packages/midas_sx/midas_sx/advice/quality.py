"""Quality checks: systematic floor, intensity drift, detector dynamic range, omega sampling, geometry consistency, contamination.

Every check measures a quantity of THIS data set and states the rule it was tested against (``THRESHOLDS`` in ``advice/__init__.py``). They read the ``Reduction`` objects of the runs
(``obs``, ``spots``, ``rejected``, ``counts``, ``geom``, ``orientation``) and the merge the user refined from (or ``merge_runs`` with its validated defaults).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..geometry import spot_directions
from ..merge import r_int, scale_groups, scale_smooth, systematic_floor
from . import THRESHOLDS, CheckUnavailable, Finding, register_check

ICE_IH_D = (3.897, 3.669, 3.439, 2.673, 2.249, 2.071, 1.917, 1.832)       # hexagonal ice Ih, d in A (lowest reflections)
REASONS_OTHER = ("frames_edge", "detector_edge", "masked_pixels", "no_background", "near_axis", "no_crossing", "factor_variation", "outside_opening")


def _th(name: str) -> float:
    return THRESHOLDS[name][0]


def _rule(name: str) -> str:
    return THRESHOLDS[name][1]


def _all_obs(ctx) -> pd.DataFrame:
    parts = []
    for name, red in ctx.runs.items():
        o = red.obs
        if o is not None and len(o):
            parts.append(o.assign(run=name))
    if not parts:
        raise CheckUnavailable("no observations")
    return pd.concat(parts, ignore_index=True)


def _merged(ctx):
    if ctx.merged_runs is None:
        from ..workflow import merge_runs
        ctx.merged_runs = merge_runs({k: v.obs for k, v in ctx.runs.items()}, ctx.space_group)
    return ctx.merged_runs


def _d_of(ctx, obs: pd.DataFrame) -> np.ndarray:
    cache: dict = {}
    out = np.empty(len(obs))
    for i, (h, k, l) in enumerate(zip(obs["h"].to_numpy(int), obs["k"].to_numpy(int), obs["l"].to_numpy(int))):
        t = (int(h), int(k), int(l))
        if t not in cache:
            cache[t] = float(ctx.lattice.d_spacing(*t))
        out[i] = cache[t]
    return out


def _spots_all(ctx, columns) -> pd.DataFrame:
    """The found blobs of every run with a ``run`` column, for linking observations to blob properties (key: run, frame, row, col rounded to 1e-6)."""
    parts = []
    for name, red in ctx.runs.items():
        sp = getattr(red, "spots", None)
        if sp is None or not all(c in sp.columns for c in columns):
            continue
        parts.append(sp[["frame", "row", "col", *columns]].assign(run=name))
    if not parts:
        raise CheckUnavailable("the spots tables lack " + ", ".join(columns))
    s = pd.concat(parts, ignore_index=True)
    for c in ("frame", "row", "col"):
        s[c] = s[c].round(6)
    return s.drop_duplicates(["run", "frame", "row", "col"])


def _link(ctx, obs: pd.DataFrame, columns) -> pd.DataFrame:
    s = _spots_all(ctx, columns)
    o = obs.copy()
    for c in ("frame", "row", "col"):
        o[c] = o[c].round(6)
    o = o.merge(s, on=["run", "frame", "row", "col"], how="left", suffixes=("", "_spot"))
    return o


def _binned(series: pd.Series, bins) -> pd.Series:
    return pd.cut(series, bins)


def _fmt(x, nd=3):
    return float(np.round(x, nd)) if np.isfinite(x) else None


# ---------------------------------------------------------------------------------------------------------------------------------------------
@register_check("systematic_floor")
def check_systematic_floor(ctx) -> List[Finding]:
    raw = _all_obs(ctx)
    m = _merged(ctx)
    scaled = m.obs
    if "run" not in scaled.columns:
        scaled = scaled.assign(run="all")
    tab = systematic_floor(scaled, ctx.space_group)
    if tab.empty or not np.isfinite(tab["floor"].iloc[0]):
        raise CheckUnavailable("fewer than 30 usable observations of reflections seen at least twice")
    f, cnt, n = float(tab["floor"].iloc[0]), float(tab["counting"].iloc[0]), int(tab["n"].iloc[0])
    plain, _ = scale_groups(raw, ctx.space_group)
    f_plain = float(systematic_floor(plain, ctx.space_group)["floor"].iloc[0])
    dims: Dict[str, pd.Series] = {}
    if "L" in scaled.columns:
        dims["Lorentz factor"] = systematic_floor(scaled, ctx.space_group, by=pd.cut(scaled["L"], [0, 1.5, 2, 3, 5, 1e9]), min_obs=50)["floor"]
    dd = _d_of(ctx, scaled)
    dims["resolution d (A)"] = systematic_floor(scaled, ctx.space_group, by=pd.cut(pd.Series(dd), [0, 0.75, 0.9, 1.1, 1.5, 2.5, 1e3]), min_obs=50)["floor"]
    if scaled["run"].nunique() > 1:
        dims["run"] = systematic_floor(scaled, ctx.space_group, by="run", min_obs=50)["floor"]
    dep, spread = None, 1.0
    shown = {}
    for name, ser in dims.items():
        v = ser.dropna()
        shown[name] = {str(k): _fmt(x) for k, x in v.items()}
        if len(v) >= 2 and v.min() > 0.005:
            r = float(v.max() / v.min())
            if r > spread:
                dep, spread = (name, str(v.idxmax())), r
    ratio = f / cnt if cnt > 0 else float("inf")
    meas = dict(floor_f=_fmt(f), counting=_fmt(cnt), ratio=_fmt(ratio, 2), usable_observations=n, floor_with_run_scales_only=_fmt(f_plain), merge_scale=m.scale, floor_by=shown)
    rule = (f"limiting when f >= {_th('floor_abs'):.0%} and f / counting >= {_th('floor_ratio'):g}; note when only the ratio is exceeded. "
            f"({_rule('floor_ratio')}; {_rule('floor_abs')})")
    ev = "merge.systematic_floor on the merged observations (leave-one-out log deviation of symmetry equivalents, robust sd, minus the counting prediction); by L, d and run in measured"
    scope = "sees only what differs between equivalents; an error common to all equivalents (wrong cell in the Lorentz factor, absorption identical in every orientation) is invisible; says nothing about accuracy against a structure"
    if f >= _th("floor_abs") and ratio >= _th("floor_ratio"):
        where = ""
        hint = ""
        if dep is not None and spread >= 1.5:
            where = f" It is largest for {dep[0]} = {dep[1]} ({spread:.1f}x the smallest bin)."
            hint = {"Lorentz factor": " High-L reflections are long in omega and near the rotation axis or the beam: check the rotation-axis alignment (wedge), the crystal centring on the axis and whether neighbouring spots overlap; a finer step helps separate them.",
                    "resolution d (A)": " A dependence on resolution points at the crystal (mosaicity, extinction on the strong low-order reflections, absorption) rather than at the instrument.",
                    "run": " A dependence on the run points at something that changed between runs: crystal centring or damage, beam stability, a different mounting."}.get(dep[0], "")
        return [Finding("systematic_floor", "limiting",
                        f"equivalent reflections disagree by {f:.1%} where counting statistics predict {cnt:.1%}: the data are limited by a systematic error, not by counts",
                        meas, rule, ev,
                        "MORE EXPOSURE WILL NOT HELP: the disagreement is not counting noise." + where + hint
                        + " More redundancy in DIFFERENT orientations lets the merge average the systematic error down.",
                        None, scope)]
    if ratio >= _th("floor_ratio"):
        return [Finding("systematic_floor", "note", f"a systematic floor of {f:.1%} exists ({ratio:.1f}x the counting prediction) but is small in absolute terms",
                        meas, rule, ev, "No acquisition change is indicated by this alone.", None, scope)]
    return [Finding("systematic_floor", "note", f"equivalents agree to counting statistics (floor {f:.1%}, counting {cnt:.1%})", meas, rule, ev,
                    "The merged precision is counting-limited: more exposure or more redundancy improves it.", None, scope)]


# ---------------------------------------------------------------------------------------------------------------------------------------------
@register_check("intensity_drift")
def check_intensity_drift(ctx) -> List[Finding]:
    raw = _all_obs(ctx)
    if "frame" not in raw.columns:
        raise CheckUnavailable("observations have no frame column")
    sg = ctx.space_group
    g, _ = scale_groups(raw, sg)
    s, tab = scale_smooth(raw, sg)
    if tab["n_shared"].sum() < 100 or (tab["n_knots"] <= 1).all():
        raise CheckUnavailable("too little redundancy for a smooth scale (fewer than 100 shared observations or one knot per run)")
    rg, rs = r_int(g, sg), r_int(s, sg)
    gain = (rg - rs) / rg if rg > 0 else 0.0
    pl = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        p = raw.copy()
        for r in p["run"].unique():
            mk = (p["run"] == r).to_numpy()
            p.loc[mk, "frame"] = rng.permutation(p.loc[mk, "frame"].to_numpy())
        ps, _ = scale_smooth(p, sg)
        pl.append(r_int(ps, sg))
    rp = float(np.mean(pl))
    pgain = (rg - rp) / rg if rg > 0 else 0.0
    rng_log = float(tab.loc[tab["n_knots"] > 1, "log_k_range"].max())
    shape = {}
    for r in s["run"].unique():
        sub = s[s["run"] == r]
        b = pd.cut(sub["frame"], 8)
        t = sub.groupby(b, observed=True)["k_smooth"].mean()
        shape[str(r)] = [float(np.round(v / t.mean(), 3)) for v in t.to_numpy()]
    i0 = ctx.acq.has_i0
    meas = dict(log_scale_range_per_run={str(r): _fmt(v, 3) for r, v in zip(tab["group"], tab["log_k_range"])}, rint_run_scales_only=_fmt(rg, 4), rint_smooth_scale=_fmt(rs, 4),
                rint_gain=_fmt(gain, 3), rint_permuted_frames=_fmt(rp, 4), placebo_gain=_fmt(pgain, 3), k_over_frame_bins=shape, i0_recorded=i0)
    rule = (f"drift when the log scale range >= {_th('drift_log_range'):g} and the R_int gain >= {_th('drift_rint_gain'):.0%}, and the permuted-frame placebo gains less than "
            f"{_th('placebo_gain_fraction'):.0%} of the real gain ({_rule('drift_log_range')}; {_rule('drift_rint_gain')})")
    ev = "merge.scale_smooth (8 knots per run over the frame index, fitted from equivalents only) against merge.scale_groups; placebo = the same fit with the frame labels permuted within each run (5 seeds)"
    scope = "the scale absorbs anything smooth in the frame index that multiplies a whole run (beam or sample-volume drift, centring, absorption by a symmetric environment); it cannot say which. Needs equivalents spread over each run's whole range"
    big_range, big_gain = rng_log >= _th("drift_log_range"), gain >= _th("drift_rint_gain")
    if not (big_range or big_gain):
        return []
    if gain > 0 and pgain > _th("placebo_gain_fraction") * gain:
        return [Finding("intensity_drift", "note", "a smooth scale lowers R_int but the permuted-frame placebo does nearly as well: not distinguishable from flexibility",
                        meas, rule, ev, "No drift is claimed. More redundancy over each run's whole range would let the test decide.", None, scope)]
    inferred = "The scale is INFERRED from equivalents, not measured: " if not i0 else "I0 is recorded: compare the fitted scale with it; if they disagree the variation is not the incident beam (crystal centring, absorption, illumination of a larger or smaller volume). "
    act = (inferred + "check that the crystal is centred on the rotation axis for the whole rotation and that the beam covers (or is smaller than) the crystal consistently; record the incident intensity "
           "(and, if possible, the transmitted beam) for every frame so the scale can be measured; if the crystal is much larger than the beam, a centring error shows up as exactly this kind of slow variation.")
    sev = "limiting" if (big_range and big_gain) else "note"
    return [Finding("intensity_drift", sev, f"the intensity scale varies by {np.expm1(rng_log):.0%} within a run (log range {rng_log:.2f}); a smooth per-run scale cuts R_int from {rg:.1%} to {rs:.1%}",
                    meas, rule, ev, act, f"on THIS data a smooth scale lowers R_int from {rg:.1%} to {rs:.1%} (in sample; placebo with permuted frames: {rp:.1%}); a held-out check is "
                    "real/a39_cv_robust.py in the validation directory", scope)]


# ---------------------------------------------------------------------------------------------------------------------------------------------
@register_check("dynamic_range")
def check_dynamic_range(ctx) -> List[Finding]:
    m = _merged(ctx)
    obs = m.obs if "run" in m.obs.columns else _all_obs(ctx)
    lk = _link(ctx, obs, ["peak_counts"])
    ok = lk["peak_counts"].notna()
    if ok.mean() < 0.5:
        raise CheckUnavailable("fewer than half of the observations link to a blob with peak_counts")
    cutoff = ctx.acq.count_cutoff
    guess = cutoff is None
    limit = 1e5 if guess else _th("peak_count_fraction") * cutoff
    hi = (lk["peak_counts"] > limit) & ok
    n_hi, n = int(hi.sum()), int(ok.sum())
    blobs_1e5 = {str(r): int((np.asarray(getattr(red.spots, "peak_counts", [])) > 1e5).sum()) for r, red in ctx.runs.items() if getattr(red, "spots", None) is not None and "peak_counts" in red.spots}
    pk = lk["peak_counts"].to_numpy(float)
    pmax = float(np.nanmax(pk[ok.to_numpy()]))
    clip_level = 0.99 * cutoff if not guess else pmax * (1.0 - 1e-3)
    clipped = (lk["peak_counts"] >= clip_level) & ok
    n_clip = int(clipped.sum())
    clip_blobs = {str(r): int((np.asarray(red.spots["peak_counts"]) >= clip_level).sum()) for r, red in ctx.runs.items() if getattr(red, "spots", None) is not None and "peak_counts" in red.spots}
    n_clip_blobs = sum(clip_blobs.values())
    saturated = n_clip_blobs >= (1 if not guess else 3) and n_clip / n >= _th("peak_obs_fraction") and (not guess or pmax >= _th("clip_min_counts"))
    above_cutoff = int((lk["peak_counts"] > cutoff).sum()) if not guess else 0
    ctx.numbers["dynamic_range"] = dict(observations_above_limit=n_hi, observations_linked=n, blobs_above_1e5_per_run=blobs_1e5, blobs_clipped_per_run=clip_blobs, observations_clipped=n_clip)      # reported also when no finding is raised
    out: List[Finding] = []
    if saturated:
        meas = dict(clip_level_counts=_fmt(clip_level, 0), cutoff=cutoff, cutoff_is_a_guess=guess, blobs_clipped_per_run=clip_blobs, observations_clipped=n_clip, observations_linked=n, fraction_clipped=_fmt(n_clip / n, 4), max_peak_counts=_fmt(pmax, 0))
        what = ((f"peak pixels at or above 99 % of the detector's count cutoff ({cutoff:.6g})" if not above_cutoff else f"peak pixels at or above 99 % of the declared count cutoff ({cutoff:.6g}), some above it: check the cutoff")
                 if not guess else f"peak pixels within 0.1 % of the maximum peak value {pmax:.6g} ({n_clip_blobs} blobs), the signature of a clipped detector")
        out.append(Finding("dynamic_range", "limiting", f"saturated: {n_clip} of {n} observations have {what}", meas,
                           (f"limiting when at least {_th('peak_obs_fraction'):.0%} of the observations have a peak pixel at or above 99 % of the declared cutoff, or, with no cutoff, at least 3 blobs tie at the maximum peak value "
                            f"within 1e-3 relative and a peak of at least {_th('clip_min_counts'):.0e} counts (a continuous peak distribution has a unique maximum). ({_rule('peak_obs_fraction')})"),
                           "spots.peak_counts of the blob behind each observation (linked by run, frame, row, col)",
                           "a clipped peak pixel means the integrated intensity of these reflections is a lower bound and cannot be corrected afterwards: attenuate or shorten the exposure so that the strongest reflections stay below the cutoff, or add a short exposure for them",
                           None, "clipping is seen in the blob finder's peak pixel; a count-rate loss below the cutoff is not seen by this test"))
    if above_cutoff:
        out.append(Finding("dynamic_range", "note", f"{above_cutoff} observations have a peak pixel above the declared cutoff {cutoff:.6g}", dict(cutoff=cutoff, observations_above_cutoff=above_cutoff),
                           "a detector cannot register more than its cutoff: the declared cutoff and the data disagree", "spots.peak_counts of the linked blobs",
                           "check the count_cutoff passed in Acquisition (header value, gain, or a summed image)", None, "none"))
    if n_hi == 0:
        return out
    peak = float(lk["peak_counts"].max())
    rate = peak / ctx.acq.exposure_s if ctx.acq.exposure_s else None
    flag = pd.Series(np.where(hi, "above", "below"), index=lk.index)
    tab = systematic_floor(lk, ctx.space_group, by=flag, min_obs=15)
    f_hi = float(tab["floor"].get("above", np.nan))
    f_lo = float(tab["floor"].get("below", np.nan))
    eff = f_hi - f_lo if np.isfinite(f_hi) and np.isfinite(f_lo) else None
    frac = n_hi / n
    meas = dict(limit_counts=_fmt(limit, 0), limit_is_a_guess=guess, cutoff=cutoff, observations_above=n_hi, fraction=_fmt(frac, 4), max_peak_counts=_fmt(peak, 0),
                max_rate_counts_per_s=_fmt(rate, 0) if rate else None, blobs_above_1e5_per_run=blobs_1e5, floor_above=_fmt(f_hi), floor_below=_fmt(f_lo), floor_effect=_fmt(eff) if eff is not None else None)
    rule = (f"limiting when at least {_th('peak_obs_fraction'):.0%} of the observations have a peak pixel above the limit AND the floor of those observations exceeds the rest by "
            f">= {_th('peak_effect_floor'):.0%}; otherwise a note. Limit = {_th('peak_count_fraction'):.0%} of the count cutoff, or 1e5 counts when no cutoff is given. ({_rule('peak_count_fraction')})")
    ev = "spots.peak_counts of the blob behind each observation (linked by run, frame, row, col); floor by merge.systematic_floor on the two groups"
    scope = "a peak pixel is measured in the blob finder's frame sums; a count-rate loss of the integrated intensity is not measured directly; the floor comparison needs about 15 observations above the limit"
    sev = "limiting" if (frac >= _th("peak_obs_fraction") and eff is not None and eff >= _th("peak_effect_floor")) else "note"
    few = frac < _th("peak_obs_fraction")
    act = (("only a few reflections are in this range (under " + f"{_th('peak_obs_fraction'):.0%}" + " of the observations); it matters only if they are the low-order reflections you need. If it does: ")
           if few else "") + ("use an attenuator or a shorter exposure, or add a second, short exposure for the strong low-angle reflections; check the detector's count-rate correction (dead time) at the highest rates"
           + (" (the cutoff was not given, so 1e5 counts is only a guess: pass Acquisition(count_cutoff=...))" if guess else ""))
    return out + [Finding("dynamic_range", sev, f"{n_hi} of {n} observations have a peak pixel above {limit:.3g} counts (max {peak:.3g})", meas, rule, ev, act, None, scope)]


# ---------------------------------------------------------------------------------------------------------------------------------------------
@register_check("omega_sampling")
def check_omega_sampling(ctx) -> List[Finding]:
    obs = _all_obs(ctx)
    lk = _link(ctx, obs, ["omega_width_frames"])
    w = lk["omega_width_frames"].dropna().to_numpy(float)
    if len(w) < 30:
        raise CheckUnavailable("fewer than 30 observations link to a blob with omega_width_frames")
    med = float(np.median(w))
    single = float(np.mean(w == 0.0))
    steps = {r: ctx.omega_step(r) for r in ctx.run_names()}
    step = float(np.median(list(steps.values())))
    meas = dict(single_frame_fraction=_fmt(single, 3), median_rms_width_frames=_fmt(med, 2), p10=_fmt(np.percentile(w, 10), 2), p90=_fmt(np.percentile(w, 90), 2), step_deg=_fmt(step, 4), n=len(w))
    rule = (f"note when at least {_th('single_frame_fraction'):.0%} of the blobs lie on a single frame (undersampled profile), or when the median RMS frame width is above {_th('oversampled_width_frames'):g} frames (oversampled). "
            f"({_rule('single_frame_fraction')})")
    ev = "spots.omega_width_frames (the intensity-weighted RMS of the integer frame indices of the blob, 0 for a blob on one frame; not an extent) of the blobs behind the observations"
    scope = "the width is an RMS over integer frames, so it is 0 for a one-frame blob and says nothing about the true angular width below that; the integrated intensity does not depend on the sampling, the profile shape and the crossing position do; the oversampled rule is untested"
    if single >= _th("single_frame_fraction"):
        new = step / 4.0
        return [Finding("omega_sampling", "note", f"{single:.0%} of the spots lie on a single frame (step {step:g} deg): the omega profile is unresolved", meas, rule, ev,
                        f"a finer rotation step would resolve the profile, so that a spot spans a few frames: try roughly {new:.3g} deg (with a proportionally shorter exposure per frame); "
                        "the total dose and scan time stay the same", None, scope)]
    if med > _th("oversampled_width_frames"):
        return [Finding("omega_sampling", "note", f"spots span {med:.0f} frames: the rotation is oversampled", meas, rule, ev,
                        f"a coarser step (about {step * med / 10:.3g} deg) with a longer exposure per frame would cut readout overhead without losing the profile", None, scope)]
    return []


# ---------------------------------------------------------------------------------------------------------------------------------------------
@register_check("geometry_consistency")
def check_geometry_consistency(ctx) -> List[Finding]:
    rows = {}
    for name, red in ctx.runs.items():
        g = red.geom
        o = getattr(red, "orientation", None)
        rows[name] = dict(tx=g.tx_deg, ty=g.ty_deg, tz=g.tz_deg, wedge=g.wedge_deg, bcz=g.bcz_px, bcy=g.bcy_px, lsd_mm=g.lsd_um / 1000.0, wavelength=g.wavelength_A,
                          rms_px=getattr(o, "rms_px", np.nan), rms_frame=getattr(o, "rms_frame", np.nan), converged=getattr(o, "converged", None),
                          eps=getattr(o, "lambda_cell_scale", 0.0) if o is not None else 0.0, eps_sigma=getattr(o, "sigma_lambda_cell_scale", np.nan) if o is not None else np.nan)
    df = pd.DataFrame(rows).T
    num = df.drop(columns=["converged"]).astype(float)
    spread = (num.max() - num.min()) if len(num) > 1 else num.iloc[0] * 0.0
    wedge_max = float(num["wedge"].abs().max())
    tilt_spread = float(max(spread["tx"], spread["ty"], spread["tz"]))
    rms_px, rms_fr = float(np.nanmax(num["rms_px"])) if num["rms_px"].notna().any() else np.nan, float(np.nanmax(num["rms_frame"])) if num["rms_frame"].notna().any() else np.nan
    notconv = [r for r, v in df["converged"].items() if v is False]
    eps_used = bool((num["eps"].abs() > 0).any())
    hg = ctx.options.get("header_geometry")
    dev = None
    if hg is not None:
        dev = {}
        for r in ctx.run_names():
            h = hg[r] if isinstance(hg, dict) else hg
            g = ctx.runs[r].geom
            dev[str(r)] = dict(centre_px=_fmt(float(np.hypot(g.bcz_px - h.bcz_px, g.bcy_px - h.bcy_px)), 2), lsd_percent=_fmt(100.0 * (g.lsd_um / h.lsd_um - 1.0), 3),
                               wedge_deg=_fmt(g.wedge_deg - h.wedge_deg, 3), tilt_deg=_fmt(float(max(abs(g.tx_deg - h.tx_deg), abs(g.ty_deg - h.ty_deg), abs(g.tz_deg - h.tz_deg))), 3))
    meas = dict(per_run={str(r): {k: _fmt(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in row.items()} for r, row in df.iterrows()},
                wedge_abs_max_deg=_fmt(wedge_max, 3), tilt_spread_deg=_fmt(tilt_spread, 3), wedge_spread_deg=_fmt(float(spread["wedge"]), 3), rms_px_max=_fmt(rms_px, 2), rms_frame_max=_fmt(rms_fr, 2),
                not_converged=notconv, wavelength_scale_fitted=eps_used, deviation_from_header=dev, calibrant_measured=ctx.acq.has_calibrant)
    gate = bool(notconv) or (np.isfinite(rms_px) and rms_px > _th("geometry_rms_px")) or (np.isfinite(rms_fr) and rms_fr > _th("geometry_rms_frame"))
    big = wedge_max > _th("geometry_spread_wedge_deg") or tilt_spread > _th("geometry_spread_wedge_deg") or float(spread["wedge"]) > _th("geometry_spread_wedge_deg")
    rule = (f"limiting when the orientation fit did not converge or its robust rms exceeds {_th('geometry_rms_px'):g} px or {_th('geometry_rms_frame'):g} frame; note when the refined wedge, a tilt "
            f"or their run-to-run spread exceeds {_th('geometry_spread_wedge_deg'):g} deg or a wavelength/cell scale had to be fitted. ({_rule('geometry_spread_wedge_deg')})")
    ev = "Reduction.geom (refined) and Reduction.orientation of every run; the header values only when options['header_geometry'] is given"
    scope = "the geometry is refined from the spots themselves; a consistent run-to-run value does not prove it is right, and a calibrant is the only independent check"
    out: List[Finding] = []
    calib = ctx.acq.has_calibrant
    cal_txt = ("" if calib else " No calibrant pattern is recorded for this data: measure one (CeO2 or LaB6) at the same energy and sample-detector distance, ideally before and after.")
    if gate:
        out.append(Finding("geometry_consistency", "limiting", "the refined geometry does not describe the spots well (orientation fit not converged or residual too large)", meas, rule, ev,
                           "check the beam centre, distance and detector tilt and the rotation-axis alignment; a calibrant pattern at this energy and distance gives the detector geometry independently of the crystal."
                           + cal_txt, None, scope))
    elif big:
        out.append(Finding("geometry_consistency", "note",
                           f"the rotation axis or detector is not where the header says: wedge up to {wedge_max:.2f} deg, tilt spread {tilt_spread:.2f} deg, wedge spread {float(spread['wedge']):.2f} deg",
                           meas, rule, ev, "the fit models this, so the merged data are not degraded by it; check the alignment of the rotation axis to the beam and the detector tilt at the beamline so that it is small to begin with, "
                           "and use scans of 180 deg or more where possible: Friedel pairs inside a run then measure tx and the wedge without the orientation fit." + cal_txt, None, scope))
    if eps_used:
        e = num["eps"].to_numpy(float)
        e_spread = float(e.max() - e.min())
        s_formal = float(np.nanmedian(num["eps_sigma"]))
        e_err = max(e_spread, _th("lambda_spread"))
        txt = (f"a wavelength / cell-scale of {float(np.mean(e)):+.2e} was fitted (run-to-run spread {e_spread:.1e}; formal sigma {s_formal:.1e}, which is known to be 20-40 times too small: use the spread). "
               "With the cell held fixed the wavelength and an isotropic cell scale are exactly degenerate, so this is NOT an energy calibration.")
        out.append(Finding("wavelength_cell_scale", "note", f"the spots need a wavelength/cell scale of {float(np.mean(e)):+.2e} ({e_err:.0e} realistic error)",
                           dict(eps_per_run={str(r): _fmt(float(v), 6) for r, v in zip(num.index, e)}, spread=_fmt(e_spread, 6), formal_sigma=_fmt(s_formal, 8), calibrant_measured=calib),
                           f"note whenever it was fitted; realistic error = max(run-to-run spread, {_th('lambda_spread'):.0e}). ({_rule('lambda_spread')})", ev,
                           "to separate the energy from the cell: a calibrant at the same energy (gives the wavelength), or a trusted energy (an absorption-edge scan of a foil) plus a trusted cell." + cal_txt,
                           None, txt))
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------
def _ring_peaks(tth: np.ndarray, bin_deg: float = 0.25, half_window_deg: float = 5.0):
    """Ring-like peaks of a 2theta histogram: a bin whose count exceeds the mean of the bins within +-5 deg (the bin itself excluded) by ring_significance Poisson sigmas, with at least 8 counts."""
    if len(tth) < 30:
        return []
    edges = np.arange(0.0, np.ceil(tth.max()) + 1.0 + bin_deg, bin_deg)
    c, _ = np.histogram(tth, edges)
    h = int(round(half_window_deg / bin_deg))
    out = []
    for i in range(len(c)):
        lo, hi = max(0, i - h), min(len(c), i + h + 1)
        nb = np.r_[c[lo:i], c[i + 1:hi]]
        e = float(nb.mean()) if len(nb) else 0.0
        sig = (c[i] - e) / np.sqrt(e + 1.0)
        if c[i] >= 8 and sig >= _th("ring_significance"):
            out.append((float(edges[i] + bin_deg / 2), int(c[i]), float(sig)))
    merged = []                                                      # one ring spans a few bins: keep the strongest of each group closer than 0.6 deg
    for t, n, sg in out:
        if merged and t - merged[-1][0] <= 0.6:
            if n > merged[-1][1]:
                merged[-1] = (t, n, sg)
            continue
        merged.append((t, n, sg))
    return merged


def _d_max_cell(lattice) -> float:
    """The largest d-spacing any reflection of this cell can have (forbidden ones included): the 2theta below it cannot be Bragg scattering of the crystal."""
    return max(float(lattice.d_spacing(h, k, l)) for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4) if (h, k, l) != (0, 0, 0))


@register_check("contamination")
def check_contamination(ctx) -> List[Finding]:
    reasons_all: Dict[str, int] = {}
    n_strong_total = 0
    tth_unexpl: List[float] = []
    lam = None
    for name, red in ctx.runs.items():
        sp = getattr(red, "spots", None)
        rej = getattr(red, "rejected", None)
        if sp is None or "integrated" not in sp.columns or rej is None:
            raise CheckUnavailable("spots lack 'integrated' or the reduction has no rejected table")
        n = len(sp)
        if n < 40:
            continue
        top = np.argsort(-sp["integrated"].to_numpy(float))[:max(20, int(np.ceil(_th("strong_top_fraction") * n)))]
        reason = pd.Series("kept", index=range(n))
        if len(rej):
            reason.loc[rej["i"].to_numpy(int)] = rej["reason"].to_numpy()
        for t in top:
            reasons_all[reason.iloc[t]] = reasons_all.get(reason.iloc[t], 0) + 1
        n_strong_total += len(top)
        top20 = np.argsort(-sp["integrated"].to_numpy(float))[:max(60, int(np.ceil(0.2 * n)))]
        un = [t for t in top20 if reason.iloc[t] in ("no_hkl", "low_angle")]
        if un:
            _, tth, _ = spot_directions(sp["row"].to_numpy(float)[un], sp["col"].to_numpy(float)[un], red.geom)
            tth_unexpl.extend(np.asarray(tth, float).tolist())
        lam = red.geom.wavelength_A
    if n_strong_total == 0:
        raise CheckUnavailable("no run has enough blobs")
    fr = lambda k: reasons_all.get(k, 0) / n_strong_total
    crowded, unexpl = fr("crowded"), fr("no_hkl") + fr("low_angle")
    other = sum(fr(k) for k in REASONS_OTHER)
    tth_unexpl = np.array(tth_unexpl, float)
    d_cell = _d_max_cell(ctx.lattice)
    d_unexpl = lam / (2.0 * np.sin(np.radians(np.clip(tth_unexpl, 1e-6, 179.0) / 2.0))) if (len(tth_unexpl) and lam) else np.array([])
    halo = d_unexpl > 1.02 * d_cell if len(d_unexpl) else np.array([], bool)
    n_halo = int(halo.sum()) if len(halo) else 0
    peaks = _ring_peaks(tth_unexpl[~halo]) if len(tth_unexpl) else []
    ring_txt = []
    for t, c, sg_ in peaks:
        d = lam / (2.0 * np.sin(np.radians(t / 2.0))) if lam else None
        ice = d is not None and any(abs(d - di) / di < 0.01 for di in ICE_IH_D)
        ring_txt.append(dict(tth_deg=_fmt(t, 2), count=c, significance=_fmt(sg_, 1), d_A=_fmt(d, 3) if d else None, consistent_with_ice_Ih=bool(ice)))
    meas = dict(strong_blobs=n_strong_total, kept=_fmt(fr("kept"), 3), crowded=_fmt(crowded, 3), unexplained_no_hkl_or_low_angle=_fmt(unexpl, 3), other_rejections=_fmt(other, 3),
                reasons={k: int(v) for k, v in sorted(reasons_all.items())}, rings=ring_txt, near_beam_unexplained=n_halo, largest_d_of_cell_A=_fmt(d_cell, 2),
                unexplained_blobs_examined=int(len(tth_unexpl)))
    rule = (f"limiting when more than {_th('crowded_strong_fraction'):.0%} of the strongest {_th('strong_top_fraction'):.0%} of blobs are rejected as crowded, or more than "
            f"{_th('unexplained_strong_fraction'):.0%} are explained by no reflection; ring-like when an unexplained 2theta bin exceeds its local median by {_th('ring_significance'):g} sigma. "
            f"({_rule('crowded_strong_fraction')})")
    ev = "Reduction.rejected (reason per blob) and spots.integrated; 2theta from the refined geometry; histogram of the unexplained strong blobs in 0.25 deg bins"
    scope = "the reason is the FIRST rule a blob failed; a blob may fail several. Ice-like d-spacings are a consistency check, not an identification"
    out: List[Finding] = []
    if crowded >= _th("crowded_strong_fraction"):
        out.append(Finding("crowded_strong_reflections", "limiting",
                           f"{crowded:.0%} of the strongest blobs are lost as 'crowded' (another spot too close in the same frames): the strongest, lowest-order reflections are missing from the merge",
                           meas, rule, ev, "a smaller rotation step separates spots in omega; a longer detector distance separates them on the detector; a smaller beam or crystal reduces the spot size; "
                           "check whether the lost reflections are the low-angle ones that matter most for the structure.", None, scope))
    elif crowded >= 0.05:
        out.append(Finding("crowded_strong_reflections", "note", f"{crowded:.0%} of the strongest blobs are lost as 'crowded'", meas, rule, ev, "a smaller rotation step or a longer detector distance would recover them.", None, scope))
    if n_halo >= 8 and n_halo >= 0.3 * max(1, len(tth_unexpl)):
        out.append(Finding("near_beam_scattering", "note",
                           f"{n_halo} of {len(tth_unexpl)} unexplained strong blobs lie at 2theta below the first possible reflection of the cell (d > {d_cell:.1f} A): scattering from near the direct beam, not from the crystal",
                           meas, rule, ev, "check the beamstop and its alignment, the air path and the collimation near the sample (a halo or air scatter close to the beam raises the background under the low-angle "
                           "reflections); a smaller beam-defining aperture or a guard slit usually reduces it.", None, scope))
    if unexpl >= _th("unexplained_strong_fraction") or peaks:
        sev = "limiting" if unexpl >= _th("unexplained_strong_fraction") else "note"
        if peaks:
            title = f"strong blobs at constant 2theta that no reflection of the crystal explains ({len(peaks)} ring-like peak(s)): powder or ice rings"
            act = ("rings at ice Ih d-spacings: check the cryostream alignment and the humidity (frost), anneal or re-coat; rings from the gasket, a loop or a holder: collimate or clean up the beam path "
                   "and mask them; they raise the background and can overlap reflections.")
        else:
            title = f"{unexpl:.0%} of the strongest blobs are explained by no reflection of this lattice: a second crystal, an extra phase or non-crystal scattering"
            act = "check for a second crystal or twin domain in the beam (a smaller beam may isolate one), air or holder scattering, and whether the lattice used is the right one."
        out.append(Finding("unexplained_strong_blobs", sev, title, meas, rule, ev, act, None, scope))
    return out
