"""``midas-grain-qa`` — one command that audits a finished FF reconstruction.

Everything here already existed as a packaged function; what was missing was a
way to run them together on a layer directory without writing ten lines of glue
each time. Five stages, each independently skippable and each degrading to a
recorded reason rather than a crash when its inputs are absent:

1. **grains** — the table itself (required).
2. **d0** — strain-free reference lattice, via
   :func:`midas_stress.equilibrium.recover_d0_cubic_free_standing`. For a cubic
   free-standing polycrystal the equilibrium correction collapses exactly to the
   volume-averaged hydrostatic strain, so no stiffness is needed. **Cubic only**
   — it refuses rather than guessing for other symmetries.
3. **twins** — Σ3 inventory with variant resolution and, crucially, the spatial
   adjacency null. A twin fraction without that null means nothing: any large
   grain list contains pairs near a CSL misorientation by chance.
4. **attribution** — which contested reflections a grain does not actually own
   (:mod:`~midas_process_grains.spot_attribution`), scored against the twin
   labels it never saw.
5. **uncertainty** — per-grain 1σ on all 12 refined parameters, plus both strain
   conventions by their own natural route.
6. **calibration** — the one stage that can tell you the error bars are wrong.
   A grain straddling the boundary between two layers is measured twice from
   independent spot sets, so ``z = (m1 - m2)/(sqrt(2)*sigma)`` must have unit
   variance if sigma is right. On 1-ID it came out at 0.35, i.e. the bars were
   ~2.9x too large, entirely because ``sigma_obs_px`` sat at its placeholder
   default of 1.0. **Reports only — it never rescales anything**, because a
   silent placeholder is what caused the problem and silently fixing it would
   be the same mistake pointed the other way.

Design rules this module follows, each learned the hard way on 1-ID LSHR:

* **No silent excepts.** Every stage records the exception type and message into
  ``warnings``; nothing is reported as "not available" when it actually threw.
  A bare ``except`` once turned a ``torch`` threading bug into a fake physical
  limit on how many grains could be measured.
* **Nulls and controls are part of the output, not an option.** The twin
  adjacency rate and the attribution's twin-agreement score are always computed
  and always printed, because they are what make the headline numbers mean
  anything.
* **Provenance.** Every number records the file it came from.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

STAGES = ("d0", "twins", "attribution", "uncertainty", "calibration")


@dataclass
class GrainQAResult:
    """Everything :func:`run_grain_qa` produced, plus why anything is missing."""

    layer_dir: Path
    n_grains: int
    space_group: int
    d0: Optional[Dict[str, Any]] = None
    twins: Any = None                       # TwinResult
    attribution: Any = None                 # SpotAttribution
    twin_agreement: Optional[Dict[str, float]] = None
    sigma: Any = None                       # PerGrainParameterSigmaResult
    sigma_gid: Any = None                   # (N,) grain ids matching `sigma` rows
    strain_sigma: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    skipped: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, str] = field(default_factory=dict)

    # ---- reporting -------------------------------------------------------
    def summary(self) -> str:
        L = [f"grain QA — {self.layer_dir}",
             f"  {self.n_grains:,} grains, space group {self.space_group}"]

        if self.d0:
            d = self.d0
            L += ["", "d0 (strain-free reference lattice)",
                  f"  recovered a0        {d['a0']:.6f} Å",
                  f"  assumed by the run  {d['a_assumed']:.6f} Å",
                  f"  mean hydrostatic    {1e6*d['eps_iso']:+.1f} µε  "
                  f"→ the reference is {1e6*d['eps_iso']:+.1f} ppm off"]
            if abs(d["eps_iso"]) > 100e-6:
                L.append(f"  ** a {1e6*d['eps_iso']:+.0f} ppm reference error biases "
                         f"HYDROSTATIC strain and any stress read from it. "
                         f"Deviatoric strain is unaffected. **")
        if self.twins is not None:
            t = self.twins
            L += ["", "twins (Σ3, variant-resolved, against a spatial null)",
                  "  " + t.summary().replace("\n", "\n  ")]
        if self.attribution is not None:
            a = self.attribution
            L += ["", "contested-spot attribution",
                  "  " + a.summary().replace("\n", "\n  ")]
            if self.twin_agreement:
                ta = self.twin_agreement
                sep = ta["keep_rate_twin"] - ta["keep_rate_accidental"]
                L += [f"  self-check vs twin labels the filter never saw:",
                      f"    twin-shared kept {100*ta['keep_rate_twin']:.1f}%  "
                      f"accidental kept {100*ta['keep_rate_accidental']:.1f}%  "
                      f"→ separation {100*sep:.1f} pp"]
                if not np.isnan(sep) and sep < 0.30:
                    L.append("    ** WEAK separation — the filter is not telling "
                             "legitimate twin sharing from accidental overlap on "
                             "this data. Do not use it to drop spots. **")
        if self.sigma is not None:
            s = self.sigma
            ok = s.ok
            if not ok.any():
                # A stage that ran and produced nothing must SAY so. Rendering
                # zero successes as silence is how a bug reads as a data limit.
                L += ["", "per-grain uncertainty — PRODUCED NOTHING",
                      f"  0 of {len(ok):,} grains yielded a covariance."]
                if getattr(s, "failures", None):
                    for msg, cnt in sorted(s.failures.items(),
                                           key=lambda kv: -kv[1])[:4]:
                        L.append(f"    {cnt:6d}  {msg}")
                else:
                    L.append("    no exceptions were recorded, so the grains were "
                             "rejected by the conditioning gate or never reached "
                             "it — check the geometry in the parameter file.")
            if ok.any():
                eu = np.degrees(s.sigma_euler_rad[ok]) * 1e3
                L += ["", f"per-grain uncertainty  ({int(ok.sum()):,} of "
                          f"{len(ok):,} grains)",
                      f"  orientation   {np.median(eu[:,0]):.1f} / "
                      f"{np.median(eu[:,1]):.1f} / {np.median(eu[:,2]):.1f} millideg",
                      f"  position      {np.median(s.sigma_pos_um[ok,0]):.1f} / "
                      f"{np.median(s.sigma_pos_um[ok,1]):.1f} / "
                      f"{np.median(s.sigma_pos_um[ok,2]):.1f} µm",
                      f"  lattice a     {1e6*np.median(s.sigma_latc[ok,0]):.0f}e-6 Å",
                      f"  ε hydrostatic {1e6*np.median(s.sigma_hydrostatic_strain[ok]):.1f} µε"]
                frac = ok.mean()
                if frac < 0.9:
                    L.append(f"  ** only {100*frac:.0f}% of grains produced a "
                             f"covariance. Before treating that as a property of "
                             f"the data, re-run with MIDAS_PG_SIGMA_JOBS=1: this "
                             f"path has a known concurrency failure. **")
        if self.strain_sigma:
            for conv in ("eFab", "eKen"):
                v = self.strain_sigma.get(conv)
                if v is not None:
                    L.append(f"  σ {conv:5s}      "
                             + " / ".join(f"{x:.0f}" for x in v[:3])
                             + " normal, "
                             + " / ".join(f"{x:.0f}" for x in v[3:])
                             + " shear (µε)")
        if self.calibration:
            c = self.calibration
            t = c.get("tight") or {}
            L += ["", "CALIBRATION — are the error bars right?",
                  f"  {t.get('n_pairs', 0)} grains re-found in "
                  f"{c['n_sibling_layers']} adjacent layer(s)",
                  f"  std(z)  X {t.get('std_z_x', float('nan')):.3f}   "
                  f"Y {t.get('std_z_y', float('nan')):.3f}   (1.000 = correct)"]
            if "loose" in c:
                lo = c["loose"]
                L.append(f"  same with a 4x looser match gate: X "
                         f"{lo['std_z_x']:.3f}  Y {lo['std_z_y']:.3f}  "
                         f"({lo['n_pairs']} pairs) — if these differ a lot, the "
                         f"match gate is shaping the answer")
            f = c.get("factor", float("nan"))
            if np.isfinite(f) and f > 1.15:
                L += [f"  ** the error bars are about {f:.1f}x TOO LARGE. **",
                      f"     sigma_obs_px was assumed "
                      f"{c['assumed_sigma_obs_px']:.3f}; this data implies "
                      f"~{c['implied_sigma_obs_px']:.3f}.",
                      f"     Re-run with sigma_obs_px={c['implied_sigma_obs_px']:.3f} "
                      f"to fix them. NOT applied automatically — a silent "
                      f"placeholder caused this, and a silent correction would be",
                      f"     the same mistake reversed.",
                      f"     NOTE this calibrates REPRODUCIBILITY. It cancels "
                      f"everything the two measurements share (detector, Lsd,",
                      f"     tilts, energy, reference lattice), so it cannot see "
                      f"a common-mode offset."]
            elif np.isfinite(f) and f < 0.85:
                L.append(f"  ** the error bars are about {1/f:.1f}x TOO SMALL. **")
            else:
                L.append("  error bars are consistent with the repeat scatter.")

        if self.skipped:
            L += ["", "skipped"]
            L += [f"  {k}: {v}" for k, v in self.skipped.items()]
        if self.warnings:
            L += ["", "warnings"]
            L += [f"  {w}" for w in self.warnings]
        if self.timings:
            L += ["", "  timings (s): " + "  ".join(
                f"{k} {v:.1f}" for k, v in self.timings.items())]
        return "\n".join(L)

    def to_json(self) -> Dict[str, Any]:
        """Plain-Python summary, safe for ``json.dump``."""
        out: Dict[str, Any] = {
            "layer_dir": str(self.layer_dir), "n_grains": self.n_grains,
            "space_group": self.space_group, "skipped": dict(self.skipped),
            "warnings": list(self.warnings), "provenance": dict(self.provenance),
            "timings": {k: round(v, 2) for k, v in self.timings.items()},
        }
        if self.d0:
            out["d0"] = {k: float(v) if isinstance(v, (int, float, np.floating))
                         else v for k, v in self.d0.items()}
        if self.twins is not None:
            t = self.twins
            out["twins"] = {
                "n_pairs": int(len(t.pairs)), "rate_near": float(t.rate_near),
                "rate_far": float(t.rate_far), "enrichment": float(t.enrichment),
                "twinned_fraction": float(t.twinned_fraction),
                "variant_counts": {k: int(v) for k, v in t.variant_counts.items()},
                "variant_balance": float(t.variant_balance),
                "trustworthy": bool(t.trustworthy)}
        if self.attribution is not None:
            a = self.attribution
            out["attribution"] = {
                "n_claims": int(len(a.grain_id)), "n_contested": int(a.n_contested),
                "n_dropped": int(a.n_dropped), "sigma_um": float(a.sigma),
                "rel_threshold": float(a.rel_threshold)}
            if self.twin_agreement:
                out["attribution"]["twin_agreement"] = {
                    k: (None if (isinstance(v, float) and np.isnan(v)) else float(v))
                    for k, v in self.twin_agreement.items()}
        if self.sigma is not None and self.sigma.ok.any():
            s, ok = self.sigma, self.sigma.ok
            out["uncertainty"] = {
                "n_ok": int(ok.sum()), "n_total": int(len(ok)),
                "median_euler_millideg": [float(np.median(
                    np.degrees(s.sigma_euler_rad[ok, k]) * 1e3)) for k in range(3)],
                "median_pos_um": [float(np.median(s.sigma_pos_um[ok, k]))
                                  for k in range(3)],
                "median_latc": [float(np.median(s.sigma_latc[ok, k]))
                                for k in range(6)],
                "median_eps_hydro": float(np.median(
                    s.sigma_hydrostatic_strain[ok]))}
        if self.calibration:
            out["calibration"] = {k: (v if not isinstance(v, dict)
                                      else {kk: float(vv) for kk, vv in v.items()})
                                  for k, v in self.calibration.items()}
        if self.strain_sigma:
            out["strain_sigma"] = {k: [float(x) for x in v]
                                   for k, v in self.strain_sigma.items()
                                   if v is not None}
        return out


# ---------------------------------------------------------------------------
#  stages
# ---------------------------------------------------------------------------

def _stage_d0(g, params, res: GrainQAResult) -> None:
    """Strain-free reference lattice. Needs no parameter file: the assumed
    reference is in the ``Grains.csv`` preamble."""
    from midas_stress.equilibrium import recover_d0_cubic_free_standing

    # Symmetry first, so the refusal is unambiguous rather than a downstream
    # complaint about a lattice we should never have looked up.
    if not (195 <= res.space_group <= 230):
        res.skipped["d0"] = (f"space group {res.space_group} is not cubic; "
                             f"recover_d0_cubic_free_standing is cubic-only and "
                             f"refuses to guess for other symmetries")
        return
    lat = np.column_stack([g.column(k) for k in
                           ("a", "b", "c", "alpha", "beta", "gamma")])
    src = None
    assumed = None
    if params is not None and getattr(params.base, "LatticeConstant", None):
        assumed = np.asarray(params.base.LatticeConstant, float)
        src = "paramstest LatticeConstant"
    if assumed is None or assumed.size < 6:
        hdr = (g.header or {}).get("lattice_parameter")
        if hdr and len(hdr) >= 6:
            assumed = np.asarray(hdr, float); src = "Grains.csv preamble"
    if assumed is None or assumed.size < 6:
        res.skipped["d0"] = ("no assumed reference lattice: neither the "
                             "parameter file nor the Grains.csv preamble "
                             "carries one")
        return
    res.provenance["d0_reference"] = src
    vol = g.grain_radius ** 3 if g.grain_radius is not None else None
    out = recover_d0_cubic_free_standing(
        lat, assumed[:6], volumes=vol, confidences=g.confidence,
        min_confidence=0.0)
    res.d0 = {"a0": float(np.asarray(out["reference_recovered"])[0]),
              "a_assumed": float(assumed[0]),
              "eps_iso": float(out["eps_iso"]),
              "scale_factor": float(out["scale_factor"]),
              "n_used": int(out.get("n_grains_used", len(lat)))}


def _stage_twins(g, res: GrainQAResult, *, max_distance_um: float,
                 tol_deg: float) -> None:
    from .twins_posthoc import find_twins
    res.twins = find_twins(g.orient_mat, g.positions, res.space_group,
                           max_distance_um=max_distance_um, tol_deg=tol_deg)


def _stage_attribution(layer_dir: Path, g, res: GrainQAResult, *,
                       rel_threshold: float, min_spots: int) -> None:
    from .spot_attribution import attribute_from_spot_matrix, twin_agreement

    sm = layer_dir / "SpotMatrix.csv"
    if not sm.exists():
        res.skipped["attribution"] = f"no {sm.name} in the layer directory"
        return
    res.attribution = attribute_from_spot_matrix(
        sm, rel_threshold=rel_threshold, min_spots=min_spots)
    res.provenance["attribution"] = str(sm)
    if res.twins is not None:
        ids = np.asarray(g.ids, dtype=np.int64)
        twin_of: Dict[int, set] = {}
        for i, j in res.twins.pairs:
            twin_of.setdefault(int(ids[i]), set()).add(int(ids[j]))
            twin_of.setdefault(int(ids[j]), set()).add(int(ids[i]))
        res.twin_agreement = twin_agreement(res.attribution, twin_of)


#: Pole-exclusion half-angle used when the parameter file names none.
DEFAULT_EXCLUDE_POLE_DEG = 6.0


def _stage_calibration(layer_dir: Path, g, res: GrainQAResult) -> None:
    """Are the error bars right? Measure it against grains seen in two layers.

    Reports, never applies. See the module docstring for why.
    """
    from scipy.spatial import cKDTree
    from midas_stress.orientation import misorientation_om
    from .io.read import read_grains_csv

    if res.sigma is None or not getattr(res.sigma, "ok", np.zeros(0, bool)).any():
        res.skipped["calibration"] = ("needs the uncertainty stage; it was "
                                      "skipped or produced nothing")
        return

    m = re.search(r"(.*LayerNr_)(\d+)\s*$", str(layer_dir).rstrip("/"))
    if not m:
        res.skipped["calibration"] = (
            f"cannot find sibling layers: {layer_dir.name!r} is not LayerNr_<n>. "
            f"This check needs at least one ADJACENT layer.")
        return
    stem, this = m.group(1), int(m.group(2))
    sibs = [Path(f"{stem}{this+d}") for d in (-1, 1)]
    sibs = [q for q in sibs if (q / "Grains.csv").exists()]
    if not sibs:
        res.skipped["calibration"] = (
            f"no adjacent layer found next to LayerNr_{this}; this check needs "
            f"LayerNr_{this-1} or LayerNr_{this+1}")
        return

    sig = {int(gid): res.sigma.sigma_pos_um[i]
           for i, gid in enumerate(np.asarray(res.sigma_gid))
           if res.sigma.ok[i]} if getattr(res, "sigma_gid", None) is not None else {}
    if not sig:
        sig = {int(gid): res.sigma.sigma_pos_um[i]
               for i, gid in enumerate(np.asarray(g.ids)[:len(res.sigma.ok)])
               if res.sigma.ok[i]}

    # Two gates on purpose. The TIGHT one is the usual choice; the LOOSE one
    # exists because selecting pairs on |m1-m2| and then measuring the scatter
    # of that same difference is selection on the dependent variable. Measured
    # on 1-ID the robust std barely moves (0.348 -> 0.360) but the TAIL
    # fractions move a lot, so tails are not reported at all.
    out = {}
    for gate_um, tag in ((60.0, "tight"), (250.0, "loose")):
        dz = {0: [], 1: [], 2: []}
        for q in sibs:
            b = read_grains_csv(q / "Grains.csv")
            tr = cKDTree(b.positions[:, :2])
            for i, gid in enumerate(np.asarray(g.ids)):
                s_i = sig.get(int(gid))
                if s_i is None:
                    continue
                for j in tr.query_ball_point(g.positions[i, :2], gate_um):
                    ang, _ = misorientation_om(g.orient_mat[i], b.orient_mat[j],
                                               res.space_group)
                    if np.degrees(ang) < 0.25:
                        for k in (0, 1):
                            if s_i[k] > 0:
                                dz[k].append((b.positions[j, k] - g.positions[i, k])
                                             / (np.sqrt(2.0) * s_i[k]))
                        break
        n = len(dz[0])
        if n < 30:
            continue
        def _rstd(v):
            v = np.asarray(v)
            return float(np.diff(np.percentile(v, [25, 75]))[0] / 1.349)
        out[tag] = dict(n_pairs=n, std_z_x=_rstd(dz[0]), std_z_y=_rstd(dz[1]))
    if not out:
        res.skipped["calibration"] = ("fewer than 30 repeat grains found; too "
                                      "few to calibrate")
        return

    t = out.get("tight") or next(iter(out.values()))
    zbar = 0.5 * (t["std_z_x"] + t["std_z_y"])
    res.calibration = dict(
        out, implied_sigma_obs_px=float(res.sigma.sigma_obs_px * zbar),
        assumed_sigma_obs_px=float(res.sigma.sigma_obs_px),
        factor=float(1.0 / zbar) if zbar else float("nan"),
        n_sibling_layers=len(sibs))


def _geometry(params):
    """Build the forward-model geometry from a paramstest.

    ``min_eta`` comes from **ExcludePoleAngle**, NOT from ``MarginEta``.
    They are different quantities and confusing them is silent: MarginEta is a
    spot-matching margin and is routinely 500 on these files, which as a
    pole-exclusion angle excludes every reflection. The forward model then
    predicts nothing and the Hessian fails with an opaque
    ``IndexError: min(): Expected reduction dim 1 to have non-zero size``
    for every grain — measured on 1-ID LSHR, 2026-09-02.
    """
    from midas_diffract.forward import HEDMGeometry
    b = params.base
    lsd = float(getattr(b, "LsdFit", 0) or getattr(b, "Lsd", 0) or 0)
    step = float(getattr(b, "OmegaStep", 0) or 0.25)
    y_bc = float(getattr(b, "YBCFit", 0) or 0)
    z_bc = float(getattr(b, "ZBCFit", 0) or 0)
    if y_bc == 0.0 or z_bc == 0.0:
        raise ValueError(
            "no beam centre in the parameter file (YBCFit/ZBCFit). Refusing to "
            "default it: a guessed beam centre produces a geometry that looks "
            "fine and is wrong everywhere.")
    raw = getattr(b, "ExcludePoleAngle", None)
    min_eta = float(raw) if raw not in (None, 0) else DEFAULT_EXCLUDE_POLE_DEG
    if not (0.0 <= min_eta < 90.0):
        raise ValueError(
            f"ExcludePoleAngle = {min_eta}° excludes every reflection "
            f"(must be in [0, 90)). This is what MarginEta looks like when it "
            f"is misread as a pole angle.")
    return HEDMGeometry(
        Lsd=lsd, y_BC=y_bc, z_BC=z_bc, px=float(b.px),
        omega_start=float(getattr(b, "OmegaStart", 0) or 0), omega_step=step,
        n_frames=int(round(360.0 / abs(step))) if step else 1440,
        n_pixels_y=int(getattr(b, "NrPixelsY", 2048) or 2048),
        n_pixels_z=int(getattr(b, "NrPixelsZ", 2048) or 2048),
        min_eta=min_eta, wavelength=float(b.Wavelength),
        tx=float(getattr(b, "txFit", 0) or 0), ty=float(getattr(b, "tyFit", 0) or 0),
        tz=float(getattr(b, "tzFit", 0) or 0), flip_y=True, apply_tilts=True), lsd, step


def _stage_uncertainty(layer_dir: Path, g, params, res: GrainQAResult, *,
                       max_grains: Optional[int], log,
                       sigma_obs_px: Optional[float] = None,
                       sigma_obs_frames: Optional[float] = None) -> None:
    import pandas as pd
    from .compute.hkl_ingest import read_hkls_csv
    from .compute.position_uncertainty import compute_per_grain_parameter_sigma
    from .compute.strain_uncertainty import fable_strain_covariance

    need = {"hkls": layer_dir / "hkls.csv",
            "ids": layer_dir / "SpotsToIndex.csv",
            "pk": layer_dir / "Results" / "ProcessKey.bin",
            "inputall": layer_dir / "InputAll.csv"}
    missing = [k for k, p in need.items() if not p.exists()]
    if missing:
        res.skipped["uncertainty"] = ("missing " +
                                      ", ".join(need[k].name for k in missing))
        return

    geom, lsd, step = _geometry(params)
    if lsd <= 0:
        res.skipped["uncertainty"] = "no Lsd/LsdFit in the parameter file"
        return
    hkls = read_hkls_csv(need["hkls"])
    sti = np.loadtxt(need["ids"], dtype=np.int64).ravel()
    s2r = {int(s): r for r, s in enumerate(np.atleast_1d(sti))}
    rows = np.array([s2r.get(int(i), -1) for i in g.ids])
    keep = np.nonzero(rows >= 0)[0]
    if keep.size == 0:
        res.skipped["uncertainty"] = ("no GrainID matched a SpotsToIndex row — "
                                      "the grain list and the seed list disagree")
        return
    if max_grains is not None and keep.size > max_grains:
        keep = keep[:max_grains]
        res.warnings.append(f"uncertainty computed on the first {max_grains} "
                            f"grains only (--max-grains)")
    inp = pd.read_csv(need["inputall"], sep=r"\s+", engine="c")
    inp.columns = [c.lstrip("%") for c in inp.columns]
    inp = inp.set_index(inp["SpotID"].astype(int))[["YLab", "ZLab", "Omega"]]
    lat = np.column_stack([g.column(k) for k in
                           ("a", "b", "c", "alpha", "beta", "gamma")])

    # The spots each grain was REFINED on, from SpotMatrix.csv. ProcessKey holds
    # only what the INDEXER matched to the representative seed -- a strict subset
    # (median 58 vs 268 on shade_LSHR), and since sigma ~ 1/sqrt(n) using it
    # inflates every sigma by ~2.15x. Fall back to ProcessKey if SpotMatrix is
    # absent, and SAY WHICH was used rather than silently differing.
    spot_ids = None
    sm_path = layer_dir / "SpotMatrix.csv"
    if sm_path.exists():
        try:
            import collections as _c
            by_gid = _c.defaultdict(list)
            with open(sm_path) as fh:
                for ln in fh:
                    if ln.startswith("%"):
                        continue
                    t = ln.split()
                    if len(t) >= 2:
                        _sid = int(float(t[1]))
                        if _sid > 0:      # skip unmatched-prediction rows
                            by_gid[int(float(t[0]))].append(_sid)
            ids_keep = np.asarray(g.ids)[keep]
            spot_ids = [np.asarray(by_gid.get(int(gid), ()), dtype=np.int64)
                        for gid in ids_keep]
            n_med = int(np.median([len(x) for x in spot_ids])) if spot_ids else 0
            log(f"  per-grain sigma: using the REFINED spot sets from "
                f"SpotMatrix.csv (median {n_med} spots/grain)")
        except Exception as e:                              # noqa: BLE001
            log(f"  per-grain sigma: could not read SpotMatrix ({e}); "
                f"falling back to ProcessKey seed spots -- sigma will be ~2x high")
            spot_ids = None
    else:
        log("  per-grain sigma: no SpotMatrix.csv; using ProcessKey SEED spots. "
            "sigma will be biased HIGH (~2.15x on shade_LSHR) because the seed "
            "carries far fewer spots than the refined grain.")

    res.sigma = compute_per_grain_parameter_sigma(
        grain_OM=g.orient_mat[keep], grain_pos_um=g.positions[keep],
        rep_cand_idx=rows[keep], pk_path=need["pk"], inputall_df=inp,
        spot_ids_per_grain=spot_ids,
        hkls=hkls, geometry=geom, latc=lat[keep],
        omega_start_deg=float(getattr(params.base, "OmegaStart", 0) or 0),
        omega_step_deg=step, return_cov=True, log=log,
        **({} if sigma_obs_px is None else {"sigma_obs_px": float(sigma_obs_px)}),
        **({} if sigma_obs_frames is None
           else {"sigma_obs_frames": float(sigma_obs_frames)}))
    res.provenance["uncertainty"] = str(need["pk"])
    res.sigma_gid = np.asarray(g.ids)[keep]

    # eFab propagated from the same covariance; eKen needs FitBest and is
    # handled by the caller when present.
    ok = res.sigma.ok
    if ok.any() and res.sigma.cov is not None:
        ref = np.array([res.d0["a0"]] * 3 + [90.0, 90.0, 90.0]) if res.d0 \
            else np.asarray(params.base.LatticeConstant, float)[:6]
        vals = []
        for j in np.nonzero(ok)[0]:
            try:
                vals.append(fable_strain_covariance(
                    lat[keep][j], ref, res.sigma.cov[j][3:9, 3:9]).sigma_voigt)
            except Exception as e:                       # noqa: BLE001
                res.warnings.append(f"eFab covariance failed for one grain: "
                                    f"{type(e).__name__}: {e}")
                break
        if vals:
            res.strain_sigma = {"eFab": (np.median(np.array(vals), axis=0) * 1e6)
                                .tolist()}


# ---------------------------------------------------------------------------
#  driver
# ---------------------------------------------------------------------------

def run_grain_qa(
    layer_dir,
    *,
    space_group: Optional[int] = None,
    skip: Sequence[str] = (),
    twin_distance_um: float = 45.0,
    twin_tol_deg: float = 1.0,
    rel_threshold: Optional[float] = None,
    min_spots: Optional[int] = None,
    max_grains_sigma: Optional[int] = None,
    sigma_obs_px: Optional[float] = None,
    sigma_obs_frames: Optional[float] = None,
    log=None,
) -> GrainQAResult:
    """Audit one finished layer directory. See the module docstring."""
    from .io.read import read_grains_csv
    from .params import read_paramstest_pg
    from .spot_attribution import DEFAULT_MIN_SPOTS, DEFAULT_REL_THRESHOLD

    if log is None:
        log = lambda *a, **k: None
    bad = [s for s in skip if s not in STAGES]
    if bad:
        raise ValueError(f"unknown stage(s) {bad}; valid: {STAGES}")
    layer_dir = Path(layer_dir)
    gpath = layer_dir / "Grains.csv"
    if not gpath.exists():
        raise FileNotFoundError(f"no Grains.csv in {layer_dir}")

    g = read_grains_csv(gpath)
    ps_path = next((layer_dir / n for n in
                    ("paramstest_pg.txt", "paramstest_refine_comp.txt",
                     "paramstest.txt") if (layer_dir / n).exists()), None)
    params = read_paramstest_pg(ps_path) if ps_path else None
    sg = int(space_group or getattr(g, "space_group", None)
             or (getattr(params.base, "SpaceGroup", None) if params else None)
             or 225)

    res = GrainQAResult(layer_dir=layer_dir, n_grains=len(g.ids), space_group=sg)
    res.provenance["grains"] = str(gpath)
    if ps_path:
        res.provenance["params"] = str(ps_path)
    else:
        res.warnings.append("no paramstest found; the uncertainty stage "
                            "needs one for the geometry (d0 does not)")

    for stage, fn in (
        ("d0", lambda: _stage_d0(g, params, res)),
        ("twins", lambda: _stage_twins(g, res, max_distance_um=twin_distance_um,
                                       tol_deg=twin_tol_deg)),
        ("attribution", lambda: _stage_attribution(
            layer_dir, g, res,
            rel_threshold=(DEFAULT_REL_THRESHOLD if rel_threshold is None
                           else rel_threshold),
            min_spots=(DEFAULT_MIN_SPOTS if min_spots is None else min_spots))),
        ("uncertainty", lambda: _stage_uncertainty(
            layer_dir, g, params, res, max_grains=max_grains_sigma,
            sigma_obs_px=sigma_obs_px, sigma_obs_frames=sigma_obs_frames, log=log)),
        ("calibration", lambda: _stage_calibration(layer_dir, g, res)),
    ):
        if stage in skip:
            res.skipped[stage] = "skipped by request"
            continue
        if stage == "uncertainty" and params is None:
            res.skipped[stage] = ("no parameter file; the geometry "
                                  "is needed to build the forward model")
            continue
        t0 = time.time()
        try:
            log(f"[grain-qa] {stage} ...")
            fn()
        except Exception as e:                            # noqa: BLE001
            # Recorded, never swallowed — see the module docstring.
            res.skipped[stage] = f"{type(e).__name__}: {e}"
            res.warnings.append(f"stage {stage!r} raised {type(e).__name__}: {e}")
        res.timings[stage] = time.time() - t0
    return res


def main(argv=None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        prog="midas-grain-qa",
        description="Audit a finished FF reconstruction: strain-free lattice, "
                    "twin inventory, contested-spot attribution and per-grain "
                    "per-parameter uncertainty.",
        epilog="Every headline number is reported with the control that makes "
               "it meaningful — the twin adjacency null and the attribution's "
               "twin-agreement self-check are always computed.")
    p.add_argument("layer_dir", help="a finished layer directory (with Grains.csv)")
    p.add_argument("--skip", nargs="*", default=[], choices=list(STAGES),
                   help="stages to skip; 'uncertainty' is much the slowest "
                        "(~0.5 s/grain)")
    p.add_argument("--space-group", type=int, default=None)
    p.add_argument("--twin-distance-um", type=float, default=45.0)
    p.add_argument("--twin-tol-deg", type=float, default=1.0)
    p.add_argument("--rel-threshold", type=float, default=None,
                   help="spot-attribution relative-likelihood cut")
    p.add_argument("--min-spots", type=int, default=None)
    p.add_argument("--sigma-obs-px", type=float, default=None,
                   help="measured spot-position noise, px. THE WHOLE SCALE of every "
                        "sigma. Default 1.0 is a placeholder, not a typical value; "
                        "the calibration stage prints the value your data imply.")
    p.add_argument("--sigma-obs-frames", type=float, default=None,
                   help="measured omega noise, in FRAMES. A frame is not a pixel; "
                        "left unset it falls back to --sigma-obs-px, which "
                        "over-assumes the omega channel.")
    p.add_argument("--max-grains-sigma", type=int, default=None,
                   help="cap the uncertainty stage; the rest still see all grains")
    p.add_argument("--json", dest="json_out", help="write the summary as JSON")
    p.add_argument("--csv", dest="csv_out",
                   help="write the per-grain uncertainty table")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    res = run_grain_qa(
        a.layer_dir, space_group=a.space_group, skip=a.skip,
        twin_distance_um=a.twin_distance_um, twin_tol_deg=a.twin_tol_deg,
        rel_threshold=a.rel_threshold, min_spots=a.min_spots,
        max_grains_sigma=a.max_grains_sigma,
        sigma_obs_px=a.sigma_obs_px, sigma_obs_frames=a.sigma_obs_frames,
        log=(None if a.quiet else lambda *m, **k: print(*m, flush=True)))
    print(res.summary())

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res.to_json(), indent=1))
        print(f"\nwrote {a.json_out}")
    if a.csv_out:
        if res.sigma is None:
            print("no uncertainty result to write (stage skipped or failed)")
        else:
            cols = res.sigma.as_columns()
            names = list(cols)
            arr = np.column_stack([np.asarray(cols[n], float) for n in names])
            # GrainID FIRST. Without it the per-grain file cannot be joined to
            # Grains.csv or to anything else -- the rows are a SUBSET (--max-grains,
            # and grains whose seed is absent from SpotsToIndex are dropped), so row
            # order is not grain order and there is no way to recover which grain a
            # row describes. Added 2026-09-03 after the file proved unusable for
            # exactly that reason.
            if getattr(res, "sigma_gid", None) is not None:
                gid = np.asarray(res.sigma_gid, float).reshape(-1, 1)
                if gid.shape[0] == arr.shape[0]:
                    arr = np.hstack([gid, arr])
                    names = ["GrainID"] + names
            np.savetxt(a.csv_out, arr, delimiter=",", header=",".join(names),
                       comments="")
            print(f"wrote {a.csv_out} ({arr.shape[0]} grains)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
