"""Data-adequacy advice: what is missing in a rotation data set, what limits the result, and what to change at the beamline.

``assess(runs, crystal, ...)`` reads the reductions of one crystal (one :class:`~midas_sx.workflow.Reduction` per rotation run, and optionally the
:class:`~midas_sx.workflow.MergedRuns` of them) and returns an :class:`AdviceReport`: a list of :class:`Finding` objects, each with the number that triggered it, the rule it was tested
against, where the number comes from, the action for the people at the beamline, what the action would buy when that is known, and what the check does NOT cover.

Rules that every check follows (they are part of the design, not style):

* A finding is raised from a MEASURED quantity of THIS data set, never from a generic checklist. A check that cannot measure its quantity (a missing column, one run only, no equivalents)
  says so as a ``note`` with ``measured['available'] = False``; it is not silent and does not guess.
* The threshold is stated in the finding, with its rationale in the module that defines it (``THRESHOLDS`` below), so that a reader can disagree with the number and not only with the text.
* Severity: ``blocking`` = the result asked for cannot be obtained from this data as it is (e.g. a lattice that does not index); ``limiting`` = the measured quantity is the main
  documented limit on the merged data's quality and an acquisition change would help; ``note`` = worth knowing, not the main limit.
* ``gain`` is filled only from something measured or simulated in this package's validation (it names the source); otherwise it is ``None``. The advice does not promise improvements.
* Advice is validated like any other result (``validation/PREREGISTER.md``): each check has a positive control (an injected defect makes it fire) and the clean control (no limiting
  finding on clean data).

The checks live in :mod:`midas_sx.advice.coverage` (rotation range, Friedel pairs, completeness, redundancy, the next setting, resolution) and :mod:`midas_sx.advice.quality`
(systematic floor, intensity drift, detector dynamic range, omega sampling, geometry consistency, contamination).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from midas_hkls import Crystal, Lattice, SpaceGroup

__all__ = ["Finding", "AdviceReport", "Acquisition", "AssessContext", "THRESHOLDS", "SEVERITIES", "assess", "register_check"]

SEVERITIES = ("blocking", "limiting", "note")

# Thresholds, all in one place. Each entry: value and the reason for it. Change a value here and the finding text changes with it.
THRESHOLDS: Dict[str, tuple] = {
    "floor_ratio": (2.0, "the systematic floor f exceeds the counting-statistics prediction by this factor: more exposure no longer helps (L-alanine, A39: 14 % against 2.7 %, ratio 5)"),
    "floor_abs": (0.08, "relative F2 floor above which equivalents disagree by more than a good data set does (L-alanine after the fixes: 5.9 %; before: 14.2 %)"),
    "drift_log_range": (0.15, "max-min of the fitted log scale within a run; +-7 % or more over a scan is a visible drift (L-alanine: 0.37-0.58)"),
    "drift_rint_gain": (0.25, "relative R_int reduction by the smooth scale; below this the scale is not doing anything the data need"),
    "friedel_min_range_deg": (180.0, "a Friedel mate crosses the Ewald sphere 180 deg away in omega; a shorter scan has (almost) no in-run pairs"),
    "completeness_n2": (0.85, "fraction of the unique reflections inside the detector's d_min seen at least twice (preregistered B1 bar of A39)"),
    "completeness_n1": (0.95, "same, seen at least once"),
    "mean_multiplicity": (4.0, "mean observations per unique observed reflection below which outlier rejection and scaling have little to work with"),
    "peak_count_fraction": (0.25, "a spot whose peak pixel exceeds this fraction of the detector's count cutoff is in the range where count-rate losses start to matter (Pilatus)"),
    "clip_min_counts": (1e4, "without a declared cutoff, ties at the maximum peak value count as clipping only above this peak (integer counts tie by chance at low values)"),
    "single_frame_fraction": (0.25, "fraction of blobs seen on one single frame (intensity-weighted RMS frame width 0) above which the omega profile is unresolved (clean 0.25 deg scene: 0.013; 1.0 deg step: 0.508)"),
    "geometry_spread_wedge_deg": (0.1, "wedge or tilt above this (or a run-to-run spread above it) means the axis or detector is not where the header says"),
    "i_over_sigma_limit": (2.0, "resolution limit is where the median I/sigma falls below this"),
    "placebo_gain_fraction": (0.5, "a permuted-frame placebo that gains more than this fraction of the real R_int gain means the smooth scale is not distinguishable from flexibility"),
    "peak_obs_fraction": (0.02, "fraction of observations with a peak pixel in the count-rate range above which it can matter for the merge"),
    "peak_effect_floor": (0.03, "difference in systematic floor (relative F2) between observations above and below the count limit that counts as a measured effect (A39: 0.002 for 43 observations)"),
    "oversampled_width_frames": (30.0, "median spot width in frames above which the rotation is oversampled"),
    "geometry_rms_px": (2.0, "robust rms of the orientation fit above which the geometry does not describe the spots (the registered gate of A37-A39)"),
    "geometry_rms_frame": (1.0, "same, in frames"),
    "lambda_spread": (5e-4, "floor on the realistic error of a fitted wavelength/cell scale: the run-to-run spread seen on L-alanine (formal sigma 1e-5 was 20-40x too small)"),
    "crowded_strong_fraction": (0.2, "fraction of the strongest blobs rejected as crowded above which the strongest reflections are missing from the merge (L-alanine: 24 of the 41 strongest unmatched)"),
    "unexplained_strong_fraction": (0.3, "fraction of the strongest blobs that no reflection of the lattice explains"),
    "strong_top_fraction": (0.05, "the 'strongest blobs' are this top fraction by integrated intensity (at least 20 per run)"),
    "ring_significance": (4.0, "sigma above the local median of the 2theta histogram of unexplained blobs for a ring-like peak"),
}


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str
    title: str
    measured: dict
    threshold: str
    evidence: str
    action: str
    gain: Optional[str] = None
    scope: str = ""

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}, got {self.severity!r}")


@dataclass
class Acquisition:
    """What the reductions cannot know. Every field is optional; a check that needs a missing one says so."""
    exposure_s: Optional[float] = None            # per frame
    count_cutoff: Optional[float] = None          # detector counts per pixel per frame (e.g. 791927 for the Pilatus header Count_cutoff)
    has_i0: Optional[bool] = None                 # is an incident-intensity (or transmitted/monitor) value recorded per frame?
    has_calibrant: Optional[bool] = None          # was a calibrant pattern measured at this energy and distance?
    max_range_deg: Optional[float] = None         # the largest rotation range the sample environment allows (a DAC opening); a shorter scan is then not a choice
    dac: Optional[object] = None                  # a midas_sx.reduce.DacModel when the sample sits in a pressure cell: reflections whose diffracted beam is blocked by the cell opening cannot be observed, and the coverage checks then count only the ACCESSIBLE ones
    notes: str = ""


@dataclass
class AssessContext:
    runs: Mapping[str, object]                    # name -> Reduction
    crystal: Crystal
    merged_runs: Optional[object] = None          # MergedRuns, or None (then the checks merge with merge_runs defaults themselves)
    acq: Acquisition = field(default_factory=Acquisition)
    options: dict = field(default_factory=dict)
    numbers: dict = field(default_factory=dict)      # per check: the numbers it measured, also when it raised no finding (a check may fill ctx.numbers[name])

    def run_names(self) -> List[str]:
        return list(self.runs.keys())

    def omega_range(self, name: str) -> tuple:
        g = self.runs[name].geom
        lo = g.omega_first_deg
        hi = lo + g.omega_step_deg * (g.n_frames - 1)
        return (min(lo, hi), max(lo, hi))

    def omega_step(self, name: str) -> float:
        return abs(float(self.runs[name].geom.omega_step_deg))

    @property
    def space_group(self) -> SpaceGroup:
        return self.crystal.space_group

    @property
    def lattice(self) -> Lattice:
        return self.crystal.lattice

    def merged(self):
        """The merge the checks read: the supplied MergedRuns, else ``workflow.merge_runs`` with its validated defaults."""
        if self.merged_runs is None:
            from ..workflow import merge_runs
            self.merged_runs = merge_runs(dict(self.runs), self.space_group)
        return self.merged_runs


@dataclass
class AdviceReport:
    findings: List[Finding]
    runs: List[str]
    checks_run: List[str]
    checks_unavailable: List[str]
    numbers: dict = field(default_factory=dict)      # measured numbers of every check that filled ctx.numbers, silent ones included

    def by_severity(self, severity: str) -> List[Finding]:
        return [f for f in self.findings if f.severity == severity]

    @property
    def ok(self) -> bool:
        """True when nothing is blocking or limiting."""
        return not (self.by_severity("blocking") or self.by_severity("limiting"))

    def text(self) -> str:
        order = {s: i for i, s in enumerate(SEVERITIES)}
        lines = [f"Data-adequacy advice for {len(self.runs)} run(s): {', '.join(self.runs)}",
                 f"  checks run: {', '.join(self.checks_run) or 'none'}" + (f"; could not measure: {', '.join(self.checks_unavailable)}" if self.checks_unavailable else ""),
                 f"  blocking {len(self.by_severity('blocking'))}, limiting {len(self.by_severity('limiting'))}, notes {len(self.by_severity('note'))}"]
        for f in sorted(self.findings, key=lambda f: (order[f.severity], f.id)):
            lines += ["", f"[{f.severity.upper()}] {f.title}  ({f.id})",
                      "  measured : " + "; ".join(f"{k} = {v}" for k, v in f.measured.items()),
                      f"  rule     : {f.threshold}",
                      f"  evidence : {f.evidence}",
                      f"  do       : {f.action}"]
            if f.gain:
                lines.append(f"  gain     : {f.gain}")
            if f.scope:
                lines.append(f"  not covered: {f.scope}")
        if self.numbers:
            lines += ["", "Measured, whether or not a finding was raised:"]
            for k, v in self.numbers.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return dict(runs=self.runs, checks_run=self.checks_run, checks_unavailable=self.checks_unavailable, ok=self.ok, numbers=self.numbers,
                    findings=[dict(id=f.id, severity=f.severity, title=f.title, measured=f.measured, threshold=f.threshold, evidence=f.evidence,
                                   action=f.action, gain=f.gain, scope=f.scope) for f in self.findings])


_CHECKS: Dict[str, Callable[[AssessContext], Sequence[Finding]]] = {}


def register_check(name: str):
    """Decorator: register ``fn(ctx) -> list[Finding]`` under ``name`` (a check raising ``CheckUnavailable`` is reported under ``checks_unavailable``)."""
    def deco(fn):
        _CHECKS[name] = fn
        return fn
    return deco


class CheckUnavailable(Exception):
    """Raised by a check that cannot measure its quantity on this input (say what is missing in the message)."""


def assess(runs: Mapping[str, object], crystal: Crystal, *, merged_runs: Optional[object] = None, acquisition: Optional[Acquisition] = None,
           checks: Optional[Sequence[str]] = None, **options) -> AdviceReport:
    """Run the registered checks on the reductions ``runs`` ({name: Reduction}) of one crystal and collect their findings.

    ``merged_runs``: the :class:`~midas_sx.workflow.MergedRuns` you refined from, if you have it (the checks otherwise merge with the validated defaults). ``acquisition``: facts the
    reductions cannot know (:class:`Acquisition`). ``checks``: a subset of check names (default all). ``options`` are passed to the checks (documented per check)."""
    from . import coverage, quality  # noqa: F401  (registers the checks)
    ctx = AssessContext(dict(runs), crystal, merged_runs, acquisition or Acquisition(), dict(options))
    names = list(checks) if checks is not None else list(_CHECKS)
    unknown = [n for n in names if n not in _CHECKS]
    if unknown:
        raise ValueError(f"unknown checks {unknown}; registered: {sorted(_CHECKS)}")
    findings: List[Finding] = []
    ran, unavailable = [], []
    for n in names:
        try:
            findings.extend(_CHECKS[n](ctx))
            ran.append(n)
        except CheckUnavailable as e:
            unavailable.append(f"{n} ({e})")
    return AdviceReport(findings, ctx.run_names(), ran, unavailable, ctx.numbers)
