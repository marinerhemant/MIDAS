"""Cross-field consistency rules.

Per-key validators live in validators.py. This module holds checks that
span multiple keys — e.g. "nDistances must match the number of Lsd entries",
or "BoxSize must be within NrPixels × px".

Rule functions take the full parsed parameter dict and a context, and
return a list of ValidationIssue.
"""

from __future__ import annotations

from typing import Any, Callable

from .schema import CrossFieldRule, Path, Severity, ValidationIssue
from .validators import Ctx


RuleFn = Callable[[Ctx], list[ValidationIssue]]


# ─── NF: multi-entry count consistency ───────────────────────────────────────


def nf_multi_entry_count_matches(ctx: Ctx) -> list[ValidationIssue]:
    """Lsd and BC must appear exactly nDistances times (one per distance).

    OmegaRange and BoxSize are NOT count-checked here: per MIDAS convention,
    they may appear one or more times per distance (single line shared across
    all distances, or multiple lines defining subset windows). There is no
    single "correct" count for them.
    """
    n = ctx.all_values.get("nDistances")
    if n is None:
        return []

    def _occurrences(key: str) -> int:
        raw = ctx.all_values.get(key)
        if raw is None:
            return 0
        return len(raw) if isinstance(raw, list) else 1

    out = []

    for key in ("Lsd", "BC"):
        count = _occurrences(key)
        if count == 0:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=key,
                message=f"{key} is required by NF when nDistances={n} but no entries found.",
                rule="nf_multi_entry_count_matches",
            ))
            continue
        if count != n:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key=key,
                line=ctx.line_of.get(key),
                message=f"{key} has {count} entries but nDistances={n}. "
                        f"Each distance needs its own {key} line.",
                suggestion=f"Add {n - count} more {key} line(s)." if count < n
                           else f"Remove {count - n} extra {key} line(s).",
                rule="nf_multi_entry_count_matches",
            ))

    return out


# ─── Omega direction consistency ─────────────────────────────────────────────


def omega_step_direction(ctx: Ctx) -> list[ValidationIssue]:
    """OmegaStep sign must match the direction from OmegaStart → OmegaEnd.

    Example: start=180, end=-180, step must be negative. Reversing the sign
    silently produces mirrored reconstructions.
    """
    start = ctx.all_values.get("OmegaStart")
    end = ctx.all_values.get("OmegaEnd")
    step = ctx.all_values.get("OmegaStep")
    if None in (start, end, step):
        return []
    if end == start or step == 0:
        return []
    travel = end - start
    if (travel > 0) != (step > 0):
        return [ValidationIssue(
            severity=Severity.ERROR,
            key="OmegaStep",
            line=ctx.line_of.get("OmegaStep"),
            message=(
                f"OmegaStep sign ({'+' if step > 0 else '-'}) does not match "
                f"travel direction OmegaStart={start} → OmegaEnd={end}."
            ),
            suggestion=f"Use OmegaStep={-step} to travel in the right direction.",
            rule="omega_step_direction",
        )]
    return []


# ─── FF indexing: OverAllRingToIndex must be in RingThresh ───────────────────


def overallring_in_ringthresh(ctx: Ctx) -> list[ValidationIssue]:
    """The ring used for seed-orientation generation must be in the active
    ring list (RingThresh)."""
    overall = ctx.all_values.get("OverAllRingToIndex")
    thresh = ctx.all_values.get("RingThresh")
    if overall is None or thresh is None:
        return []
    # RingThresh is multi-entry; each entry is [ring_nr, intensity]
    entries = thresh if thresh and isinstance(thresh[0], list) else [thresh]
    ring_nrs = {e[0] for e in entries if len(e) >= 1}
    if overall not in ring_nrs:
        return [ValidationIssue(
            severity=Severity.ERROR,
            key="OverAllRingToIndex",
            line=ctx.line_of.get("OverAllRingToIndex"),
            message=(
                f"OverAllRingToIndex={overall} but that ring is not in "
                f"RingThresh ({sorted(ring_nrs)})."
            ),
            suggestion=f"Add `RingThresh {overall} <threshold>` or change "
                       f"OverAllRingToIndex to one of {sorted(ring_nrs)}.",
            rule="overallring_in_ringthresh",
        )]
    return []


# ─── File discovery: StartNr..EndNr span ──────────────────────────────────────


def startnr_le_endnr(ctx: Ctx) -> list[ValidationIssue]:
    s = ctx.all_values.get("StartNr")
    e = ctx.all_values.get("EndNr")
    if s is None or e is None:
        return []
    if s > e:
        return [ValidationIssue(
            severity=Severity.ERROR,
            key="EndNr",
            line=ctx.line_of.get("EndNr"),
            message=f"StartNr={s} > EndNr={e}; frame range is empty.",
            rule="startnr_le_endnr",
        )]
    return []


def nf_frames_match_files_per_distance(ctx: Ctx) -> list[ValidationIssue]:
    """In NF: StartNr..EndNr is a PER-DISTANCE frame range. The span should
    equal NrFilesPerDistance + WFImages (same frame indices get reused for
    every distance with an internal offset in the binary layout)."""
    if ctx.path != "nf":
        return []
    s = ctx.all_values.get("StartNr")
    e = ctx.all_values.get("EndNr")
    nfd = ctx.all_values.get("NrFilesPerDistance")
    wf = ctx.all_values.get("WFImages") or 0
    if None in (s, e, nfd):
        return []
    expected = nfd + wf
    actual = e - s + 1
    if actual != expected:
        return [ValidationIssue(
            severity=Severity.WARNING,
            key="EndNr",
            line=ctx.line_of.get("EndNr"),
            message=(
                f"Frame span {actual} (EndNr-StartNr+1) does not match "
                f"expected {expected} = NrFilesPerDistance={nfd} + "
                f"WFImages={wf}."
            ),
            rule="nf_frames_match_files_per_distance",
        )]
    return []


# ─── Rings subset of MaxRingNumber ───────────────────────────────────────────


def omega_range_within_scan(ctx: Ctx) -> list[ValidationIssue]:
    """Each OmegaRange window must fall within the actual scanned ω range.

    Actual scan range is computed as:
      FF/PF:  [OmegaStart, OmegaEnd] (sorted), falling back to
              OmegaStart + OmegaStep × nFrames when OmegaEnd absent
      NF:     OmegaStart + OmegaStep × NrFilesPerDistance

    nFrames for FF/PF ≈ (EndNr - StartNr + 1) / NrFilesPerSweep.

    A small tolerance (|OmegaStep| + 1e-6) is allowed on each side so that
    exact-boundary windows like "-180 180" don't trip the rule due to
    floating-point rounding.
    """
    start = ctx.all_values.get("OmegaStart")
    step = ctx.all_values.get("OmegaStep")
    ome_ranges = ctx.all_values.get("OmegaRange")
    if start is None or step is None or ome_ranges is None:
        return []

    # Compute the actual scan endpoint
    end = ctx.all_values.get("OmegaEnd")
    if end is None:
        # NF path: use NrFilesPerDistance
        nframes = ctx.all_values.get("NrFilesPerDistance")
        if nframes is None:
            # FF/PF fallback: (EndNr - StartNr + 1) / NrFilesPerSweep
            snr = ctx.all_values.get("StartNr")
            enr = ctx.all_values.get("EndNr")
            per_sweep = ctx.all_values.get("NrFilesPerSweep") or 1
            if snr is None or enr is None:
                return []  # not enough info to bound
            nframes = (enr - snr + 1) // max(1, per_sweep)
        end = start + step * nframes

    lo = min(start, end)
    hi = max(start, end)
    tol = abs(step) + 1e-6

    entries = ome_ranges if ome_ranges and isinstance(ome_ranges[0], list) else [ome_ranges]
    out = []
    for i, e in enumerate(entries):
        if not (isinstance(e, list) and len(e) == 2):
            continue  # arity validator handles this
        w_min, w_max = e
        if w_min < lo - tol or w_max > hi + tol:
            out.append(ValidationIssue(
                severity=Severity.ERROR,
                key="OmegaRange",
                line=ctx.line_of.get("OmegaRange"),
                message=(
                    f"OmegaRange entry {i+1}=[{w_min}, {w_max}] is outside the "
                    f"scanned ω range [{lo:g}, {hi:g}] "
                    f"(OmegaStart={start}, OmegaStep={step}, so scan reaches {end:g})."
                ),
                suggestion=f"Clamp the window to lie within [{lo:g}, {hi:g}].",
                rule="omega_range_within_scan",
            ))
    return out


def ri_rmax_gt_rmin(ctx: Ctx) -> list[ValidationIssue]:
    """RMax must be greater than RMin, and both should fit within RhoD."""
    rmin = ctx.all_values.get("RMin")
    rmax = ctx.all_values.get("RMax")
    rhod = ctx.all_values.get("RhoD")
    out = []
    if rmin is not None and rmax is not None and rmax <= rmin:
        out.append(ValidationIssue(
            severity=Severity.ERROR,
            key="RMax",
            line=ctx.line_of.get("RMax"),
            message=f"RMax={rmax} must be greater than RMin={rmin}.",
            rule="ri_rmax_gt_rmin",
        ))
    if rmax is not None and rhod is not None and rmax > rhod:
        out.append(ValidationIssue(
            severity=Severity.WARNING,
            key="RMax",
            line=ctx.line_of.get("RMax"),
            message=f"RMax={rmax} exceeds RhoD={rhod}; radial integration "
                    f"beyond detector coverage.",
            rule="ri_rmax_gt_rmin",
        ))
    return out


def ri_etamax_gt_etamin(ctx: Ctx) -> list[ValidationIssue]:
    """EtaMax must be greater than EtaMin (within [-180, 360] range)."""
    emin = ctx.all_values.get("EtaMin")
    emax = ctx.all_values.get("EtaMax")
    if emin is None or emax is None:
        return []
    out = []
    if emax <= emin:
        out.append(ValidationIssue(
            severity=Severity.ERROR,
            key="EtaMax",
            line=ctx.line_of.get("EtaMax"),
            message=f"EtaMax={emax} must be greater than EtaMin={emin}.",
            rule="ri_etamax_gt_etamin",
        ))
    if emax - emin > 360.001:
        out.append(ValidationIssue(
            severity=Severity.WARNING,
            key="EtaMax",
            line=ctx.line_of.get("EtaMax"),
            message=f"Eta range {emax - emin}° spans more than a full revolution.",
            rule="ri_etamax_gt_etamin",
        ))
    return out


def nrpixels_either_or(ctx: Ctx) -> list[ValidationIssue]:
    """At least one of {NrPixels, (NrPixelsY AND NrPixelsZ)} must be set.

    The MIDAS text parser treats NrPixels as a shortcut that copies its
    value into both NrPixelsY and NrPixelsZ, so either form satisfies the
    detector-dimension requirement.
    """
    has_np = "NrPixels" in ctx.all_values
    has_y = "NrPixelsY" in ctx.all_values
    has_z = "NrPixelsZ" in ctx.all_values
    if has_np or (has_y and has_z):
        # Also warn if Y and Z disagree with NrPixels
        if has_np and has_y and has_z:
            np_v = ctx.all_values["NrPixels"]
            y_v = ctx.all_values["NrPixelsY"]
            z_v = ctx.all_values["NrPixelsZ"]
            if not (np_v == y_v == z_v):
                return [ValidationIssue(
                    severity=Severity.WARNING,
                    key="NrPixels",
                    line=ctx.line_of.get("NrPixels"),
                    message=(
                        f"NrPixels={np_v} disagrees with NrPixelsY={y_v} / "
                        f"NrPixelsZ={z_v}. The text parser will use NrPixels "
                        f"and overwrite Y/Z to match it."
                    ),
                    suggestion=f"Set only NrPixels (for square detectors) OR "
                               f"only NrPixelsY/NrPixelsZ (for non-square).",
                    rule="nrpixels_either_or",
                )]
        return []
    return [ValidationIssue(
        severity=Severity.ERROR,
        message=(
            "Detector size is not set. Provide either NrPixels (for a square "
            "detector) or both NrPixelsY and NrPixelsZ."
        ),
        key="NrPixels",
        suggestion="Add e.g. `NrPixels 2048` for a 2048×2048 detector, or "
                   "`NrPixelsY 2048` + `NrPixelsZ 512` for non-square.",
        rule="nrpixels_either_or",
    )]


def pf_nscans_implies_scanstep(ctx: Ctx) -> list[ValidationIssue]:
    """When nScans > 1 (PF mode), ScanStep is effectively required so
    positions.csv can be generated; warn if missing."""
    n = ctx.all_values.get("nScans")
    if n is None or n <= 1:
        return []
    if "ScanStep" not in ctx.all_values and "BeamSize" not in ctx.all_values:
        return [ValidationIssue(
            severity=Severity.WARNING,
            key="nScans",
            line=ctx.line_of.get("nScans"),
            message=f"nScans={n} (point-focus mode) but neither ScanStep nor "
                    f"BeamSize is set. positions.csv cannot be generated.",
            suggestion="Set ScanStep (µm) and BeamSize (µm) for PF-HEDM.",
            rule="pf_nscans_implies_scanstep",
        )]
    return []


def frames_exist_on_disk(ctx: Ctx) -> list[ValidationIssue]:
    """Verify that the raw frame files implied by the param set actually
    exist on disk.

    File-numbering convention:
      FF/PF: the on-disk file numbers are
             `StartFileNrFirstLayer` .. `StartFileNrFirstLayer + NrFilesPerSweep − 1`.
             Internal frame indices (StartNr..EndNr, often 1..1440) live
             INSIDE a multi-frame GE/HDF5 file — they are NOT the on-disk
             filenames. A GE container with 1440 frames is a single file.
             If `StartFileNrFirstLayer` is absent, fall back to `StartNr`.
      NF:    per-distance file numbering — `RawStartNr .. RawStartNr +
             NrFilesPerDistance + WFImages − 1`.

    Checks a small sample (first, last, up to 3 interior) rather than
    stat'ing every file.
    """
    from pathlib import Path as FsPath

    if ctx.path == "nf":
        folder = ctx.all_values.get("DataDirectory")
        stem = ctx.all_values.get("OrigFileName")
        ext = ctx.all_values.get("extOrig") or "tif"
        pad = ctx.all_values.get("Padding") or 6
        start = ctx.all_values.get("RawStartNr")
        nfd = ctx.all_values.get("NrFilesPerDistance")
        wf = ctx.all_values.get("WFImages") or 0
        if None in (folder, stem, start, nfd):
            return []
        end = start + (nfd + wf) - 1
        ext_with_dot = ext if ext.startswith(".") else f".{ext}"
        key_for_error = "OrigFileName"
    else:
        folder = ctx.all_values.get("RawFolder")
        stem = ctx.all_values.get("FileStem")
        ext = ctx.all_values.get("Ext")
        pad = ctx.all_values.get("Padding") or 6
        # On-disk file numbers: prefer StartFileNrFirstLayer; fall back to StartNr.
        start = ctx.all_values.get("StartFileNrFirstLayer")
        if start is None:
            start = ctx.all_values.get("StartNr")
        nfilessweep = ctx.all_values.get("NrFilesPerSweep") or 1
        if None in (folder, stem, ext, start):
            return []
        end = start + nfilessweep - 1
        ext_with_dot = ext if ext.startswith(".") else f".{ext}"
        key_for_error = "FileStem"

    folder_path = FsPath(folder).expanduser()
    if not folder_path.is_dir():
        return []  # directory_exists validator catches this

    # Sample check: first, last, and up to 3 evenly-spaced interior frames.
    n_total = max(0, end - start + 1)
    if n_total <= 0:
        return []
    samples = {start, end}
    if n_total > 4:
        step = n_total // 4
        samples.update({start + step, start + 2 * step, start + 3 * step})
    samples = sorted(samples)

    out = []
    missing = []
    for n in samples:
        filename = f"{stem}_{n:0{pad}d}{ext_with_dot}"
        p = folder_path / filename
        if not p.exists():
            missing.append((n, filename))

    if missing:
        # Don't spam: report the first one with a summary count.
        first_n, first_name = missing[0]
        out.append(ValidationIssue(
            severity=Severity.ERROR,
            key=key_for_error,
            line=ctx.line_of.get(key_for_error),
            message=(
                f"Expected frame file missing: {folder_path / first_name}"
                + (f" (and {len(missing) - 1} other sampled frame(s) also missing)"
                   if len(missing) > 1 else "")
            ),
            suggestion=(
                f"Check FileStem={stem!r}, Padding={pad}, Ext={ext!r}, and that "
                f"frames {start}..{end} live in {folder_path}."
            ),
            rule="frames_exist_on_disk",
        ))
    return out


def ring_numbers_below_max(ctx: Ctx) -> list[ValidationIssue]:
    maxr = ctx.all_values.get("MaxRingNumber")
    if not maxr:
        return []
    out = []
    thresh = ctx.all_values.get("RingThresh")
    if thresh:
        entries = thresh if thresh and isinstance(thresh[0], list) else [thresh]
        for e in entries:
            if len(e) >= 1 and e[0] > maxr:
                out.append(ValidationIssue(
                    severity=Severity.ERROR,
                    key="RingThresh",
                    line=ctx.line_of.get("RingThresh"),
                    message=f"RingThresh references ring {e[0]} but MaxRingNumber={maxr}.",
                    rule="ring_numbers_below_max",
                ))
    return out


# ─── Lookup table ────────────────────────────────────────────────────────────


RULES: dict[str, RuleFn] = {
    "nf_multi_entry_count_matches": nf_multi_entry_count_matches,
    "omega_step_direction": omega_step_direction,
    "omega_range_within_scan": omega_range_within_scan,
    "overallring_in_ringthresh": overallring_in_ringthresh,
    "startnr_le_endnr": startnr_le_endnr,
    "nf_frames_match_files_per_distance": nf_frames_match_files_per_distance,
    "ring_numbers_below_max": ring_numbers_below_max,
    "frames_exist_on_disk": frames_exist_on_disk,
    "ri_rmax_gt_rmin": ri_rmax_gt_rmin,
    "ri_etamax_gt_etamin": ri_etamax_gt_etamin,
    "pf_nscans_implies_scanstep": pf_nscans_implies_scanstep,
    "nrpixels_either_or": nrpixels_either_or,
}


# ─── Rule declarations (data the engine walks) ───────────────────────────────


RULE_SPECS: list[CrossFieldRule] = [
    CrossFieldRule(
        name="nf_multi_entry_count_matches",
        description="Lsd/BC/OmegaRange/BoxSize entry counts must equal nDistances.",
        applies_to=frozenset({Path.NF}),
        severity=Severity.ERROR,
        check="nf_multi_entry_count_matches",
    ),
    CrossFieldRule(
        name="omega_step_direction",
        description="Sign of OmegaStep must match direction from OmegaStart to OmegaEnd.",
        applies_to=frozenset({Path.FF, Path.NF, Path.PF}),
        severity=Severity.ERROR,
        check="omega_step_direction",
    ),
    CrossFieldRule(
        name="overallring_in_ringthresh",
        description="OverAllRingToIndex must appear in the RingThresh list.",
        applies_to=frozenset({Path.FF, Path.PF}),
        severity=Severity.ERROR,
        check="overallring_in_ringthresh",
    ),
    CrossFieldRule(
        name="startnr_le_endnr",
        description="StartNr must not exceed EndNr.",
        applies_to=frozenset({Path.FF, Path.NF, Path.PF, Path.RI}),
        severity=Severity.ERROR,
        check="startnr_le_endnr",
    ),
    CrossFieldRule(
        name="nf_frames_match_files_per_distance",
        description="(EndNr-StartNr+1) should equal (NrFilesPerDistance+WFImages)×nDistances.",
        applies_to=frozenset({Path.NF}),
        severity=Severity.WARNING,
        check="nf_frames_match_files_per_distance",
    ),
    CrossFieldRule(
        name="omega_range_within_scan",
        description="OmegaRange windows must fall within the scanned ω range "
                    "computed from OmegaStart + OmegaStep × nFrames.",
        applies_to=frozenset({Path.FF, Path.NF, Path.PF}),
        severity=Severity.ERROR,
        check="omega_range_within_scan",
    ),
    CrossFieldRule(
        name="ring_numbers_below_max",
        description="RingThresh entries must not exceed MaxRingNumber.",
        applies_to=frozenset({Path.FF, Path.PF}),
        severity=Severity.ERROR,
        check="ring_numbers_below_max",
    ),
    CrossFieldRule(
        name="frames_exist_on_disk",
        description="Sampled frame files (first, last, some interior) must exist "
                    "in RawFolder/DataDirectory. Catches off-by-one counts, wrong "
                    "FileStem, missing data.",
        applies_to=frozenset({Path.FF, Path.NF, Path.PF}),
        severity=Severity.ERROR,
        check="frames_exist_on_disk",
    ),
    CrossFieldRule(
        name="ri_rmax_gt_rmin",
        description="RMax must be greater than RMin; RMax should not exceed RhoD.",
        applies_to=frozenset({Path.RI}),
        severity=Severity.ERROR,
        check="ri_rmax_gt_rmin",
    ),
    CrossFieldRule(
        name="ri_etamax_gt_etamin",
        description="EtaMax must be greater than EtaMin; range ≤ 360°.",
        applies_to=frozenset({Path.RI}),
        severity=Severity.ERROR,
        check="ri_etamax_gt_etamin",
    ),
    CrossFieldRule(
        name="pf_nscans_implies_scanstep",
        description="PF mode (nScans > 1) needs ScanStep + BeamSize for "
                    "positions.csv generation.",
        applies_to=frozenset({Path.PF}),
        severity=Severity.WARNING,
        check="pf_nscans_implies_scanstep",
    ),
    CrossFieldRule(
        name="nrpixels_either_or",
        description="Detector size: one of NrPixels or (NrPixelsY and "
                    "NrPixelsZ) is required; warns if all three are set "
                    "with conflicting values.",
        applies_to=frozenset({Path.FF, Path.NF, Path.PF, Path.RI}),
        severity=Severity.ERROR,
        check="nrpixels_either_or",
    ),
]
