"""Cross-detector spot merge.

Takes per-detector ``InputAll.csv`` / ``InputAllExtraInfoFittingAll.csv``
/ ``SpotsToIndex.csv`` files and concatenates them into a single set in
``layer_dir``. SpotID is renumbered globally (1-based) to match
ff_MIDAS / IndexerOMP convention.

Panels do **not** overlap (pinwheel layout), so spots are concatenated
rather than deduplicated. A trailing ``DetID`` column is appended so
downstream stages can treat single-det / multi-det uniformly.

For single-detector runs this stage is a no-op (all files already live
in ``layer_dir``); we still append the ``DetID`` column for consistency.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from ._base import StageContext
from .._logging import LOG, stage_timer
from ..results import CrossDetMergeResult


def _read_ring_radii(hkls_csv: Path) -> dict[int, float]:
    """Parse hkls.csv (col 4 = RingNr, col 10 = Radius). Returns {ring_nr: radius}."""
    out: dict[int, float] = {}
    if not hkls_csv.exists():
        return out
    with hkls_csv.open() as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.replace(",", " ").split()
            try:
                rn = int(float(toks[4]))
                rad = float(toks[10])
            except (IndexError, ValueError):
                continue
            out.setdefault(rn, rad)
    return out


def _merge_paramstest(per_det_paramstest: list[Path], det_configs,
                      hkls_per_det: list[Path],
                      out_path: Path) -> None:
    """Rebuild paramstest from per-det panels:

      - bare global geometry keys (Lsd/BC/tx/.../p0..p10/RingRadii) are
        replaced with first-detector globals plus a full ``DetParams`` row
        per panel and per-detector ``RingRadii_DetN`` blocks;
      - ``OutputFolder`` / ``ResultFolder`` are repointed at ``layer_dir``;
      - ``EtaCoverage_DetN`` rows are carried through from each panel.
    """
    base = per_det_paramstest[0]
    text = base.read_text()
    lines = []
    skip_keys = {"Lsd", "Distance", "tx", "ty", "tz",
                 "YBC", "ZBC", "BC",
                 "RingRadii",
                 "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
                 "DetParams",
                 "OutputFolder", "ResultFolder"}
    for raw in text.splitlines():
        line = raw.strip().rstrip(";").rstrip()
        if not line or line.startswith("#"):
            lines.append(raw)
            continue
        key = line.split()[0]
        if key in skip_keys:
            continue
        if key.startswith("EtaCoverage_Det"):
            continue
        lines.append(raw)

    layer_dir = out_path.parent
    lines.append(f"OutputFolder {layer_dir / 'Output'}")
    lines.append(f"ResultFolder {layer_dir / 'Results'}")

    if det_configs:
        d0 = det_configs[0]
        lines.append(f"Lsd {d0.lsd}")
        lines.append(f"BC {d0.y_bc} {d0.z_bc}")
        lines.append(f"tx {d0.tx}")
        lines.append(f"ty {d0.ty}")
        lines.append(f"tz {d0.tz}")
        for i, pv in enumerate(d0.p_distortion):
            lines.append(f"p{i} {pv}")

    nominal_radii: dict[int, float] = {}
    for det, hkls in zip(det_configs, hkls_per_det):
        p = " ".join(f"{v}" for v in det.p_distortion)
        lines.append(
            f"DetParams {det.det_id} {det.lsd} {det.y_bc} {det.z_bc} "
            f"{det.tx} {det.ty} {det.tz} {p}"
        )
        radii = _read_ring_radii(hkls)
        if not radii:
            continue
        if not nominal_radii:
            nominal_radii = dict(radii)
        for rn in sorted(radii):
            lines.append(f"RingRadii_Det{det.det_id} {rn} {radii[rn]}")

    if nominal_radii:
        for rn in sorted(nominal_radii):
            lines.append(f"RingRadii {nominal_radii[rn]}")

    for det, pt in zip(det_configs, per_det_paramstest):
        if not pt.exists():
            continue
        for raw in pt.read_text().splitlines():
            line = raw.strip().rstrip(";").rstrip()
            if not line:
                continue
            key = line.split()[0]
            if key == f"EtaCoverage_Det{det.det_id}":
                lines.append(raw)

    out_path.write_text("\n".join(lines) + "\n")


def _append_detid_col(path: Path, det_id: int) -> int:
    """Append a trailing ``DetID`` column to a midas-fit-setup InputAll CSV.

    Idempotent: if already present, just count data rows. Returns the
    number of data rows.
    """
    if not path.exists():
        return 0
    text = path.read_text()
    lines = text.splitlines()
    if not lines:
        return 0
    head = lines[0].split()
    if head and head[-1] == "DetID":
        return max(0, len(lines) - 1)
    new_lines: list[str] = []
    n_rows = 0
    for i, raw in enumerate(lines):
        toks = raw.split()
        if i == 0:
            new_lines.append(raw + " DetID")
            continue
        if not toks:
            new_lines.append(raw)
            continue
        new_lines.append(raw + f" {det_id}")
        n_rows += 1
    path.write_text("\n".join(new_lines) + "\n")
    return n_rows


def _do_single_detector(ctx: StageContext, outputs: dict[str, str]):
    """No-op merge: single-detector InputAll.csv is already at layer_dir.
    Append a DetID column so binning + downstream stages can treat
    single-det / multi-det uniformly."""
    input_all = ctx.layer_dir / "InputAll.csv"
    extra_csv = ctx.layer_dir / "InputAllExtraInfoFittingAll.csv"
    det_id = ctx.detectors[0].det_id if ctx.detectors else 1
    n_total = _append_detid_col(input_all, det_id)
    _append_detid_col(extra_csv, det_id)
    outputs[str(input_all)] = ""
    ctx.merged_paramstest = ctx.layer_dir / "paramstest.txt"
    return input_all, input_all, n_total, [n_total]


def _do_multi_detector(ctx: StageContext, outputs: dict[str, str]):
    """Concatenate per-det InputAll.csv + InputAllExtraInfoFittingAll.csv
    into ``layer_dir``, renumber SpotIDs globally, and rewrite paramstest.

    Binning runs after this stage on the merged ``InputAll.csv``.
    """
    per_det_counts: list[int] = []

    merged_input_all = ctx.layer_dir / "InputAll.csv"
    merged_extra_csv = ctx.layer_dir / "InputAllExtraInfoFittingAll.csv"

    next_spot_id = 1
    fp_in = merged_input_all.open("w")
    fp_extra = merged_extra_csv.open("w")
    seed_globals: list[int] = []  # for merged SpotsToIndex.csv
    try:
        for det_idx, det in enumerate(ctx.detectors):
            det_dir = ctx.detector_dir(det)
            ia = det_dir / "InputAll.csv"
            ie = det_dir / "InputAllExtraInfoFittingAll.csv"
            sti = det_dir / "SpotsToIndex.csv"
            offset = next_spot_id - 1   # local L → global = L + offset
            n_det = 0
            if ia.exists():
                with ia.open() as fp:
                    head = fp.readline()
                    if det_idx == 0:
                        fp_in.write(head.rstrip("\n") + " DetID\n")
                    for line in fp:
                        toks = line.rstrip("\n").split()
                        if len(toks) < 5:
                            continue
                        toks[4] = str(next_spot_id + n_det)
                        toks.append(str(det.det_id))
                        fp_in.write(" ".join(toks) + "\n")
                        n_det += 1
            if ie.exists():
                with ie.open() as fp:
                    head = fp.readline()
                    if det_idx == 0:
                        fp_extra.write(head.rstrip("\n") + " DetID\n")
                    eid = next_spot_id
                    for line in fp:
                        toks = line.rstrip("\n").split()
                        if len(toks) < 5:
                            continue
                        toks[4] = str(eid)
                        toks.append(str(det.det_id))
                        fp_extra.write(" ".join(toks) + "\n")
                        eid += 1
            if sti.exists():
                with sti.open() as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            local_id = int(line.split()[0])
                        except ValueError:
                            continue
                        seed_globals.append(local_id + offset)
            per_det_counts.append(n_det)
            next_spot_id += n_det
            LOG.info("  det %d: %d spots", det.det_id, n_det)
    finally:
        fp_in.close()
        fp_extra.close()

    merged_sti = ctx.layer_dir / "SpotsToIndex.csv"
    with merged_sti.open("w") as fp:
        for sid in seed_globals:
            fp.write(f"{sid}\n")
    outputs[str(merged_sti)] = ""

    n_total = next_spot_id - 1
    outputs[str(merged_input_all)] = ""
    outputs[str(merged_extra_csv)] = ""

    per_det_paramstest = [
        ctx.detector_dir(d) / "paramstest.txt" for d in ctx.detectors
    ]
    per_det_hkls = [
        ctx.detector_dir(d) / "hkls.csv" for d in ctx.detectors
    ]
    merged_paramstest_path = ctx.layer_dir / "paramstest.txt"
    _merge_paramstest(per_det_paramstest, ctx.detectors, per_det_hkls,
                      merged_paramstest_path)
    outputs[str(merged_paramstest_path)] = ""
    ctx.merged_paramstest = merged_paramstest_path

    hkls_canonical = ctx.layer_dir / "hkls.csv"
    if not hkls_canonical.exists():
        for det in ctx.detectors:
            cand = ctx.detector_dir(det) / "hkls.csv"
            if cand.exists():
                shutil.copy2(cand, hkls_canonical)
                break

    return merged_input_all, merged_input_all, n_total, per_det_counts


def run(ctx: StageContext) -> CrossDetMergeResult:
    started = time.time()
    outputs: dict[str, str] = {}

    with stage_timer("cross_det_merge"):
        if ctx.is_multi_detector:
            input_all_path, input_all_det_path, n_total, per_det_counts = \
                _do_multi_detector(ctx, outputs)
        else:
            input_all_path, input_all_det_path, n_total, per_det_counts = \
                _do_single_detector(ctx, outputs)
        spots_path = input_all_path
        spots_det_path = input_all_det_path

    finished = time.time()
    return CrossDetMergeResult(
        stage_name="cross_det_merge",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs=outputs,
        n_total_spots=n_total,
        n_per_detector=per_det_counts,
        spots_bin=str(spots_path),
        spots_det_bin=str(spots_det_path),
        metrics={"n_detectors": len(ctx.detectors), "n_spots": n_total},
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    if ctx.is_multi_detector:
        return [
            ctx.layer_dir / "InputAll.csv",
            ctx.layer_dir / "paramstest.txt",
        ]
    return [ctx.layer_dir / "InputAll.csv"]
