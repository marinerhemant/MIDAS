"""Cross-detector spot merge.

Takes per-detector ``InputAll.csv`` / ``InputAllExtraInfoFittingAll.csv``
/ ``SpotsToIndex.csv`` files and concatenates them into a single set in
``layer_dir``. SpotID is renumbered globally (1-based) to match
ff_MIDAS / IndexerOMP convention.

Panels in a pinwheel layout do not overlap, so their spots are concatenated.
Panels that do overlap record one reflection twice. The overlap is always
measured (``panel_overlap``) and written to ``PanelOverlap.csv``: how many spots
are duplicated, the matching window the data support, and the relative
translation / scale / rotation of each panel pair with its uncertainty. The
duplicates are dropped only when ``dedup_panels`` is set. A trailing ``DetID``
column is appended so downstream stages can treat single-det / multi-det
uniformly.

For single-detector runs this stage is a no-op (all files already live
in ``layer_dir``); we still append the ``DetID`` column for consistency.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional

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
                 "DetParams", "RingNumbers",
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

    # Union of the panels' rings, not the first panel's: a ring only a later
    # panel reaches would otherwise get spots in InputAll.csv but no
    # EtaCoverage arcs and no slot in the indexer's ring list.
    ring_union: set[int] = set()
    for pt in per_det_paramstest:
        if not pt.exists():
            continue
        for raw in pt.read_text().splitlines():
            toks = raw.strip().rstrip(";").split()
            if len(toks) >= 2 and toks[0] == "RingNumbers":
                try:
                    ring_union.add(int(float(toks[1])))
                except ValueError:
                    continue
    lines.extend(f"RingNumbers {r}" for r in sorted(ring_union))

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


def _split_detid_head(head: str) -> tuple[str, bool]:
    """``(header without a trailing DetID, whether it had one)``.

    Per-panel CSVs reach the merge either bare (straight from ``transforms``)
    or already carrying ``DetID`` (a panel that went through the single-detector
    path, i.e. ``_append_detid_col``). The merge owns the column, so it strips
    one if present and writes exactly one.
    """
    toks = head.split()
    has = bool(toks) and toks[-1] == "DetID"
    return " ".join(toks[:-1] if has else toks), has


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


def _ring_d0(path: Path) -> dict[int, float]:
    """``{ring: d0}`` from one panel's ``IDsHash.csv`` (``ring id_min id_max d0``)."""
    out: dict[int, float] = {}
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        toks = raw.split()
        if len(toks) < 4:
            continue
        try:
            out.setdefault(int(toks[0]), float(toks[3]))
        except ValueError:
            continue
    return out


def _ids_hash_rows(rings: list[int], d0: dict[int, float]) -> list[str]:
    """``IDsHash.csv`` rows for the merged table, one per run of equal ring.

    ``rings[i]`` is the ring of merged SpotID ``i + 1``. A run is a maximal
    block of consecutive SpotIDs on one ring; its range is exact
    (``id_min`` .. last id + 1, exclusive), so a SpotID maps to its ring with no
    overlap between rows. Within one panel a ring is one block; across panels it
    is one block per panel, which is why a ring appears more than once. Building
    it from the final table (not by shifting each panel's file) keeps it right
    after duplicate rows have been dropped. ``process_grains`` refuses to run
    without this file: it is the only source of the per-ring reference
    d-spacing for the strain gauge.
    """
    rows: list[str] = []
    i, n = 0, len(rings)
    while i < n:
        j = i
        while j + 1 < n and rings[j + 1] == rings[i]:
            j += 1
        if rings[i] in d0:
            rows.append(f"{rings[i]} {i + 1} {j + 2} {d0[rings[i]]}")
        i = j + 1
    return rows


def _emit_eta_coverage(paramstest: Path) -> None:
    """Append ``EtaCoverage_Det<N>`` rows for every panel to the merged paramstest.

    Each row says which eta arcs of a ring one panel physically covers. They
    are written here because this stage owns the merged paramstest and has
    every panel's geometry; midas-transforms does not write them (the retired
    midas-ff-pipeline transforms stage did), so they are derived from the
    ``DetParams`` and ``RingRadii_Det<N>`` rows just written. Rows already
    present are kept.

    Consumers: ``global_powder`` (per-ring coverage). The C indexer PARSES
    these rows but does not yet use them: ``IndexerUnified.c`` stores
    ``EtaArc*`` / ``hasEtaCoverage`` and never reads them (Phase 9a
    scaffolding, Phase 9b TODO), so its completeness denominator still counts
    every predicted spot, on or off a panel.
    """
    from ..eta_coverage import parse_coverage_blocks, write_coverage_block
    from .global_powder import _coverage_from_geometry

    text = paramstest.read_text()
    if parse_coverage_blocks(text):
        return
    arcs_by_det = _coverage_from_geometry(text)
    if not arcs_by_det:
        LOG.warning("cross_det_merge: could not derive η coverage from %s; "
                    "the indexer will count every predicted spot, not just "
                    "those that land on a panel", paramstest)
        return
    for det_id, arcs in arcs_by_det.items():
        write_coverage_block(paramstest, det_id, arcs)
    LOG.info("  wrote EtaCoverage rows for %d panel(s)", len(arcs_by_det))


def _load_table(path: Path):
    """``(header without DetID, rows)`` of a per-panel CSV; rows are token lists
    without any trailing DetID. Rows with fewer than 5 tokens are skipped, as
    they always were."""
    if not path.exists():
        return None, []
    with path.open() as fp:
        head, has_detid = _split_detid_head(fp.readline())
        rows = []
        for line in fp:
            toks = line.rstrip("\n").split()
            if len(toks) < 5:
                continue
            if has_detid:
                toks.pop()
            rows.append(toks)
    return head, rows


def _kv_float(path: Path, key: str) -> Optional[float]:
    try:
        for raw in path.read_text().splitlines():
            t = raw.strip().rstrip(";").split()
            if len(t) >= 2 and t[0] == key:
                return float(t[1])
    except (OSError, ValueError):
        pass
    return None


def _edge_scores(ctx: StageContext, dets, ie_rows):
    """Per-panel, per-row distance (px) of the spot from its panel's edge, or
    None when the raw pixel position or the panel size is not available.

    Of two records of one reflection the one farther from an edge is the better
    measurement: a spot cut by the panel edge is biased in position and
    intensity.
    """
    import numpy as np
    out = []
    for k, det in enumerate(dets):
        pt = ctx.detector_dir(det) / "paramstest.txt"
        ny, nz = _kv_float(pt, "NrPixelsY"), _kv_float(pt, "NrPixelsZ")
        rows = ie_rows[k]
        if not ny or not nz or not rows or len(rows[0]) < 13:
            return None
        try:
            yz = np.array([[float(t[11]), float(t[12])] for t in rows])
        except ValueError:
            return None
        out.append(np.minimum.reduce([yz[:, 0], ny - 1 - yz[:, 0],
                                      yz[:, 1], nz - 1 - yz[:, 1]]))
    return out


def _seed_rings_requested(ctx: StageContext):
    """``None`` = keep each panel's own seeds (the ring transforms chose);
    otherwise a set of rings, or ``"all"``.

    Config ``seed_rings``: ``"auto"`` (default) -> all rings, ``"transforms"``
    -> None, a list -> those rings. ``MIDAS_SEED_RINGS`` overrides the default
    only (same spellings).
    """
    v = getattr(ctx.config, "seed_rings", "auto")
    if v == "auto":
        env = os.environ.get("MIDAS_SEED_RINGS", "").strip().lower()
        if env:
            v = env if env in ("auto", "transforms") else [int(x) for x in env.split(",") if x.strip()]
    if v == "transforms":
        return None
    if v == "auto" or not v:
        return "all"
    return set(int(x) for x in v)


def _seed_omega_window(ctx: StageContext):
    """(MinOmeSpotIDsToIndex, MaxOmeSpotIDsToIndex) of the first panel's zip."""
    try:
        from midas_transforms.params import read_zarr_params
        zp = read_zarr_params(ctx.detectors[0].zarr_path)
        return float(zp.MinOmeSpotIDsToIndex), float(zp.MaxOmeSpotIDsToIndex)
    except Exception:
        return -90.0, 90.0


def _dedup_requested(ctx: StageContext) -> bool:
    """Drop duplicate spots between overlapping panels? Off unless asked.

    Measuring the overlap is always done and reported; removing records is an
    opt-in because it changes the spot table the indexer sees. Asked for by the
    ``dedup_panels`` config field (``--dedup-panels``) or ``MIDAS_DEDUP_PANELS=1``.
    """
    if getattr(ctx.config, "dedup_panels", False):
        return True
    return os.environ.get("MIDAS_DEDUP_PANELS", "") not in ("", "0")


def _write_panel_overlap(ctx: StageContext, report) -> Path:
    path = ctx.layer_dir / "PanelOverlap.csv"
    cols = ["det_a", "det_b", "n_a", "n_b", "n_pairs", "frac_of_a", "frac_of_b",
            "d_eta_med_deg", "d_eta_sigma_deg", "d_tth_med_deg", "d_tth_sigma_deg",
            "d_ome_med_deg", "d_ome_sigma_deg", "tol_pos_um", "tol_ome_deg",
            "ty_um", "ty_se_um", "tz_um", "tz_se_um", "scale", "scale_se",
            "rot_mrad", "rot_se_mrad", "pos_sigma_um", "condition"]
    lines = [" ".join(cols)]
    for st in report:
        v = [st.det_a, st.det_b, st.n_a, st.n_b, st.n_pairs,
             st.n_pairs / max(st.n_a, 1), st.n_pairs / max(st.n_b, 1),
             st.d_eta[0], st.d_eta[1], st.d_tth[0], st.d_tth[1],
             st.d_ome[0], st.d_ome[1], st.tol_pos_um, st.tol_ome,
             st.translation_um[0], st.translation_se_um[0],
             st.translation_um[1], st.translation_se_um[1],
             st.scale, st.scale_se, st.rotation_mrad, st.rotation_se_mrad,
             st.position_sigma_um, st.condition]
        lines.append(" ".join(str(x) if isinstance(x, int) else f"{x:.6g}" for x in v))
    path.write_text("\n".join(lines) + "\n")
    return path


def _do_multi_detector(ctx: StageContext, outputs: dict[str, str]):
    """Concatenate per-det InputAll.csv + InputAllExtraInfoFittingAll.csv
    into ``layer_dir``, renumber SpotIDs globally, and rewrite paramstest.

    Panels that overlap record the same reflection twice. The overlap is always
    measured and written to ``PanelOverlap.csv`` (duplicate counts, the data-set
    matching window, and the relative translation / scale / rotation of each
    panel pair with its uncertainty); the duplicates are dropped only when
    ``dedup_panels`` is set (or ``MIDAS_DEDUP_PANELS=1``).

    Binning runs after this stage on the merged ``InputAll.csv``.
    """
    import numpy as np
    from ..panel_overlap import analyse_panels, duplicate_masks

    dets = list(ctx.detectors)
    per_det_counts: list[int] = []
    merged_input_all = ctx.layer_dir / "InputAll.csv"
    merged_extra_csv = ctx.layer_dir / "InputAllExtraInfoFittingAll.csv"

    # ---- load every panel -------------------------------------------------
    head_ia = head_ie = None
    ia_rows, ie_rows, seeds, d0 = [], [], [], {}
    for det in dets:
        d = ctx.detector_dir(det)
        h, rows = _load_table(d / "InputAll.csv")
        he, erows = _load_table(d / "InputAllExtraInfoFittingAll.csv")
        head_ia = head_ia or h
        head_ie = head_ie or he
        ia_rows.append(rows)
        ie_rows.append(erows)
        local = []
        sti = d / "SpotsToIndex.csv"
        if sti.exists():
            for line in sti.read_text().splitlines():
                try:
                    local.append(int(line.split()[0]))
                except (ValueError, IndexError):
                    continue
        seeds.append(local)
        for r, v in _ring_d0(d / "IDsHash.csv").items():
            d0.setdefault(r, v)

    # ---- measure the overlap (never blocks the merge) -----------------------
    drop = [np.zeros(len(r), dtype=bool) for r in ia_rows]
    partner: dict[tuple[int, int], tuple[int, int]] = {}
    if len(dets) > 1 and all(len(r) for r in ia_rows):
        try:
            arrays = [np.array([[float(t) for t in row[:8]] for row in rows])
                      for rows in ia_rows]
            step = _kv_float(ctx.detector_dir(dets[0]) / "paramstest.txt", "OmegaStep")
            report = analyse_panels(arrays, [d.det_id for d in dets],
                                    [d.lsd for d in dets], omega_step_deg=step)
            _write_panel_overlap(ctx, report)
            outputs[str(ctx.layer_dir / "PanelOverlap.csv")] = ""
            prefer = _edge_scores(ctx, dets, ie_rows)
            masks = duplicate_masks(arrays, report, [d.det_id for d in dets], prefer=prefer)
            pos = {d.det_id: k for k, d in enumerate(dets)}
            for st in report:
                ka, kb = pos[st.det_a], pos[st.det_b]
                LOG.info("  panels %d/%d overlap: %d duplicate spots (%.1f%% of %d, "
                         "%.1f%% of %d); relative shift b->a (%.1f +- %.1f, "
                         "%.1f +- %.1f) um, scale %.2g +- %.1g, rot %.2g +- %.1g mrad, "
                         "position scatter %.1f um",
                         st.det_a, st.det_b, st.n_pairs,
                         100.0 * st.n_pairs / max(st.n_a, 1), st.n_a,
                         100.0 * st.n_pairs / max(st.n_b, 1), st.n_b,
                         st.translation_um[0], st.translation_se_um[0],
                         st.translation_um[1], st.translation_se_um[1],
                         st.scale, st.scale_se, st.rotation_mrad,
                         st.rotation_se_mrad, st.position_sigma_um)
                for ra, rb in zip(st.idx_a, st.idx_b):
                    if masks[kb][rb] and not masks[ka][ra]:
                        partner[(kb, int(rb))] = (ka, int(ra))
                    elif masks[ka][ra] and not masks[kb][rb]:
                        partner[(ka, int(ra))] = (kb, int(rb))
            if _dedup_requested(ctx):
                drop = masks
                LOG.info("  dropping %d duplicate spot records (dedup_panels)",
                         int(sum(m.sum() for m in masks)))
            elif report:
                LOG.warning(
                    "  %d spot(s) are recorded by two panels (see PanelOverlap.csv); "
                    "they are kept. Set dedup_panels / MIDAS_DEDUP_PANELS=1 to drop "
                    "one record of each pair.", int(sum(m.sum() for m in masks)))
        except Exception as exc:                       # pragma: no cover - diagnostic
            LOG.warning("  panel-overlap analysis skipped: %s", exc)
            drop = [np.zeros(len(r), dtype=bool) for r in ia_rows]
            partner = {}

    # ---- renumber and write -------------------------------------------------
    new_id: list[np.ndarray] = []                      # local row -> global id (0 = dropped)
    next_id = 1
    for k, rows in enumerate(ia_rows):
        ids = np.zeros(len(rows), dtype=np.int64)
        for i in range(len(rows)):
            if not drop[k][i]:
                ids[i] = next_id
                next_id += 1
        new_id.append(ids)
        per_det_counts.append(int((ids > 0).sum()))
        LOG.info("  det %d: %d spots", dets[k].det_id, per_det_counts[-1])

    seed_globals: list[int] = []
    for k, local in enumerate(seeds):
        for lid in local:
            r = lid - 1
            if r < 0 or r >= len(ia_rows[k]):
                continue
            g = int(new_id[k][r])
            if g == 0 and (k, r) in partner:           # seed was the dropped copy
                kk, rr = partner[(k, r)]
                g = int(new_id[kk][rr])
            if g:
                seed_globals.append(g)
    seed_globals = sorted(set(seed_globals)) if any(m.any() for m in drop) else seed_globals

    with merged_input_all.open("w") as fp_in:
        if head_ia is not None:
            fp_in.write(head_ia + " DetID\n")
        for k, rows in enumerate(ia_rows):
            for i, toks in enumerate(rows):
                if drop[k][i]:
                    continue
                t = list(toks)
                t[4] = str(int(new_id[k][i]))
                t.append(str(dets[k].det_id))
                fp_in.write(" ".join(t) + "\n")
    with merged_extra_csv.open("w") as fp_extra:
        if head_ie is not None:
            fp_extra.write(head_ie + " DetID\n")
        for k, rows in enumerate(ie_rows):
            for i, toks in enumerate(rows):
                if i >= len(new_id[k]) or drop[k][i]:
                    continue
                t = list(toks)
                t[4] = str(int(new_id[k][i]))
                t.append(str(dets[k].det_id))
                fp_extra.write(" ".join(t) + "\n")

    seed_rings = _seed_rings_requested(ctx)
    if seed_rings:
        lo, hi = _seed_omega_window(ctx)
        seed_globals = [int(new_id[k][i]) for k, rows in enumerate(ia_rows)
                        for i, t in enumerate(rows)
                        if not drop[k][i]
                        and (seed_rings == "all" or int(float(t[5])) in seed_rings)
                        and lo <= float(t[2]) <= hi]
        LOG.info("  seeding from %s: %d seeds",
                 "all rings" if seed_rings == "all" else f"rings {sorted(seed_rings)}",
                 len(seed_globals))

    merged_sti = ctx.layer_dir / "SpotsToIndex.csv"
    with merged_sti.open("w") as fp:
        for sid in seed_globals:
            fp.write(f"{sid}\n")
    outputs[str(merged_sti)] = ""

    merged_rings = [int(float(toks[5])) for k, rows in enumerate(ia_rows)
                    for i, toks in enumerate(rows) if not drop[k][i]]
    ids_rows = _ids_hash_rows(merged_rings, d0)
    if ids_rows:
        merged_ids_hash = ctx.layer_dir / "IDsHash.csv"
        merged_ids_hash.write_text("\n".join(ids_rows) + "\n")
        outputs[str(merged_ids_hash)] = ""

    n_total = next_id - 1
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
    _emit_eta_coverage(merged_paramstest_path)
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
