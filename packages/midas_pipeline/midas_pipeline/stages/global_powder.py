"""Global powder-intensity correction (Phase C2/C3).

Per-detector ``calc_radius`` (Phase C1) computes spot-level integrated
intensity but uses *only that panel's* sum for the per-spot ``PowderIntensity``
and therefore ``GrainVolume``. For partial-coverage layouts (pinwheel:
each panel sees ~75° of η), single-panel ``GrainVolume`` is wrong by
``360°/coverage`` — and the simple uniform-powder scaling fails on
textured samples.

This stage runs **after** ``cross_det_merge`` and **before** ``binning``.
For each ring, it builds a per-η-bin intensity model
``I_hat(ring, η)`` from all panels' observed spots, with multi-distance
averaging where panels overlap. The total ring intensity is then the
arc-integral of that model over [-180°, 180°), and per-spot
``GrainVolume`` is rewritten as ``V_sample · I_spot / I_powder_total(ring)``.

Outputs land directly in ``layer_dir/InputAll.csv`` (rewriting the
``GrainRadius`` column, which downstream stages consume) and in
``layer_dir/PowderModel.csv`` (one row per (ring, η-bin) for diagnostics).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ._base import StageContext
from .._logging import LOG, stage_timer
from ..eta_coverage import parse_coverage_blocks
from ..results import StageResult


# -----------------------------------------------------------------------------
#  Bin grid
# -----------------------------------------------------------------------------

ETA_BIN_SIZE_DEG = 1.0   # 1° bins span [-180, 180); 360 bins per ring
N_ETA_BINS = int(round(360.0 / ETA_BIN_SIZE_DEG))


def _eta_bin(eta_deg: float) -> int:
    """Map eta in [-180, 180] to bin index in [0, 360)."""
    b = int((eta_deg + 180.0) / ETA_BIN_SIZE_DEG)
    if b < 0:
        b = 0
    elif b >= N_ETA_BINS:
        b = N_ETA_BINS - 1
    return b


def _coverage_per_bin(arcs_by_det: Dict[int, list]) -> Dict[int, np.ndarray]:
    """Return ``cov[ring]`` = (N_ETA_BINS,) int — number of panels covering bin.

    A bin gets credit for one panel iff that panel has an arc on this
    ring containing the bin's centre. Multi-distance overlapping panels
    give bins a count > 1; gaps (no panel covers) give count 0 and the
    spot in that bin must be discarded from the powder estimator.
    """
    out: Dict[int, np.ndarray] = {}
    for det_id, arcs in arcs_by_det.items():
        for arc in arcs:
            cov = out.setdefault(arc.ring_nr, np.zeros(N_ETA_BINS, dtype=np.int64))
            lo_b = _eta_bin(arc.eta_lo_deg)
            hi_b = _eta_bin(arc.eta_hi_deg)
            if lo_b <= hi_b:
                cov[lo_b:hi_b + 1] += 1
            else:                      # arc wraps ±180
                cov[lo_b:] += 1
                cov[:hi_b + 1] += 1
    return out


# -----------------------------------------------------------------------------
#  Spot list aggregation
# -----------------------------------------------------------------------------

def _read_radius_csvs(per_det_paths: List[Path]) -> Dict[int, dict]:
    """Read each panel's ``Radius_*.csv`` into a ``{spot_id: row_dict}`` map.

    The per-detector ``Radius_*.csv`` carries the spot-level
    ``IntegratedIntensity``, ``Eta``, ``RingNr`` etc. The local SpotID in
    that file matches the local SpotID in the per-detector
    ``InputAll.csv`` (cross_det_merge renumbered the *global* InputAll.csv
    but did not touch the per-det Radius_*.csv).
    """
    out: Dict[int, dict] = {}
    for det_idx, path in enumerate(per_det_paths):
        if not path.exists():
            continue
        with path.open() as fp:
            head = fp.readline().split()
            try:
                col_int = head.index("IntegratedIntensity")
                col_eta = head.index("Eta")
                col_ring = head.index("RingNr")
                col_id = head.index("SpotID")
            except ValueError:
                continue
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                toks = line.split()
                try:
                    out[(det_idx, int(float(toks[col_id])))] = {
                        "intensity": float(toks[col_int]),
                        "eta": float(toks[col_eta]),
                        "ring": int(float(toks[col_ring])),
                    }
                except (IndexError, ValueError):
                    continue
    return out


# -----------------------------------------------------------------------------
#  Run stage
# -----------------------------------------------------------------------------

def run(ctx: StageContext) -> StageResult:
    started = time.time()
    layer_dir = ctx.layer_dir
    input_all = layer_dir / "InputAll.csv"
    paramstest = layer_dir / "paramstest.txt"

    # Tolerant skip when upstream FF stages have nothing to feed us —
    # mirrors how zip_convert / peakfit / transforms degrade in the
    # scaffold smoke-test path (no zarr present).
    if not input_all.exists() or not paramstest.exists():
        LOG.info("global_powder: missing InputAll.csv or paramstest.txt; skip.")
        return _empty_result(started, str(input_all))

    with stage_timer("global_powder"):
        coverage_arcs = parse_coverage_blocks(paramstest.read_text())
        if not coverage_arcs:
            LOG.info("  no EtaCoverage blocks — single-detector run, "
                     "leaving GrainRadius unchanged")
            return _empty_result(started, str(input_all))

        cov_per_bin = _coverage_per_bin(coverage_arcs)
        n_panels = len(coverage_arcs)
        LOG.info("  global_powder: %d panels, rings covered: %s",
                 n_panels, sorted(cov_per_bin.keys()))

        # Per-spot intensity from per-detector Radius_*.csv.
        per_det_paths: List[Path] = []
        for det in ctx.detectors:
            det_dir = ctx.stage_dir(det)
            cands = list(det_dir.glob("Radius_StartNr_*_EndNr_*.csv"))
            if cands:
                per_det_paths.append(cands[0])
            else:
                per_det_paths.append(det_dir / "Radius_StartNr_1_EndNr_1.csv")
        radius_map = _read_radius_csvs(per_det_paths)
        LOG.info("  loaded %d per-detector spot intensity rows", len(radius_map))

        # Build per-(ring, eta-bin) intensity model.
        ring_intensity_sum: Dict[int, np.ndarray] = {}
        ring_panel_count_sum: Dict[int, np.ndarray] = {}
        for (_, _), row in radius_map.items():
            r = row["ring"]
            b = _eta_bin(row["eta"])
            ring_intensity_sum.setdefault(
                r, np.zeros(N_ETA_BINS, dtype=np.float64)
            )[b] += row["intensity"]
            ring_panel_count_sum.setdefault(
                r, np.zeros(N_ETA_BINS, dtype=np.int64)
            )[b] += 1

        # I_hat(ring, eta) = sum_obs(I) / panel_coverage_count
        # — multi-panel overlap averaged.
        i_hat: Dict[int, np.ndarray] = {}
        i_total: Dict[int, float] = {}
        for r, ising in ring_intensity_sum.items():
            cov = cov_per_bin.get(r, np.zeros(N_ETA_BINS, dtype=np.int64))
            with np.errstate(invalid="ignore"):
                hat = np.where(cov > 0, ising / cov, 0.0)
            i_hat[r] = hat
            i_total[r] = float(hat.sum() * ETA_BIN_SIZE_DEG)
            LOG.info(
                "  ring %d: total panel-corrected intensity ∫I̅(η)dη = %.3e "
                "(over %d active η-bins / %d total)",
                r, i_total[r], int((cov > 0).sum()), N_ETA_BINS,
            )

        # Rewrite GrainRadius column in InputAll.csv.
        # GrainRadius_um = ((3 * V_grain) / (4π))^(1/3)
        #   V_grain = V_sample * I_spot / I_powder_total(ring)
        v_sample = _read_vsample_um3(paramstest)
        if v_sample <= 0:
            v_sample = 1e7                       # fallback
        n_rewritten = 0
        rows_in = input_all.read_text().splitlines()
        if not rows_in:
            return _empty_result(started, str(input_all))
        header = rows_in[0]
        col_names = header.split()
        try:
            col_eta = col_names.index("Eta")
            col_ring = col_names.index("RingNumber")
            col_grad = col_names.index("GrainRadius")
            col_id = col_names.index("SpotID")
            col_det = col_names.index("DetID")
        except ValueError as e:
            LOG.warning("  InputAll.csv column missing — leaving unchanged (%s)", e)
            return _empty_result(started, str(input_all))

        out_lines = [header]
        for raw in rows_in[1:]:
            toks = raw.split()
            if len(toks) <= max(col_grad, col_ring, col_eta, col_det):
                out_lines.append(raw)
                continue
            try:
                ring = int(float(toks[col_ring]))
                eta = float(toks[col_eta])
                det_id = int(float(toks[col_det]))
                spot_id = int(float(toks[col_id]))
            except ValueError:
                out_lines.append(raw)
                continue
            # Look up local spot intensity. For multi-det, the global
            # SpotID in InputAll.csv is the cross_det_merge-renumbered
            # one; we have to re-derive the per-det local id by indexing
            # into per_det_paths according to det_id.
            radius_row = radius_map.get((det_id - 1, _global_to_local_id(
                spot_id, ctx, det_id,
            )))
            if radius_row is None:
                out_lines.append(raw)
                continue
            i_spot = radius_row["intensity"]
            tot = i_total.get(ring, 0.0)
            if tot > 0:
                v_grain = v_sample * i_spot / tot
                # protect against negative I_spot bin estimates
                v_grain = max(v_grain, 0.0)
                grad_um = (3.0 * v_grain / (4.0 * math.pi)) ** (1.0 / 3.0)
                toks[col_grad] = f"{grad_um:.6f}"
                n_rewritten += 1
            out_lines.append(" ".join(toks))

        input_all.write_text("\n".join(out_lines) + "\n")
        LOG.info("  rewrote GrainRadius for %d / %d spots", n_rewritten,
                 max(0, len(rows_in) - 1))

        # Diagnostics: per-(ring, eta-bin) intensity model.
        diag = layer_dir / "PowderModel.csv"
        with diag.open("w") as fp:
            fp.write("RingNr EtaBinLo EtaBinHi Intensity_hat NPanels\n")
            for r in sorted(i_hat):
                cov = cov_per_bin.get(r, np.zeros(N_ETA_BINS, dtype=np.int64))
                hat = i_hat[r]
                for b in range(N_ETA_BINS):
                    if hat[b] == 0 and cov[b] == 0:
                        continue
                    lo = -180.0 + b * ETA_BIN_SIZE_DEG
                    hi = lo + ETA_BIN_SIZE_DEG
                    fp.write(f"{r} {lo:.3f} {hi:.3f} {hat[b]:.6e} {int(cov[b])}\n")

    finished = time.time()
    return StageResult(
        stage_name="global_powder",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={str(input_all): "", str(diag): ""},
        metrics={
            "n_panels": n_panels,
            "n_rings": len(i_hat),
            "n_spots_rewritten": n_rewritten,
        },
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.layer_dir / "InputAll.csv"]


# -----------------------------------------------------------------------------
#  Helpers
# -----------------------------------------------------------------------------

def _empty_result(started: float, path: str) -> StageResult:
    finished = time.time()
    return StageResult(
        stage_name="global_powder",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        outputs={path: ""},
        metrics={"skipped": True},
    )


def _read_vsample_um3(paramstest: Path) -> float:
    for raw in paramstest.read_text().splitlines():
        toks = raw.strip().rstrip(";").split()
        if not toks:
            continue
        if toks[0] == "Vsample" and len(toks) >= 2:
            try:
                return float(toks[1])
            except ValueError:
                continue
    return 0.0


def _global_to_local_id(global_id: int, ctx: StageContext, det_id: int) -> int:
    """Reverse the cross_det_merge SpotID renumbering for one panel.

    Panels are concatenated in det_id order; a panel's local spot IDs
    ``1..n_d`` map to global IDs ``offset+1..offset+n_d`` where ``offset``
    is the sum of spot counts of all preceding panels.
    """
    offset = 0
    for det in ctx.detectors:
        if det.det_id == det_id:
            return global_id - offset
        # count this panel's spots
        ia = ctx.stage_dir(det) / "InputAll.csv"
        if ia.exists():
            with ia.open() as fp:
                offset += sum(1 for _ in fp) - 1
    return global_id
