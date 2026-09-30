"""Stage: find_grains_stage (PF only).

Invokes the unified ``find_grains_single`` / ``find_grains_multiple``
sub-package on the consolidated indexer output. Emits the unique-grain
table (``UniqueOrientations.csv``) + sinogram binaries that the
reconstruct stage consumes.

FF mode is a no-op (FF has no find_grains step — ``process_grains``
handles that).
"""

from __future__ import annotations

import time
from pathlib import Path

from .._logging import LOG
from ..results import FindGrainsResult, StageResult
from ._base import StageContext
from ._stub import stub_run


def _read_space_group(layer_dir: Path) -> int:
    """Best-effort space-group resolver (mirrors consolidation.py).

    Tolerates trailing punctuation like ``SpaceGroup 194;`` that legacy
    paramstest files carry over from the C param-file format.
    """
    p = layer_dir / "paramstest.txt"
    if not p.exists():
        return 225
    for line in p.read_text().splitlines():
        toks = line.split()
        if len(toks) >= 2 and toks[0] == "SpaceGroup":
            digits = "".join(c for c in toks[1] if c.isdigit())
            if digits:
                return int(digits)
    return 225


def _read_omega_step(layer_dir: Path):
    """|OmegaStep| from paramstest.txt, or None."""
    p = layer_dir / "paramstest.txt"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        toks = line.replace(";", " ").split()
        if len(toks) >= 2 and toks[0] == "OmegaStep":
            try:
                return abs(float(toks[1]))
            except ValueError:
                return None
    return None


def _sino_tolerances(recon_cfg, layer_dir: Path) -> tuple[float, float, str]:
    """(tol_ome_deg, tol_eta_deg, how) for tolerance-mode sinograms.

    A configured value > 0 wins; otherwise 2 x |OmegaStep|; otherwise the
    historical 1.0 deg.
    """
    step = _read_omega_step(layer_dir)
    out, how = [], []
    for name in ("sino_tol_ome_deg", "sino_tol_eta_deg"):
        v = float(getattr(recon_cfg, name, -1.0))
        if v > 0:
            out.append(v); how.append("configured")
        elif step:
            out.append(2.0 * step); how.append("2xOmegaStep")
        else:
            out.append(1.0); how.append("default 1 deg (no OmegaStep)")
    return out[0], out[1], "/".join(how)


def run(ctx: StageContext) -> StageResult:
    if ctx.is_ff:
        # FF doesn't run find_grains; record a clean skipped row.
        return stub_run("find_grains", ctx)

    started = time.time()
    layer_dir = Path(ctx.layer_dir)
    space_group = _read_space_group(layer_dir)

    # Soft skip when indexing hasn't produced the consolidated triplet.
    index_best_all = layer_dir / "Output" / "IndexBest_all.bin"
    if not index_best_all.exists():
        LOG.info("find_grains(PF): missing %s → skip.", index_best_all)
        return stub_run("find_grains", ctx)

    LOG.info("find_grains(PF): layer_dir=%s, space_group=%d",
             layer_dir, space_group)

    from ..find_grains import find_grains_single, find_grains_multiple

    # P7 hook: build a sino-soft weight fn from the SoftAttributionConfig.
    # ``soft_attribution`` is an optional field on PipelineConfig; tolerate
    # absence gracefully so legacy / partial config builds still run.
    soft_cfg = getattr(ctx.config, "soft_attribution", None)
    soft_weight_fn = None
    emit_softsum = False
    if soft_cfg is not None and getattr(soft_cfg, "enable", False):
        import numpy as np
        sigma_w = soft_cfg.omega_sigma_deg
        if sigma_w > 0:
            def soft_weight_fn(ome_d, eta_d):
                # 1-D Gaussian in ω (η is unweighted — eta-binning handles it).
                return np.exp(-(ome_d * ome_d) / (2.0 * sigma_w * sigma_w))
        emit_softsum = True

    tol_ome, tol_eta, how = _sino_tolerances(ctx.config.recon, layer_dir)
    LOG.info("find_grains(PF): sinogram window omega %.3f deg, eta %.3f deg (%s)",
             tol_ome, tol_eta, how)

    tb_margin = float(getattr(ctx.config.recon, "brightness_tiebreak_margin", 0.0))
    if getattr(ctx.config.recon, "candidate_brightness", True) or tb_margin > 0:
        _write_candidate_brightness(layer_dir, ctx.config.scan.n_scans,
                                    required=tb_margin > 0)

    if ctx.config.one_sol_per_vox:
        # soft-attribution kwargs only when the optional plumbing is wired —
        # keep parity with versions of find_grains_single that predate it.
        soft_kwargs = (
            {"emit_softsum": emit_softsum, "soft_weight_fn": soft_weight_fn}
            if emit_softsum else {}
        )
        artifacts = find_grains_single(
            layer_dir,
            space_group=space_group,
            sino_mode=ctx.config.recon.sino_source,    # "tolerance" | "indexing"
            confidence_min=ctx.config.recon.sino_conf_min,
            scan_tolerance_um=ctx.config.recon.sino_scan_tol_um,
            cluster_misorientation_deg=ctx.config.fusion.max_ang_deg,
            n_scans=ctx.config.scan.n_scans,
            tol_ome_deg=tol_ome, tol_eta_deg=tol_eta,
            conc_threshold=getattr(ctx.config.recon, "sino_conc_threshold", 0.0),
            conc_min_band_um=getattr(
                ctx.config.recon, "sino_conc_min_band_um", 4.0,
            ),
            # Per-voxel clustering is O(n_sol^2) and dominates this stage on
            # dense maps; it once ran 94 min on s5/L3 with no way to tell it
            # from a hang.
            progress_cb=(ctx.progress.update if ctx.progress else None),
            cluster_device=str(ctx.config.device),
            brightness_tiebreak_margin=tb_margin,
            sibling_merge_deg=getattr(ctx.config.fusion, "sibling_merge_deg", 0.0),
            **soft_kwargs,
        )
    else:
        artifacts = find_grains_multiple(
            layer_dir,
            space_group=space_group,
            cluster_misorientation_deg=ctx.config.fusion.max_ang_deg,
            progress_cb=(ctx.progress.update if ctx.progress else None),
        )

    sine = _write_sine_consistency(layer_dir / "Output")

    finished = time.time()
    return FindGrainsResult(
        stage_name="find_grains",
        started_at=started, finished_at=finished, duration_s=finished - started,
        unique_orientations_csv=str(getattr(artifacts, "unique_orientations_csv", "")),
        unique_index_single_key_bin=str(getattr(
            artifacts, "unique_index_single_key_bin", "",
        )),
        spots_to_index_csv=str(getattr(artifacts, "spots_to_index_csv", "")),
        n_unique_grains=int(getattr(artifacts, "n_unique_grains", 0)),
        outputs={},
        metrics={"scan_mode": "pf", "space_group": space_group,
                 "one_sol_per_vox": ctx.config.one_sol_per_vox,
                 "sine_consistency": sine},
    )


def _write_sine_consistency(output_dir) -> dict:
    """Output/SineConsistency.csv: does each grain trace one sine? Never fails the stage."""
    from ..diagnostics.sine_consistency import sine_consistency
    try:
        return sine_consistency(output_dir)
    except Exception as e:                                # noqa: BLE001 - diagnostic only
        LOG.warning("find_grains: sine consistency skipped (%s)", e)
        return {}


def _write_candidate_brightness(layer_dir, n_scans: int, required: bool) -> None:
    """Write Output/CandidateBrightness.npz. Missing inputs (per-scan CSVs,
    IDsMergedScanning.csv) skip it with a warning, unless the tie-break needs
    it, in which case the stage fails rather than silently not tie-breaking."""
    from ..diagnostics.candidate_brightness import candidate_brightness
    try:
        candidate_brightness(layer_dir, int(n_scans))
    except Exception as e:      # default-on diagnostic: it must never abort the stage
        if required:
            raise RuntimeError(
                f"find_grains: --brightness-tiebreak-margin needs "
                f"CandidateBrightness.npz, which could not be built: {e}") from e
        LOG.warning("find_grains: candidate brightness skipped (%s: %s)",
                    type(e).__name__, e)
