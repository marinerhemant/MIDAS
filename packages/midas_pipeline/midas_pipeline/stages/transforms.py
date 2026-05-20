"""Stage: transforms (per-scan: merge + radius + fit_setup + aggregate).

For PF mode this stage owns three sub-stages per scan — all wrapped
inside ``midas_transforms``:

* ``merge_overlapping_peaks`` → consumes ``Temp/AllPeaks_PS.bin``
  written by ``peakfit``, emits the merged Result CSV + MergeMap.
* ``calc_radius`` → emits ``Radius_StartNr_*.csv``.
* ``fit_setup`` → emits ``InputAllExtraInfoFittingAll.csv``,
  ``InputAll.csv``, ``paramstest.txt``, ``IDRings.csv``,
  ``SpotsToIndex.csv`` (per-scan).

After per-scan work, this stage performs pf_MIDAS.py:760-906's
aggregation step:

* Y-offset the per-scan ``InputAllExtraInfoFittingAll.csv`` rows
  (the scan-Y position is added to YLab and YOrig).
* Recompute ``Eta`` and ``Ttheta`` from the shifted coords.
* Write the result to ``layer_dir/InputAllExtraInfoFittingAll{s-1}.csv``
  (0-indexed by scan within the layer) so the existing
  ``merge_scans`` stage picks it up.
* Copy one scan's ``paramstest.txt`` and ``hkls.csv`` up to
  ``layer_dir`` so downstream stages (indexing/refinement/etc.) have a
  single canonical copy.

The ``merge_overlaps`` and ``calc_radius`` stage modules are kept as
no-op stubs since this stage subsumes them (matches the natural
chunking inside ``midas_transforms.Pipeline``).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from .._logging import LOG
from ..results import StageResult
from ._base import StageContext
from ._stub import stub_run

RAD2DEG = 57.2957795130823


def run(ctx: StageContext) -> StageResult:
    started = time.time()
    if ctx.is_pf:
        return _run_pf(ctx, started)
    return _run_ff(ctx, started)


def _run_pf(ctx: StageContext, started: float) -> StageResult:
    from .._pf_scans import iter_pf_scans
    try:
        from midas_transforms.params import read_zarr_params
        from midas_transforms.merge import merge_overlapping_peaks
        from midas_transforms.radius import calc_radius
        from midas_transforms.fit_setup import fit_setup
    except ImportError as e:
        LOG.warning("transforms(PF): midas_transforms imports failed (%s); skip.", e)
        return stub_run("transforms", ctx)

    cfg = ctx.config
    layer_dir = ctx.layer_dir

    try:
        scans = iter_pf_scans(
            params_file=cfg.params_file,
            layer_dir=layer_dir,
            layer_nr=ctx.layer_nr,
            raw_dir=cfg.raw_dir,
            n_scans_hint=cfg.scan.n_scans,
        )
    except (FileNotFoundError, ValueError) as e:
        LOG.warning("transforms(PF): scan discovery failed (%s); skip.", e)
        return stub_run("transforms", ctx)

    device = cfg.device or "cpu"
    dtype = cfg.dtype or "float64"

    written: list[Path] = []
    ok = 0
    failed = 0
    representative_paramstest: Path | None = None

    for s in scans:
        layer_extra = layer_dir / f"InputAllExtraInfoFittingAll{s.scan_nr - 1}.csv"
        if layer_extra.exists():
            written.append(layer_extra)
            ok += 1
            if (s.scan_dir / "paramstest.txt").exists() and representative_paramstest is None:
                representative_paramstest = s.scan_dir / "paramstest.txt"
            continue
        if not s.zip_path.exists():
            LOG.warning("transforms(PF): scan %d zip missing (%s); skip.",
                        s.scan_nr, s.zip_path)
            failed += 1
            continue
        if not s.allpeaks_ps_bin.exists():
            LOG.warning("transforms(PF): scan %d AllPeaks_PS.bin missing — "
                        "peakfit stage must have failed; skip.", s.scan_nr)
            failed += 1
            continue

        t0 = time.time()
        try:
            zp = read_zarr_params(s.zip_path)
            # ── merge ──────────────────────────────────────────────────
            merge = merge_overlapping_peaks(
                allpeaks_ps_bin=s.allpeaks_ps_bin,
                allpeaks_px_bin=(s.allpeaks_px_bin
                                 if s.allpeaks_px_bin.exists() else None),
                result_folder=s.scan_dir,
                skip_frame=zp.SkipFrame,
                use_maxima_positions=bool(zp.UseMaximaPositions),
                use_pixel_overlap=bool(zp.UsePixelOverlap),
                nr_pixels=zp.NrPixels,
                device=device, dtype=dtype,
                write=True,
            )
            # ── radius ─────────────────────────────────────────────────
            rad = calc_radius(
                result_folder=s.scan_dir,
                zarr_params=zp,
                result_array=merge.peaks.detach().cpu().numpy().astype(np.float64),
                start_nr=1,
                end_nr=zp.EndNr if zp.EndNr > 0 else len(merge.peaks),
                device=device, dtype=dtype,
                write=True,
            )
            # ── fit_setup ──────────────────────────────────────────────
            fs = fit_setup(
                result_folder=s.scan_dir,
                zarr_params=zp,
                radius_array=rad.spots.detach().cpu().numpy().astype(np.float64),
                device=device, dtype=dtype,
                write=True,
            )
        except Exception as e:
            LOG.warning("transforms(PF): scan %d failed: %s", s.scan_nr, e)
            failed += 1
            continue

        # ── aggregate per-scan: Y-offset + Eta/Ttheta recompute ─────
        try:
            _write_layer_extra(
                src_extra=s.input_all_extra_csv,
                dst_extra=layer_extra,
                y_position=s.y_position_um,
                Lsd=zp.Lsd,
            )
        except Exception as e:
            LOG.warning("transforms(PF): scan %d aggregate failed: %s",
                        s.scan_nr, e)
            failed += 1
            continue

        written.append(layer_extra)
        ok += 1
        if representative_paramstest is None and (s.scan_dir / "paramstest.txt").exists():
            representative_paramstest = s.scan_dir / "paramstest.txt"
        LOG.info("transforms(PF): scan %d/%d done in %.1fs (%d spots)",
                 s.scan_nr, len(scans), time.time() - t0,
                 fs.spots_inputall.shape[0] if fs.spots_inputall is not None else 0)

    # ── promote paramstest.txt + hkls.csv to layer dir ─────────────
    layer_paramstest = layer_dir / "paramstest.txt"
    if representative_paramstest is not None and not layer_paramstest.exists():
        shutil.copy2(representative_paramstest, layer_paramstest)
        LOG.info("transforms(PF): promoted paramstest.txt → %s",
                 layer_paramstest)

    LOG.info("transforms(PF): %d ok + %d failed (of %d scans)",
             ok, failed, len(scans))
    return _result(started, written, ok, failed)


def _write_layer_extra(*, src_extra: Path, dst_extra: Path,
                       y_position: float, Lsd: float) -> None:
    """Per pf_MIDAS.py:849-906 — apply Y-offset + recompute Eta/Ttheta.

    ``InputAllExtraInfoFittingAll.csv`` columns (1-row example):
        %YLab ZLab Omega GrainRadius SpotID RingNumber Eta Ttheta
        OmegaIni(NoWedgeCorr) YOrig(NoWedgeCorr) ZOrig(NoWedgeCorr)
        YOrig(DetCor) ZOrig(DetCor) OmegaOrig(DetCor)
        IntegratedIntensity(count) RawSumIntensity maskTouched FitRMSE
    """
    import pandas as pd

    # ``sep=r"\s+"`` (not the removed-in-pandas-3.0 ``delim_whitespace=True``).
    df = pd.read_csv(src_extra, sep=r"\s+", skipinitialspace=True)
    if df.empty:
        df.to_csv(dst_extra, sep=" ", header=True, float_format="%.6f", index=False)
        return

    # Match pf_MIDAS.py's column-name tolerance (older binaries lose the % prefix).
    ylab_col = "%YLab" if "%YLab" in df.columns else "YLab"

    # GrainRadius > 0.001 → spot was found (vs zero-padded rows).
    valid_mask = df["GrainRadius"] > 0.001
    df.loc[valid_mask, ylab_col] += y_position
    if "YOrig(NoWedgeCorr)" in df.columns:
        df.loc[valid_mask, "YOrig(NoWedgeCorr)"] += y_position

    # Recompute Eta + Ttheta with the shifted y.
    y = df[ylab_col].to_numpy()
    z = df["ZLab"].to_numpy()
    # CalcEtaAngleAll: alpha = rad2deg * arccos(z / |yz|), sign-flipped where y > 0.
    norm = np.linalg.norm(np.stack([y, z], axis=0), axis=0)
    safe = norm > 0
    eta = np.zeros_like(y)
    eta[safe] = RAD2DEG * np.arccos(np.clip(z[safe] / norm[safe], -1.0, 1.0))
    eta[y > 0] *= -1.0
    df["Eta"] = eta
    df["Ttheta"] = RAD2DEG * np.arctan(norm / float(Lsd))

    # Fill NaN (from ragged C output lines) with 0 before writing.
    df = df.fillna(0)
    dst_extra.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_extra, sep=" ", header=True, float_format="%.6f", index=False)


def _run_ff(ctx: StageContext, started: float) -> StageResult:
    """FF: midas_transforms.Pipeline.from_zarr(zip).run() + dump."""
    try:
        from midas_transforms import Pipeline
    except ImportError as e:
        LOG.warning("transforms(FF): midas_transforms import failed (%s); skip.", e)
        return stub_run("transforms", ctx)

    cfg = ctx.config
    layer_dir = ctx.layer_dir
    zip_path = cfg.zarr_path
    if not zip_path:
        for p in layer_dir.glob("*.MIDAS.zip"):
            zip_path = str(p)
            break
    if not zip_path or not Path(zip_path).exists():
        LOG.info("transforms(FF): no zarr/zip; skip.")
        return stub_run("transforms", ctx)

    pipe = Pipeline.from_zarr(
        zip_path,
        result_folder=layer_dir,
        device=cfg.device, dtype=cfg.dtype,
    )
    pipe.run()
    pipe.dump(layer_dir)
    LOG.info("transforms(FF): pipeline.dump → %s", layer_dir)
    return _result(started, [layer_dir / "InputAllExtraInfoFittingAll.csv"], 1, 0)


def _result(started: float, written, ok: int, failed: int) -> StageResult:
    finished = time.time()
    return StageResult(
        stage_name="transforms",
        started_at=started, finished_at=finished, duration_s=finished - started,
        outputs={"per_scan_extra": [str(p) for p in written]},
        metrics={"n_scans_ok": ok, "n_scans_failed": failed},
    )
