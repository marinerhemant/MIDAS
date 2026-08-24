"""HKL list generation via ``midas-hkls``.

Runs once per detector (each detector's zarr has its own
``analysis/process/analysis_parameters`` group that must be patched
with the hkls table for downstream peak finding).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from ._base import StageContext, run_subprocess, hash_inputs
from .._logging import LOG, stage_timer
from ..results import HKLResult


def run(ctx: StageContext) -> HKLResult:
    started = time.time()
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}

    # Atom basis, if Parameters.txt declares one. This stage shells out to the
    # midas-hkls CLI (unlike midas_pipeline, which calls the function), so the
    # basis has to be re-expressed as flags. Absent a basis the argv is
    # unchanged and hkls.csv stays the historical 11-column file; present, it
    # gains an F2 column, which by itself changes nothing — the weighting is a
    # separate opt-in (ConfidenceMetric) and trailing columns are ignored by
    # every existing reader.
    basis_args: list[str] = []
    try:
        from midas_hkls import read_phase_basis

        basis = read_phase_basis(ctx.config.params_file)
        for a in (basis.get("atoms") or []):
            basis_args += [
                "--phase-atom",
                f"{a.element} {a.fract[0]!r} {a.fract[1]!r} {a.fract[2]!r} "
                f"{a.occupancy!r} {a.B_iso!r}",
            ]
        if basis.get("cif_path"):
            basis_args += ["--phase-cif", str(basis["cif_path"])]
        if basis.get("drop_forbidden"):
            basis_args += [
                "--drop-forbidden",
                "--forbidden-f2", repr(basis.get("forbidden_f2_threshold", 1e-6)),
            ]
        if basis_args:
            LOG.info("hkl: atom basis declared; hkls.csv will carry F2.")
    except ImportError:
        pass

    with stage_timer("hkl"):
        # Run midas-hkls per zarr file (each detector has its own zarr).
        for det in ctx.detectors:
            zarr_path = Path(det.zarr_path)
            if not zarr_path.exists():
                raise FileNotFoundError(
                    f"detector {det.det_id} zarr not found: {zarr_path}"
                )
            inputs[str(zarr_path)] = ""    # actual hash filled in by caller if needed
            cmd = ([sys.executable, "-m", "midas_hkls", "zarr", str(zarr_path)]
                   + basis_args)
            run_subprocess(
                cmd,
                cwd=ctx.stage_dir(det),
                stdout_path=ctx.log_dir / f"hkl_det{det.det_id}_out.csv",
                stderr_path=ctx.log_dir / f"hkl_det{det.det_id}_err.csv",
            )

        # Pick up hkls.csv from wherever midas-hkls dropped it. It honours the
        # zarr's analysis_parameters['ResultFolder'] (often the directory of
        # the zarr itself), so we search per-det stage dirs first, then the
        # zarr's own directory, and finally mirror the file into layer_dir.
        hkls_canonical = ctx.layer_dir / "hkls.csv"
        if not hkls_canonical.exists():
            for det in ctx.detectors:
                cand = ctx.stage_dir(det) / "hkls.csv"
                if cand.exists():
                    if cand != hkls_canonical:
                        hkls_canonical.write_bytes(cand.read_bytes())
                    break
        if not hkls_canonical.exists():
            for det in ctx.detectors:
                cand = Path(det.zarr_path).parent / "hkls.csv"
                if cand.exists():
                    hkls_canonical.write_bytes(cand.read_bytes())
                    break
        if not hkls_canonical.exists():
            raise FileNotFoundError(
                f"hkls.csv not produced by midas-hkls in {ctx.layer_dir}"
            )
        outputs[str(hkls_canonical)] = ""
        # Mirror hkls.csv into each per-detector dir so peakfit_torch /
        # midas-merge-peaks / midas-calc-radius find it via their
        # ResultFolder lookups.
        if ctx.is_multi_detector:
            for det in ctx.detectors:
                dst = ctx.stage_dir(det) / "hkls.csv"
                if dst.resolve() != hkls_canonical.resolve():
                    dst.write_bytes(hkls_canonical.read_bytes())

    finished = time.time()
    return HKLResult(
        stage_name="hkl",
        started_at=started,
        finished_at=finished,
        duration_s=finished - started,
        inputs=inputs, outputs=outputs,
        hkls_csv=str(hkls_canonical),
        metrics={"n_detectors": len(ctx.detectors)},
    )


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.layer_dir / "hkls.csv"]
