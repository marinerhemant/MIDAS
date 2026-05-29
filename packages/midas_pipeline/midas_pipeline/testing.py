"""Test-data helpers: synthesize a forward-simulated FF-HEDM dataset on demand.

Ported from ``midas_ff_pipeline.testing`` as part of the consolidation of
ff-pipeline into the unified ``midas-pipeline`` package (see
``MIDAS_FF_PIPELINE_DEPRECATION_PLAN.md``). The ``midas_ff_pipeline.testing``
module is now a back-compat shim that re-exports from here.

Public API:

    generate_synthetic_dataset           single-detector synthetic FF
    generate_pinwheel_synthetic_dataset  4-detector pinwheel synthetic
    generate_multidet_synthetic_dataset  N-detector synthetic (arbitrary layout)

All three require a working MIDAS C build (``ForwardSimulationCompressed`` on
PATH or ``$MIDAS_HOME`` set). Tests that depend on them should skip when the
binary is unavailable; the package install itself never requires the C build.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from ._logging import LOG


def _find_midas_home() -> Path:
    """Find the MIDAS source tree (for ForwardSimulationCompressed binary).

    Order of preference:
      1. ``MIDAS_HOME`` env var
      2. ``$HOME/opt/MIDAS``
      3. discover via ``shutil.which("ForwardSimulationCompressed")``
    """
    env = os.environ.get("MIDAS_HOME")
    if env:
        return Path(env)
    candidate = Path.home() / "opt" / "MIDAS"
    if (candidate / "FF_HEDM" / "bin" / "ForwardSimulationCompressed").exists():
        return candidate
    fs = shutil.which("ForwardSimulationCompressed")
    if fs:
        # bin/ is parent of the file, MIDAS_HOME is two levels up.
        return Path(fs).parent.parent.parent
    raise FileNotFoundError(
        "Cannot locate MIDAS_HOME — set MIDAS_HOME env var or install MIDAS at "
        "$HOME/opt/MIDAS (need ForwardSimulationCompressed binary on PATH)."
    )


def _generate_grains_csv(grains_csv: Path, n_grains: int,
                         lattice: List[float],
                         rsample: float, hbeam: float, beam_thickness: float,
                         space_group: int, seed: int) -> None:
    """Write GrainsSim.csv with `n_grains` random orientations + positions.

    Reuses ``tests/generate_grains.py`` from the MIDAS source tree.
    """
    midas_home = _find_midas_home()
    sys.path.insert(0, str(midas_home / "tests"))
    try:
        from generate_grains import generate_grains_csv  # type: ignore
        generate_grains_csv(
            grains_csv, n_grains, lattice,
            rsample, hbeam, beam_thickness,
            space_group=space_group, seed=seed,
        )
    finally:
        sys.path.pop(0)


def _enrich_zarr_metadata(zip_path: Path, params: dict) -> None:
    """Inject parameter metadata into the simulated zarr.

    Mirrors ``tests/test_ff_hedm.py::enrich_zarr_metadata`` — used so the
    downstream pipeline can read params from ``analysis_parameters``.
    """
    midas_home = _find_midas_home()
    sys.path.insert(0, str(midas_home / "utils"))
    try:
        import numpy as np
        import zarr  # type: ignore
        from ffGenerateZipRefactor import write_analysis_parameters  # type: ignore
        with zarr.ZipStore(str(zip_path), mode="a") as store:
            try:
                root = zarr.group(store=store)
            except Exception:
                root = zarr.group(store=store, overwrite=True)
            if "analysis" not in root:
                root.create_group("analysis/process/analysis_parameters")
            if "measurement" not in root:
                root.create_group("measurement/process/scan_parameters")
            sp_ana = root.require_group("analysis/process/analysis_parameters")
            sp_pro = root.require_group("measurement/process/scan_parameters")
            data_dtype = str(root["exchange/data"].dtype)
            sp_pro.create_dataset(
                "datatype",
                data=np.bytes_(data_dtype.encode("UTF-8")),
            )
            write_analysis_parameters(
                {"sp_pro_analysis": sp_ana, "sp_pro_meas": sp_pro},
                params,
            )
    finally:
        sys.path.pop(0)


def _parse_parameter_file(path: Path) -> dict:
    """Loose parse of a Parameters.txt — repeated keys collected into lists."""
    params: dict = {}
    with path.open() as fp:
        for raw in fp:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tokens = line.split()
            key = tokens[0]
            vals: list = []
            for v in tokens[1:]:
                try:
                    vals.append(int(v))
                except ValueError:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(v)
            value: object = vals if len(vals) > 1 else (vals[0] if vals else "")
            if key in params:
                if not isinstance(params[key], list) or not any(isinstance(i, list) for i in params[key]):
                    params[key] = [params[key]]
                params[key].append(value)
            else:
                params[key] = value
    return params


def _make_test_paramfile(template: Path, work_dir: Path,
                        out_file_name: str = "midas_ff_pipeline_synth") -> tuple[Path, dict]:
    """Rewrite a Parameters.txt template with absolute paths under work_dir.

    Returns (test_param_file, params_dict).
    """
    params = _parse_parameter_file(template)

    in_file = params.get("InFileName")
    if isinstance(in_file, list):
        in_file = in_file[0] if in_file else None
    if in_file:
        full_in = template.parent / str(in_file)
        if not full_in.exists():
            # Allow a freshly generated GrainsSim.csv in work_dir to satisfy this.
            full_in = work_dir / str(in_file)

    new_param = work_dir / f"test_{template.name}"
    with new_param.open("w") as out:
        for raw in template.open():
            stripped = raw.split()
            if not stripped:
                out.write(raw)
                continue
            key = stripped[0]
            if key == "InFileName":
                out.write(f"InFileName {full_in}\n")
            elif key == "OutFileName":
                out.write(f"OutFileName {work_dir / out_file_name}\n")
            elif key == "PositionsFile":
                pos_val = params.get("PositionsFile")
                if isinstance(pos_val, list):
                    pos_val = pos_val[0]
                out.write(f"PositionsFile {template.parent / str(pos_val)}\n")
            else:
                out.write(raw)
    return new_param, params


def generate_synthetic_dataset(*,
                               out_dir: Path,
                               params_template: Path,
                               n_grains: int = 50,
                               seed: int = 42,
                               n_cpus: int = 8,
                               device: str = "cpu") -> Path:
    """End-to-end synthetic data generation (pure-Python). Returns the .MIDAS.zip path.

    Uses ``midas_diffract.simulate_panel_zarrs`` with a single panel and
    ``midas_hkls`` for the ring list — no C ``ForwardSimulationCompressed``
    or ``GetHKLList`` shell-out, so the path works on any machine that has
    the Python sibling packages installed.

    Reads geometry / lattice / scan keys from ``params_template`` so the
    generator stays driven by the parameter file the user already curates
    for their experiment. Anything not in the template falls back to the
    Au-FCC defaults baked into ``simulate_panel_zarrs``. The returned zarr
    has the full ``analysis/process/analysis_parameters`` group baked in,
    so the pipeline doesn't need a separate paramstest.txt — but we still
    write one next to the zarr (a copy of the template with absolute
    paths) for users who pass ``--params`` to the CLI.
    """
    from midas_diffract.simulate_panel_zarrs import (  # type: ignore
        PanelGeom, SimConfig, simulate_panel_zarrs,
    )

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy template params into out_dir so users who pass ``--params`` to
    # the pipeline get a sane absolute-path Parameters.txt next to the
    # zarr.
    template_local = out_dir / params_template.name
    if not template_local.exists() or template_local.resolve() != params_template.resolve():
        shutil.copy2(params_template, template_local)
    params = _parse_parameter_file(template_local)

    def _scalar(key: str, default: float) -> float:
        v = params.get(key, default)
        if isinstance(v, list):
            v = v[0] if v else default
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    lat = params.get("LatticeConstant", params.get("LatticeParameter",
                                                  [4.08, 4.08, 4.08, 90, 90, 90]))
    if not isinstance(lat, list):
        lat = [lat]
    lat = [float(v) for v in lat] + [90.0] * 6
    sg = int(_scalar("SpaceGroup", 225))

    bc = params.get("BC", [1024.0, 1024.0])
    if isinstance(bc, list) and len(bc) >= 2:
        y_bc, z_bc = float(bc[0]), float(bc[1])
    else:
        y_bc = _scalar("YBC", 1024.0)
        z_bc = _scalar("ZBC", 1024.0)

    panel = PanelGeom(
        det_id=1,
        lsd=_scalar("Lsd", _scalar("Distance", 1_000_000.0)),
        y_bc=y_bc,
        z_bc=z_bc,
        tx=_scalar("tx", 0.0),
        ty=_scalar("ty", 0.0),
        tz=_scalar("tz", 0.0),
        p_distortion=[_scalar(f"p{i}", 0.0) for i in range(11)],
    )
    cfg = SimConfig(
        space_group=sg,
        lattice_a=lat[0],
        wavelength_A=_scalar("Wavelength", 0.22291),
        n_pixels=int(_scalar("NrPixels", 2048)),
        px_um=_scalar("px", _scalar("PixelSize", 200.0)),
        omega_start_deg=_scalar("OmegaStart", 180.0),
        omega_end_deg=_scalar("OmegaStart", 180.0)
            + _scalar("OmegaStep", -0.25) * int(_scalar("EndNr", 1440)),
        omega_step_deg=_scalar("OmegaStep", -0.25),
        rings_max=int(_scalar("OverAllRingToIndex", 9)) + 6,
        wedge_deg=_scalar("Wedge", 0.0),
        rho_d=_scalar("RhoD", _scalar("MaxRingRad", 2_000_000.0)),
        rsample_um=_scalar("Rsample", 1_000.0),
        hbeam_um=_scalar("Hbeam", 1_000.0),
        peak_intensity=_scalar("PeakIntensity", 50_000.0),
        gauss_sigma_px=_scalar("GaussWidth", 1.5),
        pos_noise_px=0.0,
    )

    LOG.info("simulate_panel_zarrs: 1 panel, %d grains, seed=%d, device=%s",
             n_grains, seed, device)
    summary = simulate_panel_zarrs(
        out_dir=out_dir,
        n_grains=n_grains,
        panels=[panel],
        cfg=cfg,
        seed=seed,
        out_stem="midas_ff_pipeline_synth",
        device=device,
        log=lambda msg: LOG.info(msg),
    )
    zips: list[Path] = list(summary["zips"])
    if not zips:
        raise RuntimeError("simulate_panel_zarrs produced no zarr zip")
    final_zip = zips[0]
    LOG.info("synthetic dataset ready: %s", final_zip)
    return final_zip


# ---------------------------------------------------------------------------
#  Multi-detector synthetic data
# ---------------------------------------------------------------------------

DEFAULT_PINWHEEL_TX_DEG = (15.0, 105.0, -75.0, -165.0)
HYDRA_PANEL_TX_DEG = (297.0, 27.0, 117.0, 207.0)
HYDRA_PANEL_BC_PX = (
    (2300.578631, 2172.142632),
    (2270.011492, 2167.351149),
    (2293.938108, 2097.179088),
    (2404.608066, 2141.325841),
)
HYDRA_PANEL_LSD_UM = (3298727.507767, 3299134.008645, 3298243.246691, 3298414.13366)
HYDRA_PANEL_TY_DEG = (-0.207605, 0.066873, 0.205411, -0.092397)
HYDRA_PANEL_TZ_DEG = (0.17269, 0.381165, 0.145752, 0.181459)


def generate_pinwheel_synthetic_dataset(*,
                                         out_dir: Path,
                                         n_grains: int = 100,
                                         seed: int = 42,
                                         n_panels: int = 4,
                                         use_hydra_geometry: bool = True,
                                         lattice_a: float = 4.08,
                                         wavelength_A: float = 0.22291,
                                         space_group: int = 225,
                                         n_pixels: int = 2048,
                                         px_um: float = 200.0,
                                         omega_start_deg: float = 180.0,
                                         omega_end_deg: float = -180.0,
                                         omega_step_deg: float = -0.25,
                                         rings_max: int = 9,
                                         rsample_um: float = 100.0,
                                         hbeam_um: float = 100.0,
                                         peak_intensity: float = 50000.0,
                                         gauss_sigma_px: float = 1.5,
                                         pos_noise_px: float = 0.0,
                                         device: str = "cuda",
                                         out_stem: str = "midas_diffract_pinwheel",
                                         ) -> tuple[list[Path], Path]:
    """Forward-simulate a true multi-detector pinwheel via midas_diffract.

    Returns ``(zip_paths, detectors_json)`` ready for
    ``midas-ff-pipeline run --detectors detectors.json``.
    """
    from midas_diffract.simulate_panel_zarrs import (  # type: ignore
        PanelGeom, SimConfig, simulate_panel_zarrs,
    )

    if use_hydra_geometry and n_panels != 4:
        raise ValueError("hydra geometry presets are 4-panel; set use_hydra_geometry=False or n_panels=4")

    if use_hydra_geometry:
        panels = [
            PanelGeom(
                det_id=det_id,
                lsd=HYDRA_PANEL_LSD_UM[i],
                y_bc=HYDRA_PANEL_BC_PX[i][0],
                z_bc=HYDRA_PANEL_BC_PX[i][1],
                tx=HYDRA_PANEL_TX_DEG[i],
                ty=HYDRA_PANEL_TY_DEG[i],
                tz=HYDRA_PANEL_TZ_DEG[i],
                p_distortion=[0.0] * 11,
            )
            for i, det_id in enumerate(range(1, n_panels + 1))
        ]
    else:
        # Simplified pinwheel: shared BC/Lsd, different tx by 90°.
        panels = [
            PanelGeom(
                det_id=det_id,
                lsd=3_300_000.0,
                y_bc=2300.0,
                z_bc=2150.0,
                tx=DEFAULT_PINWHEEL_TX_DEG[i % len(DEFAULT_PINWHEEL_TX_DEG)],
                ty=0.0,
                tz=0.0,
            )
            for i, det_id in enumerate(range(1, n_panels + 1))
        ]

    cfg = SimConfig(
        space_group=space_group,
        lattice_a=lattice_a,
        wavelength_A=wavelength_A,
        n_pixels=n_pixels,
        px_um=px_um,
        omega_start_deg=omega_start_deg,
        omega_end_deg=omega_end_deg,
        omega_step_deg=omega_step_deg,
        rings_max=rings_max,
        wedge_deg=0.0,
        rho_d=2_000_000.0,
        rsample_um=rsample_um,
        hbeam_um=hbeam_um,
        peak_intensity=peak_intensity,
        gauss_sigma_px=gauss_sigma_px,
        pos_noise_px=pos_noise_px,
    )

    summary = simulate_panel_zarrs(
        out_dir=out_dir,
        n_grains=n_grains,
        panels=panels,
        cfg=cfg,
        seed=seed,
        out_stem=out_stem,
        device=device,
        log=lambda msg: LOG.info(msg),
    )
    return summary["zips"], summary["detectors_json"]


def _make_per_det_paramfile(template: Path,
                            grains_csv: Path,
                            out_dir: Path,
                            det_id: int,
                            *,
                            tx: float, ty: float, tz: float,
                            lsd: float,
                            y_bc: float, z_bc: float,
                            out_file_name: str) -> tuple[Path, dict]:
    """Rewrite a Parameters.txt template with one detector's geometry."""
    params = _parse_parameter_file(template)
    new_param = out_dir / f"params_det{det_id}.txt"
    keys_we_handle = {
        "InFileName", "OutFileName", "Lsd", "BC", "tx", "ty", "tz",
        # Suppress per-panel jitter sigmas + multi-det layout keys when
        # forward-simulating a single detector.
        "PanelTxJitterSigma", "PanelTyJitterSigma", "PanelTzJitterSigma",
        "PanelLsdJitterUm", "PanelTxNominal", "NumDetectors",
    }
    with new_param.open("w") as out:
        for raw in template.open():
            stripped = raw.split()
            if not stripped:
                out.write(raw)
                continue
            key = stripped[0]
            if key in keys_we_handle:
                continue
            out.write(raw)
        out.write(f"\n# --- per-detector overrides (det {det_id}) ---\n")
        out.write(f"InFileName {grains_csv}\n")
        out.write(f"OutFileName {out_dir / out_file_name}\n")
        out.write(f"Lsd {lsd}\n")
        out.write(f"BC {y_bc} {z_bc}\n")
        out.write(f"tx {tx}\n")
        out.write(f"ty {ty}\n")
        out.write(f"tz {tz}\n")
    return new_param, params


def generate_multidet_synthetic_dataset(*,
                                        out_dir: Path,
                                        params_template: Path,
                                        n_grains: int = 50,
                                        seed: int = 42,
                                        n_cpus: int = 8,
                                        n_detectors: int = 4,
                                        tx_nominal_deg: tuple[float, ...] = DEFAULT_PINWHEEL_TX_DEG,
                                        lsd_jitter_um: float = 2000.0,
                                        ty_sigma_deg: float = 0.05,
                                        tz_sigma_deg: float = 0.05,
                                        ) -> tuple[list[Path], Path]:
    """Forward-simulate a pinwheel multi-detector dataset.

    All ``n_detectors`` panels see the same grain population (one shared
    ``GrainsSim.csv``) but each panel has its own (Lsd, ty, tz, tx) so the
    captured spots differ. Panels share the same beam center in their own
    local frame; their tx jitter rotates them into different eta wedges.

    Returns
    -------
    zips : list[Path]
        Per-detector ``.MIDAS.zip`` paths, one per detector (det_id 1..N).
    detectors_json : Path
        Path to a ``detectors.json`` ready to feed
        ``midas-ff-pipeline run --detectors``.
    """
    import json
    import numpy as np

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(tx_nominal_deg) < n_detectors:
        raise ValueError(
            f"tx_nominal_deg has {len(tx_nominal_deg)} entries but n_detectors={n_detectors}"
        )

    # Copy template + parse.
    template_local = out_dir / params_template.name
    if not template_local.exists() or template_local.resolve() != params_template.resolve():
        shutil.copy2(params_template, template_local)
    params = _parse_parameter_file(template_local)

    # ---- One shared GrainsSim.csv ----
    grains_csv = out_dir / "GrainsSim.csv"
    rsample = float(params.get("Rsample", 200))
    hbeam = float(params.get("Hbeam", 200))
    beam_thickness = float(params.get("BeamThickness", 200))
    sg = int(params.get("SpaceGroup", 225))
    lat = params.get("LatticeConstant", [4.08, 4.08, 4.08, 90, 90, 90])
    if not isinstance(lat, list):
        lat = [lat]
    lat = [float(v) for v in lat]
    while len(lat) < 6:
        lat.append(90.0)
    _generate_grains_csv(grains_csv, n_grains, lat,
                         rsample, hbeam, beam_thickness,
                         space_group=sg, seed=seed)

    # ---- Detector geometry truth ----
    rng = np.random.default_rng(seed)
    lsd_nominal = float(params.get("Lsd", 1_000_000.0))
    bc = params.get("BC", [1024.0, 1024.0])
    if not isinstance(bc, list):
        bc = [bc, bc]
    bc_y_nom, bc_z_nom = float(bc[0]), float(bc[1])
    lsd_truth = lsd_nominal + rng.normal(0, lsd_jitter_um, n_detectors)
    tx_truth = np.array(tx_nominal_deg[:n_detectors], dtype=float)
    ty_truth = rng.normal(0, ty_sigma_deg, n_detectors)
    tz_truth = rng.normal(0, tz_sigma_deg, n_detectors)

    midas_home = _find_midas_home()
    fs_bin = midas_home / "FF_HEDM" / "bin" / "ForwardSimulationCompressed"
    if not fs_bin.exists():
        raise FileNotFoundError(f"missing {fs_bin}")

    # ---- Forward-simulate each panel ----
    out_zips: list[Path] = []
    for det_id in range(1, n_detectors + 1):
        idx = det_id - 1
        out_file_base = f"midas_ff_pipeline_synth_det{det_id}"
        per_det_param, _ = _make_per_det_paramfile(
            template_local, grains_csv, out_dir, det_id,
            tx=float(tx_truth[idx]), ty=float(ty_truth[idx]), tz=float(tz_truth[idx]),
            lsd=float(lsd_truth[idx]), y_bc=bc_y_nom, z_bc=bc_z_nom,
            out_file_name=out_file_base,
        )
        # Re-parse the just-written per-det param file so the zarr's stored
        # analysis_parameters reflects this detector's geometry, not the
        # template defaults.
        det_params = _parse_parameter_file(per_det_param)
        LOG.info("Running ForwardSimulationCompressed for det %d (tx=%.2f°, Lsd=%.1f um)…",
                 det_id, tx_truth[idx], lsd_truth[idx])
        raw_zip = out_dir / f"{out_file_base}_scanNr_0.zip"
        # FS can SIGSEGV at libhdf5 cleanup when launched as a subprocess of a
        # Python parent that has loaded torch/h5py/zarr. Retry up to 3 times;
        # the binary is deterministic so a clean second invocation almost
        # always succeeds.
        last_rc = -1
        for attempt in range(3):
            if raw_zip.exists():
                break
            proc = subprocess.run(
                [str(fs_bin), str(per_det_param), str(n_cpus)],
                cwd=str(out_dir), check=False,
            )
            last_rc = proc.returncode
            if raw_zip.exists():
                break
            LOG.warning("FS attempt %d for det %d failed (rc=%d, no zip); retrying",
                        attempt + 1, det_id, last_rc)
        if not raw_zip.exists():
            raise FileNotFoundError(f"FS didn't produce {raw_zip} for det {det_id} after 3 attempts (last rc={last_rc})")
        if last_rc != 0:
            LOG.warning("FS rc=%d for det %d but %s exists; continuing",
                        last_rc, det_id, raw_zip)

        final_zip = out_dir / f"{out_file_base}.analysis.MIDAS.zip"
        if final_zip.exists():
            final_zip.unlink()
        shutil.move(str(raw_zip), str(final_zip))
        _enrich_zarr_metadata(final_zip, det_params)
        out_zips.append(final_zip)

    # ---- detectors.json ready for the pipeline ----
    detectors_json = out_dir / "detectors.json"
    entries = []
    for det_id in range(1, n_detectors + 1):
        idx = det_id - 1
        entries.append({
            "det_id": det_id,
            "zarr_path": str(out_zips[idx]),
            "lsd": float(lsd_truth[idx]),
            "y_bc": bc_y_nom,
            "z_bc": bc_z_nom,
            "tx": float(tx_truth[idx]),
            "ty": float(ty_truth[idx]),
            "tz": float(tz_truth[idx]),
            "p_distortion": [0.0] * 11,
        })
    with detectors_json.open("w") as fp:
        json.dump(entries, fp, indent=2)

    LOG.info("multi-detector synthetic dataset ready: %d zips + %s",
             n_detectors, detectors_json)
    return out_zips, detectors_json
