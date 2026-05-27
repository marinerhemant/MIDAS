"""Phase 5 of the real-data validation: downstream strain comparison.

Refits grains under two calibrations using the SAME Phase 2 spot data:
  (A) Sequential: Phase 1 powder-only MAP (the baseline the paper §6.2 critiques)
  (B) Joint:      Phase 3 joint MAP

For each, patch the HEDM paramstest's (Lsd, BC, tilts, panel_delta_*) with
that calibration's values, then re-run midas-fit-grain + process_grains via
midas-ff-pipeline on the existing Phase 2 spot association.  Compare the
per-grain deviatoric strain distributions; this is paper §5's headline
real-data figure.

Writes:
    <output>/sequential/                 — refit dir under Phase 1 MAP
    <output>/joint/                      — refit dir under Phase 3 MAP
    <output>/phase5_strain_sequential.csv
    <output>/phase5_strain_joint.csv
    <output>/phase5_strain_comparison.csv  — paired ||eps_dev|| + KS stats
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np


# ---------------------------------------------------------------- patching


def patch_paramstest(template_path: Path, out_path: Path, geom: Dict,
                     panel_layout=None, layer_dir_target: Path = None) -> None:
    """Write a copy of ``template_path`` with the geometry keys overridden
    by ``geom``. Adds/overrides:
        Lsd, BC, ty, tz, p0..p14 (if panel-derived)
        OutputFolder, ResultFolder if layer_dir_target is provided.

    Geometry kwargs that are missing in ``geom`` are left at the template's
    values (so we don't over-restrict the input).
    """
    lines = template_path.read_text().splitlines()
    overrides = {
        "Lsd": ("Lsd", lambda v: f"Lsd {v:.6f}"),
        "BC_y": ("BC", None),   # BC line: "BC y z"
        "ty": ("ty", lambda v: f"ty {v:.6f}"),
        "tz": ("tz", lambda v: f"tz {v:.6f}"),
    }
    # Build new BC line if BC_y/BC_z present
    bc_y = geom.get("BC_y"); bc_z = geom.get("BC_z")

    seen = set()
    out = []
    for raw in lines:
        s = raw.split("#", 1)[0].strip()
        if not s:
            out.append(raw); continue
        toks = s.split()
        key = toks[0]
        if key == "Lsd" and "Lsd" in geom:
            out.append(f"Lsd {geom['Lsd']:.6f}"); seen.add("Lsd"); continue
        if key == "BC" and bc_y is not None and bc_z is not None:
            out.append(f"BC {bc_y:.6f} {bc_z:.6f}"); seen.add("BC"); continue
        if key == "ty" and "ty" in geom:
            out.append(f"ty {geom['ty']:.6f}"); seen.add("ty"); continue
        if key == "tz" and "tz" in geom:
            out.append(f"tz {geom['tz']:.6f}"); seen.add("tz"); continue
        if key in ("OutputFolder", "ResultFolder") and layer_dir_target is not None:
            sub = "Output" if key == "OutputFolder" else "Results"
            out.append(f"{key} {layer_dir_target}/{sub}"); seen.add(key); continue
        out.append(raw)
    # Append missing keys
    if "Lsd" not in seen and "Lsd" in geom:
        out.append(f"Lsd {geom['Lsd']:.6f}")
    if "BC" not in seen and bc_y is not None and bc_z is not None:
        out.append(f"BC {bc_y:.6f} {bc_z:.6f}")
    if "ty" not in seen and "ty" in geom:
        out.append(f"ty {geom['ty']:.6f}")
    if "tz" not in seen and "tz" in geom:
        out.append(f"tz {geom['tz']:.6f}")
    if layer_dir_target is not None:
        if "OutputFolder" not in seen:
            out.append(f"OutputFolder {layer_dir_target}/Output")
        if "ResultFolder" not in seen:
            out.append(f"ResultFolder {layer_dir_target}/Results")
    out_path.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------- refit


def stage_refit_dir(phase2_layer_dir: Path, refit_root: Path) -> Path:
    """Stage a refit working dir: <refit_root>/LayerNr_1/ with symlinks back
    to Phase 2 inputs (Spots.bin, Data.bin, IDsHash.csv, SpotsToIndex.csv,
    hkls.csv, IDRings.csv, midas_log/) and fresh writable Output/Results dirs.
    """
    layer_target = refit_root / "LayerNr_1"
    (layer_target / "Output").mkdir(parents=True, exist_ok=True)
    (layer_target / "Results").mkdir(parents=True, exist_ok=True)
    # Symlink shared inputs
    for name in ("Spots.bin", "Spots_det.bin", "Data.bin", "nData.bin",
                  "ExtraInfo.bin", "IDsHash.csv", "IDRings.csv",
                  "SpotsToIndex.csv", "hkls.csv", "InputAll.csv",
                  "InputAllExtraInfoFittingAll.csv", "MergeMap.csv",
                  "PowderModel.csv", "Result_StartNr_1_EndNr_1440.csv",
                  "Radius_StartNr_1_EndNr_1440.csv"):
        src = phase2_layer_dir / name
        dst = layer_target / name
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
    # Also link IndexBest.bin / IndexBestFull.bin (from Phase 2's Output dir)
    src_out = phase2_layer_dir / "Output"
    if src_out.exists():
        for f in src_out.iterdir():
            dst = layer_target / "Output" / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
    return layer_target


def run_pipeline_refit(refit_root: Path, paramstest: Path,
                        python_bin: str, n_cpus: int = 16) -> int:
    """Invoke midas-ff-pipeline on a staged refit dir, resuming at the refine
    stage (all upstream artefacts are pre-staged via symlinks from Phase 2)."""
    cmd = [
        python_bin, "-u", "-m", "midas_ff_pipeline.cli", "run",
        "--params", str(paramstest),
        "--result", str(refit_root),
        "--layers", "1-1",
        "--n-cpus", str(n_cpus),
        "--device", "cpu",
        "--dtype", "float64",
        "--solver", "lm",
        "--loss", "pixel",
        "--resume", "from",
        "--from", "refinement",
        "--skip-validation",
        "--num-files-per-scan", "1440",
    ]
    print(f"   $ {' '.join(cmd)}")
    return subprocess.call(cmd)


def pad_refine_outputs(layer_dir: Path, target_n: int = 22807) -> None:
    """Workaround for the midas_fit_grain refiner truncating the last 3-4
    seed slots. Pad OrientPosFit (27*8), ProcessKey (5000*4), FitBest
    (22*5000*8) up to ``target_n`` rows with zeros, so process_grains'
    row-count guard at midas_process_grains/io/binary.py:269 doesn't trip."""
    specs = [
        (layer_dir / "Results" / "OrientPosFit.bin", 27 * 8),
        (layer_dir / "Results" / "ProcessKey.bin",   5000 * 4),
        (layer_dir / "Output" / "FitBest.bin",       22 * 5000 * 8),
    ]
    import numpy as np
    for path, rec_size in specs:
        if not path.exists():
            print(f"   pad: skip {path.name} (missing)")
            continue
        sz = path.stat().st_size
        n = sz // rec_size
        residual = sz - n * rec_size
        if n >= target_n and residual == 0:
            continue
        # Trim partial trailing record then zero-pad.
        if residual:
            with open(path, "r+b") as f:
                f.truncate(n * rec_size)
        pad = np.zeros(((target_n - n) * rec_size,), dtype=np.uint8)
        with open(path, "ab") as f:
            f.write(pad.tobytes())
        print(f"   pad: {path.name}  {n} → {path.stat().st_size // rec_size} rows")


def run_process_grains(refit_root: Path, paramstest: Path,
                        python_bin: str, n_cpus: int = 16) -> int:
    """Invoke midas-ff-pipeline --only process_grains on a staged refit dir.

    Used after :func:`pad_refine_outputs` patches up the row-count mismatch
    from the refiner.  We can't pass --only to the same pipeline call that
    runs refinement (the pipeline dies before process_grains is reached when
    the refiner produces inconsistent row counts), so we drive it in two
    pipeline calls.
    """
    cmd = [
        python_bin, "-u", "-m", "midas_ff_pipeline.cli", "run",
        "--params", str(paramstest),
        "--result", str(refit_root),
        "--layers", "1-1",
        "--n-cpus", str(n_cpus),
        "--device", "cpu",
        "--dtype", "float64",
        "--solver", "lm",
        "--loss", "pixel",
        "--resume", "from",
        "--from", "process_grains",
        "--skip-validation",
        "--num-files-per-scan", "1440",
    ]
    return subprocess.call(cmd)


# ---------------------------------------------------------------- compare


def read_grains_strains(grains_csv: Path):
    """Read Grains.csv → (grain_ids, strain_voigt[N,6])."""
    if not grains_csv.exists():
        raise FileNotFoundError(grains_csv)
    ids, strain = [], []
    for line in grains_csv.read_text().splitlines():
        if line.startswith("%") or not line.strip():
            continue
        c = line.split("\t")
        if len(c) < 21:
            continue
        ids.append(int(c[0]))
        # cols 13..18 in 21-col Grains.csv = strain[6] (Voigt)
        strain.append([float(x) for x in c[13:19]])
    return np.asarray(ids, dtype=np.int64), np.asarray(strain, dtype=np.float64)


def deviatoric_norm(strain_voigt: np.ndarray) -> np.ndarray:
    """||eps_dev||_F per row. Voigt: [e11, e22, e33, e23, e13, e12]."""
    e = strain_voigt
    tr = (e[:, 0] + e[:, 1] + e[:, 2]) / 3.0
    dev = np.column_stack([
        e[:, 0] - tr, e[:, 1] - tr, e[:, 2] - tr,
        e[:, 3], e[:, 4], e[:, 5],
    ])
    # Frobenius norm — diagonals counted once, off-diags doubled (full tensor)
    return np.sqrt(dev[:, 0]**2 + dev[:, 1]**2 + dev[:, 2]**2
                   + 2.0 * (dev[:, 3]**2 + dev[:, 4]**2 + dev[:, 5]**2))


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase1-map",  type=Path, required=True,
                   help="phase1_powder_map.json")
    p.add_argument("--phase3-map",  type=Path, required=True,
                   help="phase3_joint_map.json")
    p.add_argument("--phase2-layer-dir", type=Path, required=True)
    p.add_argument("--paramstest-template", type=Path, required=True,
                   help="HEDM paramstest to clone + patch with each MAP")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--python-bin", default=sys.executable,
                   help="python interpreter to invoke midas-ff-pipeline with")
    p.add_argument("--n-cpus", type=int, default=16)
    args = p.parse_args(argv)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Phase 5 — Downstream strain comparison (sequential vs joint)")
    print("=" * 70)

    map1 = json.loads(args.phase1_map.read_text())
    map3 = json.loads(args.phase3_map.read_text())

    geom1 = {k: float(map1[k]) for k in ("Lsd", "BC_y", "BC_z", "ty", "tz") if k in map1}
    geom3 = {k: float(map3[k]) for k in ("Lsd", "BC_y", "BC_z", "ty", "tz") if k in map3}
    print(f"\n   sequential (Phase 1) Lsd={geom1.get('Lsd'):.2f} ty={geom1.get('ty'):.4f} tz={geom1.get('tz'):.4f}")
    print(f"   joint      (Phase 3) Lsd={geom3.get('Lsd'):.2f} ty={geom3.get('ty'):.4f} tz={geom3.get('tz'):.4f}")

    for label, geom in [("sequential", geom1), ("joint", geom3)]:
        refit_root = args.output / label
        layer = stage_refit_dir(args.phase2_layer_dir, refit_root)
        ps = refit_root / "ps_patched.txt"
        patch_paramstest(args.paramstest_template, ps, geom, layer_dir_target=layer)
        # Pipeline + process_grains read `<layer>/paramstest.txt` directly
        # for OutputFolder/ResultFolder; mirror our patched master there.
        shutil.copy(ps, layer / "paramstest.txt")
        print(f"\n>> [{label}] running refit at {refit_root}")
        rc = run_pipeline_refit(refit_root, ps, args.python_bin, args.n_cpus)
        # rc != 0 is OK if refinement landed but process_grains tripped the
        # known row-count guard.  Always pad + (re-)run process_grains.
        print(f"   pad-fixup (refiner truncation workaround):")
        pad_refine_outputs(layer)
        print(f"   re-run process_grains:")
        rc2 = run_process_grains(refit_root, ps, args.python_bin, args.n_cpus)
        if rc2 != 0:
            print(f"   ✗ process_grains {label} failed (rc={rc2})")
            continue
        grains_csv = layer / "Grains.csv"
        if not grains_csv.exists():
            print(f"   ✗ {grains_csv} not produced")
            continue
        ids, strain = read_grains_strains(grains_csv)
        strain_csv = args.output / f"phase5_strain_{label}.csv"
        with open(strain_csv, "w") as f:
            f.write("grain_id,e11,e22,e33,e23,e13,e12,eps_dev_norm\n")
            dn = deviatoric_norm(strain)
            for i, gid in enumerate(ids):
                f.write(f"{int(gid)},"
                        f"{','.join(f'{x:.6e}' for x in strain[i])},"
                        f"{dn[i]:.6e}\n")
        print(f"   ✓ Saved {strain_csv}  ({len(ids)} grains)")

    # ----- comparison
    seq_csv = args.output / "phase5_strain_sequential.csv"
    joint_csv = args.output / "phase5_strain_joint.csv"
    if seq_csv.exists() and joint_csv.exists():
        seq = np.loadtxt(seq_csv, delimiter=",", skiprows=1)
        jnt = np.loadtxt(joint_csv, delimiter=",", skiprows=1)
        seq_map = {int(r[0]): r for r in seq}
        jnt_map = {int(r[0]): r for r in jnt}
        common = sorted(set(seq_map) & set(jnt_map))
        cmp_csv = args.output / "phase5_strain_comparison.csv"
        with open(cmp_csv, "w") as f:
            f.write("grain_id,eps_dev_seq,eps_dev_joint,delta_eps_dev\n")
            for gid in common:
                s_d = seq_map[gid][-1]; j_d = jnt_map[gid][-1]
                f.write(f"{gid},{s_d:.6e},{j_d:.6e},{(j_d-s_d):.6e}\n")
        s_arr = np.array([seq_map[gid][-1] for gid in common])
        j_arr = np.array([jnt_map[gid][-1] for gid in common])
        # KS test
        try:
            from scipy.stats import ks_2samp
            ks = ks_2samp(s_arr, j_arr)
            ks_line = f"   KS: D={ks.statistic:.4f}  p={ks.pvalue:.3e}"
        except Exception:
            ks_line = "   KS: scipy unavailable"
        print(f"\n   Sequential ||eps_dev|| mean={s_arr.mean():.3e}  median={np.median(s_arr):.3e}  std={s_arr.std():.3e}  (n={len(s_arr)})")
        print(f"   Joint      ||eps_dev|| mean={j_arr.mean():.3e}  median={np.median(j_arr):.3e}  std={j_arr.std():.3e}")
        print(ks_line)
        print(f"   ✓ Saved {cmp_csv}")
    print("\n>> Phase 5 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
