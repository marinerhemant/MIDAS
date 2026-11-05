#!/usr/bin/env python3
"""Prepare one arm of a null-controlled FF run: transforms with an optional omega scramble, then a seed subsample.

Usage:
  python run_arm.py --src <real LayerNr_1 dir> --params <Parameters.txt> --out <arm dir>
                    [--null-seed N] [--f 0.1] [--f-seed 20261026] [--rings 1,2,4,5,6,8,9,11] [--band-px 12]
                    [--px 150] [--device cuda] [--dtype float32]

--src must hold the zarr (*.MIDAS.zip), hkls.csv and Temp/AllPeaks_PS.bin of a finished peakfit (they are symlinked,
never modified). The arm directory then looks like a normal result folder that has completed peakfit and transforms, so
`midas-pipeline run --resume auto` skips straight to binning/indexing (every stage guard is "output exists").

What it does (this is the same code path as `Pipeline.run()` in midas_transforms, with ONE hook):
  merge_overlapping_peaks -> [--null-seed: permute (Omega, MinOme, MaxOme) among spots of the same selected ring]
  -> calc_radius -> fit_setup -> bin_data -> dump
The scramble is applied to the merged spot table, i.e. AFTER overlapping peaks are merged across frames (scrambling
before the merge would break the merge, not test the indexer) and BEFORE anything geometric is computed. A spot is in a
ring if its Radius (px) is within --band-px of a selected ring's radius (hkls.csv); other spots are left untouched.
Without --null-seed the table is untouched, so the arm must reproduce the real run's InputAll.csv: that is the check
that the harness itself changes nothing.
--f < 1 replaces SpotsToIndex.csv by a uniform random subset (the full list is kept as SpotsToIndex_full.csv).
"""
import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

# Result_*.csv / merge_arr columns (midas_transforms.merge): SpotID IntegratedIntensity Omega YCen ZCen IMax MinOme MaxOme
# SigmaR SigmaEta NrPx NrPxTot Radius Eta RawSumIntensity maskTouched FitRMSE ReturnCode
C_OMEGA, C_MINOME, C_MAXOME, C_RADIUS = 2, 6, 7, 12


def ring_radii_px(hkls_csv: Path, px_um: float):
    rows = [r.split() for r in hkls_csv.read_text().splitlines() if r.strip()]
    hdr = rows[0]
    ir, irad = hdr.index("RingNr"), hdr.index("Radius")
    out = {}
    for r in rows[1:]:
        out.setdefault(int(r[ir]), float(r[irad]) / px_um)
    return out


def scramble(arr: np.ndarray, radii: dict, rings, band_px: float, seed: int):
    """Permute (Omega, MinOme, MaxOme) among the spots of each selected ring. Returns (new array, per-ring counts)."""
    new = arr.copy()
    rng = np.random.default_rng(seed)
    rad = arr[:, C_RADIUS]
    R = np.array([radii[r] for r in rings])
    d = np.abs(rad[:, None] - R[None, :])
    nearest = d.argmin(1)
    inband = d[np.arange(len(arr)), nearest] <= band_px
    cols = [C_OMEGA, C_MINOME, C_MAXOME]
    counts = {}
    for k, ring in enumerate(rings):
        idx = np.where(inband & (nearest == k))[0]
        counts[ring] = int(len(idx))
        if len(idx) > 1:
            perm = rng.permutation(idx)
            new[np.ix_(idx, cols)] = arr[np.ix_(perm, cols)]
    return new, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--null-seed", type=int, default=None)
    ap.add_argument("--f", type=float, default=1.0)
    ap.add_argument("--f-seed", type=int, default=20261026)
    ap.add_argument("--rings", default="1,2,4,5,6,8,9,11")
    ap.add_argument("--band-px", type=float, default=12.0)
    ap.add_argument("--px", type=float, default=150.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    a = ap.parse_args()

    import torch
    from midas_transforms import Pipeline
    from midas_transforms.fit_setup import fit_setup
    from midas_transforms.merge import merge_overlapping_peaks
    from midas_transforms.radius import calc_radius
    from midas_transforms.bin_data import bin_data

    src, out = Path(a.src), Path(a.out)
    lay = out / "results" / "LayerNr_1"
    (lay / "Temp").mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.params, out / "Parameters.txt")
    for f in list(src.glob("*.MIDAS.zip")) + [src / "hkls.csv"]:
        t = lay / f.name
        if not t.exists():
            os.symlink(f.resolve(), t)
    for f in (src / "Temp").glob("AllPeaks_P*.bin"):
        t = lay / "Temp" / f.name
        if not t.exists():
            os.symlink(f.resolve(), t)
    zip_path = next(lay.glob("*.MIDAS.zip"))

    pipe = Pipeline.from_zarr(zip_path, result_folder=lay, device=a.device, dtype=a.dtype)
    zp = pipe.zarr_params
    merge = merge_overlapping_peaks(
        allpeaks_ps_bin=pipe.allpeaks_ps_bin, allpeaks_px_bin=pipe.allpeaks_px_bin, result_folder=lay,
        overlap_length=(pipe.overlap_length if pipe.overlap_length is not None else zp.OverlapLength),
        skip_frame=zp.SkipFrame, use_maxima_positions=bool(zp.UseMaximaPositions),
        use_pixel_overlap=int(zp.UsePixelOverlap), nr_pixels=zp.NrPixels,
        device=pipe.device, dtype=pipe.dtype, write=False, include_saturated=bool(zp.IncludeSaturatedSpots))
    merge_arr = merge.peaks.detach().cpu().numpy().astype(np.float64)
    report = {"n_merged_spots": int(len(merge_arr)), "null_seed": a.null_seed, "f": a.f}
    if a.null_seed is not None:
        rings = [int(x) for x in a.rings.split(",")]
        radii = ring_radii_px(lay / "hkls.csv", a.px)
        merge_arr_s, counts = scramble(merge_arr, radii, rings, a.band_px, a.null_seed)
        report["scrambled_per_ring"] = counts
        report["frac_omega_changed"] = float(np.mean(merge_arr_s[:, C_OMEGA] != merge_arr[:, C_OMEGA]))
        merge.peaks = torch.as_tensor(merge_arr_s, dtype=merge.peaks.dtype, device=merge.peaks.device)
        merge_arr = merge_arr_s
    radius = calc_radius(result_folder=lay, zarr_params=zp, result_array=merge_arr, start_nr=1,
                         end_nr=zp.EndNr if zp.EndNr > 0 else len(merge_arr), device=pipe.device, dtype=pipe.dtype, write=False)
    radius_arr = radius.spots.detach().cpu().numpy().astype(np.float64)
    fs = fit_setup(result_folder=lay, zarr_params=zp, radius_array=radius_arr, device=pipe.device, dtype=pipe.dtype, write=False)
    spots_inputall = fs.spots_inputall.detach().cpu().numpy().astype(np.float64)
    spots_extra = fs.extra.detach().cpu().numpy().astype(np.float64)
    bins = bin_data(result_folder=lay, paramstest=fs.paramstest, spots_inputall=spots_inputall,
                    extra_inputall=spots_extra, device=pipe.device, dtype=pipe.dtype, write=False)
    from midas_transforms.pipeline import PipelineResult
    pr = PipelineResult(merge=merge, radius=radius, fit_setup=fs, bins=bins)
    pipe._result = pr
    pipe.dump(lay)

    # seed subsample
    sti = lay / "SpotsToIndex.csv"
    lines = [ln for ln in sti.read_text().splitlines() if ln.strip()]
    shutil.copy2(sti, lay / "SpotsToIndex_full.csv")
    report["n_seeds_full"] = len(lines)
    if a.f < 1.0:
        rng = np.random.default_rng(a.f_seed)
        keep = np.sort(rng.choice(len(lines), size=int(np.floor(a.f * len(lines))), replace=False))
        sti.write_text("\n".join(lines[i] for i in keep) + "\n")
    report["n_seeds_used"] = len([l for l in sti.read_text().splitlines() if l.strip()])
    report["n_inputall_rows"] = int(len(spots_inputall))
    (out / "arm_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
