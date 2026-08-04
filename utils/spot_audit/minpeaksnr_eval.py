"""What would MinPeakSNR do to the Au3 spot list, at detection time?

Runs the production peak search over sampled frames at several MinPeakSNR
values and reports how many blobs survive, plus how many of the blobs that
correspond to KNOWN-REAL (indexed) spots survive. The second number is the one
that matters: a filter that cleans the list by deleting real spots is useless.
"""
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/gdata/dm/1ID/2026/bt_1id_jul26/analysis/au3_cubes_ff_000008")
RESULT = R / "results/LayerNr_1"
ZIP = RESULT / "Au3_cubes_ff_000008.MIDAS.zip"
AUD = R / "spot_noise_audit/spot_audit_snr.csv"
NFRAMES = 40

from midas_peakfit.background import (
    bins_from_params, estimate_cell_stats, region_snr,
)
from midas_peakfit.connected import filter_regions_by_size, find_regions
from midas_peakfit.geometry import compute_good_coords, load_ring_radii
from midas_peakfit.orchestrator import _build_panels
from midas_peakfit.preprocess import (
    apply_threshold, correct_frame, prepare_dark, prepare_flood,
)
from midas_peakfit.zarr_io import (
    frame_omega, load_corrections, parse_zarr_params, read_frame,
)

p = parse_zarr_params(str(ZIP)); p.ResultFolder = str(RESULT)
panels = _build_panels(p); load_corrections(str(ZIP), p)
rads = load_ring_radii(p, p.ResultFolder)
bins = bins_from_params(p, panels, rads, n_sectors=36)
gc = compute_good_coords(p, panels, rads)
dark = prepare_dark(p.dark, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)
flood = prepare_flood(p.flood, p.NrPixels, p.NrPixelsY, p.NrPixelsZ, p.TransOpt)

m = pd.read_csv(AUD)
omes = np.array([frame_omega(p, i + p.skipFrame) for i in range(p.nFrames)])
m["frame"] = [int(np.argmin(np.abs(omes - o))) for o in m["Omega"]]
idxs = np.linspace(0, p.nFrames - 1, NFRAMES).astype(int)

# blob -> "is this a known-real (indexed) spot?" by proximity, in RAW coords
snrs, is_real = [], []
for fi in idxs:
    raw = read_frame(str(ZIP), int(fi) + p.skipFrame)
    corr = correct_frame(raw, NrPixels=p.NrPixels, NrPixelsY=p.NrPixelsY,
                         NrPixelsZ=p.NrPixelsZ, transform_options=p.TransOpt,
                         dark=dark, flood=flood, good_coords=gc, bc=p.bc,
                         bad_px_intensity=p.BadPxIntensity, make_map=p.makeMap)
    regs = filter_regions_by_size(find_regions(apply_threshold(corr, gc), gc),
                                  p.minNrPx, p.maxNrPx)
    if not regs:
        continue
    med, sig = estimate_cell_stats(corr, bins)
    sub = m[(m["frame"] == int(fi)) & m["indexed"]]
    for reg in regs:
        snrs.append(region_snr(reg, corr, bins, med, sig))
        # corrected (row=Y, col=Z) -> raw (row=Z, col=Y)
        rz, ry = reg.pixel_cols.mean(), reg.pixel_rows.mean()
        near = False
        if len(sub):
            d = np.hypot(sub["ZRawPx"].to_numpy() - rz,
                         sub["YRawPx"].to_numpy() - ry).min()
            near = d <= 3.0
        is_real.append(near)

snrs = np.asarray(snrs); is_real = np.asarray(is_real)
print(f"{len(snrs)} blobs over {len(idxs)} frames; "
      f"{int(is_real.sum())} match an indexed spot within 3 px\n")
print(f"  {'MinPeakSNR':>11s} {'blobs kept':>11s} {'% of all':>9s} "
      f"{'known-real kept':>16s}")
n_real = max(int(is_real.sum()), 1)
for cut in (0, 2, 3, 5, 8, 10, 20, 50):
    k = snrs >= cut
    print(f"  {cut:11d} {int(k.sum()):11d} {k.mean():8.1%} "
          f"{int((k & is_real).sum()):8d}/{n_real} "
          f"({(k & is_real).sum()/n_real:5.1%})")
