# DFXM: more than one orientation population per pixel (`midas_dfxm.populations`)

A single Gaussian per pixel is wrong when a pixel's rocking curve holds two separated populations (two
orientations along the beam, or a boundary seen in projection): its centroid lands between them and its
width reports the separation. `midas_dfxm.populations` decomposes each pixel's curve into up to four
populations, model-free, and tests the two-population reading against single-population nulls built from
the scan's own data. Numbers below come from ONE 6-ID-C dataset (datasetJ Dec-2025, scan S168, Ba122) and from
phantoms built on it; `$ANALYSIS` paths are provenance, not links (see `README.md`).

**Status (2026-09-29).** The S168 two-population result is VERIFIED (four adversarial lenses). The performance
and false-positive tables below are measurements on phantoms and real signal-free pixels from the same scan;
they went through two adversarial verification rounds (2026-09-29): the first REFUTED the original "0 false pairs"
wording (rounded rates), the second left the corrected text PROVISIONAL (all counts, recall ranges and lone-population
rates reproduce; the flux-jitter, real-noise and sharp-boundary numbers are heavy-tailed single-scan measurements).

## Use

```python
import midas_dfxm as dx
from midas_dfxm import populations as pp
scan = dx.load_6idc_scan(frames_dir, motor_csv, roi=None)          # frames sorted by angle x
P  = pp.pixel_populations(frames, x, baseline, lit, sigma_own)     # per-pixel populations (I, C, W, snr, peak_snr)
tp = pp.two_population(P, edge)                                    # pairs straddling `edge` +- 15 mdeg; T, share, centres
conf = pp.confirmed_populations(P)                                 # lone populations gated on peak SNR >= 5
null = pp.single_population_null(frames, x, baseline, lit, sigma_frame, P["fwhm_single"], kind="gauss", edge=edge)
sep  = pp.separation_bootstrap(tp, lit)                            # median separation (mdeg) with a block-bootstrap CI
```

Pick `edge` at the valley between the two populations (the intensity-weighted histogram of centres,
`pp.band_edges`, is DESCRIPTIVE only). `sigma_own` is the per-pixel noise of the repeat-averaged frames
(`support.reduce_support(...).sigma_frame[lit]`). Units: degrees in, degrees out; separations in mdeg.

## What it delivers, and how well (all measured on phantoms with known truth unless marked REAL)

| Quantity | Result | Source (`$ANALYSIS/datasetJ_dfxm_dec2025/`) |
|---|---|---|
| False two-population pixels, single-population truth | 160 of 5,858,557 (2.7e-5; pooled over brightness 1, 0.3, 0.1, 0.03 x real, 3 seeds each; at most 6.7e-5 in any peak-SNR bin, none below SNR 1.5); brightness down to 0.03 x real. 4 of 741,904 pure-noise pixels; 869 of 19.6 M truly single pixels in the recall phantoms (4.4e-5). The false pixels are spatially clustered, so the effective sample is smaller than the pixel count | `domains/h2h/h2h_A.json`, `h2h_C.json` |
| False two-population pixels, REAL signal-free pixels | 14 of 467,241 (3.0e-5) across the edge; 19 of 466,541 with >= 2 populations anywhere | `domains/h2h/h2h_real.json`, `h2h_multi.json` |
| Recall of true pairs (share 0.2-0.8), peak SNR >= 5, full brightness | 100 mdeg: 0.01-0.04 (not resolved); 150: 0.47-0.69 (ramp phantom; the thin-strip phantom gives 0.60-0.83); 200: 0.87-0.99; 300: 0.95-1.00. These are ranges over peak-SNR bins pooled over 3 seeds; seed-to-seed spread is about +-0.04, and the top of a range can rest on a few hundred pixels | `domains/h2h/h2h_C.json` |
| Same at 0.1 x brightness | 150: 0.21; 200: 0.51-0.69; 300: 0.58-0.81 | `domains/h2h/h2h_C.json` |
| Centre error of detected populations | 1.5-5.5 mdeg (full brightness), 6-8 (0.1 x) | `domains/h2h/h2h_C.json` |
| Thin (10 px) second-population strip at 150 mdeg | found: recall 0.60-0.83 | `domains/h2h/h2h_C.json` (P-D) |
| REAL S168 | 49 % of the lit intensity in pixels with two populations (repeats 48 %); high population 15.890 deg; separation median 157 mdeg (90 % CI 117-197) | `domains/S168_v4/s168_v4.json`; claim verified (4 lenses), `PREREGISTER_S168_v4_narrow.md` |

## Known limits (read before quoting a number)

* **The two-population fraction T is a LOWER BOUND.** With S168's separation (150-200 mdeg) the detector finds
  only ~0.5-0.9 of true pairs even when bright, and almost none at 100 mdeg (< ~0.8 FWHM the sum of two peaks
  is unimodal; only width and shape can tell, and single-population widths scatter by about 25 %). Never state
  "the rest is single-population".
* **A lone spurious population is common in dim pixels.** On real signal-free pixels a population with z >= 5
  appears in 1.7 / 8.4 / 22.8 / 47.8 % of pixels at peak SNR 1-1.5 / 1.5-2 / 2-3 / 3-5 (worse than white
  Gaussian noise: 3.4 / 10.2 / 15.5 %), but two or more populations in only 0.00 / 0.00 / 0.03 / 0.23 % (19 of 466,541 overall).
  Use `confirmed_populations` (peak SNR >= 5) for lone populations; pairs need no such gate
  (rate ~3e-5, see the table; `domains/h2h/h2h_multi.json`).
* **Where the second population sits is NOT readable from where it is detected.** Dim pixels (grain edges)
  miss it, so "distance to the edge" of detected pixels followed brightness and reversed sign under a matched
  null. Location claims need a matched-detection-power subset (or the threshold-free share map, below).
* **A histogram split into bands is not evidence of two populations**: one smooth population also splits into
  two bands. Only the per-pixel test counts.
* **The separation depends on the detector** (S168: 157 / 179 / 197 mdeg median with three different
  detectors): quote it as a range with the method.
* Sampling: about 12 scan points per population FWHM (S168: 12.4) is at the floor for resolving curve shape.
* Untested here: motor backlash (no angle readback logged), harmonic contamination, a second reflection in the
  field. The main recall table uses Gaussian phantom peaks; on held-out asymmetric shapes (skew-normal,
  pseudo-Voigt; full brightness, peak SNR >= 5) the detector's recall was 0.24-0.38 at 125 mdeg, 0.52-0.68 at
  150 and 0.91-0.95 at 200, with 369 false pairs in 7.46 M single-population pixels on six shape families (4.9e-5; at most 1.3e-4 in
  any condition, largest for Lorentzian peaks) (`domains/lrt/lrt_C.json`, `lrt_A.json`, column M1).
* **False pairs concentrate at sharp orientation steps.** The decision curve pools a 5 x 5 neighbourhood, so a pixel
  within ~2 px of a step in centre position across the edge sees a mixed curve. On all-single-population phantoms
  with a +-100 mdeg step (about 2 px wide) across the edge (artifact lens, `refute_perf2_artifact/t7.json`, one seed,
  full brightness, peak SNR >= 5, not re-run by the author): 7-16 % of pixels within 2.5 px of the boundary are
  flagged, 0.10 % (lens), 0.12 % (line), 0.19 % / 0.40 % (Voronoi mosaic, 12 / 40 domains) of the whole image, and
  <= 1e-5 beyond 6 px. Earlier, +-100 mdeg stripes 4 px wide reached 4.6 %. A pair on or next to a sharp boundary
  is not evidence of two populations along the beam; a fine mosaic can put more than 5 % of pixels in that zone.
* **Frame-to-frame flux jitter creates false pairs.** The detector does not normalise flux. On single-population
  phantoms at S168 brightness with a common multiplicative per-frame factor 1 + s N(0,1), the false-pair rate was
  1.4e-5 (s = 0), median 4e-5 (s = 1.25 %, 8 draws; range 1.4e-5 - 6e-4),
  median 0.28 % (16 draws, mean 1.1 %) and 0.36 % (24 independent draws, range 4e-5 - 12 %) at s = 2.5 %, median 0.74 %
  (s = 3.5 %, 10 draws; range 9e-5 - 12 %), median 4.3 % (s = 5 %, 10 draws; range 0.2 - 35 %), 60.8 % (s = 10 %, one
  draw). Rates are among pixels with peak SNR >= 5; each draw is one realisation of 61 frame factors and the rate is
  heavy-tailed and set by where the largest frame deviations fall, not by the realised sd (two draws at sd 2.44 % gave
  2.8 % and 1e-4). Draws from two verification rounds (`refute_perf_artifact/t2..t4*.json`,
  `refute_perf2_reproduction/jitter_results*.json` under the verify workdir); not re-run by the author. A pair fraction below ~10 % is therefore not interpretable without the scan's flux
  stability; S168 has no flux monitor (its jitter is estimated from the repeat halves, next bullet), and its measured fraction 0.49 is ~5x the 0.105 a
  single-population null multiplied by a pessimistic flux series reaches.
* **S168 itself is not flux-stable.** The per-frame multiplicative difference between S168's two repeat halves has
  sd 5.6 % (frame 0: +18 %), about 2.8 % at the level of the averaged frames (physics lens, round 2,
  `refute_perf2_physics/e3_halves.py`, not re-run by the author; the two repeats' per-frame factors also correlate
  at 0.85, and some excess repeat-to-repeat variance is spatially coherent, origin not separated). That is the
  regime where the phantom false-pair rate is ~0.03 % - 3 %, so 3e-5 is the flux-stable rate, not S168's; the
  measured S168 pair fraction (0.49) is far above it. An additive per-frame pedestal offset (2.8 counts sd) or
  3 mdeg angle jitter does NOT produce the effect (<= 1.4e-4); it is specific to a multiplicative per-frame factor.
* Real detector noise (injected S168 signal-free noise cube, rescaled per pixel; physics lens, `refute_perf_physics/
  realnoise_phantom.py`, not re-run by the author): false pairs 19 of 489,895 (3.9e-5) at full brightness, 568 of
  489,895 (0.12 %) at 0.1 x brightness for the first phantom seed but 213 (0.043 %) for another and 0.02-0.04 % in 12 further
  draws (round-2 reproduction: use ~0.02-0.12 %; about 70 % of the false pairs in the 'signal-free' window sit within 60 px of
  lit pixels, so on strictly far pixels the real-noise rate is ~1e-4, about 2x white noise), against ~5e-5 for white noise; the overall figure hides higher
  rates in the dimmest bins (full brightness: 2 of 510 = 3.9e-3 at peak SNR 3-5, 3 of 3,102 = 9.7e-4 at 5-8); recall at 200 / 150 mdeg 0.82-0.99 / 0.42-0.60.
  S168's per-frame factors estimated from the data (sd 0.07, up to +-13 % between frames, correlation 0.85 between
  the two repeats) injected into a single-population phantom gave 2.8 % false pairs (0.08 % at half amplitude);
  whether they are flux or sample structure is not established.
* Flux, S168 specifics: no independent flux monitor exists for S168 (motor table constant, off-sample pedestal
  tracks the grain signal, r = 0.92). Dividing out a pessimistic per-frame flux series left T unchanged
  (0.489 vs 0.490); a single-population null multiplied by it reached 0.105.
* Bug class to remember: an excluded frame leaves an EMPTY bin in the histogram of centres, and a valley
  finder puts the band edge on it; `band_edges(..., valid=...)` interpolates across excluded bins.

## Not delivered (do not promise)

Boundary WIDTH between the populations (a fit that fills its window is a rail; a share estimator that needs
detection cannot see the dim transition zone), sub-domains inside the high-angle band, an independent flux
correction. The threshold-free share estimator (`midas_dfxm.population_share`) addresses the first; its phantom
validation and its real-data result are in `PREREGISTER_S168_boundary_templates.md`.
