# Phase 6 — Reconstruction space: sinograms, shapes, and what not to quote

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
> **Read this before quoting a grain shape, a grain-ID map, or anything from `Recons/`.**

Phases 3–5 end at the **point-by-point** result: one orientation and lattice per voxel,
segmented into grains. That map is the robust product. This phase covers the *other* half
of the pipeline — the tomo/vmap tail (`voxel_cleanup → sinogen → reconstruct → fuse →
potts → em_refine`), which turns each grain's reflections into a **sinogram** and
reconstructs a **shape**.

It is documented here for two reasons, and neither is "so you can use the shapes":

1. Two shipped diagnostics live in this space (§6.5, §6.6) and they **improve the
   point-by-point result** — the concentration filter took a grain's fitted position from
   5.59 µm to 1.11 µm. They are worth running even when shapes are not the goal.
2. **Grain shapes from this reconstruction are not currently quotable** (§6.7). That is a
   measured, adversarially-tested conclusion, not a caveat, and it is the single most
   likely thing for a session to get wrong here — the shapes render, they look like
   grains, and some of them are even right.

Everything below was measured on the **20-ID Varex reference campaign** (notebook §7:
51 × 51 voxels at 1 µm, FCC, `nf709` set A) unless another dataset is named.

## 6.0 The gate — do you need this phase at all?

| goal | run this phase? |
|---|---|
| orientation / KAM / GROD / pole figure | **no** — phases 3–5 are complete without it |
| per-voxel strain (pf-odf) | **no** — phase 4 does not consume sinograms |
| better fitted **grain positions**, contaminated-row diagnosis | **yes** — §6.5, §6.6 |
| grain **shapes**, grain-ID maps from tomography | **read §6.7 first, then decide** |
| the **sample boundary** | not here — [`phase-1b-sample-boundary.md`](phase-1b-sample-boundary.md) |

## 6.1 What is in reconstruction space

`find_grains` already builds the sinograms; the tail stages consume them.

| file | shape / meaning |
|---|---|
| `Output/sinos_<variant>_<nG>_<nH>_<nS>.bin` | float64 `(n_grains, max_n_hkls, n_scans)` — one **row per reflection**, one column per scan position |
| `Output/omegas_<...>.bin` | mean ω per row, degrees; `-10000` = unpopulated row |
| `Output/nrHKLs_<...>.bin` | reflections per grain |
| `Output/sinoOccupancy_<nG>.bin` | float64 per grain — §6.6. **Always written** |
| `Output/sinoConc_<...>.bin` | per-row concentration, only when the filter is on — §6.5 |
| `Output/spotPos_<nG>_<nH>_<nS>.bin` | per-cell `(yCen, zCen)` lab positions, µm |
| `Recons/recon_grNr_*.tif`, `Sinos/`, `Thetas/` | per-grain reconstructions and their inputs |

`spot_meta` is `(nG, nH, nS, 4)` and its columns are **`eta, 2theta, yCen, zCen`**
(`midas_pipeline/find_grains/_patches.py:76`) — cols 2 and 3 are lab positions in µm.
Use these. See §6.9 for the legacy-C file of the same name that is 97.7 % unwritten.

## 6.2 The projector convention, and the crop that was off by a voxel

The reconstruction's forward model is

```
s(ω) = x·sin ω + y·cos ω
```

pinned by `midas_pipeline/recon/fbp.py:176` and its regression test
(`tests/unit/test_recon_fbp_registration.py`). **This is the reconstruction code's
convention and it is settled.** What is *not* settled is which sense the spot-count
sinogram is indexed in — that open item lives in
[`phase-1b-sample-boundary.md`](phase-1b-sample-boundary.md) §1b.6, and it does not
affect anything in this file.

**Version floor: `midas_pipeline ≥ 0.11.0`.** Below it, the crop that extracts the
`n_scans × n_scans` field from the upscaled reconstruction was computed as
`recon_dim // 2 - n_scans // 2` instead of `(recon_dim - n_scans) // 2`. Those differ
whenever `n_scans` is odd — which is every realistic pf scan — and **every feature came
back one voxel low in both axes**, verified with point phantoms across 17 odd `n_scans`.
A constant offset, so nothing looked broken; it silently mis-registered every shape
reconstruction against the voxel map it was being compared to.

Two things to carry forward:

- **Do not "simplify" the crop** to `round((recon_dim-1)/2 - (n_scans-1)/2)`. That is an
  exact half-integer for odd `n_scans`, so banker's rounding decides it, and it
  reproduces the bug at `n_scans` = 33, 37, 45, 49, 53, 65, 97, 101, 129 while passing at
  31, 35, 47, 51. The comment at `fbp.py:168` says so; the test sweeps both families.
- **For EVEN `n_scans` an irreducible −0.5 voxel offset remains** — the true crop is a
  half-integer and no integer crop can remove it. Stated, not hidden. Prefer odd
  `n_scans` when a shape comparison is planned.

In practice also floor at **0.12.0**, which added "a run that finds nothing must not exit
0" — a pf-relevant silent-failure fix, and the version the beamline sheet quotes.

## 6.3 `do_tomo` — leave it OFF for the point-by-point map

Seeding the re-index from a tomographic starting map makes the point map **worse**:

| | direct (`do_tomo=False`) | tomo-seeded (`do_tomo=True`) |
|---|---|---|
| voxels refined | **2601** (all) | 2433 |
| voxels below completeness 0.5 | **11** | 367 |

Measured on the reference campaign, same data, same geometry, everything else identical.
Whatever the tomography is wanted for, it is not this. The tail is skipped by default in a
grain-map / pf-odf run (phase 3.1) and that default is correct.

The stage is also a clean no-op without sinograms: `reconstruct` soft-skips when
`do_tomo=False`, and again when no `sinos_*` are on disk (`stages/reconstruct.py:88`).

## 6.4 The sinogram variants

Written by `generate_sinograms_tolerance` (`midas_pipeline/find_grains/_sinogen.py:344`):

| variant | transform | use |
|---|---|---|
| `raw` | none | the default, and the input to §6.5 |
| `norm` | each row ÷ its own max | destroys the volume information — **not** a physical normalisation |
| `abs` | absorption transform | **came back degenerate on the reference run** (residual 1.000/1.000/1.000/0.000). Check it is populated before trusting it |
| `normabs` | both | as above |
| `softsum` | Σ w·I per cell | only when soft attribution is enabled |
| `clean` | `raw` with contaminated rows zeroed | §6.5, only when the filter is on |

> **A row-wise *physical* normalisation was tried and refused to help.** Each row is a
> different reflection with its own |F|² and 2θ, so FBP is being fed projections on
> inconsistent scales — a textbook streak cause. Dividing by |F|²·Lp was tested twice:
> powder-Lp (spread 21.6× across rings) gave mean +(−0.004) in dice-above-chance against a
> ≥ +0.10 bar and cost 0.082 in half-split; the η-resolved version (spread 36.2×) gave
> +0.010 and +0.044. **Both REFUTED.** Do not re-derive this.

## 6.5 The concentration filter — contaminated "vertical stripe" rows

**The mechanism.** A clean reflection puts its intensity in the few scan positions where
the beam actually crosses the grain: a compact blob riding a sinusoid. A row that also
collected a *neighbour's* spot, or a spot from a grain outside the scanned field, smears
across every scan position and reads as a vertical stripe. Those rows drag the fitted
grain position badly.

`sinogram_concentration` (`midas_pipeline/find_grains/_sinogen.py:91`) fits the offset-free
sinusoid `s(ω) = a·sin ω + b·cos ω` to each populated row's intensity-weighted centroid,
estimates the grain width `D` from the median second moment, and returns the fraction of
row intensity within `±max(D, min_band_um)/2` of the fitted track.

```python
from midas_pipeline.find_grains import sinogram_concentration, apply_concentration_filter
conc          = sinogram_concentration(raw_sino, omegas, nr_hkls, scan_positions=s_values)
clean, dropped = apply_concentration_filter(raw_sino, conc, 0.35)
```

or in a pipeline run, `--sino-conc-threshold 0.35` (`--sino-conc-min-band` floors the
acceptance band, default 4.0 µm). The filter **only ever removes rows it has positive
evidence against**: `NaN` concentration — unpopulated, or an unfittable grain — is kept
(`_sinogen.py:216`).

**What it bought, and what it did not.** On the reference campaign it flagged **16 of 958
rows (1.7 %)**:

| grain | position fit rms, before → after |
|---|---|
| 3 | 5.59 → **1.11 µm** |
| 8 | 5.27 → 2.14 µm |
| 9 | 3.41 → 2.06 µm |
| 2 | 5.42 → 3.29 µm |

and the MLEM residual moved **0.798 → 0.797** — i.e. **not at all**. That is the honest
shape of this result: it is a large win for **position** and does nothing for the shape
problem (§6.7). The residual is carried by the other 98 % of rows.

**Do not retune the 0.35.** Tested out-of-sample on five independent layers of a different
sample at 21× coarser sampling (16 positions at 21.3 µm), driven through the shipped
pipeline with nothing retuned: 4 of 5 layers landed inside the preregistered 1–6 % flagged
band. **The threshold transfers.**

**But the benefit is smaller off-sample, and the verdict there was INCONCLUSIVE.** Median
gain 16–34 % on every layer — never reaching the preregistered 40 % CONFIRM bar, never
dropping below the 15 % REFUTE bar. It is bimodal within a layer (46–91 % on some grains,
~0 % on others, and **−1 % / −2 % on two — it can very slightly harm**). The reference
campaign's flagged grains gained up to 80 %; do not promise that elsewhere.

> **Characterisation of the stripe rows, for anyone tempted to re-diagnose them.** They
> are real contamination, not a plotting artefact: they are *stronger* than clean rows
> (median total I 470 k vs 310 k; `corr(conc, log I) = −0.254`) and span 44 % more scan
> bins (36 vs 25). They are **not** from the other listed grains (contested-spot
> enrichment 0.8×) and **not** ring-clustered (per-ring rates 0.0–3.6 % against 1.7 %
> overall, but only 16 events — low power). **Untested: intruders from grains outside the
> scanned field.** That is the live hypothesis.

## 6.6 The occupancy flag — grains that fill the scanned field

`sinoOccupancy_<nG>.bin` gives, per grain, the **median fraction of the scan line its
reflections light up** (`sinogram_occupancy`, `_sinogen.py:237`; written unconditionally by
`write_occupancy`, `_sinogen.py:291`). Above ~**0.65** the grain is as large as or larger
than the scanned window, so its *shape* cannot be recovered from these sinograms — the
projections never see it end.

`reconstruct` warns and names the grains (`stages/reconstruct.py:58`); the cutoff is
`--out-of-field-occupancy`, default 0.65, `0` disables.

On the reference campaign **2 of 10** grains were flagged, at 0.84 and 0.78, with every
other grain at or below 0.51. Out-of-sample on the five coarse layers: **zero** flagged,
max 0.31–0.50 — the correct answer for a ±160 µm field of small grains, and predicted in
advance.

### ⚠ Flag, never filter

**Do not delete flagged grains from the map.** Excluding them takes agreement with the
point-by-point result from **47.8 % down to 11.0 %**, because the largest grain is most of
the material and its voxels then go to whichever small grain wins by default. The flag
tells you which *shapes* to distrust. It is not a filter, and the code deliberately leaves
the grain in the grain-ID competition (`stages/reconstruct.py:58` docstring).

## 6.7 Grain shapes — the verdict, and the ledger behind it

**Positions: believe them.** A grain's position fitted from its sinogram agrees with its
voxel map to about **1.3–2.1 µm** rms on clean single grains.

**Shapes: do not quote them.** Some grains come back as clean compact objects in the right
place and some do not; on the reference campaign the two grains whose reference is a single
compact region reconstruct well and the two largest do not. The reconstruction residual
sits at **0.82–0.84 and is invariant** across FBP / SIRT / MLEM, with and without a
support, across every sinogram variant, and for self-fitted vs borrowed reference masks.

**What has been tested against it.** Eleven candidate mechanisms were each preregistered
and refuted, then a **metric audit requalified four of them**:

| tested | outcome |
|---|---|
| scan-window truncation; contamination from outside the window; angular coverage; spot density; same-orientation blends; peak-finder merging; mosaic spread | refuted |
| absorption; primary extinction; detector spot overlap; intensity thresholding | **requalified** — contributing but insufficient (see below) |
| `RawSumIntensity` instead of `IntegratedIntensity` | refuted, decisively |
| |F|²·Lp normalisation, powder and η-resolved | refuted, twice (§6.4) |
| MIDAS edge-padding replication (`extra_pad`) | refuted — mean half-split change −0.025 against a ≥ +0.05 bar |

The requalification matters more than the count. All eleven were scored primarily with
**dice**, which thresholds at the true voxel count and is blind to anything below the blob
amplitude. Re-scored on out-of-mask energy, four mechanisms moved the artifact level by
47–396 % while dice moved 0.003–0.042. **"Eleven mechanisms refuted" overstates it.**

**What survives is sharper than the count.** The full simulated stack (geometry +
absorption + extinction + detector merging + thresholding) reaches an artifact level of
**0.239**. Real grains: 0.094 and 0.211 for the two that reconstruct **well**; 0.476 and
0.896 for the two that **fail**. So the modelled physics accounts for the grains that work
and is 2.0× and 3.8× short of the ones that do not. **The cause of the failure is
unknown.** If shapes are needed, absorption tomography of the same sample is the better
instrument.

### Two methodology rules this campaign paid for

1. **Do not use dice on this problem again.** Four separate requalifications trace to it.
2. **Half-split consistency is the metric that survived.** Reconstruct each grain from two
   disjoint halves of its rows and correlate the results. It needs no ground truth, no
   mask and no chance floor, and a disc cannot game it. Of five metrics tried it is the
   only one that survived contact with an adversarial case: it cleanly separated a
   self-consistent sinogram (0.82–0.92) from a noise-dominated one (≈ 0) where dice would
   have reported merely "somewhat worse".

## 6.8 Comparing two maps — report the majority-class null

**Any** comparison between a grain-ID map and another labelling — tomographic vs
point-by-point, filtered vs unfiltered, this layer vs a neighbour — must be reported
against the **constant-map null**: the score you get by calling every voxel the most common
grain.

On the reference campaign the tomographic map agreed with the point-by-point map on
**60.1 %** of voxels, which sounds reasonable until you notice that "call every voxel grain
0" scores **65.2 %** on the same voxels. Cohen's κ was 0.399. **Agreement below its own
constant-map null is not agreement**, and a raw percentage hides that completely.

This is a spine hard rule (README §"Hard rules"), not a local convention — it applies
wherever a map is scored, including the completeness and vacuum comparisons in
[`phase-1b-sample-boundary.md`](phase-1b-sample-boundary.md).

## 6.9 If you are on the legacy C path

The beamline install may be the v11 C `pf_MIDAS.py` rather than `midas_pipeline`
([`INSTRUMENT.md`](INSTRUMENT.md) §I3). One defect matters in this phase:

**`spotPositions_*.bin` is 97.7 % unwritten in the C path** — 604 of 26 119 spot-bearing
bins carry values, the rest sit at the `-1` initialiser. The write is gated on
`idMap_scanNr[gid] == scanNr` (`FF_HEDM/src/findSingleSolutionPFRefactored.c:2655`) and a
missing `Result_*.csv` is silently skipped, with the error logged only for `scanNr < 3`
(`findSingleSolutionPFRefactored.c:2683-2687`).

**Do not patch the C.** `midas_pipeline/find_grains/_patches.py:115-122` has no such guard
and takes positions straight from `spot_meta`, so the maintained path is already correct.

⚠ **The filename changed across the migration**: C writes `spotPositions_*.bin`, Python
writes `spotPos_*.bin`. Anything that reads by name will silently find *nothing* after
migration rather than erroring.

## 6.10 What to hand forward

```
occupancy flagged grains  = <ids + values>   (cutoff 0.65)
conc filter               = on/off, threshold, <n flagged>/<n rows> (<%>)
position fit rms          = <µm> per clean grain
shapes quoted?            = NO  (or: which, and against which null)
map comparisons           = <score> vs constant-map null <score>, κ = <...>
```

Then [`phase-5-read-report.md`](phase-5-read-report.md) §5.3 for the report itself.
