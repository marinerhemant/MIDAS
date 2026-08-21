# Phase 4 — Read the result, and report

> Part of the **FF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 8. STEP 7 — Read the result

### 8-0. The spots that were never found

`SpotMatrix.csv` is **28 columns** from `midas-process-grains` 0.10.0, and the
important part is not the extra columns — it is the extra **rows**. Col 12
`Matched` is 1 for an observed spot that matched a prediction and **0 for a
reflection the grain was predicted to produce and which was never found**. Until
now that population was recorded nowhere: `Grains.csv`, `SpotMatrix.csv` and
`FitBest.bin` all described matched spots only, so completeness could be read as a
number but never explained.

Un-found rows carry `-1` in the two integer columns (`SpotID`, observed `RingNr`)
because those are `%d` and cannot hold NaN, and **NaN** in every other observed
column. The prediction is in cols 14-18 (`theorRingNr`, `theorEta`, `YExp`,
`ZExp`, `OmegaExp`). Cols 0-11 on *matched* rows are byte-identical to the legacy
12-column layout, so a parser taking the first 12 tab fields is unaffected.

The first thing to do with it is ask **which rings are losing spots**:

```python
import numpy as np
d = np.genfromtxt("SpotMatrix.csv", skip_header=1)
u = d[:, 12] <= 0.5                       # predicted but not found
for r in sorted(set(d[:, 14][np.isfinite(d[:, 14])])):
    tot = (d[:, 14] == r).sum(); un = (d[u, 14] == r).sum()
    print(f"ring {int(r)}: {un}/{tot} un-found ({100*un/tot:.1f}%)")
```

On the reference Ni layer that reads 2.7 / 6.9 / 9.1 / 9.7 / **21.6 %** for rings
1-5 — the outer ring is where the completeness deficit actually lives, which no
existing output would have told you. Cols 19-21 and 22-27 carry the per-spot
residual before and after the fit (590.27 -> 300.39 µm median, 80.4 % of
individual spots improving). Post-fit columns are NaN where the post-fit matcher
did not keep that spot: 0.03 % on the reference layer, and notably **not** the
worst spots, which is unexplained.

### 8a. Check the refiner version before reading the residual columns

**`midas-fit-grain` < 0.5.7 writes `DiffPos`, `DiffOme`, `DiffAngle` cyclically
mislabeled.** `driver.py` assigned `calc_angle_errors`'s `(mean_angle, mean_pos, mean_ome)`
straight into the `(pos, ome, angle)` slots, so every FF/PF run through the python/torch
refiner has the three columns rotated (commit `44394e61`; the classic C refiner path was
unaffected).

The tell is obvious once you look: an ω residual of **223°** is impossible on a 0.25° step.
Post-fix the same grain reads `DiffPos 202 µm, DiffOme 0.054°, DiffAngle 0.090°` — all
physical.

0.5.7 fixes the *labels*. **It is not the floor** — the floor is **`>= 0.7.0`** (§0), and
the three reasons stack:

| version | what it fixes |
|---|---|
| 0.5.7 | the cyclic column mislabel described above |
| 0.6.0 | the `pos_scale` fp32 scaling bug — below this the refiner **returns its seed positions unrefined** and reports success (Lab Notebook §3c) |
| 0.7.0 | the `c_recipe` refine mode and its NLopt Nelder-Mead port (`06dd3241`) — the mode that reproduces the C refiner (Lab Notebook §7n) |

Passing the 0.5.7 check and stopping there is the trap: the labels are right and the
positions are still the indexer's seeds.

If you are stuck on 0.5.6, the mapping is: printed `DiffPos` = true DiffAngle, printed
`DiffOme` = true DiffPos, printed `DiffAngle` = true DiffOme.

### 8b. What to check in `Grains.csv`

Before interpreting it:

1. **Grain count vs expectation** — but rule out the plumbing before you blame the physics.
   A calibration cube should give a handful of grains, not thousands. Too many grains has
   **three** causes, in this order:
   1. **Your grain-selection keys were discarded.** `FitSetup` writes `paramstest.txt` for
      the indexer and refiner, which have no use for `Completeness`/`MinNrSpots`, so those
      keys are simply absent from it and every downstream consumer falls back to its own
      default. Measured on a Ni layer: the same refiner output gave **23710 grains via
      `paramstest.txt` and 6132 via the archive that carries the keys** — 3.9×, no error
      anywhere. Fixed by `360cc09e` + `midas-process-grains >= 0.7.0` (§0). **Check this
      first** — it is free, and it looks exactly like a bad peak search.
   2. Genuinely permissive `Completeness` / `MinNrSpots` for the sample.
   3. Only then: the peak search is finding noise (§6b).
2. **Completeness distribution**, not just the mean — a bimodal distribution means two
   populations, usually real grains plus junk. **`midas-process-grains >= 0.7.0` will read
   the cut off that distribution for you**: the antimode of the log₁₀ histogram of the
   quality metric. It is deliberately data-driven, because a fixed threshold does not
   transfer — the EBSD-optimal `DiffPos` cut on one `shade_LSHR` layer was 195.4 µm for the
   C chain and 222.8 µm for the python chain on the *same raw data* (`296368d2`). The gate
   **refuses rather than guesses** when the distribution is not bimodal; a refusal is
   information, not a failure.
3. **Position envelope.** If grain positions pile up against ±`Rsample` or ±`Hbeam`/2, the
   envelope is binding and the positions are not physical. The fix is to make the envelope
   MORE generous, never less — see the hard rule in §6.
4. **Strain sanity.** Whole-grain strains far above ~10⁻³ on an annealed calibration sample
   mean the geometry, not the sample.
5. **What fraction of spots got indexed?** `wc -l InputAll.csv` versus the spots actually
   assigned. A handful of grains explaining a few hundred of several thousand spots is an
   *under-indexed* run, not a sparse sample — confidence 1.0 on the few grains found says
   nothing about the ones missed.
6. **Re-run and compare grain-by-grain.** Grains that appear in one run and not the next
   are indexing noise. On `Au3_cubes_ff_000008` two runs shared only one of their two
   grains; that instability is the signal that `Completeness`, `MinNrSpots`,
   `OverAllRingToIndex` still need work.
7. **`indexing: 0 / N seeds with non-zero data`** in the log deserves an explanation before
   any grain list is trusted.
8. **Bin `DiffPos` against grain position** — specifically the radial offset
   `r = sqrt(X² + Y²)` from the rotation axis. If the well-fitting grains all sit within
   roughly the beam half-width and there are none beyond it, the reconstruction is
   **illumination-limited**: only the near-axis core is determined and the rest of the
   population is manufactured. This is the single cheapest check for the failure in
   DIAGNOSIS `split.illumination_radial`, and it is free.

> **`Confidence` is not a grain-quality metric — do not filter on it.** It is dominated by
> the chance-match floor whenever the spot list is dense. Measured on a 20-ID alumina layer
> (1652 grains): median `Confidence` was **flat at ~0.72 from r = 0 to 600 µm** while
> `DiffPos` climbed 544 → 783 µm over the same range, and one grain carried `Confidence`
> **1.000** with `DiffPos` **688 µm**. Rank and cut on **`DiffPos`**; treat a high
> `Confidence` as no evidence at all. (Item 2 above still holds for the *shape* of the
> completeness distribution — bimodality is informative — but the absolute value is not.)

---


## 11. Validation status — what is measured, what is convention

Short form. The evidence, the failed hypotheses and the retracted claims live in the
companion **`LAB_NOTEBOOK.md`**; this section is only what you need in order to
know how far to trust a number.

**Measured on this beamtime** — ω sign (`aero`, all 7297 par rows), the throwaway first
frame (~1.5 % baseline offset, three files), `SkipFrame` as a consumer-side skip,
`DetZ` − `Lsd` = +181 mm by ring ratios, energy 95.0 keV (three instrument records +
beamline), CeO₂ 0/180 repeatability, the `RingThresh` table in §6b.

**Convention, NOT measured** — `ImTransOpt 0` (chosen so the recon matches the frame the
calibration was fitted on; a self-consistent calibration + recon pair can still be
globally *mirrored*, and nothing here pins the absolute handedness). `OmegaStart` as "ω of
the first USED frame".

**Could not verify** — whether `DetZ`'s +181 mm offset is stable across the beamtime
(measured at one distance); whether the 95-vs-96 keV strain gap is a genuine energy
discriminator or partly distortion re-fitting (the distortion-frozen control was not run).

**How far to trust the output** — orientation and lattice parameter are solid; grain
**position is good to ~100 µm, no better** (Lab Notebook §2d). Everything else is
conditional on the install passing §0: `GrainRadius` needs `midas-process-grains >= 0.6.1`
and the **grain-selection keys** need `>= 0.7.0`; bit-reproducibility needs
`midas-peakfit >= 0.4.6` and `midas-transforms >= 0.8.2`; the refiner refines position at
all only from `midas-fit-grain >= 0.6.0` and reproduces the C recipe only from `>= 0.7.0`.
**Run the §0 check and quote its output** — do not assert these from this list.

**Do not judge a reconstruction by the fraction of the spot list it indexes.** On this
dataset 2 grains index 8.9 % of the rows and the recon is nevertheless *complete* — ~98 %
of that list is noise, zero-intensity padding, and over-segmented haloes of the two grains
themselves. A low indexed fraction is a statement about `RingThresh`, not about missing
grains. Classify the spots by own-frame SNR against the raw frames before concluding
anything from it; the method and the numbers are in Lab Notebook §4d.

**Bottom line.** The geometry is trustworthy in *magnitude* — `Lsd` and `BC` repeat to
0.01 % / 0.01 px across an independent 180° repeat, and the rings overlay. Its
**handedness** rests on convention, not measurement. The ω sign and the frame-0 skip
are the two settings that will silently ruin a reconstruction, and both are now pinned
by tests or by measurement rather than by memory.

### `indexing(FF): 0 / N seeds with non-zero data` — no longer cosmetic. Read it.

**This section previously said the message was cosmetic and could be ignored. That is
wrong as of `8a594ea5`, and ignoring it now hides a real failure.**

The history: the stage counted non-zero rows in `Output/IndexBest.bin` only, while both
modern backends — python and c-omp — write the consolidated `IndexBest_all.bin` family
instead. Measured on a Ni layer: the python backend wrote **only** the consolidated
family, the classical `IndexerOMP` **only** the legacy pair. So the counter found no file
and printed `0 / N` on every c-omp and python FF run, then advertised seed paths that had
never been written.

The stage now counts from **either** family and distinguishes the two cases:

| what you see | meaning |
|---|---|
| **hard error**, "no recognisable seed file" | the indexer exited 0 having written nothing. A real fault — do not proceed |
| **warning**, an honest `0 / N` | indexing ran and genuinely seeded nothing. A real, if disappointing, result — investigate `RingThresh`, `Completeness`, the geometry |
| a non-zero count | normal |

`IndexResult.n_seeds_attempted` / `n_seeds_indexed` are also populated now; FF only ever
wrote them into metrics, so **both fields read 0 on every FF run before this fix** — do
not quote those numbers from an older run. Refinement now warns when its own
`Results/OrientPosFit.bin` is absent or empty, and `process_grains` checks its **size**
rather than its existence, so a 0-byte `OrientPosFit.bin` no longer surfaces as
`cannot mmap an empty file` two stages downstream of the actual fault.

Still judge the stage by `Results/OrientPosFit.bin` and the grain count — but the seed
line is now evidence, not noise.

---


## 14. Report — and what "done" means

### 14a. What to hand back

A grain list is not the deliverable; a grain list **with its provenance and its caveats**
is. Write these into the report, not just into the chat:

- The `§0` install-gate output, verbatim. Every claim below is conditional on it.
- `SURVEY.md` (§0b) — the measured inventory, including which file is which and where each
  number came from.
- The calibration result: strain median **and** 5 %-trimmed, the 0/180 spread if you have
  one, and **the ring overlay image** (§5d). The overlay is evidence, not decoration.
- The parameter file actually used, and the `RingThresh` measurement that set it (§6b).
- `Grains.csv` with the §8b checks answered, each with the number that answers it.
- Every assumption you made where this document said *stop and ask* and you proceeded
  anyway — name it explicitly.

**Every quantitative claim names the file and the command that produced it.** A number you
cannot re-derive does not go in the report.

### 14b. Say which bucket each number falls in

§11 splits this pipeline's output three ways — **measured on this beamtime**,
**convention, not measured**, and **could not verify**. Put every number you report into
one of them. The geometry's *magnitude* is measured; its *handedness* is convention. Do
not let a §11 item become a fact by being quoted often enough.

### 14c. Done means

- [ ] §0 install gate run, output pasted, **no package below floor**
- [ ] `SURVEY.md` written, every number read from a file rather than a name
- [ ] ω sign established from par field 9 — or **stopped and asked** if it was not `aero`
- [ ] `SkipFrame` set, and the peakfit banner's `nFrames` = logged frames − `SkipFrame`
- [ ] dark verified **non-zero in the zarr**, not merely configured (§3d)
- [ ] energy from three instrument records, never the filename
- [ ] calibrant strain ≤ 100 µε, reported as median + trimmed
- [ ] **ring overlay produced and looked at** (§5d) — the one check that catches a
      well-converged fit on the wrong rings
- [ ] `RingThresh` measured with `midas-ring-thresh`, not copied
- [ ] sample lattice constant + space group replace the calibrant's (§6)
- [ ] `Rsample`/`Hbeam` left generous; grain positions checked for pile-up at the bounds
- [ ] `Grains.csv` read against all seven §8b checks
- [ ] reconstruction re-run once and compared grain-by-grain (§8b item 6)
- [ ] every number bucketed per §14b, with its provenance

**If a box cannot be ticked, say so in the report rather than leaving it blank.** An
unticked box is a known limit; a silently skipped one becomes a false claim.
