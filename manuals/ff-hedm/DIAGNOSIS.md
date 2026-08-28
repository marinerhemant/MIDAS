# FF-HEDM diagnosis reference

Symptom → discriminating test → cause → lever, for far-field HEDM. Read by
`midas_ff_report_beamreport.py` via beamreport; each entry attaches to a symptom the
generic diagnostics detect.

Source of the content: `manuals/Reconstruction_Reports.md` §4. Every entry carries a
test that can come back the other way — an entry that cannot exonerate the cause it
names does not belong here.

## Local symptoms

These are emitted by **the FF pipeline's own checks**, not by `beamreport`'s generic
diagnostics, which key off per-observation residuals against declared coordinates. A
single refined geometry parameter finishing on its bound, or an indexing stage that
reports zero seeds in 0.1 s, is real and useful, and nothing generic will ever detect
it — so it is declared here rather than renamed into the wrong shape.

Every row names where the check lives. A symptom nothing produces is dead text that
reads as coverage, which is exactly what the generic vocabulary existed to prevent.

| symptom | emitted by |
|---|---|
| `bound.parameter_railed` | `midas_joint_ff_calibrate.grain_refine` names any refined parameter that finished on a bound (`grain_refine.py:344`) and exits 1 |
| `count.zero_indexed` | `n_seeds_indexed` in `<result>/LayerNr_N/midas_state.h5` (`stages/indexing/metrics`), written by the pipeline's indexing stage, read against the indexer's wall time |
| `split.illumination_radial` | this entry's own two tests — per-grain `DiffPos` binned by `r = sqrt(X²+Y²)` from `Grains.csv`, and the per-grain lit-ω-arc duty-cycle enrichment (its own null) |
| `systematic.mirrored_beam_centre` | this entry's comparison of the `midas_calibrate_v2` refined beam centre against its mirror `N-1 − BC`, since strain does not diagnose it |
| `resid.population_mixture` | this entry's own test — per-grain `DiffPos` from `Grains.csv` binned by `Confidence`, with the per-spot internal angle from `residuals/spot_table` col 9 as the censoring check. Both read from files the default run already writes |

Strain railing at the Kenesei `MargStrain` bound is **not** listed here: it is the
generic `bound.pileup` (objects piling against a declared parameter bound), and both
the total and partial cases use it, discriminated by their own tests.

---

## Detector centre offset

symptom: trend.amplitude_constant
coord: eta_deg

**Test.** The amplitude of the azimuthal sinusoid is compared across bins of ring
radius. A rigid detector-centre shift displaces every ring by the same *absolute*
distance, so the amplitude in µm stays constant with radius. If instead it grows with
radius, the centre is not the cause and this entry does not apply — see the
sample-displacement entry.

Confirm it is global rather than per-grain: compare the mean of the per-grain (dy, dz)
offsets against their scatter. Mean much larger than scatter means a common offset.
Mean near zero with large scatter is genuine per-grain position spread, which is not a
bug.

**Cause.** The beam centre used in reconstruction differs from the true one. Amplitude
divided by pixel size gives the offset in pixels; Δtan shows the same amplitude 90° out
of phase.

**Lever.** Recalibrate BC, Lsd, tilts and distortion against a powder calibrant taken
with the same geometry, then re-index. Transplant `{Lsd, BC, tx, ty, tz, p0..p10,
RhoD}` as one block — a new distortion set with an old RhoD silently corrupts the
distortion.

## Sample displacement or distance error

symptom: trend.amplitude_growing
coord: eta_deg

**Test.** Same comparison as above, read the other way: the sinusoid amplitude grows
with ring radius rather than staying constant in µm. If it is flat in absolute units,
this entry does not apply and the detector-centre entry does.

**Cause.** A sample displacement or an Lsd error, both of which scale the ring pattern
rather than translating it.

**Lever.** Refine Lsd against a calibrant. Powder cannot constrain `tx` (rotation about
the beam) — keep it fixed there and refine it from grains in a second pass with
`midas_joint_ff_calibrate.grain_refine`, whose reported `tx` is a **residual** that must be
composed onto the applied one and iterated (`midas_joint_ff_calibrate/grain_refine.py:426`).
Put the resulting `tx` in `Parameters.txt` and re-run from `transforms` — that is the stage
that applies it (`midas_transforms/fit_setup/core.py:376`). Do not look for `tx` in the
indexer or refiner: neither backend carries one, and neither should.

## Position refinement runaway

symptom: param.residual_correlated
coord: Z

**Test.** Correlate each grain's fitted Z against its own vertical residual. Near-zero
correlation, with residual flat against Z, means the Z values are supported by the
spots and the spread is physical. A strong negative correlation — core grains at
roughly zero residual, tail grains carrying residuals pointing back toward the beam
centre — means the spots contradict the assigned Z.

Rule out geometry before concluding: compare the ring composition of the core against
the tail. If they are identical, a ring-dependent tilt or distortion error is not the
cause.

**Cause.** The position fit is placing grains outside the illuminated slab, where they
could not have diffracted. The tail is a fitting artifact, not structure.

**Lever.** Set `Hbeam` / `BeamThickness` to the true per-layer beam rather than the
full sample height. A ten-layer 100 µm scan often carries `Hbeam 1000`, which lets Z
roam ±500 µm. Grains outside the beam cannot diffract, so this is a physical prior, not
a fudge. Then re-check that the dz residual stays flat against Z.

## Bound-limited positions

symptom: bound.pileup
coord: Z

**Test.** Divergence-to-bound leaves a pile-up *at* the bound. If the outer shell holds
close to zero percent of grains, the bound is not being reached and this entry does not
apply — look instead at whether the residual supports the fitted positions.

**Cause.** The optimiser is running into `Hbeam/2` or `Rsample` rather than converging.

**Lever.** Widen the bound only if the physics justifies it; usually the correct move is
the opposite, since a generous bound is what let positions roam in the first place.
Never set these to the actual sample dimensions.

## Reference-lattice or wavelength scale error

symptom: systematic.common_offset

**Test.** Look at how the per-ring radial bias behaves in ppm. Constant ppm across
rings points at Lsd, since `δR/R = δLsd/Lsd` is ring-independent. Growing ppm with 2θ
points at the reference lattice or wavelength instead. If the ppm range across rings is
under roughly 200, neither is worth chasing.

**Cause.** A shared offset in the strain-free reference, the wavelength, or the
detector distance. For a cubic free-standing polycrystal the equilibrium condition
reduces exactly to a zero volume-averaged hydrostatic strain, so any nonzero mean *is*
the d0 error.

**Lever.** Recover d0 with `midas_stress.recover_d0_cubic_free_standing`. The
correction is purely isotropic, so deviatoric strain is unchanged: it fixes bias, never
scatter. Report the stress impact as `eps_iso × 3K` — it is usually the headline number
and often hundreds of MPa.

> **Non-cubic: one scale factor is not enough.** `recover_d0` and
> `recover_d0_cubic_free_standing` both scale `a`, `b` and `c` by a single factor,
> which is exact only for cubic. A hexagonal/trigonal cell has **two** independent
> reference lengths, and a real error is often *anti*-correlated between them —
> measured on NMC811 (SG 166): `a` low by 6316 µε while `c` was high by 7476 µε.
> No single scale can absorb that. Use
> **`midas_stress.recover_d0_anisotropic(..., crystal_system="hexagonal")`**, which
> solves ⟨σ⟩ = σ_applied for one unknown per symmetry-allowed length.
>
> Read its `condition_number`, and note the counter-intuitive part: a **weak**
> texture is the ill-conditioned case. Uniform orientation averaging projects onto
> the isotropic subspace, so the `a` and `c` responses collapse onto `C{I}` and
> `cond` *grows with N* (2.8 single-orientation or 10° fibre; 23 at N=100 uniform;
> 142 at N=1000). Sharp texture separates them cleanly.
>
> Reassuringly, the answer barely depends on the elastic constants: scaling the
> whole stiffness tensor ±30 % changes it **not at all** (the factor cancels), and
> swinging C33 140→260 / C13 20→90 moved `a` by 198 µε and `c` by 718 µε. Badly
> known single-crystal constants are not load-bearing here.

## A *fraction* of grains rail at ±10000 µε — the reference cell is wrong

symptom: bound.pileup
coord: strain

**Test.** Count components within ~1 µε of `1.000e+04`. **Total** railing (100 %,
with `RMSErrorStrain ≈ 1e36`) is the missing-`IDsHash.csv` defect below — a
different entry. A *partial* rail — measured 11.9 % of voxels, worst on one
component — means the reference cell is wrong but not absent. Confirm by fitting
the cell from the observed rings and comparing with `LatticeParameter`:

```python
from midas_hkls import refine_lattice_from_d_spacings
fit = refine_lattice_from_d_spacings(hkls, d_obs, "hexagonal")   # no starting cell
```

A mismatch beyond ~2000 µε is the cause. **Do not widen `MargStrain`** — the box
is not the problem, and a wider one hides a bad reference instead of exposing it.

**Cause.** `StrainTensorKenesei` (`FitUnified.c:1061`) gauges
`(dsObs − ds0)/ds0` against the `ds0` implied by `LatticeParameter`, inside
`MargStrain` (default ±0.01 = ±10000 µε; a compiled-in constant before
2026-08-21). A reference wrong by ~0.7 % — e.g. pristine values used for a
charged battery cathode, c/a 4.95 vs the actual 5.07 — spends most of the box
before any real strain is measured.

It is **not** only a strain problem: on the reference dataset pinning the cell
took solved voxels 84 → 123 and completeness median 0.618 → 0.833. A static
"the ring shift fits inside `MarginRadial`" argument said it would cost no spots,
and was wrong.

**Lever.** Pin `LatticeParameter` from the observed ring positions — `Ttheta` and
`RingNumber` in `InputAllExtraInfoFittingAll*.csv`, which involve no indexing and
no per-grain refinement — then re-run. Two traps in that fit:

- **Drop or down-weight the lowest-angle ring.** `dd/d = cot(θ)·dθ`; at 2θ = 2.85°
  a 0.006° systematic in 2θ is **2105 µε in d** versus 596 µε at 10°. Its residual
  was −1696 µε where four other rings sat inside ±340 µε; dropping it took the fit
  RMS 776 → 171 µε.
- **Never weight by the ring centroid's statistical error.** With ~160 k spots the
  SEM is ~6 µε, so 1/σ² weighting gives the least reliable ring the largest weight.
  Use uniform or `tan²θ`.

**Never recover the cell by averaging refined per-grain cells** — the refiner
starts from `LatticeParameter` and only partly leaves it, so that average returns
roughly what you fed in (measured: a further −3740 µε in `a`, +6361 µε in `c`,
not converging). Gate on the powder fit and `recover_d0_anisotropic` agreeing;
on the reference dataset they closed to −994 / +587 µε.

## Strain pegged at its bound for every grain

symptom: bound.pileup
coord: strain (all)

**Test.** Read `RMSErrorStrain` (Grains.csv col 42) and the largest `|eKen|`
(cols 33–41). The signature is unmistakable and admits no partial version:
**every** grain sits at exactly `1.000e+04` µε — the ±0.01 Kenesei bound — and
`RMSErrorStrain` lands around `1e36`. Then check the run directory for
`IDsHash.csv`:

```bash
ls -la <layer_dir>/IDsHash.csv
```

Absent is the whole diagnosis. Present, and this entry does not apply — look at
*Reference-lattice or wavelength scale error* instead, which is the physical
version of a strain problem.

**Cause.** `IDsHash.csv` carries the reference d-spacing d₀ per ring, and it is
the only source process-grains has for the Kenesei gauge
`ε = (d_obs − d₀)/d₀`. A missing file used to be answered with a fabricated
`d₀ = 0`, which does not degrade the strain — it destroys it, by dividing by
zero. Measured on both datasetA and shade_LSHR: 100 % of grains railed, in runs
whose grain count, positions, orientations, sizes and completeness were **all
correct and inside their acceptance bands**. Nothing warned.

The file was written by `fit_setup(write=True)` but not by
`Pipeline.dump()` — and `dump()` is the path `midas-pipeline run --scan-mode ff`
takes. The two writers had silently diverged.

**Lever.** Re-run the transforms stage (`--resume from --from transforms`) with
`midas-transforms` ≥ the version whose `Pipeline.dump()` calls
`write_ring_tables`, then re-run process-grains. Verify before trusting
anything: `IDsHash.csv` present, its fourth column a real d-spacing per ring
(for FCC Ni: 2.0785, 1.8000, 1.2728, 1.0854, 1.0392 Å), and `RMSErrorStrain`
back in the hundreds of µε with **0 %** of grains at the bound.

Current versions raise `FileNotFoundError` naming the file rather than
fabricating d₀, so this can only be met on output produced by an older tree.

## Two populations in completeness or spot count

symptom: split.bimodal

**Test.** Check whether the split is spatial. Map the two populations onto grain
positions: if the split follows position, it is the illumination footprint — which part
of the sample the beam actually covered — and not a reconstruction defect.

If the split is *not* spatial, bin grains by radial distance from the rotation axis and
histogram within each bin. Modes that move with radius indicate a smooth geometric
effect. Mode positions fixed across radius, with only the population fraction shifting,
indicate a discrete algorithmic branch.

**Cause.** Spatial split: illumination coverage. Non-spatial with fixed modes: a solver
branch, most often the Friedel-pair position path succeeding versus falling back.

**Lever.** For the footprint case, see the next section before writing "nothing to fix" —
a beam narrower than the sample does not merely limit coverage, it manufactures grains.
For the branch case, re-run a subset with `UseFriedelPairs 0`; if the split
collapses, the Friedel path is the branch. Expect the bad branch to also carry inflated
|Z|, internal angle and strain error, and verify they co-move before blaming one cause.

## Most grains do not fit, and the good ones are all near the rotation axis

symptom: split.illumination_radial
coord: r = sqrt(X² + Y²) from `Grains.csv`

The horizontal analogue of the `Hbeam` problem, and a different animal. `Hbeam` bounds Z,
which does not change with ω. **A beam narrower than the sample is ω-dependent**: the beam
is fixed in the lab while the sample rotates, so a grain at radial offset *r* is only lit
while `|X·sin ω + Y·cos ω| < hw`. Its *achievable* completeness is geometric:

```
f(r) = 1                        r <= hw
f(r) = (2/pi)*arcsin(hw/r)      r >  hw
```

With a 100 µm beam (hw = 50 µm) on a 1 mm rod: r=100 → 33 %, r=250 → 13 %, r=500 → 6 %.

**This collides with `MinMatchesToAcceptFrac` (default 0.5).** A grain that can physically
produce only 13 % of its reflections cannot reach the bar on real spots — so the only
candidates that clear it at large *r* are ones padded out with coincidental matches. The
result is not "fewer grains", it is **fabricated grains**, and the population median
`DiffPos` then describes those artifacts rather than the instrument.

**Test.** Two tests, cheap then decisive: bin per-grain `DiffPos` by radial
offset *r*, then — for the grains that matter — compare each grain's assigned
spots against the ω arcs where it was actually lit. The second carries its own
null, because the arcs' duty cycle *is* the chance rate. Details below.

**Test 1 — free, `Grains.csv` only.** Bin per-grain `DiffPos` by *r*. Measured on a 20-ID
alumina rod (1 mm dia, 100 µm beam): good grains (`DiffPos` < 150 µm) sat at r p50 = 44 µm,
p90 = 60 µm against a 50 µm beam half-width, and there were **zero** good grains beyond
r = 100 µm, while `DiffPos` climbed 544 → 783 µm across the bins.

**Test 2 — decisive, per grain, with a built-in null.** For each grain compute the ω arcs
where it was actually lit, then ask what fraction of its assigned spots fall inside them.
The arcs' **duty cycle is exactly the chance rate**, so the comparison needs no separate
null:

* `obs >> duty` → the spots track the illumination; they are that grain's real
  reflections, and the ones it is missing are missing because it left the beam.
* `obs ≈ duty` → the spots are spread independently of when the grain was lit; they are
  coincidences and the "grain" is an artifact.

On that alumina layer the informative grains (r > 75 µm) gave enrichment `obs/duty` p50 =
**1.23** with 40 % consistent with pure chance — i.e. a grain lit for 8.8 % of the rotation
had 85 % of its assigned spots recorded while it was **outside the beam**.

**Cause.** Not the instrument, not the geometry, not the peak search — the matched spots
are real, strong diffraction (median SNR 51 on the raw frames), they simply belong to
other grains.

**Lever.** There is no parameter that fixes this; the data under-determines those grains.
Report the near-axis subset (r ≲ hw) as the reconstruction and say so explicitly. Widening
the matching margins is *not* the cure and tightening them is worse — see rule 9 and the
trap table. To reconstruct the full cross-section you need a **translation scan**
(scanning 3DXRD / pf-HEDM), not a re-run of the same data.

**Check whether `Confidence` is SATURATED before you either trust it or dismiss it.**
This entry used to say flatly "do not use `Confidence` to find the good grains — it
measures the chance floor". That is true of some runs and false of others, on the *same
sample at the same station*, so the rule is the check, not the verdict:

| 20-ID alumina run | grains | `Confidence` median | frac ≥ 0.999 | `DiffPos` all | `DiffPos` of the ≥ 0.999 set |
|---|---|---|---|---|---|
| saturated | 7132 | 0.992 | **46.9 %** | 602.0 µm | **619.7 µm** — *worse* than the population |
| not saturated | 1729 | 0.836 | **0.6 %** | 655.7 µm | **35.2 µm** |

**The one-line test is the fraction of grains at `Confidence` ≥ 0.999.** Percent-level or
below, the column is live and is the sharpest discriminator in `Grains.csv`; tens of
percent, it is pinned and reading it will actively mislead — there, use `DiffPos`. Do not
attribute the difference to any single setting: those two runs differ in `tx`,
`MinNrSpots`, `MinNrPx` and `--pg-mode` at once. **The boring explanation was checked and
fails** — neither run contains a single grain built from ≤ 2 spots (minimum 77 and 42
spots respectively), so this is not the under-determined-grain artefact of the
`MinNrSpots` rule.

Re-derive both rows before quoting them:
`$ANALYSIS/nfdev_jul26_20id_ff/scripts/verify_for_docs.py` (reads `Grains.csv`
and `processgrains_diagnostics.h5` only, seconds to run).

## A population `DiffPos` that will not come down

symptom: resid.population_mixture
coord: per-grain `DiffPos` from `Grains.csv`, binned by `Confidence`

The station's residual has been "chronically high" for a while, geometry has been
re-checked, and nothing moves it. Before attributing it to the instrument: **a population
median describes a mixture as faithfully as it describes a population, and says nothing
about which you have.**

**Test — free, `Grains.csv` only, seconds.** Bin per-grain `DiffPos` by `Confidence` and
read the shape. A gradient is a systematic; a **cliff** is a mixture. Measured on a 20-ID
alumina layer (1729 grains, population median 655.7 µm):

| `Confidence` | n | `DiffPos` (µm) | `DiffAngle` | n_spots |
|---|---|---|---|---|
| < 0.60 | 18 | 896.8 | 0.556 | 52 |
| 0.70–0.80 | 172 | 744.2 | 0.572 | 73 |
| **0.80–0.85** | **974** (the bulk) | **661.9** | 0.554 | 80 |
| 0.90–0.95 | 65 | 531.6 | 0.501 | 96 |
| **≥ 0.95** | **37 (2.1 %)** | **57.1** | **0.081** | **126** |

A **9.3× step across one 0.05 bin** — bimodal, not a gradient — and the good grains carry
a nearly complete spot set (126 against a population median of 80) while the bulk does
not. **655.7 µm is a correct statistic and it describes neither population.** Never quote
it as "the fit quality".

**Run the control.** A sample whose residual *is* instrumental shows no such split: gold
on the same detector and geometry gave 5 grains all at ~237 µm, uniformly. Its residual is
real, and traceable to 0.43° azimuthal streaks measured on the raw frames. Alumina's
*good* grains fit **4× better than gold**, which is what the raw frames say too — alumina
spots are compact (NrPx 4–11), gold's are 350-lit-pixel streaks.

**Cause — the matcher's 1.0° internal-angle cap is what ADMITS the bad grains**, not what
censors good ones. `calc_angle_errors` (`midas_fit_grain/midas_fit_grain/c_port.py`,
mirrored in the c-omp binary) keeps a spot only if its best candidate is within **1.0°**, searching
the same ring within a **±5° ω** window. Both are hardcoded, deliberately, and are not
parameters. With ~398 candidates per prediction in that window on this sample, a
random-orientation null puts a chance spot within 500 µm **42 %** of the time — which
passes 1° easily. So a wrong orientation accumulates enough accidental matches to clear
completeness. Measured: the refiner picks the geometrically nearest candidate **41.5 %** of
the time on a random alumina sample against **79.2 %** on the high-confidence grains.

**Corollary — matcher statistics on such a population are CENSORED, and the censoring is
invisible in `Grains.csv`.** The per-spot internal angle is truncated at the cap, so every
statistic derived from it is biased low. Read it from `residuals/spot_table` column 9 in
`processgrains_diagnostics.h5`, never from the `DiffAngle` column, which is a per-grain
*mean* and smooths the truncation away:

| run | per-spot max | p99 | frac at the cap | `Grains.csv` `DiffAngle` max |
|---|---|---|---|---|
| alumina | **1.0000** | 0.9878 | 0.82 % | 0.6977 |
| gold | 0.9427 | 0.7070 | 0.00 % | 0.1956 |

A per-spot max of exactly 1.0000 means truncated. Gold's 0.9427 is a real maximum.

**Lever.** Filter, or tighten acceptance — do **not** chase a detector parameter, and do
not tighten the matching margins to make the number look better (rule 9 and the trap
table: the refiner never reads them, and what they actually change is which candidates
survive indexing). If the split follows radial position rather than confidence, it is
illumination and the entry above applies instead.

> **Estimator warning — "distance to the nearest observed spot" needs its null.**
> It is a minimum over the candidates in the window, so it shrinks as the peak list gets
> denser and is **not comparable between samples**. Raw, it ranked alumina 21.9 µm against
> gold 106 µm — backwards, because alumina had ~398 candidates per prediction and gold
> ~30. Against a random-orientation null the margins invert correctly: gold chance
> 44,358 µm (**418×**), alumina 392 µm (**18×**). Always run the null, and never select
> grains by confidence while measuring a population residual — that alone moved alumina
> 618 → 20 µm.

## Zero seeds indexed, run exits 0

symptom: count.zero_indexed
coord: —

**Test.** Read `n_seeds_indexed` from `<result>/LayerNr_N/midas_state.h5`
(`stages/indexing/metrics`) and the indexer's wall time. An honest zero on a real
search takes tens of seconds; **0 seeds in ~0.1 s means the search never ran**.
Then check the ring count:
`awk 'NR>1{print $5}' hkls.csv | sort -n | tail -1`. Under 500 and this entry does
not apply — look instead at the geometry, the ring assignment, or `Completeness`.

**Cause.** `RhoD`/`MaxRingRad` far larger than the detector generated more rings
than the indexer's fixed `MAX_N_RINGS` (500) table holds. On builds before
2026-08-16 those rows were written unbounded, through `RingTtheta` and into the
`data`/`ndata` bin pointers, after which every seed matched nothing. Measured:
`RhoD 2000000` on a 2880 px detector → 745 rings → 0 of 4569 seeds.

**Lever.** Set `RhoD` to `corner_px × px` (README rule 15, §6d) and regenerate —
the cap only takes effect when `hkls.csv` is *written*, so changing the parameter
without re-running `hkl` changes nothing. Upgrade `midas-index` so the condition
warns instead of corrupting. The downstream `process-grains`
`TypeError: ufunc 'invert'` is a symptom of the empty grain list, not the cause.

## Refined geometry parameter sitting on a bound

symptom: bound.parameter_railed
coord: —

**Test.** Compare each refined value against its bounds. `tx`/`Wedge` are bounded
±5°, the distortion amplitudes ±0.05. A value *at* a limit is the optimiser saying
it wanted to keep going. Cross-check `matched spots`: a large count with a railed
parameter means the parameter is unconstrained; a tiny count means the prediction
is not landing on the data at all, and that is the real fault.

**Cause.** Either the parameter is not constrained by the data (too few grains,
or a quantity this objective cannot see), or matching has failed upstream so the
fit is optimising noise.

**Lever.** Refine fewer parameters, or supply more grains. Distortion belongs on
the powder calibrant, not on a handful of grains. Never report a bound value as a
measurement — `midas-joint-ff-calibrate` ≥ 0.1.9 names it and exits 1.

## Beam centre mirrored about the detector centre

symptom: systematic.mirrored_beam_centre
coord: —

**Test.** Compare the refined beam centre with the previous one and with its
mirror, `N-1 − BC`. Landing within a pixel or two of the mirror rather than of
the prior value is decisive. **Strain does not diagnose this**: measured on 20-ID
Varex, the mirrored fit scored 47.2 µε against the correct fit's 58.2 — the wrong
geometry looked *better*.

**Cause.** The image transform used in calibration does not match the one the
reconstruction uses. `ImTransOpt 2` (flip-Z) applied in one and not the other
flips an axis, and the fit follows it happily.

**Lever.** Make `ImTransOpt` identical in both. Read it from the parameter file
text or `v1.extra` — `CalibrationParams` has no such attribute, so `getattr`
returns nothing and the calibration silently runs with no transform at all.
