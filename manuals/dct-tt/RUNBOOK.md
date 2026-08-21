# Runbook — a real scan, end to end

A from-scratch reconstruction of a multi-tens-of-GB rotation series whose file header
contained **nothing usable for geometry**. Calibration, lattice, indexing, spot assignment and
grain shapes were all derived from the data. Written against what actually ran, not from the
API.

**Use this as the shape of a run, and as the thing to compare your numbers against.**

## The scan

```
3600 projections over 360 deg, 0.1 deg/step, 0.4 s exposure
2048 x 2048 uint16 frames, sCMOS detector
no flat, no dark, sample never clears the beam at any omega
```

That last line matters: **no absorption tomography is obtainable from this scan.** The sample
outline had to be inferred, and FBP gives the classic truncation wedge.

## Current state — the pick-up point

**As of 2026-08-21.** The scan below is complete and its numbers are final;
this section says where the *work* stopped, not where the scan did.

| what | state |
|---|---|
| Grain map (orientation + position) | **DONE** — 862 grains, spot intensity explained 19.6 % → 77.1 % |
| Grain shapes | **DONE** — 455 grains with a validated shape |
| Domain unambiguously one grain | **~22 %** at any threshold — the information ceiling of this data, not a tuning target |
| Orientation refinement | **NEXT** — the open step; the map is built but per-grain orientations have not been refined against their assigned spots |
| TT intragranular field | gated on the conditioning check in `ENVELOPE.md` §3 — do not promise a rotation tensor before running it |

**Where the next session should start:** orientation refinement. Everything
above it is closed; nothing below it should be attempted until the §3
conditioning gate is run for the specific reflection pairs available.

## Where it ended up

| | start | final |
|---|---|---|
| grains (orientation + position) | 405 | **862** |
| spot intensity explained | 19.6 % | **77.1 %** |
| spots per grain | ~16 | **46** |
| grains with a validated shape | 0 | **455** |

with **~22 % of the domain unambiguously one grain at any threshold** — the information
ceiling of this data.

## The chain

```
segment            71554 spots at 8 sigma
Friedel pairs      29295 pairs (82%); ring constraint applied DURING matching
                   (on-ring pairs 5679 -> 16282 once fixed)
self-calibrate     fcc, s = lambda/2a = 0.037257, Lsd 6.775 mm, 5 rings,
                   2 free parameters, 0.91 px rms
                   effective pixel 1.653 um; rotation-axis column 1016.53
joint index        orientation AND the position the grain's own pairs imply
forward assign     full 360 deg, both convention bugs fixed first
residual index     -> merge -> re-assign
per-frame extract  164887 single-omega views
SIRT               per grain, 100 iterations
threshold          0.10 + largest + fill + open + dilate  (Otsu, not 0.5*max)
voxel index        pair-free, GPU: 40000 orientations per voxel in 164 ms
                   -> 256 found, 220 REDISCOVERIES, 36 new, 19 verified
```

## Order of operations, with the checks that closed each step

**1. Geometry, from the data.** Effective pixel from the slit box: 134.0 × 420.6 px against
0.2199 × 0.700 mm gaps — **the two axes agree to 1.4 %**, which is what makes it a
measurement rather than a number. The header pixel was the **sensor** pixel, a factor 6.65
away.

**2. Rotation-axis column** = the value making Friedel-pair ring radii sharpest. Diagnostic
that it worked: paired radii sharp (0.9–2.9 px) where unpaired radii are broad.

**3. Lattice by joint fit.** fcc on 5 rings with 2 free parameters at 0.91 px rms. An earlier
"hcp c/a = 1.856, L = 123 mm" was a small-angle local minimum where only the product `s·L` is
determined — retracted.

**4. Index on virtual spots.** `(y+y')/2 − c` and `(z−z')/2`. Raw-spot ring assignment was
impossible: rings 84 px apart, position moving a spot up to 150 px.

**5. Set the margin from the null.** At the first (loose) margin, real and ω-scrambled null
indexed **identically** — 2761/2902, completeness 0.250 both. Adopted: 0.52° on ring 1 with a
0.09 minimum-match fraction, above the null's 0.069 maximum. Real 488 seeds (ring-1 seeding),
814 (all-ring); **null 0**.

**6. Fix the symmetry frame** before clustering: `Uaᵀ Ub S`, S on the right. Wrong frame gave
29.8° for a 0.33° pair and 367 clusters where there were 205. Caught by the `midas_stress`
cross-check, which then agreed to 5.7e-14°.

**7. Solve positions in the sample frame.** `Rz(σω)` in the design matrix: residual 52 → 41 µm,
and the centre cloud became a recognisable sample cross-section (~350 µm across, z ∈ [−62, 58]
µm inside a 220 µm beam).

**8. Fix the forward model before assigning.** The antipode inversion and the unwrapped ω both
had to go first; three retractions rest on them.

**9. SIRT, one thread per worker.** 20 workers × 64 threads finished zero grains in 32 min;
one thread each finished 121 in ~3 min.

**10. Validate the shapes twice.** Internal: spot-swap null, **15.0×** volume separation
(12.2×–16.4× across thresholds 0.05–0.125, no cliff). External: phantom from a published grain
map with matched spot physics — SIRT **91.4 %** vs hull 77.7 % vs no-shape floor 56.3 %.

**11. Report the map honestly.** 86 % dilation, ~22 % uncontested.

## Timing and hosts

Reconstruction ran on a 64-core CPU host; the pair-free voxel indexing on a single large GPU
(40 000 orientations per voxel in 164 ms). Long jobs launched detached with output redirected
to a log — an SSH hangup otherwise kills them. Scripts were rsynced from the local source of
truth before each run; results copied back.

## What would be done differently

* **Run the ω-scramble null before believing any indexing number**, not after. That one
  ordering change would have prevented the largest retraction.
* **Verify the forward model against one trusted grain** before building assignment on it.
* **Measure the coherent/incoherent split of the ray-direction residual early.** It was
  measured before building a refinement stage and showed refinement could not help
  (0.031° coherent vs 0.173° incoherent). That saved the most time of anything here.
* **Decide the material question at the start.** Everything except absolute strain was
  reachable without it, but that was established late.
