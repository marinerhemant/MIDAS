# Phase 3 — DCT: assignment → extraction → SIRT → grain map

**Goal:** a 3-D shape per grain and a labelled map, with an honest statement of how much of it
is measurement.

## 3.1 Forward-assign spots to grains

With orientations and positions known, predict where each grain's reflections land and assign
observed spots to them. Two forward-model bugs invalidated results here before they were
found, so verify the forward model **before** trusting an assignment:

* **The antipode.** One forward path placed spots at the exact inversion of the other about
  the beam centre. `flip_y` cannot express that transformation, so no combination of flip
  flags fixes it. Check by overlaying predicted and observed spots for one grain you trust.
* **The ω wrap.** Converting ω to a frame index without wrapping silently discards **half of a
  360° scan**. This reads as a *physics* limit on completeness — and did, until it was found.

Assignment quality here: completeness **0.465** against a **0.117** null; spot intensity
explained rose **19.6 % → 77.1 %**, and spots per grain **~16 → 46**, as residual indexing and
merging stages were added.

## 3.2 Extract per-frame, not ω-summed

Extract each grain's spot on **single-ω frames** rather than from an ω-summed patch — here
**164 887** single-ω views. The ω-sum is a reasonable approximation (see the withdrawal in
§3.6), but per-frame views are what the reconstruction wants, and the ω-summed route was
entangled with both bugs above.

## 3.3 Reconstruct with SIRT

```python
from midas_dct_tt import sirt
vol = sirt(views, geometry, n_iter=100)
```

**Threshold with Otsu, never a fraction of the maximum.** Streak artefacts put a few voxels
far above the grain's own level, so `0.5 × max` reported **60 µm grains as 6 µm**.

**One thread per worker.** `torch` grabs every core in every worker: 20 workers × 64 threads
did not finish a *single* grain in 32 minutes; with `torch.set_num_threads(1)` and
`OMP_NUM_THREADS=1`, all 121 finished in ~3 min.

## 3.4 Validate the shape against a null — and against truth if you can

**Internal null (always).** A spot-swap null: give each grain another grain's spots and
reconstruct identically. Adopted result: SIRT cores separate **15.0×** in volume from that
null. Across thresholds 0.05–0.125 the separation ran 12.2× → 16.4× with **no cliff**, so the
threshold is a choice within a broad valid range, not a discovery.

**External validation (if a reference map exists).** Build a phantom from a published grain
map with spot physics matched to yours (view area 107 px vs 107, wander ratio 0.717 vs 0.572,
grain size 33 vs 34 µm), then score against the true labels — 500 grains, 38 spots each:

| method | accuracy | boundary error |
|---|---|---|
| nearest grain centre (no shape information) | 56.3 % | 6.0 µm |
| visual hull | 77.7 % | 2.0 µm |
| **SIRT (adopted)** | **91.4 %** | **2.0 µm** |

On *identical* voxels SIRT beats the no-shape floor by **+27.8 points** and the hull by +13.6.
Note the honest caveat: the phantom's grains are slightly better ordered than the real ones,
so real performance is probably somewhat below 91.4 %.

**The no-shape floor is the comparison that matters.** 56.3 % accuracy from grain centres
alone means a shape method must beat 56 %, not beat zero.

## 3.5 The two numbers that must accompany every map

**Dilation fraction.** In the adopted map **86 % of the labelled volume is dilation**, not
measurement. Its clean, space-filling appearance is that interpolation. A map without this
number overstates itself by roughly a factor of seven.

**Uncontested fraction.** Voxels claimed by exactly one grain:

| threshold + mode | measured | contested | **uncontested** |
|---|---|---|---|
| 0.125 + largest+fill | 31.2 % | 33.7 % | **20.7 %** |
| 0.100 + largest+fill | 38.4 % | 44.0 % | **21.5 %** |
| 0.100 raw | 47.2 % | 50.3 % | **23.4 %** |
| 0.075 raw | 58.2 % | 63.2 % | **21.4 %** |

**~22 % is the information ceiling of this dataset regardless of threshold.** Lowering the
threshold converts uncontested → contested, not uncontested → measured — and contested-voxel
assignment does *not* separate from a null. Any fill past ~22 % is a reproducible choice, not
a measurement.

## 3.6 Why only some grains yield a shape — and why that is physical

Requiring one compact body dropped **810 SIRT reconstructions to 455**. Four candidate causes
were injected into the validated phantom one at a time, holding everything else fixed:

| candidate | injected at the real value | fragmentation | real |
|---|---|---|---|
| spot count | 29 spots | 2.0 % | 44 % |
| position error | 12 µm | 1.4 % | 44 % |
| silhouette roughness | — | ruled out | 44 % |
| **ray-direction scatter** | **0.10°** | **20.0 %** | 44 % |

Only ray-direction scatter reproduces it; the observed rate implies **~0.14°**.

**And it is not fixable by refining orientations** — checked before building anything.
Decomposing the measured ray-direction residual per grain, the **coherent** part (a wrong `U`,
removable by refinement) is **0.031°**, while the **incoherent** scatter about it is
**0.173°**, a ratio of 0.18. The orientations are already good; the scatter is *within* each
grain. Independent confirmation: 0.173° predicts ~71 % fragmentation on an injection curve
built from completely different data, against 44 % observed.

So **fragmentation and the coverage ceiling are the same phenomenon** — claimed volume halves
(22.8 % → 11.1 %) at the same 0.1° that causes 20 % fragmentation — and ~0.17° of intragranular
orientation spread means a grain is not one coherent diffracting object. No estimator and no
extra spots make it one.

*Caveat kept honest:* the incoherent term also contains per-spot assignment error. Assignment
contamination is **not** the driver — contested fraction separates fragmented from intact
grains at exactly **0.00** — but it is not zero, so 0.17° is an **upper bound** on the true
mosaic spread.

**One withdrawn mechanism.** "The ω-summed patch is smeared because the spot travels across
the detector" was withdrawn: measured centroid displacement is **27 px**, not the ~107 px
claimed, and the net/path ratio (0.572, falling to 0.216 for long spots) says the spot
**wanders about a fixed point**. The *measurements* stood; that *explanation* did not. Treat
causal accounts as provisional even when the numbers they explain are solid.

## 3.7 Exit criteria

- [ ] forward model verified (antipode and ω-wrap both checked) before assignment is trusted
- [ ] assignment completeness quoted against its null
- [ ] shapes validated against a spot-swap null, with the separation factor
- [ ] threshold shown to sit in a **range** without a cliff, not tuned to a value
- [ ] **dilation fraction** and **uncontested fraction** stated with the map
- [ ] grains without a compact core reported as position+orientation only, with the count

**Halt** before publishing a map that looks clean and space-filling without knowing its
dilation fraction. That appearance is what dilation of a sparse measurement produces, and it
is the most convincing wrong picture this pipeline makes.
