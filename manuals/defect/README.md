# Diffuse-scattering defect metrology — from a rotation scan to a defect inventory

**Use this doc to start a fresh session on a dataset this pipeline has never seen.**
Paste it in together with `LAB_NOTEBOOK.md`, then give, or work out from the data:

```
Data:      <ABSOLUTE PATH>   # raw rotation frames, OR an existing q-space voxel cloud
Grains:    <ABSOLUTE PATH>   # Grains.csv / orientations, or "index it from the cloud"
Material:  <e.g. FCC Cu a=3.6356 -- or "tell me from the data">
Goal:      dislocation density | fault rods | polytype ladder | intensity budget | sub-grains
```

**Scope.** The **diffuse field around and between the Bragg peaks** of a discrete-spot
rotation series, through the `midas_defect` chain. The standard MIDAS chain
(`midas_index` → `midas_fit_grain` → `midas_process_grains`) operates on detected peaks and
returns grains; it discards everything between them. For a deformed material that discarded
field is the defect signal:

* **Asterism** — extended intensity hugging each Bragg core: dislocation strain field and
  orientation gradient, hence a per-grain dislocation density.
* **Rods** in q-space — 1-D streaks threading several Bragg shells along low-index
  directions: the signature of planar defects (stacking faults, twin walls).
* **Satellite ladders** — discrete `n·G/m` reflections in symmetry-forbidden gaps: a
  polytype (9R, 4H, …) rather than a continuum.
* **The full intensity budget** — classifying *every* above-threshold voxel against the
  predicted reciprocal lattice, so scattered intensity can be decomposed and closed.

**This doc set does not get you the grains.** Indexing a far-field HEDM dataset is
`ff-hedm`; scanning 3DXRD is `pf-hedm`; near-field is `nf-hedm`. **And if the CELL itself is
unknown or disputed — ab initio indexing, lattice symmetry, an a/b splitting or a shear, which
member of a structural series, a pressure — that is `solve-cell`, not `ff-hedm`,** which
assumes a known cell across many grains. What is new here is that
`midas_defect.ingest` can now take you from **raw frames** to a spot list and a voxel cloud
without leaving the package — see `phase-1-ingest.md`. If your rings are continuous powder
rings, this is the wrong doc set entirely: that is `xrd-ct`.

> **On sources.** Every number here is a measurement made by this project on data it
> processed. The real-data anchor is one material family — a deformed Cu-9at%Al single
> crystal measured by FF-HEDM over 10 layers, referred to throughout as *the reference
> sample*. `ENVELOPE.md` §0 says exactly how far that one anchor licenses you to go.

## The doc set — what to read when

| File | Holds | Read it |
|---|---|---|
| **`README.md`** (this) | scope gate, install gate, the order, hard rules, halt conditions | always |
| `phase-0-survey.md` | what you actually have; is there a diffuse field at all? | first |
| `phase-1-ingest.md` | raw frames → mask → background → 3-D spots → voxel cloud, plus the **calling contract** (exact signatures) and the calibration trap table | **if you are starting from frames — copy the chain from here** |
| `phase-2-index.md` | orientation from the cloud: bright-core seeding, **row and pair seeding for weak domains**, **every domain at a position in one call (`find_domains`)**, targeted extraction at predicted sites, the search null, refine-to-convergence, and the **completeness audit** | before any per-grain number |
| `phase-3-classify.md` | Bragg/diffuse split, intensity budget, and what closure means | always |
| `phase-4-rods.md` | rods in the cloud and on the frame stack; **which (h,k) rods carry intensity** (the in-plane fault-vector test); an open split / one-sided record; satellites; polytype; **the Friedel/crossing quartet and how to verify a pairing**; variants vs a fixed q-offset; **a ladder can be right in \|q\| and not be a ladder** | fault / polytype work |
| `phase-5-asterism.md` | asterism fit, Williamson–Hall, sub-grains, Burgers population | dislocation work |
| `phase-6-mechanics.md` | GND, stress, Schmid, variants, energy, Mecking–Kocks, CPFEM handoff | if the goal is mechanics, not just defects |
| `phase-7-report.md` | what to state, what to label provisional, provenance | at the end |
| `ENVELOPE.md` | what this measurement **can** determine, and what it cannot — including §13, why a per-group difference cannot be validated by reproduction, and **§16 why a label sum at a fixed threshold is not a measurement** | **before promising an answer** |
| `DIAGNOSIS.md` | symptom → discriminating test → cause → lever | **when something looks wrong** |
| `LAB_NOTEBOOK.md` | evidence ledger, **retracted results**, and one live package bug | before re-investigating anything |
| `RUNBOOK.md` | where it runs, healthy ranges with their conditions, pick-up point | on resume |

## STOP — read this before touching anything

### The governing fact

**Every scalar you are about to compute is a sum over voxels you have not looked at.**
A dislocation density, a budget percentage, a coherence length, a satellite enhancement —
each is a number produced by pooling a field. The failure mode of this technique is not a
crash; it is a plausible number pooled over the wrong voxels, and it looks identical to a
right one.

The reference sample makes the point quantitatively. An auto-classifier built on this data
reported **99.8 % intensity-budget closure**. It was **~18 % wrong** against ground truth
and was retracted. Root cause: at high `|q|` the 18°-mosaic node cloud of 232 indexed
orientations makes `5G/3`, `220`, asterism and fault-rod tails **overlap in every scalar
feature**, so no scalar decision tree can separate them. Closure was arithmetic, not
attribution. See `LAB_NOTEBOOK.md` entry R1.

### Install gate

```bash
python -c "import midas_defect as m; print(m.__version__)"          # expect >= 0.1.5
python -c "from midas_defect import ingest, completeness, rod_profile, \
           residual_decomposition, rod_detect, seed_index, honesty, domains, selfcal; print('ok')"
python -c "from midas_defect.bragg_diffuse import check_orientation_convention; print('ok')"
```

If any of the first four fail to import you are on a tree from before the 2026-09-01 port and
this doc set does not describe it. The real-data regressions are **env-gated**:

```bash
MIDAS_DEFECT_REAL_DATA=1 pytest packages/midas_defect/tests -q     # 564 pass, 7 skip
```

### The command line

Four entry points ship with the package. `midas-defect-inventory` is the flagship and the
fastest way to see the whole chain on data you have:

```bash
midas-defect-inventory --voxels /path/voxels_layerXXXX.npz --grains /path/Grains.csv
#   geometry QC -> Bragg/diffuse split -> 100 % intensity budget
#              -> forbidden-reflection test -> <111> rod enrichment -> fault-alpha
midas-defect-rods        # rod detection alone
midas-defect-asterism    # asterism fit alone
midas-defect-polytype    # satellite ladder / polytype alone
```

The inventory driver is **phase-agnostic** — swap the `crystal` and the distortion/geometry
block to point it at another material. It reuses
`midas_transforms.fit_setup.transform.apply_tilt_distortion` for the validated,
distortion-aware pixel→lab map: the same path that produced the published numbers.

> **A live convention trap in that driver.** It uses the validated ω map
> `ω = 180 − 0.25·frame`, **not** the sign stored in some older NPZ files. An ω sign error
> does not crash; it mirrors reciprocal space and every direction you go on to measure.

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** Every condition below finishes and
looks right.

| Condition | Why you cannot decide it yourself |
|---|---|
| A deprecated estimator is about to be reached with `allow_deprecated=True` | `polytype_satellite_enhancement` and `per_grain_lamella_thickness` **raise by default** for good reason: the first inflated a real ~5× excess to **700–1600×** and, separately, gives false negatives in textured samples. The escape hatch exists to reproduce history, not to get an answer. Use `satellite_excess.satellite_radial_excess`. `LAB_NOTEBOOK.md` D1. |
| A **budget-closure percentage** is about to be quoted | Closure is arithmetic; attribution is the claim. 99.8 % closure was ~18 % misattributed. Quote the per-class fractions *and* how they were separated, or quote neither. |
| The **indexer's grain count** is being used as a grain count | On the reference sample ~230 "grains"/layer are a **~100× over-fragmentation** of a 2-family, ~18°-mosaic continuum. Per-grain statistics over those are statistics over fragments. |
| A **coherence length** is about to be quoted as a value | Satellite widths there were mosaic-contaminated: FWHM 0.075–0.12 Å⁻¹ → L ≈ 5–10 nm is a **lower bound**. `rod_profile.transverse_width` returns a lower bound rather than a number when the feature is resolution-limited; do not convert it back. |
| Bulk is to be separated from **boundary** on a **shared** reciprocal direction | Intrinsic limit, not a resolution problem. On the reference sample the 9R sits on the ⟨111⟩ **shared** by parent and twin, and FF-HEDM cannot tell parent-bulk from twin-bulk from the boundary film there. Needs pf-HEDM or DFXM. `ENVELOPE.md` §4. |
| A **control** was chosen without asking whether it can fail | The azimuthal-median control on polar-median-subtracted data returns exactly zero **by construction**. `rod_profile.rod_significance` refuses a zero-scatter control for this reason; do not work around it. |
| Rod strength is about to be quoted as a **ratio** | Per-frame background-subtracted data is centred on **zero**; a ratio then divides by noise about zero. It returned **−1550×** once. Quote σ above a matched control. |

Finish everything not blocked by the halt before reporting it.

## Hard rules

1. **Set every threshold from a null that could have failed, or not at all.** The nulls that
   matter here keep the *support* and destroy only the *structure* — ω-scramble, spot-swap,
   matched (h+½,k+½) walk. A null that cannot produce a hit proves nothing.
2. **Never ring-filter before a 3-D search.** Assigning spots to rings first discards exactly
   the off-ring intensity the diffuse layer exists to measure. On one dataset it cost 17 of
   45 reflections.
3. **Segment in 3-D, in (ω, row, col).** A reflection sweeps several ω frames; per-frame
   labelling counts it once per frame and cannot distinguish a genuine second reflection from
   the same one a frame later.
4. **Not-observable is not not-there.** Masked, off-detector and no-ω-solution are three
   different things and none of them is zero intensity. A silent gap looks exactly like a
   real minimum in the quantity being measured.
5. **Whenever a step selects, report how many it discarded.** In the source analysis
   "keep one, discard the rest" silently drove a result four times; the worst case discarded
   43 of 45 reflections and then quoted a rate over the surviving 2 as if over 45.
6. **A discrete feature needs a discrete test.** A 15° cone diluted <5° satellites into
   invisibility on the reference sample. Look at raw voxels and nearest-axis attribution, not
   along-axis-versus-random enhancement.
7. **Report each candidate's line count in any d-matching comparison.** A residual without a
   line count is not evidence — `midas_hkls.phase_id` enforces this and supplies the
   volume-correct null.
8. **Beating a null is not enough if a deliberately WRONG model beats it too.** Run a decoy —
   a cell or an axis you know to be wrong — through the identical machinery. If the decoy
   also clears the null, the null is measuring the machinery rather than the model, and the
   question has to be settled structurally instead.
9. **A `return None` floor will silently answer for your null.** A refinement that gives up
   below *n* matches makes the null *and* the decoy report zero, which reads as a crushing
   p-value. Re-run every null with the floor lowered. Uncensored, one such "p = 0.000,
   decoy 0" became null median 3 / max 5, decoy 4.
10. **Look at the raw per-frame data before concluding a feature is absent.** Summing over ω
    superimposes every grain's rod at every orientation and smears it into a ring; the
    feature resolves only per-frame, per-grain-cluster. A real ⟨111⟩ relrod was reported
    absent on exactly this frame-of-reference error.
11. **A null that can fail is not enough — a positive control must show the test can see the
    feature PRESENT.** Plant it, run it through the pipeline unchanged, and check it is
    recovered. A directional test here returned 29.1° against a 28.95° isotropic null and read
    as a clean absence; a planted rod through the same machinery scored 29.02° against 28.34°
    for a planted isotropic blob. It had no power, and nothing in the output said so.
12. **Before promising any per-voxel attribution, compare the predicted node spacing against
    the width of the feature.** 232 orientations × 282 allowed hkl tile reciprocal space at
    0.0688 Å⁻¹ here, finer than the 0.05–0.15 Å⁻¹ halo — so no voxel can be attributed to a
    reflection, by distance or by direction. One line at phase 0 decides which deliverables
    are reachable.
13. **Two populations may only be compared if they were scored the same way.** Far-only
    against whole-shell produced a 58× "ranking" that was a tie under matched accounting.
14. **Let the data referee the orientation convention** — `check_orientation_convention`. It is
    a property of the voxel cloud, not a rule; two products of one experiment disagree.

## The order, and why each step is where it is

| # | Step | Why here |
|---|---|---|
| 0 | Survey — is there a diffuse field, and of what kind? | Asterism, rods and satellites need different machinery and different nulls. Choosing wrong wastes the campaign. |
| 1 | Ingest — mask, background, 3-D spots, voxel cloud | Everything downstream is a pooling over these voxels. The background model decides what "between the peaks" even means. |
| 2 | Index + **completeness audit** | Attribution needs a predicted lattice. The audit is what tells you the orientation is not silently missing reflections that *are* present. |
| 3 | Bragg/diffuse classification and the budget | Defines the denominators every later fraction is quoted against. |
| 4 | Rods / satellites | Needs the lattice (step 2) to know which directions are low-index and which gaps are forbidden. |
| 5 | Asterism, WH, sub-grains | Needs the Bragg/diffuse split (step 3) to know which voxels are "near-Bragg". |
| 6 | Mechanics — GND, stress, Schmid, energy, hardening | Consumes step 5's widths and the per-grain orientations. It inherits the phase, so it inherits the Burgers vector. |
| 7 | Report | `ENVELOPE.md` decides what may be claimed; `phase-7` decides how to say it. |

## Sibling doc sets

`manuals/solve-cell/` (**where the CELL comes from when it is unknown or disputed** — ab
initio indexing, lattice symmetry, distortion mode, phase ID and pressure, skill
`solve-cell`; note that `midas_defect.rows`, `.seed_index`, `.geometry` and
`.completeness` are that chain's machinery living in this package),
`manuals/ff-hedm/` (far-field HEDM — **where the grains come from** at a KNOWN cell,
skill `ff-hedm`),
`manuals/pf-hedm/` (scanning 3DXRD — **the escalation when bulk and boundary share a
reciprocal direction**, skill `pf-hedm`), `manuals/nf-hedm/` (near-field, skill `nf-hedm`),
`manuals/dct-tt/` (DCT and topotomography, skill `dct-tt`), `manuals/dfxm/` (dark-field X-ray
microscopy — the other escalation for intragranular detail, skill `dfxm`),
`manuals/xrd-ct/` (**the right doc set if your rings are continuous**, skill `xrd-ct`),
`manuals/calibrate-integrate/` (the geometry this doc set consumes — and note that
`midas_calibrate_v2.friedel` can supply a beam centre and **tx** from the sample's own
Friedel pairs when no calibrant exists, skill `calibrate-integrate`), `manuals/tomo/` (the
**coordinate-system reference** across FF, NF, PF and tomo, skill `tomo`), and
`manuals/dct-tt/` (DCT and topotomography, skill `dct-tt`).
