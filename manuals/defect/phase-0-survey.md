# Phase 0 — survey: is there a diffuse field, and of what kind?

**Read before choosing any machinery.** Asterism, rods and satellites need different
estimators and different nulls; choosing wrong costs the campaign, not an afternoon.

## What you must establish here

1. **Discrete spots or continuous rings?** Continuous rings are `xrd-ct`, not this doc set.
2. **Is there intensity between the peaks at all**, above the background model?
3. **Which kind**, because the three are told apart by *shape in q-space*:

| kind | shape | what it encodes | phase |
|---|---|---|---|
| **asterism** | anisotropic cloud hugging each Bragg core | dislocation strain field, orientation gradient | `phase-5` |
| **rod / streak** | 1-D line threading several Bragg shells along a low-index direction | planar defects — faults, twin walls | `phase-4` |
| **satellite ladder** | *discrete* peaks at `n·G/m` in symmetry-forbidden gaps | a polytype (9R, 4H, …) | `phase-4` |

A satellite is **compact in ω** and has a Friedel mate at ω+180. A relrod is not compact and
its apparent position moves with ω. That single test decides which half of `phase-4` you are
in, and it needs an ω step fine enough to resolve it — on the reference sample the satellites
were σ ≈ 0.6° at a 1° step, which is marginal.

## Inputs to write down before you start

```
Frames or cloud:  <path>
Grains:           <path or "index from the cloud">
Material:         <lattice type + a, or "unknown">
omega:            first, step, count
Detector:         pixel um, distance um, beam centre, mask source
```

If the material is unknown you can still get orientations and rod *directions*; you cannot get
any absolute length. `ENVELOPE.md` §0.

## The two things to check first

**Is the sample textured?** Cluster the indexed orientations. If they collapse into a few
families rather than filling orientation space, every random-direction null downstream is
suspect and `DIAGNOSIS.md` entry 1 applies. On the reference sample ~230 indexed "grains" per
layer were **two** Σ3-related families at ~18° mosaic.

**Is the indexer over-fragmenting?** Compute the pairwise misorientation distribution and
compare against Mackenzie. A large small-angle excess means your `n` is the fragmentation of
the indexer, not the sample. `DIAGNOSIS.md` entry 3.

## Output of this phase

A one-paragraph statement: what kind of diffuse signal is present, whether the sample is
textured, how many genuine orientation families there are, and which phase file you are going
to next. Write it down — it is the thing that stops a later step being applied to the wrong
kind of feature.
