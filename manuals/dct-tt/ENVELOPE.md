# Envelope — what these measurements can and cannot determine

**Owner:** Hemant Sharma. **Last reviewed 2026-08-21.**

## Tiers — which limits can move, and which cannot

The tier decides what a report is allowed to say: a *configured* limit may be
suggested as a change, an *intrinsic* one must be called unobtainable rather
than tuned at.

| tier | meaning | sections |
|---|---|---|
| **Fixed** | set by the instrument or the material for this cycle; not changeable by this analysis | §1 (λ/2a degeneracy — the ring pattern determines `s = λ/2a` and cannot separate λ from `a`), §5 (the physical ceiling on grain coherence) |
| **Configured** | chosen per run, changeable next time — these are the only ones a report may propose changing | §7 (what these recipes assume: tolerances, thresholds, model order), and the acceptance set from the ω-scrambled null in `DIAGNOSIS.md` entry 2 |
| **Intrinsic** | a property of what the measurement can determine at all; no parameter recovers it | §2 (how much of a grain map is measurement rather than dilation), §3 (the conditioning envelope for intragranular rotation), §4 (the spatial envelope for a recovered field), §6 (the detection floor) |

Each section below states what imposes its limit, which is the part that makes
a bound actionable rather than decorative.


A dataset can be squarely in scope and still unable to support what is being asked of it.
**Read this before promising an answer.** Section 0 is the table to check first.

## §0 Deliverable vs what it requires

| Deliverable | Requires | Available from a single DCT/TT scan? |
|---|---|---|
| Grain **orientations** | lattice type + `λ/a` | ✅ yes, without naming the material |
| Grain **positions** | pairing + sample-frame solve | ✅ yes (~41 µm residual here) |
| Grain **map / shapes** | assignment + a shape estimator + a null | ✅ but see §2 on how much is measurement |
| **Absolute** d-spacing, any **strain** | λ and `a` **separately** | ❌ only `λ/2a` is measurable |
| Overall **handedness** | something outside the diffraction | ❌ only `y_sign × ω_sign` is fixed |
| **Sample outline** by absorption | sample clearing the beam at some ω | ❌ if projections are truncated |
| Intragranular **rotation field** (well-determined components) | one TT scan | ✅ minus the component along `G` |
| Intragranular **rotation tensor** (all three components) | two reflections with γ ≥ 60° | ⚠️ stage-limited — see §3 |
| Intragranular **strain** by TT | a demonstration that does not exist here | ❌ not demonstrated on real data |

## §1 What only the material can give you

The ring pattern determines `s = λ/2a`. It does **not** determine λ or `a` separately.

Consequences:
* Indexing, orientations, the grain map: unaffected — they use lattice **type** and `λ/a`.
* Any absolute length in the crystal, any d-spacing, any strain: **unavailable** until the
  material (or the energy) is named.

Narrowing the candidates by other evidence is legitimate, but an argument like "the strong
ω-dependent attenuation argues against a light alloy" is **not a measurement** until it is
made quantitative. Do not let it become one by repetition.

## §2 How much of a grain map is measurement

Three different numbers, and reports routinely conflate them:

| Quantity | Meaning | Measured here |
|---|---|---|
| **labelled** | any voxel with a grain ID | 100 % after dilation |
| **measured** | voxel came from a reconstructed core | 31–58 % depending on threshold |
| **uncontested** | claimed by exactly **one** grain | **~22 %, at any threshold** |

**~22 % is the information ceiling of that dataset.** Lowering the threshold moves volume from
uncontested into contested, not into measured — and contested-voxel assignment does not
separate from a null. So any fill past ~22 % is a *reproducible choice*, not a measurement.

Separately: **86 % of the labelled volume was dilation**. A map is not wrong for being
dilated; it is wrong for not saying so.

## §3 The conditioning envelope for intragranular rotation

A single TT scan is blind to rotation about its own `G` — exactly null, not merely weak. Two
reflections separated by γ give sensitivity eigenvalues

```
(1 - cos g)/4 ,   (1 + cos g)/4 ,   1/2
```

so the weakest component scales as **γ²/8**. Practical thresholds:

| γ | weakest/strongest | verdict |
|---|---|---|
| 13.3° | 0.0067 vs 0.5 | **75× worse in one direction** — no tensor |
| 60° | within a factor 2 | usable tensor |
| 90° | parity across all three | ideal |

**And reachability is a separate question from desirability.** Audited over 55 grains: at the
tilt envelope actually used, **0 of 55** grains could reach a γ ≥ 60° pair; at ±30°, **51 of
55** could. Nothing below ±25° works for any grain. The binding constraint is the **stage**.

So the honest sequence is: compute γ from the grain's orientation and the stage envelope
*before* the experiment, and if γ < 60° report the well-determined components and say the
tensor is out of reach. That is a fact about the instrument, not a shortcoming of the fit.

## §4 The spatial envelope for a recovered field

Recovery is only per-voxel information where it beats a **polynomial ceiling**. Measured
window: **1.2–2.0 µm** (2–4 voxels), peaking at 1.5 µm; above ~2.8 µm a low-order polynomial
does better, meaning a few dozen numbers would have reproduced the "field".

Report the window with the field. Without it, a smooth convincing map is unbounded — it may be
carrying no per-voxel information at all.

## §5 The physical ceiling on grain coherence

~**0.17°** of intragranular orientation spread was measured (an **upper bound**, since the
estimate also contains per-spot assignment error). At that spread a grain is not one coherent
diffracting object: 44 % of grains fragment, and claimed volume halves at the same 0.1° that
causes 20 % fragmentation.

This is not a fitting failure and **no refinement fixes it** — the coherent, refinable part of
the residual is 0.031° against 0.173° incoherent. Deformed material has a lower ceiling than
annealed material, and that is a property of the specimen.

## §6 The detection floor

A space-filling structure at the measured grain size would have held ~2010 grains; 862 were
found. Four independent routes each hit the same wall, including pair-free voxel indexing
which added only **19 verified** new grains (+2.3 %) while **rediscovering 220** already
known.

A genuinely missing grain is **small, weak or deep enough that it never produces the ~10
matched spots any method needs**. Do not promise that more indexing effort closes this gap on
comparable data.

## §7 What these recipes assume

* Kinematical diffraction. Extinction is handled as a constant prefactor for a mosaic crystal
  (`midas_dct_tt.extinction`); strong dynamical effects are out of scope.
* Rigid grains for shape reconstruction — §5 is exactly where that assumption starts to fail.
* A rotation series over a full 360° with pairing available. Partial scans lose the Friedel
  cancellation that makes phase 1 and 2 work.
