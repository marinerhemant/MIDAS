# Phase 4 — rods, satellites and polytypes

Two different machineries, told apart by the `phase-0` test: a **satellite** is compact in ω
with a Friedel mate at ω+180; a **rod** is continuous. They have different nulls and must not
be swapped.

## Rods in the cloud

```python
from midas_defect.rod_detect import find_rods, find_rods_iterative_residual
```

Pair-seeded RANSAC over bright cores, scored by counting bright voxels within a tube, then
NMS and a differentiable soft-tube refinement of direction and pivot. Per rod it returns the
integrated intensity, the line-projected profile, and which Bragg shells it crosses. The rod
direction is the **defect-plane normal**.

## Rods on the frame stack

```python
from midas_defect.rod_profile import (rod_path, matched_control_path, profile_along,
                                      rod_significance, ring_L_marks, transverse_width)
```

Use this when the question is *how intensity runs along a known rod*, or *is this rod real*.

**Walk the rod where it is straight.** A rod is a straight line in reciprocal space, but the
map to a flat detector goes through the Ewald sphere and is nonlinear, so on the detector it
is a **curve**. Walking a straight detector line drifts off it, worst at large |L| — which is
exactly where the profile is read. `rod_path` parameterises `(h, k, L)` with L continuous,
solves ω per point, and projects. Each L is then read from **its own frame**, not from a max
projection that piles every frame's background under every point.

**Three ways a point drops out, none of them "no intensity":** no ω solution in the delivered
range, off the detector, or the transverse slice mostly masked. All are counted separately and
returned as **NaN**. A silent gap looks exactly like a real minimum in the quantity being
measured.

**Quote σ, never a ratio.** Per-frame background-subtracted data is centred on **zero**, so a
ratio divides by noise about zero — it returned **−1550×** once. `rod_significance` gives
`(rod − median(control)) / (1.4826 × MAD)`.

**The control must be able to fail.** `matched_control_path` is the same walk at
`(h+½, k+½, L)`: not a lattice rod, but the same |q| range, curvature, ω range and detector
regions. An earlier control — the azimuthal median at the same radius — returned exactly zero
for every row, because a polar-median-subtracted stack *is* an azimuthal median.
`rod_significance` raises on a zero-scatter control rather than returning a number.

**Mark ring crossings, do not delete them.** `ring_L_marks` maps detected powder rings onto L.
A ring puts a bump at one L, imitating exactly the modulation that would encode fault
statistics. Get the radii from the **raw** data, not from a stack the rings were removed from.

**Width is a comparison, not an absolute.** `transverse_width(diffuse_fwhm, bragg_fwhm)` —
the Bragg peaks *on the same rod* carry the resolution function, so
`excess² = diffuse² − bragg²`. If the two are equal the rod is **resolution-limited** and the
function returns a **lower bound** with `coherence_length_A = None`. That is a real result.
Do not convert it back into a value. `ENVELOPE.md` §2.

## A null that can fail is necessary, and not sufficient

**You also need a positive control proving the test can see the feature when it is PRESENT.**
A null tells you the test does not fire on noise. It says nothing about whether the test can
fire at all.

On the reference sample a directional rod test returned **29.1°** against **28.95°** isotropic
and read as a clean absence. Planting a real ⟨111⟩ rod through the identical machinery scored
**29.02°** against **28.34°** for a planted isotropic blob — the test had **no power**, and the
"absence" was meaningless. `LAB_NOTEBOOK.md` R14, `ENVELOPE.md` §1a.

So: plant the feature at a real node, run it through unchanged, and only then believe an
absence. If the planted feature is invisible, the honest report is **"not obtainable"**, not
"absent".

## Satellites and polytypes

```python
from midas_defect.polytype.satellite_excess import satellite_radial_excess
```

**Use this, not `satellite_intensity.polytype_satellite_enhancement`**, which raises by
default and for two opposite reasons — voxel-count inflation (~5× became 700–1600×) and
texture-contaminated nulls (false negatives). `LAB_NOTEBOOK.md` D1.

`satellite_radial_excess` is a **count-normalised** per-voxel mean-intensity ratio between
on-axis and off-axis voxels **at the same |q|**, so detector sampling density and texture
cancel. It carries the discriminator the old metric lacked:

* a real periodic polytype is **peaked at the thirds** (`G/3`, `2G/3`) and **dips to
  background at the half-integers** (`G/6`, `G/2`, `5G/6`);
* a continuous relrod **rises monotonically inward** from the Bragg.

The verdict comes back with the numbers.

**In a textured sample, never use an along-axis-versus-random-direction statistic.** A 15°
cone dilutes a <5° satellite into invisibility and the "random" baseline is contaminated by
the textured axes themselves — both push toward a false negative, and together they produced
one on a phase that is unambiguously present. Use raw voxels and nearest-axis attribution with
a cluster-label cross-tabulation that makes no shared/unique assumption. `DIAGNOSIS.md` 1.

## Finding the axis, and the rest of the polytype API

```python
from midas_defect.polytype.activated_axis import detect_activated_111_axis
from midas_defect.polytype.cell_index import ...          # supercell indexing of the ladder
from midas_defect.polytype.finite_stack import ...        # finite-stack (Hendricks-Teller) model
from midas_defect.polytype.doublet_survey import ...      # doublets across the whole ladder
from midas_defect.polytype.aggregate_thickness import ... # lamella thickness, aggregated
```

`detect_activated_111_axis` finds the modulation axis **from the data** rather than being told
it — the data-refereed route that, cross-checked against the shared-⟨111⟩ construction, gave
99/99 Σ3 pairs agreeing to 0.0° on the reference sample.

A **finite-stack / Hendricks–Teller fit is prone to local minima** — an α = 0.16 on the
reference sample was self-retracted as exactly that. Sweep the starting point and report the
landscape, not the converged value alone.

## Checking a rod against a simulation, and looking at it

```python
from midas_defect.forward_sim import ...                  # simulate the expected diffuse field
from midas_defect.viz import render_rod_overlay_html      # rods over the data, interactive
from midas_defect.asterism.family_asterism import reflection_directions, family_asterism_arc
```

Related modules: `polytype.ladder`, `polytype.satellite_doublet` (a doublet's signature is an
**ω-split**, not a q-position split — `DIAGNOSIS.md` 6), `polytype.modulation_tilt`,
`polytype.modulation_type`, `polytype.fault_balance`, `delta_pdf` for the r-space view.
