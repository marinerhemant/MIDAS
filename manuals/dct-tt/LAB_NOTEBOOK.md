# Lab notebook — evidence, and what died

**Read this before re-investigating anything.** Seven results in this project were retracted
and one mechanism withdrawn. **None died of new physics.** Every one died of a broken input
that produced a plausible number, or of a threshold carried into a regime it was not tuned
for.

The recurring failure modes, stated once:

1. **Comparing quantities computed differently on the two sides.**
2. **Carrying a threshold tuned at one projection count into a regime with ten times as many.**
3. **Believing a positive before running its null.**

## §1 RETRACTED — do not resurrect these numbers

| # | Claim | Why it died |
|---|---|---|
| R1 | **"2761 seeds indexed → 287 grains"** at a loose indexing margin | The ω-scrambled null indexed **identically**: 2761/2902, max completeness 0.250 on *both*. An artefact of the tolerance, not grains |
| R2 | **"hcp, c/a = 1.856, L = 123 mm"** from an early ring fit | Small-angle-limit local minimum where only the **product** `s·L` is determined. The individual values were meaningless. Earlier hcp / simple-cubic "wins" were overfitting artefacts of nearest-neighbour scoring |
| R3 | **Grain 3-D shapes, first attempt** | Built on top of R4 and R5 below |
| R4 | *(input bug)* forward model placing spots at the **antipode** — exact inversion about the beam centre | `flip_y` cannot express that transformation, so no flip flag fixes it. Every comparison looked "nearly right" |
| R5 | *(input bug)* ω → frame conversion **without wrapping**, silently discarding **half a 360° scan** | Read as a *physics* limit on completeness |
| R6 | **"Silhouettes are merged blobs"** diagnosis | Silhouette area grows as frames^0.82 *within* a single grain while correlating with local crowding at **+0.002**. The growth was a frame-count effect, not merging |
| R7 | Two verdicts on SIRT | Both drawn from R4 and R5. On per-frame views the same estimator separates **15×** from a null and is the one adopted |

**On the intragranular-field side**, separately: an intensity-based claim that the topograph
depth *is* the 3-D shape, and a claim of no extinction saturation, were both retracted; a
reported advantage over controls was found to be 5–9× inflated; and an apparent intensity
drift attributed to edge blur turned out to be a background over-subtraction.

## §2 WITHDRAWN mechanism, surviving measurement

**"The ω-summed patch is smeared because the spot travels across the detector."** Measured
centroid displacement is **27 px**, not the ~107 px claimed, and the net/path ratio (0.572,
falling to 0.216 for long spots) says the spot **wanders about a fixed point** — the mosaic
picture, in which an ω-sum is a reasonable approximation to the grain's projection after all.

The measurements it was invoked to explain all stood; the explanation did not. **Treat causal
accounts as provisional even when the numbers they explain are solid.**

## §3 ESTABLISHED — survived a null or an exact cross-check

**DCT**

* **Friedel pairing works.** 29 295 pairs from 71 554 spots (82 %). Paired ring radii are sharp
  (widths 0.9–2.9 px) where unpaired radii are broad — the position blur cancels exactly, as
  the algebra says.
* **fcc**, 5 rings, 2 free parameters, **0.91 px rms**; `s = λ/2a = 0.037257`;
  Lsd 6.775 mm (6.725–6.826 at Δχ² < 9); effective pixel **1.653 µm**; rotation-axis column
  1016.53.
* **Indexing is above chance.** 488 seeds with ring-1 seeding, 814 with all-ring; ω-scrambled
  null indexed **0**. Internal angle 0.33°. Tolerance set *from the null*: 0.52° on ring 1 with
  a 0.09 minimum-match fraction, above the null's maximum completeness of 0.069.
* **Misorientation verified** against `midas_stress.misorientation` to **5.7e-14°**, after the
  symmetry-frame fix.
* **Shapes:** SIRT cores separate **15.0×** in volume from a spot-swap null; 12.2×–16.4× across
  thresholds 0.05–0.125 with no cliff. External phantom validation: SIRT **91.4 %** vs hull
  77.7 % vs no-shape floor 56.3 %.
* **The coverage ceiling is physical.** Ray-direction scatter of ~0.14–0.17° reproduces the
  observed 44 % fragmentation; the *coherent* (refinable) part is 0.031° against 0.173°
  incoherent, ratio 0.18. Refining orientations cannot fix it.
* **220 rediscoveries** by pair-free voxel indexing — an independent route re-deriving known
  orientations from unassigned spots. The strongest validation in the reconstruction.

**TT**

* **The geometry chain reproduces 74 real goniometer settings**: median 0.043°/0.050° against
  a 25–40° random-grain null; discrimination 985×/526×; 0 of 200 null draws beat the truth.
* **Tilt sibling branch is unreachable**: minimum `|up|` of the sibling is **90.0°** over
  18 810 grain × reflection cases.
* **Field is determined, not merely reproducible**: disjoint halves 0.856 / 0.853 with
  cross-half NCC **+0.903**.
* **The domain matters less than it feels**: wrong support scores 0.810 against 0.860, fields
  agreeing +0.940 on 79 % overlap.
* **Resolution window 1.2–2.0 µm**, measured against a polynomial ceiling.
* **z-folding factor 1/√(1−2/π) = 1.6589** is an analytic identity, not an empirical constant.

## §4 KNOWN LIMITS that must NOT be upgraded

* **λ and `a` are individually undetermined** — only `λ/2a` is measured. The grain map does not
  depend on the split; absolute d-spacings and any strain do.
* **Overall handedness is undecidable from a single scan.** Only `y_sign × ω_sign` is fixed;
  flipping both mirrors every grain and changes no residual. A config file from a *different*
  acquisition on the same instrument is suggestive and is **not** evidence.
* **No absorption tomography** when the sample never clears the beam: projections are
  truncated, and FBP gives the classic truncation wedge.
* **One ring unexplained by fcc** (flat in η, ~1000 net pairs, sitting between two indexed
  rings). A second phase is not excluded.
* **86 % of the published map is dilation**, and **~22 %** of the domain is uncontested at any
  threshold.
* **0.17° is an upper bound** on intragranular mosaic spread, because the incoherent term also
  contains per-spot assignment error.
* **The TT tilt envelope** attributed to a campaign is the range it *used*, not provably its
  hardware limit.
* **Absolute strain has not been demonstrated on real data** by the TT pipeline; only `c/a` is
  identifiable from goniometer tilts.

## §5 DEFECTS IN DEPENDENCIES — found here

* **`midas_stress.rodrigues_to_orient_mat` returned the wrong rotation angle, below
  midas-stress 0.9.0.** It gave a proper rotation (det 1, orthogonal to 1e-16) about the
  **exactly correct axis**, but with the angle inflated by `1/cos²(θ/2)`: 5°→5.010°,
  30°→32.154°, 60°→**80°**, 90°→**180°**. Both numpy and torch backends, identically. Cause:
  the quaternion vector part was built as `rod/cos(θ/2)` where it should be `n·sin(θ/2)`.
  Its own tests missed it because they checked the identity (where the bug vanishes),
  structural properties the bug preserves, and numpy-vs-torch parity (both wrong the same
  way). There was no inverse function, so no round-trip masked it.
  **Fixed in midas-stress 0.9.0**, which now agrees with
  `midas_dct_tt.rodrigues_to_crystal_to_sample` to 2.7e-15.
  Two things survive the fix. First, **check your installed version** — substituting the old
  converter moved the 74-scan tilt residual from 0.043° to **26.5°**, indistinguishable from
  assigning scans to random grains, and it is silent at small angles (5°→5.010°). Second, the
  **convention** question is separate from the defect: grain maps written by the common Python
  microstructure toolchain store the negated Rodrigues convention, so keep using
  `midas_dct_tt.rodrigues_to_crystal_to_sample` to read them.
  *Lesson worth more than the bug: a test suite can check determinant, orthogonality and
  backend parity and still never assert the quantity the function exists to produce.*

* **Threshold counts at a saturation point are rounding censuses.** Counting `γ ≥ 90°` exactly
  counts how many dot products round to *bit-exact* zero (`acos(0.0)` is exactly 90;
  `acos(1e-17)` is a hair under). Two algebraically identical reciprocal bases gave **39** and
  **34** of 55 under a bare `>=`, but **40** and **42** under a 1e-9 tolerance. Always compare
  against a threshold with a tolerance.

## §6 THINGS THAT COST TIME — do not repeat

* `IndexBest.bin` **col 13 = `n_t_spots`, col 14 = `n_matches`** — the module docstring says the
  opposite; `_seed_record` is authoritative. Reading col 13 makes every seed look perfect.
* `midas-index` treats `OutputFolder` as the directory **holding** `Spots.bin`, not its parent.
* **Misorientation symmetry belongs on the right**: `Uaᵀ Ub S`. `S` in the middle reported
  29.8° for a pair 0.33° apart and left 367 clusters where there were 205.
* **Grain position is fixed in the SAMPLE frame.** Adding `Rz(σω)` cut the residual 52 → 41 µm.
* **One thread per worker.** 20 workers × 64 threads finished **zero** grains in 32 min; with
  one thread each, 121 finished in ~3 min.
* **Otsu, not `0.5 × max`.** Streaks made 60 µm grains read as 6 µm.
* **MIDAS FF margins are µm against the ring radius** — at an 880 µm radius, 150 µm is ~10°.
* **Feed the indexer virtual Friedel spots, not raw spots** — rings 84 px apart while position
  moves a spot up to 150 px.
* **Grain-map references erode.** A reconstruction ~30 % larger than a reference grain was
  *correct*; growing the reference by 2 voxels improved agreement.
