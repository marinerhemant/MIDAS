# Instrument — conventions that cannot be recovered from a finished reconstruction

**Read before touching a new instrument's data.** Everything here is a property of the
acquisition that leaves no trace in the reconstruction: get it wrong and you get a complete,
plausible, wrong answer with no residual to warn you.

## §1 The five conventions

| Convention | Why it cannot be recovered later | How to pin it |
|---|---|---|
| **Effective pixel** | Scales every length linearly and consistently, so nothing looks wrong | A known physical length in the field of view, on **two independent axes** |
| **ω sign** | Mirrors the reconstruction; residuals are identical | Documented stage sense. `DCT_OMEGA_SIGN_CCW = +1`, `DCT_OMEGA_SIGN_AERO = −1` |
| **Detector handedness / flip** | Mirrors the reconstruction; residuals are identical | A flip flag for *this* pipeline, or an external chiral fiducial |
| **Dark already subtracted?** | Double subtraction suppresses weak spots and reads as a detection limit | Compare frame median to dark median |
| **Per-frame ω** | A plausible wrong ω gives a plausible wrong map | The scan command, not the motor block (§3) |

Only the **product** `y_sign × ω_sign` is determined by the diffraction itself. Flipping both
mirrors every grain and changes nothing measurable. Treat the individual signs as external
information.

## §2 The pixel-size trap, twice

The header pixel is the **sensor** pixel. With any magnifying optic the imaging pixel differs
and the header does not know it.

* One DCT dataset: header value was **6.65×** the true effective pixel.
* One TT dataset: the metadata file disagreed with the camera actually in use by **2×**, and
  the correct value was only in the associated publication.

**Check the instrument documentation, not just the header**, and confirm against a known
length. Two axes agreeing to a couple of per cent is the standard.

## §3 ω may not be recorded at all

On one TT set the rotation motor read a **single constant value in all 2880 frames**, and the
per-frame angle list in the configuration file was **empty**. The angles were never recorded.

Motor blocks in frame headers can be a **static snapshot** written once at scan start. Read
the value from several frames spread across the series before trusting it; if it never
changes, fall back to the scan command (start, step, count), which is usually the only honest
source.

## §4 Detector distance

Fit it jointly with the lattice rather than reading it. Quote the confidence interval — here
6.775 mm with 6.725–6.826 at Δχ² < 9 — and quote rings used and free parameters, because a
small-angle-limit fit can determine only the *product* of scale and distance while reporting
confident individual values.

## §5 Truncation: does the sample clear the beam?

If the sample never fully clears the beam at any ω, projections are truncated:

* **No absorption tomography** is obtainable — no sample outline from this scan.
* FBP produces the classic truncation wedge, which is an artefact and not a feature of the
  specimen.
* Any domain boundary is **inferred**, not measured, and must be labelled so.

Check for a flat and a dark too; one dataset here had neither.

## §6 Reference grain maps

Useful as a format reference and, carefully, as ground truth:

* **DCT erodes grain boundaries.** A reconstruction came out ~30 % larger than a reference
  grain, and growing the **reference** by 2 voxels improved agreement. Do not "correct" the
  reconstruction toward the reference.
* **Rodrigues vectors carry a convention.** Maps written by the common Python microstructure
  toolchain use the negated convention. Read them with
  `midas_dct_tt.rodrigues_to_crystal_to_sample`. Note also that **below midas-stress 0.9.0**
  `rodrigues_to_orient_mat` returned the right axis at the wrong angle (60°→80°, 90°→180°);
  fixed in 0.9.0, but check what you have installed.
* A reference map from a **different specimen** is a format reference only, never ground truth.

## §7 Open-data archives

Several facilities publish DCT and topotomography raw data under open licences, some
retrievable anonymously over a catalogue API. Two operational notes:

* Datasets may be **on tape**; a request returns an error plus an automatic restore, so plan
  for latency on anything not marked online.
* A fetch helper lives in `packages/midas_dct_tt/dev/data_hunt/`. Record the dataset
  identifier and licence in your provenance file at download time — reconstructing it
  afterwards is far harder than writing it down.

## §8 Compute

* **One thread per worker.** `torch.set_num_threads(1)` and `OMP_NUM_THREADS=1`. Without it,
  20 workers × 64 threads finished zero grains in 32 minutes; with it, 121 in ~3 min.
* **Launch long jobs detached** with output redirected to a log, or an SSH hangup kills them.
* **Keep one source of truth** for scripts and sync outward before each run; copy results
  back. Divergent copies on compute hosts is how two versions of an analysis start disagreeing
  quietly.
* GPU work here: pair-free voxel indexing scanned 40 000 orientations per voxel in **164 ms**.
