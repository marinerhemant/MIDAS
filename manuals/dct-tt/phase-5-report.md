# Phase 5 — report

**Goal:** a statement a stranger can check, in which every number can be re-derived and every
caveat survives into the text that leaves the session.

## 5.1 Every number carries its provenance

For each quantity, record the **file** and the **command** that produced it. Read numbers from
source when writing, never from working memory — the numbers in a conversation drift, and a
figure that was provisional three steps ago becomes a fact by repetition.

A minimal provenance table:

| Quantity | Value | Produced by |
|---|---|---|
| effective pixel | 1.653 µm | slit-box measurement, two axes, agree to 1.4 % |
| lattice, `λ/2a` | fcc, 0.037257 | 5-ring fit, 2 free parameters, 0.91 px rms |
| indexed grains | 862 | index + verification, ω-scrambled null = 0 |
| shape null separation | 15.0× | spot-swap null |
| uncontested volume | ~22 % | contested-voxel count at four thresholds |

## 5.2 The labels that must survive

These are the ones that get quietly upgraded between the notes and the report:

| Say | Not |
|---|---|
| "orientations and positions for 862 grains" | "862 grains reconstructed" |
| "86 % of the labelled volume is dilation" | *(silence)* |
| "~22 % of the domain is uncontested at any threshold" | "the map fills the domain" |
| "`λ/2a` measured; λ and `a` not separated" | "the material is X" |
| "mirror-ambiguous: only `y_sign × ω_sign` is fixed" | *(silence)* |
| "the well-determined rotation components" | "the rotation tensor" (if γ < 60°) |
| "recovery exceeds a polynomial ceiling over 1.2–2.0 µm" | "the intragranular field" |
| "0.17° is an upper bound on mosaic spread" | "the mosaic spread is 0.17°" |
| "the tilt envelope the campaign used" | "the stage limit" |

**The rule:** the status a result has in the working notes is the status it has in the
document you send. A hedge that disappears at the last step is the most expensive kind of
error, because nobody downstream can see it was ever hedged.

## 5.3 What a shape claim must carry

* the estimator and its **null separation** (spot-swap or ω-scramble), not just an image
* evidence the threshold sits in a **range** without a cliff, rather than being tuned
* the **no-shape floor** for comparison — nearest-grain-centre assignment scored 56.3 % here,
  so a shape method must beat 56 %, not beat zero
* the count of grains **without** a validated shape, reported as position+orientation only

## 5.4 What a field claim must carry

* the reflection pair's **separation γ** and the sensitivity eigenvalues
* a **held-out** score, plus shuffled-training and constant-field baselines
* the **wrong-support control**
* the **transfer function** — the length scale above which a polynomial does better
* a statement on registration, since jitter manufactures amplitude

## 5.5 Figures

Keep the generator with the figure, checked in the same day. A figure whose script has drifted
from the numbers in the text is unverifiable, and re-deriving it later costs more than writing
it down now. Where a figure reads from a log rather than recomputing, say so — the log is the
provenance, and re-running a long sweep to redraw a plot invites the two to diverge.

## 5.6 A closing check

Before sending, re-read the report against this doc set's `ENVELOPE.md` and ask: **is anything
claimed here outside what the measurement can determine?** The common ones are an absolute
strain without the material, a tensor from a poorly conditioned pair, a space-filling map, and
a smooth field with no resolution bound.

If you halted at a gate, say which one, what you measured, and what would unblock it. A halt
is a result.
