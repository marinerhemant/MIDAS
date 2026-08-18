# Phase 2 — extraction: background, vetting, area and centroid

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Goal:** for every ring and every azimuth, an integrated **area**, an intensity-weighted
**centroid**, and a **per-azimuth SNR** — plus the two numbers that decide what you can
promise: peak-to-background and the singlet verdict.

Everything here is `midas_dt.azimuthal`.

---

## 2.0 The order matters

```
1. choose windows          in PIXELS, bounded by the gap to the neighbour
2. mask the known rings    widened, so the background never sees a peak
3. estimate the background from ring-FREE radii, PER AZIMUTH, interpolated across
4. vet for singlets        count maxima with a PROMINENCE criterion
5. area + centroid         on the subtracted net
6. per-azimuth SNR         and gate on it per bin, never on the ring median
7. MAD-reject outliers     Mürer's 3-MAD, to suppress large crystallites
```

Steps 2–3 before 4–5 is not stylistic: the multiplet test only works on the *subtracted*
profile, and at 1.17× contrast the un-subtracted profile is 85 % background.

## 2.1 Windows, in pixels

```python
idx, half = ring_windows(r_axis, centres_px, max_half_px=16.0, gap_frac=0.45)
```

* **In pixels, converted to bins.** A cake at 0.25 px/bin makes a bin-specified window ~4×
  too narrow, and the part lost is the *tails* — which is where area lives.
* **Bounded by the gap** to the nearest neighbour (`gap_frac`), so a window never reaches onto
  another ring.
* **Never runs off the axis**, which matters for the innermost and outermost rings.

Set `max_half_px` from the measured FWHM (phase 0), roughly 2× FWHM. Too narrow truncates the
tails; too wide imports background.

## 2.2 Background — the dominant error term, not a refinement

```python
mask = ring_free_mask(n_r, idx, half, widen=1.6)
net, bg = background_from_ring_free(cake, mask, block_bins=30)
```

**Three properties, each of which fixes a specific measured failure:**

1. **From ring-free radii, interpolated across the peaks.** A rolling low percentile over
   blocks comparable to the peak width sits on the peak *flank*, biasing the background **up**
   exactly where the peak is. An 8-px FWHM peak is ~27 % of a 30-px block. At 1 % contrast a
   1 % background error is a **~20 % area error**.
2. **Per azimuth.** The background varies with **both** R and η — Compton from anvils or a
   furnace falls with angle, absorption varies with path length. A radial-only background
   leaves an η pattern that looks exactly like texture, and that produced a conclusion which
   had to be withdrawn (`LAB_NOTEBOOK.md` §5g).
3. **`widen=1.6`.** The tails carry area, so a background estimated from radii that still hold
   tail is biased up.

**`net` is not clipped at zero.** Clipping would bias the centroid, and clipping is the
caller's decision. The moments in `area_and_centroid` clip internally, which is exactly why the
centroid is robust and the area is not.

**Do not use `rings.rolling_baseline` here.** It is for ring *finding*, where the flank bias
does not matter.

## 2.3 Vet for singlets

```python
n_max = count_maxima(net_window.mean(axis=1), min_frac=0.5, min_prominence=0.10)
```

**Multiplets are reported and EXCLUDED, not fitted.** A sub-pixel ring assignment matches a
doublet as one line: hcp Ti α(101) held maxima at 381.6 and 393.6 px with a dip between, and
was fitted as a single ring at 393.6 px through four analyses.

**Height alone is not enough.** Photon noise on a peak's own plateau readily gives several
samples that each clear half height and each exceed their two neighbours — so a height-only
count calls a clean singlet a *triplet* and the ring gets rejected for a multiplet it is not.
A genuine doublet is defined by the **dip**: the Ti doublet's dip sits ~60 % below its lobes,
while noise ripple is a few percent. The gap is wide, so the prominence criterion is robust
rather than tuned.

## 2.4 Area, centroid, and the asymmetry between them

```python
ext = extract_ring(cake, net, bg, r_axis, idx[c], half[c])
# ext.area, ext.centroid, ext.snr, ext.n_maxima, ext.contrast
```

| Quantity | Arithmetic | Feeds | Robustness |
|---|---|---|---|
| `area` | **difference** of large numbers | texture (phase 4) | inherits the background model's error in full |
| `centroid` | **ratio** | strain (phase 3) | a slowly varying background largely cancels |

Measured at 2 % contrast with no planted azimuthal structure: **area 36 %, centroid 0.85 %**.
This is the fact that decides the deliverable.

## 2.5 The moving-peak trap

Under differential stress the peak *position* varies with azimuth:

```
d(ψ) = d₀ [1 + (1 − 3cos²ψ) Q(hkl)]        (Singh's lattice-strain relation)
```

In a DAC this is **physics, not an artefact** — it is how differential stress is measured. But
a fixed radial window converts that movement into azimuth-dependent **intensity**: fake
texture.

```python
c = radial_half_correlation(net[lo:hi])
```

**The sign is the discriminator:**

| value | meaning |
|---|---|
| strongly **negative** | the peak is **moving** — intensity leaves one radial half as it enters the other. Azimuthal "texture" from a fixed window is largely truncation |
| **positive** | the **amplitude** is changing while the position holds — what a genuine pole figure looks like |
| near zero | neither dominates, or the ring is noise |

Measured **−0.72** on the 11-ID-C CeO₂ scan — a standard that should have no texture at all —
which is what identified peak movement as the cause of a spurious structured ODF.

Verified on synthetics with Poisson noise: a 6-px azimuthal shift gives −0.40, a 40 %
amplitude modulation gives +0.59. The separation is by sign, with a wide margin.

**Why this and not an edge-occupancy test.** An earlier version compared the window's edge level
against a fraction of the peak. That cannot work at realistic contrast: at 26 % contrast on a
~200-count background the 3σ floor on an edge mean is ~21 counts against a peak amplitude of
~45, so a perfectly stationary ring scored 0.05 and the statistic was measuring noise. Summing
each **half** averages over many bins, making this a ratio of large numbers — which survives the
same contrast where the edge test collapses. (Same argument as area-vs-centroid, §2.4.)

If the correlation is negative: widen the window, or move to peak-fitted areas. And read the
movement itself as the signal — that is your strain.

## 2.6 Gate on per-azimuth SNR

```python
snr = snr_per_eta(cake[lo:hi], net[lo:hi])      # (n_eta,)
live = ext.live_mask(min_snr=3.0)               # per-azimuth, USE THIS
usable = ext.usable(min_snr=3.0)                # ring-level: singlet AND median SNR
```

**Gate per bin, never on the ring's median.** The median lets dead azimuths through, and any
peak-to-peak spread over η is a max-minus-min that a handful of them dominate. Measured
consequence: one reflection reported **107,410 µε** and another **638,282 µε** (11 % and 64 %
strain) purely from this.

Why the gate is decisive rather than cosmetic: at ~220 counts of background a bin carries ~15
counts of Poisson noise, while a peak at `peak/bg = 0.02` has an amplitude of ~4 counts —
**below** the per-bin noise. Signal only emerges after integrating the window, and then only
for the strongest rings.

## 2.7 Reject azimuthal outliers

```python
keep = mad_filter(area, cut=3.0)
binned = azimuthal_rebin(area, factor, keep=keep)
```

Mürer et al. 2021's 3-MAD rejection, to suppress a few large crystallites making a ring spotty.
A spot is not a pole figure.

**Known behaviour worth understanding before you test it:** with identical values the MAD is
exactly zero, so every deviation is infinitely many MADs. The filter deliberately passes
everything through in that case rather than emptying the window. Synthetic tests built on a
noiseless baseline conclude the filter is broken; it is not.

## 2.8 The gate before phase 3 and phase 4

Report this table. It **is** the decision:

```
ring        half-win(px)  n_maxima  peak/bg   median SNR/eta   live eta   verdict
alpha(100)         16.0          1    0.170            14.9      48/50    SINGLET
alpha(101)         12.0          2    0.145             9.1      44/50    multiplet -> DROP
...
```

Then:

* **Strain (phase 3)** — proceed for every vetted singlet. It works down to ~1 % contrast.
* **Texture (phase 4)** — check `ENVELOPE.md` §0 against the measured peak/background. Below
  ~0.05, do not proceed to a per-voxel map; a global texture may still be reachable and the
  ladder will tell you.

## 2.9 What to hand forward

```
Per ring:  centre bin, half-width (px), n_maxima, peak/bg, median + per-bin SNR,
           live azimuth count, radial half-correlation (SIGN)
Vetted set: which rings passed, which were dropped and WHY
Arrays:    area, centroid, snr, per (translation, omega, eta, ring)
Decision:  strain only | strain + global texture | strain + per-voxel texture
```

Then `phase-3-strain.md` (always) and `phase-4-texture.md` (only if the contrast allows).
