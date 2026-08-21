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

### ★ Centre the window on the PEAK, not on the catalogue

`ring_windows` places each window at the bin nearest the position **you gave it** — normally a
catalogued or previously-assigned ring radius. If that is off the real peak, the window is
asymmetric about the peak, and then any change in peak **width** between measurements is
converted into an apparent change in **centroid**, i.e. into strain that is not there.

```python
idx = refine_ring_centres(net, r_axis, idx, half)     # measured peak, not catalogue
# or, per ring:
ext = extract_ring(cake, net, bg, r_axis, idx[c], half[c], recentre=True)
```

`extract_ring` **always** reports `centre_offset_bins` and warns above 10 % of the half-width,
so the defect cannot pass unnoticed; `recentre` defaults to `False` so it never silently moves
an existing analysis's numbers.

**Measured on the DAC Ti S1 scan**, whose centres came from a 2021 assignment: offsets of
−0.563, **+0.044**, **−1.524** and −0.848 px on four rings, the largest 19 % of that ring's
FWHM. Re-centring moved one ring's apparent strain by **55 %** and **flipped another's sign**,
while the ring already centred to +0.044 px was unchanged to the bit — which is what makes the
comparison a controlled test rather than a tweak.

**Do not over-claim it.** For a Gaussian the artefact is second-order at a comfortably wide
window (`half/σ` 5 → 0.02–0.22 bins) and first-order only near the safety line (`half/σ` 2.5 →
0.72–2.38 bins). Where real data moves far more than that, the cause is non-Gaussian content
entering the window — background residual and neighbour tails — not tail truncation. And
because `centre_bin` is an integer, re-centring leaves up to ~0.5 bin of residual; widen the
window rather than chasing it.

**A window-width sweep cannot find this.** The offset is a fixed *fraction* of the window, so
it does not dilute as the window grows — which is exactly why it survived one on this dataset.

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

### ★ This estimator has its own error floor, and on weak rings it may dominate

Interpolating a low percentile *across* the peak leaves a residual whose **absolute** size is set
by the background field's curvature (detector + air + cell) — **not** by how much sample is in the
beam. So as a *fraction* of a ring's area it grows as `1/intensity`, and it is ω-locked,
translation-specific and roughly white in η.

On the 11-ID-C CeO₂ scan this is the leading explanation for a 0.9 % azimuthal floor that survives
any amount of ω averaging: the two **weakest** rings carry the two **largest** floors, and
`corr(floor, 1/ring intensity) = +0.859`. It was mistaken for a sample property for two days.

**Two consequences.** A parameter sweep over `block_bins` and `percentile` tests *sensitivity*,
**not correctness** — an η-structured error common to every setting scores zero spread and looks
excluded. And any azimuthal floor should be checked against `1/intensity` across rings before it
is attributed to the sample.

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

**Three bands, calibrated against planted truth** — not two. A *small* positive is the **null
baseline**, not amplitude variation:

| value | meaning |
|---|---|
| **≲ −0.4** | the peak is **moving** — intensity leaves one radial half as it enters the other |
| **−0.4 … +0.3** | **no coherent azimuthal signal.** The baseline drifts upward with window width |
| **≳ +0.5** | the **amplitude** is genuinely changing while the position holds |

Calibration on synthetics with Poisson noise, where the truth is known
(`~/Desktop/analysis/11idc_ceo2_dt/peakfit/control_subpixel.py`):

| planted | half-correlation |
|---|---|
| pure sub-pixel **shift** | **−0.98 … −0.99** |
| pure **amplitude** modulation (20 %) | **+0.99** |
| **neither** | **+0.02 … +0.24** ← the baseline |

The same control recovers planted sub-pixel shifts at **0.014 px RMSE** down to 0.05 px, so the
statistic is calibrated in position as well as in sign.

Measured **−0.72** on the 11-ID-C CeO₂ scan, reproduced independently at −0.61 to −0.74 on five
rings. **Read CeO₂ (111) at +0.219 and (222) at +0.265 as *no signal*** — they sit inside the
baseline band, not in the amplitude band.

### ★ What a negative value does NOT mean

**It does not mean the windowed AREA is corrupted.** Movement and area corruption are separate
questions, and at any sensible window width the area is *immune*: a planted 0.26 px shift inflates
the area RMS by **1.00×** from 2.3× FWHM out to 16.3× (only 1.37× at 1.7× FWHM), while the
half-correlation reads −0.98 at *every* width. A window wider than ~2× FWHM captures all the
intensity wherever the peak sits inside it, so the sum is invariant while the *distribution within*
the window is not.

So: **do not reach for peak-fitting on the strength of a negative half-correlation alone.** That
inference was made on CeO₂ and refuted on mechanism (`LAB_NOTEBOOK.md` §5b-ter).

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
