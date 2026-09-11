# Phase 6 — Model the G(r)

> Part of the **pdf doc set**. Spine: [`README.md`](README.md).
> The code is lifted from `$ANALYSIS/scripts/06a_smallbox.py`, `09a/09b`, `10a/10b`, `11a_rmc.py`
> and `12a_bayes.py`, which ran on the reference Ni and CeO2 G(r), with names made generic.

**Run §5's physical checks first.** If they failed, every number below is conditional on the
reduction. Say so next to the number.

## §6.1 Small box

```python
import math, numpy as np, torch
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_pdf import build_pair_list, pdffit_gr, refine_structure

def cubic(name, a, atoms, sg=225):
    return Crystal(lattice=Lattice(a, a, a, 90, 90, 90), space_group=SpaceGroup.from_number(sg),
                   atoms=[Atom(element=e, fract=f, B_iso=0.0) for e, f in atoms], name=name).to_torch()

ct = cubic("Ni", 3.524, [("Ni", (0, 0, 0))])
sel = (r >= R_MIN) & (r <= R_MAX)                       # refine_structure has no r-range argument
r_t, G_t, s_t = (torch.as_tensor(x[sel]) for x in (r, G, sigma_G))
pairs = build_pair_list(ct, r_max=R_MAX + 1.0)
Gm = pdffit_gr(ct, r_t, pairs, scale=1.0, u_iso=0.005)
s0 = float((G_t * Gm).sum() / (Gm * Gm).sum())          # a linear start for the scale
res = refine_structure(ct, r_t, G_t, pairs, sigma_obs=s_t, init_a=3.524, init_u_iso=0.005,
                       init_scale=s0, bg_order=0, steps=300)
a, u_iso = float(res.fitted["a"]), float(res.fitted["u_iso"])
```

**What it refines:** cubic a (a = b = c), **one** U_iso shared by every site, a scale, and
optionally a polynomial background.

**What it does not model:** δ1/δ2, Qdamp, Qbroad, Q_max termination, the Lorch window, per-site
ADPs. For per-site ADPs use `midas_pdf.aniso_refine.refine_aniso_occupancy`.

**Consequences:**
- Fit the unwindowed G(r) for U_iso, or label it window-inflated (hard rule 8).
- Memory scales with points × pairs: CeO2 to 16 Å is ~15.6k pairs. Keep R_MAX modest, or
  subsample r.
- Reference values, Ni (r 1.5–15 Å, Lorch): a 3.524673 Å, U_iso 0.00931 Å², χ²_ν 4787. The same
  fit on the unwindowed G(r) gives U_iso 0.00506.

## §6.2 The uncertainty — a fixed recipe, not the Hessian

```python
n_eff = (R_MAX - R_MIN) * Q_MAX / math.pi               # Fourier-correlated points
sigma_stat = float(res.uncertainty["a"]) * math.sqrt(float(res.chi2_reduced)) * math.sqrt(sel.sum() / n_eff)
sigma_rscale = RING_MEAN_D_RMS_UE * 1e-6 * a            # the calibration's own residual (§3)
sigma_chain = abs(A_CALIBRANT_FIT / A_CALIBRANT - 1) * a  # the calibrant refined through this chain
sigma_choice = 0.5 * (max(a_arms) - min(a_arms))        # {FT Q_max} x {window} x {bg order}
sigma_total = math.sqrt(sigma_stat**2 + sigma_rscale**2 + sigma_chain**2 + sigma_choice**2)
```

On Ni the terms were 4.5e-4, 1.8e-4, 3.9e-3 and 1.8e-4 Å, giving σ_total 3.9e-3 Å against a
raw Hessian σ of 1.7e-6 (Lab Notebook §7).

`sigma_chain` carries the calibrant's positional bias through the same chain. On CeO2 that was
−1107 ppm, and it may not transfer to a cleaner sample. **Report σ_total with and without it,
never only the smaller one.** Here, without it, σ_total was 5.1e-4 Å.

## §6.3 Is a second phase there? Raw data first

**Do this before any multiphase or core-shell fit.**

1. Take the 1-D I(Q) of the sample (§5.1).
2. At each Q0 over a range free of known peaks, fit (quadratic background + Gaussian of fixed
   width centred at Q0) over ±0.15 Å⁻¹. The width comes from a parent-phase peak. The
   Gaussian's coefficient is A(Q0).
3. The null is the robust std (1.4826 × MAD) of A(Q0) away from the candidate's peak.
   z = A / null.
4. Controls: a planted Gaussian of amplitude 5 × null std must come back at z ≈ 5, and the
   parent phase's own peak must be enormous.

**Present** means z > 5 at the candidate position ± 0.01 Å⁻¹. **Absent** means z < 3. Repeat at
3× the width for a thin, broadened shell.

On the Ni sample, NiO(111) at 2.605 gave z −0.8, the plant 5.1 and Ni(111) 3419
(`$ANALYSIS/scripts/10a_nio_raw.py`).

## §6.4 Multiphase — with a decoy, read as an amplitude fraction

```python
from midas_pdf.multi_phase import refine_multi_phase

ce = cubic("CeO2", 5.4116, [("Ce", (0, 0, 0)), ("O", (0.25, 0.25, 0.25))])
p_ce = build_pair_list(ce, r_max=R_MAX + 1.0)
mp = refine_multi_phase([ct, ce], r_t, G_t, [pairs, p_ce], sigma_obs=s_t, init_a=[3.524, 5.4116],
                        init_u_iso=[0.005, 0.005], init_scale=[s0, s0], init_weights=[0.5, 0.5],
                        steps=200, lr=0.05)
w = mp.weights_normalised
amp = [w[i] * mp.fitted[f"scale_{i}"] for i in range(2)]
f_decoy = amp[1] / sum(amp)                              # the only identifiable "fraction"
chi2_gain = 1 - mp.chi2_reduced * (N - 8) / (single.chi2_reduced * (N - 3))
```

- **Weights and scales are degenerate.** Weight σ is NaN (hard rule 9).
- **Run a decoy that is known to be absent, from two starts.** Then repeat with the decoy's
  structure **fixed**: add its pdffit_gr peaks (baseline −4πρ₀r removed) with one linear
  amplitude c, and compute z = c / (σ_c · √χ²_ν · √(N/N_eff)).
- **A free decoy whose lattice constant runs away is absorbing misfit, not detecting a phase.**
  On Ni the free decoy took 3–7 % off χ² at a 5.35–5.97 Å; the fixed one gave 0.3 ± 0.4 %.

## §6.5 Core-shell — only as a null, and never trust its zeros

```python
from midas_pdf.multi_phase import refine_core_shell
cs = refine_core_shell(core_ct, shell_ct, r_t, G_t, core_pairs, shell_pairs, sigma_obs=s_t,
                       init_a_core=A_CORE, init_a_shell=A_SHELL, init_scale_core=s0, init_scale_shell=s0)
cs.volume_fractions, cs.fitted, cs.uncertainty
```

- **An uncertainty of 0.0 is a degenerate Hessian**, and `--pin-geometry` on the CLI does
  nothing (hard rule 17).
- **Read the shell's U_iso before its fraction.** On Ni with a NiO shell and no NiO in the data:
  shell fraction 0.36, U_iso 25 Å², χ² −47 %. That is a smooth term, not an oxide.

## §6.6 strain-PDF — read the Fisher eigenbasis, not `recover_strain`

```python
import midas_pdf.strain_pdf as sp
# Y: (D, R) wedge G(r), each through §5 with its own eta wedge, then a linear scale/offset onto the
# isotropic model; a, u_iso, scale from §6.1 on the full-azimuth G(r), background off.
qhats = sp.probe_directions([-eta for eta in WEDGE_ETAS_MIDAS], theta_deg=0.0)   # package eta = -MIDAS eta
f = lambda e6: sp.sliced_gr_stack(ct, r_t, pairs, qhats, e6, u_iso=u_iso, scale=scale,
                                  kernel_m=8, n_quad=96).reshape(-1)
zero = torch.zeros(6, dtype=torch.float64)
J = torch.func.jacfwd(f)(zero)                                   # (D*R, 6)
d = torch.as_tensor(Y).reshape(-1) - f(zero)
evals, evecs = torch.linalg.eigh(J.T @ J)
keep = evals > evals.max() * 1e-6
estimate = (evecs.T @ (J.T @ d)) / evals                         # per eigen-direction
sigma = SIGMA_G / torch.sqrt(evals)                              # x sqrt(N / N_eff) as well
# in-plane tensor only (one frame): least squares on J[:, 1:4] (e22, e33, e23)
```

- **From one frame, e11, e12 and e13 are blind.** `strain_crlb` then marks all six components
  infinite, and `recover_strain` returns percent-level numbers in the blind directions
  (hard rule 11).
- **Wedges near the vertical may not reach Q_max** on an edge-centred beam: η 15° and 165° did
  not.
- Reference, unloaded Ni: e22 +69 ± 8, e33 −123 ± 15, e23 −25 ± 6 µε (raw CRLB; ×3.87 for
  √(N/N_eff)).

## §6.7 RMC

```python
from midas_pdf.rmc.supercell import Supercell
from midas_pdf.rmc.ensemble import rmc_refine_ensemble

sc = Supercell.from_crystal(ct, size=(4, 4, 4))
sc.species = ["Ni"] * int(sc.positions.shape[0])                # from_crystal labels them "X"
sel6 = (r >= 1.5) & (r <= 6.5)                                   # below half the box
# no scale in the RMC chi2: divide by a small-box scale fitted on the same r range
G_target = G_t6 / scale6
sig_target = s_t6 / scale6 * math.sqrt(chi2nu6)
ens = rmc_refine_ensemble(sc, r_t6, G_target, n_chains=4, sigma_G=sig_target, n_moves=10000,
                          u_iso=0.001, temperature=1.0, min_distance_A=2.0, initial_jitter_A=0.05, seed=0)
[c.final_chi2 for c in ens.chains]; ens.chains[0].supercell
```

- **No X-ray weighting.** The forward model is an unweighted pair sum, so it is not an X-ray
  G(r) for more than one element.
- **Cost.** Every move rebuilds the O(N²) pair list; CPU only, chains run sequentially.
- **Set the kernel `u_iso` small** so the configuration carries the disorder. Then compare the
  first-shell width, sqrt(var + 2·u_kernel), with √(2·U_iso) from §6.1.
- **Correct the first-shell mean before comparing it with a/√2.** Under disorder it sits above
  the static distance by about s²/d, where s is the spread of the shell distances. On Ni that
  was +0.006 Å with s 0.117 Å. In a box fixed at the small-box a, that offset is all the
  position can show; it is not a second measurement of a.
- **Cost:** 48 min for 4 chains × 10 000 moves at 256 atoms, chains run one after another
  (Lab Notebook §11).

## §6.8 Bayesian posterior and model ranking

```python
from midas_pdf.bayesian_refine import bayesian_refine_svi, bayesian_refine_nuts
from midas_pdf.model_comparison import waic, loo, compare_models

s_eff = s_t * math.sqrt(float(res.chi2_reduced))                 # likelihood sigma at the residual level
svi = bayesian_refine_svi(ct, r_t, G_t, pairs, sigma_obs=s_eff,
                          map_init={k: float(res.fitted[k]) for k in ("a", "u_iso", "scale")},
                          bg_order=0, n_steps=2000, lr=5e-3, n_posterior_samples=500)
svi.summary()                                                    # mean, std, q05, q95 per parameter
ic = waic(G_t, s_eff, svi.G_samples)                             # repeat per model, then compare_models
```

- **The likelihood is independent Normal per r point.** WAIC/LOO SEs are √(N·var), so inflate
  any Δ-IC z by √(N/N_eff) before calling a ranking decisive.
- **Priors.** a ~ Normal(init, 0.02). The docstring says HalfNormal for u_iso and scale; the code
  uses LogNormal (`packages/midas_pdf/midas_pdf/bayesian_refine.py:81-82` against `:113-119`).
  `prior_widths` for u_iso and scale is ignored.
- **Pass `map_init` for anything but Ni.** The a prior otherwise centres on 3.524 Å.
- **Check each model's posterior mean against that model's own MAP.** On Ni, SVI seeded from the
  bg-0 MAP landed −1.79σ from it (bg 0) and +0.48σ from the bg-None MAP, but **+4.13σ** from the
  bg-2 MAP. Seed each model from its own MAP. Whether that closes the gap is untested (DIAGNOSIS).
- **Cost.** On the reference Ni G(r) (r 1.5–15 Å, 2 threads, other fits running), SVI took
  7.5–13 min per model and NUTS (200 warmup + 500 samples) took 59 min.
- **Prefer NUTS for the posterior you report.** On Ni, NUTS sat on the MAP (\|z\| ≤ 0.06) with
  σ(a) 1.20e-4 Å, within 3 % of the Hessian σ × √χ²_ν. SVI's mean-field posterior was 1.7×
  wider, and its mean sat 3.1 NUTS-σ away.
- **NUTS with `bg_order` set crashes twice before it runs.**
  - A float32/float64 mismatch: call `torch.set_default_dtype(torch.float64)` first.
  - A `KeyError` on `bg_0`: put the MAP's background coefficients into `map_init` as `bg_0`, … .

  See DIAGNOSIS.
- Result: Lab Notebook §12.

## §6.9 CIF and the `midas-pdf-*` CLIs

- **CIF.** `midas_pdf.cif.write_crystal_to_cif` / `read_cif_to_crystal` round-trip a, space
  group, atoms, occupancy and B_iso. **`_atom_site_U_iso_or_equiv` is dropped on read**: convert
  to B (= 8π²U) first.
- **`midas-pdf-refine`, `-multiphase`, `-coreshell`, `-rmc`** read an ASCII `r G [σ]` file and
  apply no correction.
  - Without a σ column they invent σ = 5 % of |G|max.
  - `-refine` reports the raw Hessian σ.
  - `-coreshell --pin-geometry` does nothing.
  - `-rmc` writes `X` atoms.
- **`midas-pdf-joint`** needs SAXS data; not exercised.
