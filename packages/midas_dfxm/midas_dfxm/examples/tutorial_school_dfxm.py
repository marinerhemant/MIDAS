# %% [markdown]
# # DFXM with a differentiable digital twin — a hands-on tutorial
#
# **Dark-Field X-ray Microscopy (DFXM)** is a real-space microscope built on a
# *diffracted* beam: an objective lens (a compound refractive lens) images the
# beam Bragg-diffracted by a chosen lattice plane inside a bulk crystal. Rocking
# the crystal through the Bragg condition and reading each pixel's rocking curve
# gives, per pixel:
# - the **center of mass (COM)** -> the local lattice *orientation* (mosaicity),
# - the **width (FWHM)** -> the local *mosaic spread*,
# - a **2-theta shift** -> one component of *strain*.
#
# `midas-dfxm` is a *differentiable* twin of this whole measurement: the forward
# model (optics -> crystal -> detector) and an inverse that recovers the full
# **deformation-gradient tensor F** (all nine components), not just the three
# scalars above.
#
# Run this file cell by cell (VSCode: click "Run Cell" above each `# %%`; Jupyter:
# it is a valid notebook via jupytext). Everything is CPU-only and synthetic — no
# data files, no GPU. On Windows set `KMP_DUPLICATE_LIB_OK=TRUE` first.
#
# **To reduce a scan measured at the beamline**, use the rocking-curve notebook instead
# (its Part B). Copy both tutorials to a folder you can edit with
# `python -m midas_dfxm.examples.get_notebooks`.

# %%
import torch
import matplotlib.pyplot as plt
from midas_dfxm import (
    make_uniform_field, with_orientation_gradient,
    GoniometerSetting, reference_q_nom, aligned_resolution,
    ObjectiveOptics, bragg_two_theta_deg, dfxm_image,
)
from midas_dfxm.field_inverse import deformation_observable, recover_deformation_direct

torch.manual_seed(0)
print("torch", torch.__version__, "- CPU is fine for this tutorial")

# %% [markdown]
# ## 1. Build a crystal grain
# We make a small grain whose lattice **rotates smoothly across x** (a sub-grain
# boundary / bend), the canonical thing DFXM images. `field.F` is the per-voxel
# deformation gradient — the ground truth we will try to recover later.

# %%
field = make_uniform_field(shape=(64, 64, 1), spacing_um=0.5)
field = with_orientation_gradient(field, axis=(0, 0, 1), deg_per_um=0.02, along=0)
print("field F shape:", tuple(field.F.shape), "(N_voxels, 3, 3)")

# %% [markdown]
# ## 2. Render a realistic DFXM image
# The forward model needs the reflection (hkl), the goniometer setting, the
# instrument resolution, and the objective optics. `dfxm_image` returns a
# differentiable image: only the sub-region satisfying the Bragg condition lights
# up, which is what gives DFXM its orientation contrast. The energy is 20 keV, a typical
# APS 6-ID-C setting; the geometry here is the package default (vertical scattering plane),
# and the rocking-curve notebook shows the horizontal plane used at 6-ID-C.

# %%
WAVELENGTH_A = 12.398419843320026 / 20.0          # 20 keV
hkl, center = (1, 1, 1), GoniometerSetting()
q_nom = reference_q_nom(field, hkl, center)
res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-3)
tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), wavelength_A=WAVELENGTH_A)
print(f"2theta of {hkl} at 20 keV: {tt:.2f} deg")
# 0.5 um voxels x 10 = 5 um on the detector: 2.5 um pixels sample that twice. With 1 um
# pixels the voxels land 5 px apart and the image breaks into a comb of vertical lines.
optics = ObjectiveOptics(two_theta_deg=tt, magnification=10.0, pixel_um=2.5,
                         detector_shape=(160, 160))

image = dfxm_image(field, hkl, center, res, optics)
plt.figure(figsize=(4, 4))
plt.imshow(image.detach().T, origin="lower", cmap="magma")
plt.title("a DFXM image: only the Bragg-satisfying strip lights up")
plt.colorbar(shrink=0.8); plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Rock the crystal — the diffracting region sweeps across the bend
# Because the lattice orientation varies across x, different columns satisfy the
# Bragg condition at different rocking angles. Stepping the goniometer sweeps the
# bright strip — this sweep *is* the per-pixel rocking curve DFXM measures.

# %%
fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
for a, dchi in zip(ax, [-0.05, 0.0, 0.05]):
    g = GoniometerSetting(chi=dchi)
    img = dfxm_image(field, hkl, g, res, optics)
    a.imshow(img.detach().T, origin="lower", cmap="magma")
    a.set_title(f"chi = {dchi:+.2f} deg"); a.set_xticks([]); a.set_yticks([])
fig.suptitle("rocking the crystal sweeps the diffracting region across the bend")
fig.tight_layout(); plt.show()

# %% [markdown]
# ## 4. The full-F inverse — what several reflections add
# The measured per-pixel reciprocal-space shift is exactly linear in F:
# $\Delta Q = (F^{-T}-I)\,Q_0$. With **>= 3 non-coplanar reflections** all nine
# components of F are recoverable per voxel. Below we simulate the shifts for four
# reflections (with a little noise) and invert them.
#
# **Read the error correctly.** Recovering a phantom that the same model generated shows
# that the inverse is consistent with its own forward model. It is *not* the accuracy you
# would get on measured frames, which carry a detector pedestal, noise, registration between
# reflections and model error. On real data the error bar comes from a split-half or from
# injecting a known shift into the measured frames (the rocking-curve notebook, Part B).

# %%
refls = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3)]
meas = deformation_observable(field, refls)
meas = meas + 1e-3 * meas.abs().mean() * torch.randn_like(meas)
F_rec = recover_deformation_direct(meas, refls, field=field)

err = (F_rec - field.F).abs()
print(f"phantom round-trip (a consistency check, not accuracy): "
      f"max |dF| = {float(err.max()):.2e}, mean = {float(err.mean()):.2e}")

# compare a recovered component vs truth
rot_true = (0.5 * (field.F[:, 1, 0] - field.F[:, 0, 1])).reshape(64, 64) * (180 / torch.pi) * 1e3
rot_rec = (0.5 * (F_rec[:, 1, 0] - F_rec[:, 0, 1])).reshape(64, 64) * (180 / torch.pi) * 1e3
fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
for a, d, t in [(ax[0], rot_true, "true lattice rotation (mdeg)"),
                (ax[1], rot_rec, "recovered (full-F inverse)")]:
    im = a.imshow(d.detach().T, origin="lower", cmap="twilight")
    a.set_title(t); a.set_xticks([]); a.set_yticks([]); fig.colorbar(im, ax=a, shrink=0.8)
fig.tight_layout(); plt.show()

# %% [markdown]
# ## 5. What one reflection cannot give you
# A DFXM study often wants twin domains, elastic strain and mosaicity on one grain. One
# limit worth learning early: **a single reflection measures only the strain projected onto
# its g** (`eps_gg = ĝ·ε·ĝ`) — and in a solid solution even that is a d-spacing change, not
# necessarily an elastic strain. The **full six-component** strain tensor needs a diverse
# set of reflections, and the code tells you whether a set is enough before you take the
# data.

# %%
from midas_dfxm.inverse import strain_identifiability

one_reflection = [(2, 0, 2)]
rank6_set = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3), (3, 1, 1), (1, 3, 1)]
for name, refls in [("single reflection", one_reflection), ("6-reflection set", rank6_set)]:
    info = strain_identifiability(refls)
    print(f"{name:18s}: rank {info['rank']}/6  full-tensor recoverable = {info['recoverable']}")
# -> the single reflection is rank 1 (only one projected component); the diverse set is
#    rank 6 (all six components).

# %% [markdown]
# ## 6. Where to go next
# - **A scan you measured:** the rocking-curve notebook. Part A simulates and reconstructs
#   rocking scans; Part B loads an APS 6-ID-C scan, checks that every frame is paired with
#   its own angle, subtracts the detector pedestal, and makes a tilt or strain map with an
#   error bar.
# - **Dislocation typing** (simulation):
#   `python -m midas_dfxm.examples.tutorial_dislocation_typing` recovers a Burgers vector,
#   direction *and* sign, from the anisotropic contrast.
# - **Your own field:** replace `field.F` with any deformation gradient (from crystal
#   plasticity, from a measurement) and re-render. The forward is differentiable, so you
#   can also *fit* instrument and sample parameters.
#
# Everything here ran on CPU in seconds.
print("tutorial complete — every cell ran on CPU.")
