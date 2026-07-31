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
# up, which is what gives DFXM its orientation contrast.

# %%
hkl, center = (1, 1, 1), GoniometerSetting()
q_nom = reference_q_nom(field, hkl, center)
res = aligned_resolution(q_nom, sigma_par=5e-3, sigma_perp=5e-3)
tt = bragg_two_theta_deg(float(torch.linalg.vector_norm(q_nom)), wavelength_A=0.172979)
optics = ObjectiveOptics(two_theta_deg=tt, magnification=10.0, detector_shape=(256, 256))

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
# ## 4. The full-F inverse — the extra information the twin gives
# The measured per-pixel reciprocal-space shift is exactly linear in F:
# $\Delta Q = (F^{-T}-I)\,Q_0$. With **>= 3 non-coplanar reflections** we can
# solve for all nine components of F per voxel. Below we simulate the shifts for
# four reflections (with a little noise) and recover F. The round-trip error is
# tiny — this is the capability a single COM scan does not give.

# %%
refls = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3)]
meas = deformation_observable(field, refls)
meas = meas + 1e-3 * meas.abs().mean() * torch.randn_like(meas)
F_rec = recover_deformation_direct(meas, refls, field=field)

err = (F_rec - field.F).abs()
print(f"full-F round-trip: max |dF| = {float(err.max()):.2e}, mean = {float(err.mean()):.2e}")

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
# ## 5. Where to go next
# - **Dislocation typing:** `examples/tutorial_dislocation_typing.py` — recover a
#   Burgers vector (direction *and* sign) from the anisotropic contrast.
# - **Physics-regularized inverse:** fit F through a dislocation model, not
#   per-voxel, for far lower variance under noise.
# - **Your own field:** replace `field.F` with any deformation gradient (from
#   crystal plasticity, from a measurement) and re-render — the forward is
#   differentiable, so you can also *fit* instrument and sample parameters.
#
# Everything here ran on CPU in seconds. The same code differentiates end to end,
# which is what lets the twin do design, self-calibration, and regularized
# inversion — see the paper for the full story.
print("tutorial complete — every cell ran on CPU.")
