"""Gates on the node-basis texture operator, solver and the planted recovery.

The generators here are separate code from the operator: crossings by brute-force root bracketing on a fine omega grid (the operator
uses the closed form), spot weights from the textbook rotation-axis Lorentz factor ``1 / |e(eta) . (k x p)|`` (the operator never forms a
Lorentz factor), crystallites drawn from a Gaussian of the rotation vector (the operator evaluates a von Mises-Fisher density). They are
NOT a third-party check: they share the convention being tested (``ghat_lab = cos(theta) e(eta) - sin(theta) p``, omega an active
right-handed rotation about ``k``), and that convention and the operator's design were taken from reading diffractom's documented
frame conventions. The convention was compared with diffractom's operator numerically in a development script that is not shipped
(repository ``dev/``, git-ignored), not here.

Statistical tests (Monte Carlo pole figure, aliasing, planted recovery) are written as claims over several seeds: per-seed bounds are
loose, the strict bounds are on a median or a mean over seeds, because a single seed set was found to fail at 5 of 8 other seeds.

Simulation only. Nothing here is a measured result.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("midas_hkls")
pytest.importorskip("midas_stress")
from scipy.spatial.transform import Rotation  # noqa: E402

from midas_dt.odf_readout import voxel_odf  # noqa: E402
from midas_dt.texture_nodes import (  # noqa: E402
    NodeBasis,
    NodeOperator,
    TextureScan,
    make_rings,
    pole_figure_density,
    ring_normalise,
)
from midas_dt.texture_solve import estimate_lipschitz, fista_nonneg  # noqa: E402

A_CU = 3.6149
E_KEV = 50.0
HKLS = ((1, 1, 1), (2, 0, 0), (2, 2, 0))


def make_scan(n_omega=90, step=2.0, n_eta=60, n_trans=1, **kw):
    # diffractom-style eta origin (-y), so the same vectors serve the dev validation against the third-party geometry
    return TextureScan(n_omega=n_omega, omega_start_deg=0.0, omega_step_deg=step, n_eta=n_eta, n_trans=n_trans,
                       wavelength_a=12.398 / E_KEV, eta_zero=(0, -1, 0), eta_ninety=(0, 0, -1), **kw)


def make_ring_set(scan, hkls=HKLS):
    return make_rings(hkls, (A_CU,) * 3 + (90.0,) * 3, 225, scan.wavelength_a)


# ----------------------------------------------------- independent generator
def _ghat(scan, eta, theta):
    e = np.cos(eta)[:, None] * scan.vec("eta_zero") + np.sin(eta)[:, None] * scan.vec("eta_ninety")
    return np.cos(theta) * e - np.sin(theta) * scan.vec("beam")


def sample_orientations(centre, sigma_deg, n, rng):
    rv = rng.normal(scale=np.radians(sigma_deg), size=(n, 3))
    return Rotation.from_rotvec(rv).as_matrix() @ centre


def crossings_with_owner(scan, u, theta):
    """(owner index, omega_deg, eta_deg, lorentz) of every Ewald crossing in [0, 360) of unit sample vectors ``u`` (n,3): sign changes of
    ``(Omega(w) u) . p + sin(theta)`` on a 1 deg grid, polished by two Newton steps on that function. A crossing pair closer than the grid
    (a near-tangent one) can be missed; it is a negligible fraction of the sphere. Lorentz = 1 / |e(eta) . (k x p)|."""
    k, p = scan.vec("rot_axis"), scan.vec("beam")
    grid = np.radians(np.arange(0.0, 361.0, 1.0))
    owners, oms, ets, lors = [], [], [], []
    for lo in range(0, len(u), 4000):
        uu = u[lo:lo + 4000]
        a, b = uu @ p, np.cross(k, uu) @ p
        f = a[:, None] * np.cos(grid) + b[:, None] * np.sin(grid) + np.sin(theta)
        ii, jj = np.nonzero(np.sign(f[:, :-1]) * np.sign(f[:, 1:]) < 0)
        w = grid[jj] - f[ii, jj] * (grid[jj + 1] - grid[jj]) / (f[ii, jj + 1] - f[ii, jj])
        for _ in range(2):
            w = w - (a[ii] * np.cos(w) + b[ii] * np.sin(w) + np.sin(theta)) / (-a[ii] * np.sin(w) + b[ii] * np.cos(w))
        ur = uu[ii]
        g = ur * np.cos(w)[:, None] + np.cross(k, ur) * np.sin(w)[:, None] + k * (ur @ k)[:, None] * (1 - np.cos(w))[:, None]
        eta = np.arctan2(g @ scan.vec("eta_ninety"), g @ scan.vec("eta_zero"))
        e = np.cos(eta)[:, None] * scan.vec("eta_zero") + np.sin(eta)[:, None] * scan.vec("eta_ninety")
        owners.append(ii + lo)
        oms.append(np.degrees(w) % 360.0)
        ets.append(np.degrees(eta) % 360.0)
        lors.append(1.0 / np.abs(e @ np.cross(k, p)))
    return np.concatenate(owners), np.concatenate(oms), np.concatenate(ets), np.concatenate(lors)


def simulate(scan, rings, U, weight, xy=(0.0, 0.0)):
    """Independent forward of crystallites ``U`` (n,3,3) with per-crystallite ``weight`` at voxel centre ``xy`` (x, y; voxel units).
    Spot weight = weight * Lorentz; omega / eta by exact deposit into the bin the crossing falls in; translation by a box of the voxel's
    projected width ``|sin w| + |cos w|`` centred on ``t = x sin w + y cos w``. Returns (n_omega, n_trans, n_eta, n_rings)."""
    data = np.zeros((scan.n_omega, scan.n_trans, scan.n_eta, rings.n_rings))
    wts = np.broadcast_to(np.asarray(weight, dtype=float), (len(U),))
    for r in range(rings.n_rings):
        u = np.einsum("nij,mj->nmi", U, rings.normals[r]).reshape(-1, 3)
        own, om, et, lor = crossings_with_owner(scan, u, rings.theta_rad[r])
        own = own // len(rings.normals[r])
        fr = np.floor((om - scan.omega_start_deg) / scan.omega_step_deg).astype(int)
        eb = np.floor((et - scan.eta_start_deg) / scan.eta_step_deg).astype(int) % scan.n_eta
        keep = (fr >= 0) & (fr < scan.n_omega)
        w = wts[own] * lor
        wr = np.radians(om)
        t = xy[0] * np.sin(wr) + xy[1] * np.cos(wr) + (scan.n_trans - 1) / 2.0          # detector-bin coordinate of the voxel centre
        half = 0.5 * (np.abs(np.sin(wr)) + np.abs(np.cos(wr)))
        lo, hi = t - half + 0.5, t + half + 0.5                                              # bin j covers [j - 0.5, j + 0.5) about its centre
        for j in range(scan.n_trans):
            ov = np.clip(np.minimum(hi, j + 1.0) - np.maximum(lo, float(j)), 0.0, None) / (hi - lo)
            sel = keep & (ov > 0)
            np.add.at(data[:, j, :, r], (fr[sel], eb[sel]), (w * ov)[sel])
    return data


# ----------------------------------------------------------------- the tests
def test_adjoint_dot_product_full_operator():
    """<A x, y> = <x, A^T y> for the full operator (Radon x pole-figure blocks), several node batches, sub-sampling on."""
    scan = make_scan(n_omega=40, step=4.5, n_eta=36, n_trans=15)
    rings = make_ring_set(scan)
    U = Rotation.random(7, random_state=3).as_matrix()
    rng = np.random.default_rng(0)
    mask = np.ones((11, 11), bool)
    mask[0, :] = mask[:, 0] = False
    op = NodeOperator(scan, rings, NodeBasis(U, 2.0, 225), size=11, mask=mask, sub_omega=2, sub_eta=3, node_batch=3,
                      dtype=torch.float64)
    x = torch.from_numpy(rng.random((op.n_vox, op.K)))
    y = torch.from_numpy(rng.random(op.data_shape))
    lhs = float((op.forward(x) * y).sum())
    rhs = float((x * op.adjoint(y)).sum())
    assert lhs > 0 and abs(lhs - rhs) < 1e-10 * abs(lhs)
    assert op.nnz > 0


def test_batching_invariance():
    """Results do not depend on the node batch size, and cache=False gives the same numbers as cache=True."""
    scan = make_scan(n_omega=40, step=4.5, n_eta=36, n_trans=11)
    rings = make_ring_set(scan)
    U = Rotation.random(9, random_state=5).as_matrix()
    basis = NodeBasis(U, 1.5, 225)
    rng = np.random.default_rng(1)
    x = rng.random((49, 9))
    y = rng.random((40, 11, 36, 3))
    ref = None
    for nb, cache in ((9, True), (4, True), (1, False), (2, True)):
        op = NodeOperator(scan, rings, basis, size=7, sub_omega=2, sub_eta=2, node_batch=nb, cache=cache, dtype=torch.float64)
        f, a = op.forward(x).numpy(), op.adjoint(y).numpy()
        if ref is None:
            ref = (f, a)
        else:
            np.testing.assert_allclose(f, ref[0], rtol=1e-12, atol=1e-12 * np.abs(ref[0]).max())
            np.testing.assert_allclose(a, ref[1], rtol=1e-12, atol=1e-12 * np.abs(ref[1]).max())


def test_convention_against_independent_ewald_crossings():
    """One node, one off-centre voxel: the operator puts intensity in the (omega, translation, eta, ring) bins that an independent
    brute-force Ewald solution predicts, to within one bin, and conversely."""
    scan = make_scan(n_omega=360, step=0.5, n_eta=180, n_trans=21)
    rings = make_ring_set(scan)
    U = Rotation.random(1, random_state=11).as_matrix()
    size = 15
    row, col = 3, 10                                      # image (row, col): y = row - 7, x = col - 7
    mask = np.zeros((size, size), bool)
    mask[row, col] = True
    op = NodeOperator(scan, rings, NodeBasis(U, 0.3, 225), size=size, mask=mask, sub_omega=2, sub_eta=2, dtype=torch.float64)
    d = op.forward(np.ones((1, 1))).numpy()
    obs = np.argwhere(d > 0.05 * d.max())
    xs, ys = col - (size - 1) / 2, row - (size - 1) / 2
    pred = []                                               # (omega bin, t bin, eta bin, ring, omega extent in bins), continuous
    for r in range(rings.n_rings):
        u = (U[0] @ rings.normals[r].T).T
        _, om, et, lor = crossings_with_owner(scan, u, rings.theta_rad[r])
        tb = xs * np.sin(np.radians(om)) + ys * np.cos(np.radians(om)) + (scan.n_trans - 1) / 2.0 + 0.5
        # a crossing of angular width sigma lasts sigma * Lorentz in omega (slow, grazing crossings are long); cutoff 4.5 sigma
        ext = 4.5 * 0.3 * lor / scan.omega_step_deg
        for o, e, t, x in zip((om - scan.omega_start_deg) / scan.omega_step_deg, et / scan.eta_step_deg, tb, ext):
            if 0 <= o < scan.n_omega and 0 <= t < scan.n_trans:
                pred.append((o, t, e, r, x))
    pred = np.array(pred)
    assert len(pred) > 15 and len(obs) > 15
    obs_set = {tuple(i) for i in obs}

    def pred_seen(q):
        o, t, e, r = int(np.floor(q[0])), int(np.floor(q[1])), int(np.floor(q[2])), int(q[3])
        return any((o + do, t + dt, (e + de) % scan.n_eta, r) in obs_set for do in (-1, 0, 1) for dt in (-1, 0, 1) for de in (-1, 0, 1))

    def obs_explained(b):
        pr = pred[pred[:, 3] == b[3]]
        de = np.abs((pr[:, 2] - (b[2] + 0.5) + scan.n_eta / 2) % scan.n_eta - scan.n_eta / 2)
        return bool(np.any((np.abs(pr[:, 0] - (b[0] + 0.5)) <= 1.5 + pr[:, 4]) & (np.abs(pr[:, 1] - (b[1] + 0.5)) <= 1.5) & (de <= 1.5)))

    assert np.mean([pred_seen(q) for q in pred]) > 0.97
    assert np.mean([obs_explained(b) for b in obs]) > 0.97


def test_kernel_pole_figure_matches_monte_carlo():
    """The binned pole-figure model (sub-sampled) against crystallites SAMPLED from the Gaussian on SO(3), with independent crossings and
    Lorentz weights. Bin-integrated counts / (N m dw de) must equal the operator's density.

    Five node orientations (one of them, 40, is the worst of 50 scanned: a near-tangent crossing makes its {111} bins sensitive to the
    sub-sampling and tail cutoff, corr 0.981 / rel 0.13). Per orientation and ring: corr > 0.97, rel < 0.2. Median over orientations:
    corr > 0.995, rel < 0.06.
    """
    scan = make_scan(n_omega=90, step=2.0, n_eta=60, n_trans=3)      # 3 detector bins so the voxel's box is not clipped
    rings = make_ring_set(scan, ((1, 1, 1), (2, 2, 0)))
    sigma, n = 2.5, 40000
    dw, de = np.radians(scan.omega_step_deg), np.radians(scan.eta_step_deg)
    corrs = [[] for _ in range(rings.n_rings)]
    rels = [[] for _ in range(rings.n_rings)]
    for k, cs in enumerate((21, 30, 40, 41, 42)):
        centre = Rotation.random(1, random_state=cs).as_matrix()[0]
        rng = np.random.default_rng(cs)
        U = sample_orientations(centre, sigma, n, rng)
        mc = simulate(scan, rings, U, 1.0).sum(axis=1)                   # (o, e, r), summed over the translation bins
        op = NodeOperator(scan, rings, NodeBasis(centre[None], sigma, 225), size=1, sub_omega=4, sub_eta=4, dtype=torch.float64,
                          cutoff_sigma=6.0)
        g = op.forward(np.ones((1, 1))).numpy().sum(axis=1)
        for r in range(rings.n_rings):
            m = len(rings.normals[r])
            # the Lorentz weight of `simulate` is 1/|e.(k x p)| = cos(theta) / J; J cancels, leaving cos(theta) * P d omega d eta
            mc_r = mc[..., r] / (n * m * dw * de * np.cos(rings.theta_rad[r]))
            big = g[..., r] > 0.05 * g[..., r].max()
            assert big.sum() > 30
            corr = np.corrcoef(mc_r[big], g[..., r][big])[0, 1]
            rel = np.linalg.norm(mc_r[big] - g[..., r][big]) / np.linalg.norm(g[..., r][big])
            assert corr > 0.97 and rel < 0.2, (cs, r, corr, rel)
            corrs[r].append(corr)
            rels[r].append(rel)
        if k == 0:
            # the density helper agrees with the operator's centre value to the sub-sampling accuracy
            peak = np.unravel_index(np.argmax(g[..., 0]), g[..., 0].shape)
            pf = pole_figure_density(centre[None], rings.normals[0], rings.theta_rad[0],
                                     scan.omega_start_deg + (peak[0] + 0.5) * scan.omega_step_deg, (peak[1] + 0.5) * scan.eta_step_deg,
                                     sigma, scan)
            assert 0.5 < g[peak[0], peak[1], 0] / pf[0, 0] < 2.0
    for r in range(rings.n_rings):
        assert np.median(corrs[r]) > 0.995, (r, corrs[r])
        assert np.median(rels[r]) < 0.06, (r, rels[r])


def dense_sampled_density(scan, rings, ring, U, sigma_deg, S):
    """Exact (untruncated) family-averaged pole-figure density of nodes U at every sub-sample of every bin of one ring, averaged per bin."""
    om = np.radians(scan.omega_start_deg + (np.arange(scan.n_omega * S) + 0.5) / S * scan.omega_step_deg)
    et = np.radians(scan.eta_start_deg + (np.arange(scan.n_eta * S) + 0.5) / S * scan.eta_step_deg)
    k, p = scan.vec("rot_axis"), scan.vec("beam")
    e = np.cos(et)[:, None] * scan.vec("eta_zero") + np.sin(et)[:, None] * scan.vec("eta_ninety")
    gl = np.cos(rings.theta_rad[ring]) * e - np.sin(rings.theta_rad[ring]) * p                       # (ne, 3)
    kappa = 1.0 / np.radians(sigma_deg) ** 2
    norm = kappa / (2 * np.pi * (1 - np.exp(-2 * kappa)))
    out = np.zeros((scan.n_omega * S, scan.n_eta * S))
    for Uk in U:
        u = (Uk @ rings.normals[ring].T).T                                                          # (m, 3)
        for i in range(len(om)):
            uo = u * np.cos(om[i]) + np.cross(k, u) * np.sin(om[i]) + k * (u @ k)[:, None] * (1 - np.cos(om[i]))
            out[i] += norm * np.exp((uo @ gl.T - 1.0) * kappa).sum(axis=0) / len(u)
    return out.reshape(scan.n_omega, S, scan.n_eta, S).mean(axis=(1, 3))


def test_operator_equals_dense_sampled_kernel_including_slow_crossings():
    """The sparse footprint construction reproduces the exact kernel sampled on the same sub-sample grid, in every bin -- including the bins
    of slow (near-tangent) crossings, where the two crossings of one normal have overlapping footprints (counted once) and the footprint is
    long in omega and eta. (A first version added the overlap twice: 35-70 % too much there, 2e-5 elsewhere.)"""
    scan = make_scan(n_omega=90, step=2.0, n_eta=60, n_trans=3)
    rings = make_ring_set(scan, ((2, 2, 0),))
    U = Rotation.random(40, random_state=5).as_matrix()
    sigma, S = 1.0, 2
    op = NodeOperator(scan, rings, NodeBasis(U, sigma, 225), size=1, sub_omega=S, sub_eta=S, dtype=torch.float64, cutoff_sigma=8.0, node_batch=16)
    mine = op.forward(np.ones((1, len(U)))).numpy().sum(axis=1)[..., 0]
    ref = dense_sampled_density(scan, rings, 0, U, sigma, S)
    e = np.cos(np.radians((np.arange(scan.n_eta) + 0.5) * scan.eta_step_deg))
    graz = np.abs(e) < 0.15                                                   # study geometry: |e . (k x p)| = |cos eta|
    assert ref[:, graz].sum() > 0.01 * ref.sum()
    assert np.linalg.norm(mine - ref) / np.linalg.norm(ref) < 1e-6
    assert np.linalg.norm(mine[:, graz] - ref[:, graz]) / np.linalg.norm(ref[:, graz]) < 1e-5


def _mismatch(scan, rings, centre, sigma, S, truth):
    op = NodeOperator(scan, rings, NodeBasis(centre[None], sigma, 225), size=1, sub_omega=S, sub_eta=S, dtype=torch.float64)
    g = op.forward(np.ones((1, 1))).numpy().sum(axis=1)
    s = (g * truth).sum() / (g * g).sum()
    return float(np.linalg.norm(s * g - truth) / np.linalg.norm(truth))


def test_subsampling_reduces_aliasing():
    """A basis narrower than the bin: one sample per bin mismatches bin-integrated data; sub-sampling removes most of the mismatch.

    Bins 1 deg (omega) x 3 deg (eta), sigma 0.5 deg. 'Data' = crystallites sampled from the same Gaussian, binned (bin-integrated by
    construction). The direction is the claim; the numbers printed on failure are the measurement.

    Six node orientations (aliasing depends on where the spots fall in their bins). Over 24 scanned orientations: S=1 0.32-0.80, S=2
    0.08-0.26, S=6 0.009-0.064; the order S=1 > S=2 > S=6 and S=6 < 0.2 S=1 held for every one, so those are per orientation. The
    absolute levels are on the median.
    """
    scan = make_scan(n_omega=180, step=1.0, n_eta=120, n_trans=3)
    rings = make_ring_set(scan, ((1, 1, 1),))
    sigma = 0.5
    rows = []
    for seed in range(4, 10):
        centre = Rotation.random(1, random_state=seed).as_matrix()[0]
        rng = np.random.default_rng(3)
        truth = simulate(scan, rings, sample_orientations(centre, sigma, 50000, rng), 1.0).sum(axis=1)
        truth = truth / truth.sum()
        rows.append([_mismatch(scan, rings, centre, sigma, S, truth) for S in (1, 2, 6)])
    rows = np.array(rows)
    assert (rows[:, 0] > 0.2).all() and (rows[:, 2] < 0.1).all(), rows
    assert (rows[:, 0] > rows[:, 1]).all() and (rows[:, 1] > rows[:, 2]).all(), rows
    assert (rows[:, 2] < 0.2 * rows[:, 0]).all(), rows
    assert np.median(rows[:, 0]) > 0.3 and np.median(rows[:, 2]) < 0.03, rows


def test_ring_normalise_equalises_rings():
    rng = np.random.default_rng(0)
    d = rng.random((4, 5, 6, 3)) * np.array([1.0, 10.0, 0.0])
    out, scale = ring_normalise(d)
    tot = out.sum(axis=(0, 1, 2))
    assert tot[0] == pytest.approx(tot[1], rel=1e-5) and tot[2] == 0.0 and scale[2] == 0.0
    out2, _ = ring_normalise(d, q99=50.0)
    assert np.quantile(out2[out2 > 0], 0.99) == pytest.approx(50.0, rel=1e-4)


# ------------------------------------------------------------------- solver
def _tiny_problem(seed=0, noise=False):
    """3 grain sectors of a disc (one true node each), plus one mixed voxel pair; K = 8 nodes (3 true + 5 decoys)."""
    rng = np.random.default_rng(seed)
    scan = make_scan(n_omega=90, step=2.0, n_eta=60, n_trans=13)
    rings = make_ring_set(scan)
    nodes = Rotation.random(8, random_state=100 + seed).as_matrix()
    size = 9
    yy, xx = np.mgrid[:size, :size] - (size - 1) / 2.0
    disc = (xx ** 2 + yy ** 2) <= 4.3 ** 2
    ang = (np.degrees(np.arctan2(yy, xx)) + 360.0) % 360.0
    lab = np.where(ang < 120, 0, np.where(ang < 240, 1, 2))
    vox = np.flatnonzero(disc.ravel())
    truth = {}
    data = np.zeros((scan.n_omega, scan.n_trans, scan.n_eta, rings.n_rings))
    for v in vox:
        row, col = divmod(int(v), size)
        g = int(lab[row, col])
        members = [g]
        if (row, col) in ((4, 4), (4, 5)):             # two mixed voxels: grains 0 and 1 half and half
            members = [0, 1]
        truth[int(v)] = members
        for gi in members:
            Uc = sample_orientations(nodes[gi], 0.4, 120, rng)
            data += simulate(scan, rings, Uc, 1.0 / len(members), xy=(col - (size - 1) / 2.0, row - (size - 1) / 2.0))
    if noise:
        data = rng.poisson(data / data.max() * 200.0).astype(float)
    return scan, rings, nodes, size, disc, vox, truth, data


def _mass_in_truth(coef, vox, truth, nodes):
    out = []
    for i, v in enumerate(vox):
        odf = voxel_odf(coef[i], nodes, space_group=225, floor=0.0)
        if len(odf) == 0:
            out.append(0.0)
            continue
        out.append(odf.mass_within(nodes[truth[int(v)]], 1.5))
    return np.array(out)


@pytest.fixture(scope="module")
def tiny():
    return _tiny_problem()


def _solve(scan, rings, nodes, size, disc, data, niter=120):
    basis = NodeBasis(nodes, 1.5, 225)
    op = NodeOperator(scan, rings, basis, size=size, mask=disc, sub_omega=2, sub_eta=3, node_batch=4, dtype=torch.float32)
    d, _ = ring_normalise(data, q99=10.0)
    return op, fista_nonneg(op, d, n_iter=niter, loss="huber", delta=3.0), d


def test_planted_recovery_and_omega_shuffle_null():
    """Planted nodes are recovered from independently simulated data; the same data with the omega frames shuffled recover nothing.

    Six planted problems (different node lists, crystallite draws and shuffles). Over 16 scanned seeds: mass-in-truth 0.925-0.966,
    omega-shuffled 0.15-0.47 (one shuffle), objective ratio 0.47-0.58, and the two mixed voxels carried both of their nodes in 30 of 32
    voxels (seed 12 puts ~0 on one node of both). So per problem only loose bounds; the strict ones are on means over problems.
    """
    reals, nulls, both = [], [], []
    for seed in range(6):
        scan, rings, nodes, size, disc, vox, truth, data = _tiny_problem(seed)
        op, res, d = _solve(scan, rings, nodes, size, disc, data)
        assert res.objective[-1] < 0.7 * res.objective[0]     # the data are 120 discrete crystallites per voxel and node: a floor of shot noise remains
        real = _mass_in_truth(res.coef, vox, truth, nodes)
        rng = np.random.default_rng(9 + seed)
        null = np.mean([_mass_in_truth(fista_nonneg(op, d[rng.permutation(scan.n_omega)], n_iter=120, loss="huber", delta=3.0).coef,
                                       vox, truth, nodes).mean() for _ in range(3)])
        # Chance is 1/8 per single-node voxel. The shuffled fit sits above it because a permutation of the frames keeps everything that does
        # not depend on omega order (the omega-marginal of the (t, eta, ring) data), which already carries some orientation information.
        print(f"seed {seed}: mass in truth: real {real.mean():.3f}, omega-shuffled {null:.3f}")
        assert real.mean() > 0.8 and real.mean() - null > 0.3, (seed, real.mean(), null)
        reals.append(real.mean())
        nulls.append(null)
        for i, v in enumerate(vox):
            if len(truth[int(v)]) == 2:                       # the two mixed voxels: grains 0 and 1 half and half
                odf = voxel_odf(res.coef[i], nodes, space_group=225, floor=0.05)
                both.append(set(odf.index.tolist()) >= {0, 1})
    assert np.mean(reals) > 0.9, reals
    assert np.mean(nulls) < 0.45, nulls
    assert np.mean(reals) - np.mean(nulls) > 0.5, (reals, nulls)
    assert len(both) == 12 and np.mean(both) >= 0.75, both


def test_solver_is_nonnegative_and_objective_decreases(tiny):
    scan, rings, nodes, size, disc, vox, truth, data = tiny
    op, res, _ = _solve(scan, rings, nodes, size, disc, data, niter=40)
    assert (res.coef >= 0).all()
    assert res.n_increases <= 4                       # FISTA is not monotone, but not by much
    assert res.objective[-1] < res.objective[0]
    assert res.lipschitz > 0


def test_weights_mask_and_l1(tiny):
    """Zero-weight data do not constrain the fit; an l1 penalty reduces the support."""
    scan, rings, nodes, size, disc, vox, truth, data = tiny
    basis = NodeBasis(nodes, 1.5, 225)
    op = NodeOperator(scan, rings, basis, size=size, mask=disc, sub_omega=2, sub_eta=2, node_batch=8, dtype=torch.float32)
    d, _ = ring_normalise(data, q99=10.0)
    corrupt = d.copy()
    corrupt[:, :, :, 0] = 1000.0
    w = np.ones_like(d)
    w[:, :, :, 0] = 0.0
    a = fista_nonneg(op, corrupt, n_iter=40, loss="l2", weights=w)
    b = fista_nonneg(op, np.where(w > 0, d, 0.0), n_iter=40, loss="l2", weights=w)
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-3, atol=1e-4 * b.coef.max())
    c = fista_nonneg(op, d, n_iter=40, loss="l2", l1=5.0, lipschitz=a.lipschitz)
    e = fista_nonneg(op, d, n_iter=40, loss="l2", l1=0.0, lipschitz=a.lipschitz)
    assert c.coef.sum() < e.coef.sum()


def test_power_iteration_bounds_the_norm(tiny):
    scan, rings, nodes, size, disc, vox, truth, data = tiny
    op = NodeOperator(scan, rings, NodeBasis(nodes, 1.5, 225), size=size, mask=disc, node_batch=8, dtype=torch.float64)
    lam = estimate_lipschitz(op, n_iter=40)
    rng = np.random.default_rng(2)
    for _ in range(5):                                  # a random vector can never exceed the largest eigenvalue
        x = torch.from_numpy(rng.random((op.n_vox, op.K)))
        q = float((op.forward(x) ** 2).sum() / (x ** 2).sum())
        assert q <= lam * 1.001


def test_lazy_exports():
    """The node-basis names resolve through the package's PEP 562 lazy exports and are the module objects' own."""
    import midas_dt
    from midas_dt import odf_readout, texture_nodes, texture_solve

    assert midas_dt.NodeOperator is texture_nodes.NodeOperator
    assert midas_dt.fista_nonneg is texture_solve.fista_nonneg
    assert midas_dt.voxel_odf is odf_readout.voxel_odf
    assert midas_dt.merge_nodes is texture_nodes.merge_nodes
    with pytest.raises(AttributeError):
        midas_dt.not_a_node_thing


# ------------------------------------------- gates added after a mutation check (each pins a way the code could silently go wrong)
def test_footprint_build_is_independent_of_chunking(monkeypatch):
    """The footprint of a node batch is built in chunks of ``_CHUNK_ELEMENTS`` sub-sample values; the chunk size must not change the operator
    (a chunk that drops its last spot, or is empty, silently loses signal)."""
    scan = make_scan(n_omega=60, step=3.0, n_eta=36, n_trans=7)
    rings = make_ring_set(scan)
    U = Rotation.random(5, random_state=3).as_matrix()
    x = np.random.default_rng(0).random((25, 5))

    def build():
        op = NodeOperator(scan, rings, NodeBasis(U, 1.0, 225), size=5, sub_omega=2, sub_eta=2, node_batch=5, dtype=torch.float64)
        return op.forward(x).numpy(), op.nnz

    ref, nnz = build()
    assert nnz > 500
    for chunk in (1, 400, 1500):
        monkeypatch.setattr(NodeOperator, "_CHUNK_ELEMENTS", chunk)
        got, nnz_c = build()
        assert nnz_c == nnz, (chunk, nnz_c, nnz)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_eta_start_shifts_the_bins():
    """Moving the eta origin of the bins by a whole number of bins rolls the data along eta and changes nothing else (n_eta 60 -> 6 deg
    bins, 30 deg = 5 bins)."""
    U = Rotation.random(3, random_state=8).as_matrix()
    x = np.random.default_rng(1).random((25, 3))
    out = []
    for eta_start in (0.0, 30.0):
        scan = make_scan(n_omega=40, step=4.5, n_eta=60, n_trans=7, eta_start_deg=eta_start)
        op = NodeOperator(scan, make_ring_set(scan), NodeBasis(U, 1.5, 225), size=5, sub_omega=2, sub_eta=3, dtype=torch.float64)
        out.append(op.forward(x).numpy())
    assert out[0].max() > 0
    np.testing.assert_allclose(out[1], np.roll(out[0], -5, axis=2), rtol=1e-9, atol=1e-9 * out[0].max())
    assert np.abs(out[1] - out[0]).max() > 0.1 * out[0].max()               # and the shift is not a no-op


class _DenseOp:
    """A small dense non-negative linear operator with the interface ``fista_nonneg`` needs."""

    def __init__(self, a, shape, dtype=None):
        self.a = torch.as_tensor(a, dtype=torch.float64)
        self.n_vox, self.K = shape
        self.data_shape = (self.a.shape[0], 1, 1, 1)
        self.dtype, self.device = torch.float64, torch.device("cpu")

    def forward(self, x):
        return (self.a @ x.reshape(-1)).reshape(self.data_shape)

    def adjoint(self, y):
        return (self.a.T @ y.reshape(-1)).reshape(self.n_vox, self.K)


def _reference_fista(a, y, w, n_iter, L, loss, delta, l1):
    """Textbook FISTA (Beck & Teboulle) in plain numpy: gradient of ``sum w rho(Az - y)`` at the extrapolated point, prox of l1 + c >= 0."""
    x = np.zeros(a.shape[1])
    z, t = x.copy(), 1.0
    for _ in range(n_iter):
        r = a @ z - y
        psi = r if loss == "l2" else np.clip(r, -delta, delta)
        x_new = np.maximum(z - (a.T @ (w * psi)) / L - l1 / L, 0.0)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + (t - 1.0) / t_new * (x_new - x)
        x, t = x_new, t_new
    return x


@pytest.mark.parametrize("loss,l1", [("huber", 0.0), ("huber", 0.3), ("l2", 0.0)])
def test_fista_matches_textbook_reference(loss, l1):
    """The solver's iterates equal an independent textbook FISTA on a dense problem, with per-datum weights (not all 1) and outliers that
    activate the Huber clip: pins the momentum, the clip, the clip-before-weight order, the step size and the prox."""
    rng = np.random.default_rng(5)
    a = rng.random((30, 12)) * (rng.random((30, 12)) < 0.5)
    c_true = rng.random(12) * (rng.random(12) < 0.5)
    y = a @ c_true + 0.02 * rng.standard_normal(30)
    y[[3, 17]] += 5.0                                         # outliers: |residual| > delta
    w = 0.2 + 1.8 * rng.random(30)                            # non-uniform weights, some > 1
    L = 1.3 * float(np.linalg.eigvalsh(a.T @ (w[:, None] * a)).max())
    delta, n_iter = 0.4, 25
    op = _DenseOp(a, (1, 12))
    res = fista_nonneg(op, y.reshape(op.data_shape), n_iter=n_iter, loss=loss, delta=delta, weights=w.reshape(op.data_shape), l1=l1,
                       lipschitz=L)
    ref = _reference_fista(a, y, w, n_iter, L, loss, delta, l1)
    np.testing.assert_allclose(res.coef.reshape(-1), ref, rtol=1e-5, atol=1e-6)
    if loss == "huber":                                        # the clip is active: the l2 solution differs
        l2 = _reference_fista(a, y, w, n_iter, L, "l2", delta, l1)
        assert np.abs(l2 - ref).max() > 1e-3


def test_node_basis_from_grid(monkeypatch):
    """``from_grid`` builds a proper-rotation grid with one width per node, finer step -> more nodes, covering random orientations."""
    pytest.importorskip("midas_stress", minversion="0.16.0")
    from midas_stress.orientation import misorientation_om_batch

    coarse = NodeBasis.from_grid(15.0, 225, 2.0)
    fine = NodeBasis.from_grid(10.0, 225, 2.0)
    assert 20 < len(coarse) < len(fine)
    assert fine.sigma_deg.shape == (len(fine),) and (fine.sigma_deg == 2.0).all() and fine.space_group == 225
    R = fine.orientations
    np.testing.assert_allclose(R @ np.swapaxes(R, 1, 2), np.broadcast_to(np.eye(3), R.shape), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-9)
    rnd = Rotation.random(60, random_state=0).as_matrix()
    ia, ib = np.meshgrid(np.arange(60), np.arange(len(R)), indexing="ij")
    d = np.degrees(misorientation_om_batch(rnd[ia.ravel()], R[ib.ravel()], 225)).reshape(60, -1)
    assert d.min(axis=1).max() < 16.0                         # measured over 300 random orientations: max 13.1 deg at step 10
    # without orientation_grid the error says which release is needed
    import sys
    monkeypatch.setitem(sys.modules, "midas_stress", None)
    with pytest.raises(ImportError, match="midas-stress >= 0.16.0"):
        NodeBasis.from_grid(10.0, 225, 2.0)
