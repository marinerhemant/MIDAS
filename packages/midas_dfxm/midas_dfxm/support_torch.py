"""torch backend for :mod:`midas_dfxm.support` -- the SAME estimator, line for line, on any torch device.

``support_curve_torch`` reproduces :func:`midas_dfxm.support.support_curve` (float64 by default, so CPU
numpy and GPU results agree to rounding; tested in tests/test_support_torch.py). Medians use
``nanquantile(0.5)`` (numpy's midpoint for even counts; ``torch.median`` returns the lower element).
The neighbourhood box (``support_smooth``) is replicate-pad + avg_pool2d = ``uniform_filter(mode="nearest")``.
Used through ``reduce_support(..., backend="torch", device="cuda")``.

Measured agreement on real data (datasetJ S1050, full 2560 x 2160 frame, RTX A6000, 2026-09-28): identical on
a 300k-px lit region (CPU and CUDA); over the full frame 1 of 656,480 LIT pixels and 0.003-0.011 % of
unlit pixels differ -- knife-edge decisions (argmax / threshold) flipped by last-bit summation order in
flat noise curves. 60.4 s numpy -> 3.1 s CUDA (19x); 6 sensitivity variants 11 s.
"""
import math

import numpy as np
import torch
import torch.nn.functional as Fnn

NAN = float("nan")


def _t(a, dev, dt):
    return torch.as_tensor(np.asarray(a), device=dev, dtype=dt)


def _shift(a, ax, step):
    out = torch.full_like(a, NAN)
    src = [slice(None)] * a.ndim; dst = [slice(None)] * a.ndim
    if step > 0:
        src[ax] = slice(None, -step); dst[ax] = slice(step, None)
    else:
        src[ax] = slice(-step, None); dst[ax] = slice(None, step)
    out[tuple(dst)] = a[tuple(src)]
    return out


def _smooth(D, shape):
    out = D
    for ax, n in enumerate(shape):
        if n >= 3:
            st = torch.stack([_shift(out, ax, 1), out, _shift(out, ax, -1)])
            fin = torch.isfinite(st)
            cnt = fin.sum(0)
            s = torch.where(fin, st, torch.zeros_like(st)).sum(0)
            out = torch.where(torch.isfinite(out), s / torch.clamp(cnt, min=1), torch.full_like(out, NAN))
    return out


def _median0(x):
    """numpy-compatible median over dim 0 (linear midpoint), ignoring NaN; chunks the columns so
    quantile's input-size limit is never hit."""
    out = torch.empty(x.shape[1], device=x.device, dtype=x.dtype)
    step = max(1, (8_000_000 // max(x.shape[0], 1)))
    for c0 in range(0, x.shape[1], step):
        out[c0:c0 + step] = torch.nanquantile(x[:, c0:c0 + step], 0.5, dim=0)
    return out


def _noise(D, shape):
    d2 = []
    for ax, n in enumerate(shape):
        if n >= 3:
            v = _shift(D, ax, 1) - 2 * D + _shift(D, ax, -1)
            v = v.reshape(-1, v.shape[-1])
            v = v[torch.isfinite(v).all(1)]
            d2.append(v)
    N = D.shape[-1]
    if not d2:
        return torch.full((N,), NAN, device=D.device, dtype=D.dtype)
    d2 = torch.cat(d2)
    if d2.shape[0] < 5:
        return torch.full((N,), NAN, device=D.device, dtype=D.dtype)
    med = _median0(d2)
    return 1.4826 * _median0(torch.abs(d2 - med)) / math.sqrt(6.0)


def _floor(sm, n=3):
    v = sm.reshape(-1, sm.shape[-1])
    v = torch.where(torch.isfinite(v), v, torch.full_like(v, float("inf")))
    lo = torch.sort(v, 0).values[:n]
    fin = torch.isfinite(lo)
    nf = fin.sum(0)
    m = torch.where(fin, lo, torch.zeros_like(lo)).sum(0) / nf.clamp(min=1).to(lo.dtype)
    return torch.where(nf > 0, m, torch.full_like(m, NAN))          # numpy nanmean of all-NaN = NaN


def _ends_floor(D, shape, peak_idx, k=3):
    N = D.shape[-1]
    acc = torch.zeros(N, device=D.device, dtype=D.dtype); cnt = torch.zeros(N, device=D.device, dtype=D.dtype); used = 0
    for ax, n in enumerate(shape):
        if n < 2 * k:
            continue
        used += 1
        far_hi = peak_idx[:, ax].to(D.dtype) < (n - 1) / 2.0
        for sl, pick in ((slice(0, k), ~far_hi), (slice(n - k, n), far_hi)):
            idx = [slice(None)] * D.ndim; idx[ax] = sl
            band = D[tuple(idx)].reshape(-1, N)
            fin = torch.isfinite(band)
            nb = fin.sum(0).to(D.dtype)
            m = torch.where(fin, band, torch.zeros_like(band)).sum(0) / nb.clamp(min=1)
            acc += torch.where(pick, m * nb, torch.zeros_like(m))
            cnt += torch.where(pick, nb, torch.zeros_like(nb))
    if used == 0:
        return _floor(D), torch.full((N,), 3.0, device=D.device, dtype=D.dtype)
    return torch.where(cnt > 0, acc / cnt.clamp(min=1), _floor(D)), cnt.clamp(min=1)


def _dilate(reg, shape, measured):
    grow = reg.clone()
    for ax, n in enumerate(shape):
        if n < 2:
            continue
        a = [slice(None)] * reg.ndim; b = [slice(None)] * reg.ndim
        a[ax] = slice(1, None); b[ax] = slice(None, -1)
        grow[tuple(a)] |= reg[tuple(b)]
        grow[tuple(b)] |= reg[tuple(a)]
    return grow & measured


def _support(sm_s, thr, shape, measured):
    N = sm_s.shape[-1]
    ninf = torch.full_like(sm_s, -float("inf"))
    flat = torch.where(torch.isfinite(sm_s), sm_s, ninf).reshape(-1, N)
    k = flat.argmax(0)
    above = torch.where(torch.isnan(sm_s), ninf, sm_s) >= thr
    if len(shape) == 1:
        M = shape[0]
        idx = torch.arange(M, device=sm_s.device)[:, None]
        below = ~above
        left = torch.where(below & (idx < k[None]), idx, torch.full_like(idx, -1)).max(0).values + 1
        right = torch.where(below & (idx > k[None]), idx, torch.full_like(idx, M)).min(0).values - 1
        return (idx >= left[None]) & (idx <= right[None]), k
    seed = torch.zeros(flat.shape, dtype=torch.bool, device=sm_s.device)
    seed[k, torch.arange(N, device=sm_s.device)] = True
    reg = seed.reshape(sm_s.shape) & above
    for _ in range(int(sum(shape))):
        grow = _dilate(reg, shape, measured) & above
        if torch.equal(grow, reg):
            break
        reg = grow
    return reg, k


def _faces(shape):
    for ax, n in enumerate(shape):
        for side, i in ((0, 0), (1, n - 1)):
            sl = [slice(None)] * len(shape); sl[ax] = i
            yield ax, side, tuple(sl)


def _fwhm(s, x, k):
    M, N = s.shape
    cols = torch.arange(N, device=s.device)
    lo, hi = (k - 1).clamp(0, M - 1), (k + 1).clamp(0, M - 1)
    peak = torch.maximum(torch.maximum(s[lo, cols], s[k, cols]), s[hi, cols])
    half = 0.5 * peak
    idx = torch.arange(M, device=s.device)[:, None]
    below = s < half[None]
    left = torch.where(below & (idx < k[None]), idx, torch.full_like(idx, -1)).max(0).values
    right = torch.where(below & (idx > k[None]), idx, torch.full_like(idx, M)).min(0).values

    def cross(e, step):
        e = e.clamp(0, M - 1); e2 = (e + step).clamp(0, M - 1)
        y1, y2 = s[e, cols], s[e2, cols]
        dy = y2 - y1
        t = torch.where(torch.abs(dy) > 0, (half - y1) / torch.where(dy == 0, torch.ones_like(dy), dy), torch.zeros_like(dy))
        return x[e] + t.clamp(0, 1) * (x[e2] - x[e])

    xl = torch.where(left >= 0, cross(left, +1), x[0].expand(N))
    xr = torch.where(right < M, cross(right, -1), x[-1].expand(N))
    return xr - xl


def _safe_div(a, b):
    return torch.where(b > 0, a / torch.where(b > 0, b, torch.ones_like(b)), torch.full_like(a, NAN))


def support_curve_torch(D, shape, coords, *, frac=0.0, nsig=2.0, pad=3, min_out=6, end_frac=0.10,
                        halves=None, Dw=None):
    """Torch twin of support.support_curve. D, Dw, halves: torch tensors (*shape, N) on one device.
    Returns a dict of numpy arrays (same keys as the numpy version)."""
    dev, dt = D.device, D.dtype
    Dw = D if Dw is None else Dw
    same_w = Dw is D
    N = D.shape[-1]; d = len(shape)
    measured = torch.isfinite(D[..., 0])
    meas_n = measured[..., None]
    nsm = math.sqrt(3.0 ** sum(1 for n in shape if n >= 3))
    sig = _noise(D, shape)
    sigw = sig if same_w else _noise(Dw, shape)
    sigw_sm = sigw / nsm
    sm0w = _smooth(Dw, shape)
    ninf = torch.full_like(sm0w, -float("inf"))
    kw = torch.where(torch.isfinite(sm0w), sm0w, ninf).reshape(-1, N).argmax(0)
    peak_idx = torch.stack(torch.unravel_index(kw, shape), 1)
    b0w, n_floor = _ends_floor(Dw, shape, peak_idx)
    bw = b0w.clone()
    for _ in range(2):
        smw = sm0w - bw
        pkw = torch.where(torch.isnan(smw), ninf, smw).reshape(-1, N).max(0).values
        thr = torch.maximum(frac * pkw, nsig * sigw_sm)
        reg, k = _support(smw, thr, shape, meas_n)
        W = reg
        for _p in range(pad):
            W = _dilate(W, shape, meas_n)
        n_out = (meas_n & ~W).reshape(-1, N).sum(0)
        med_w = _median0(torch.where(W | ~meas_n, torch.full_like(Dw, NAN), Dw).reshape(-1, N))
        bw = torch.where(n_out >= min_out, med_w, b0w)
    smw = sm0w - bw
    pkw = torch.where(torch.isnan(smw), ninf, smw).reshape(-1, N).max(0).values
    thr = torch.maximum(frac * pkw, nsig * sigw_sm)
    reg, k = _support(smw, thr, shape, meas_n)
    W = reg
    for _p in range(pad):
        W = _dilate(W, shape, meas_n)
    n_out = (meas_n & ~W).reshape(-1, N).sum(0)
    ok = n_out >= min_out
    sm0 = sm0w if same_w else _smooth(D, shape)
    med_out = _median0(torch.where(W | ~meas_n, torch.full_like(D, NAN), D).reshape(-1, N))
    b = torch.where(ok, med_out, _ends_floor(D, shape, peak_idx)[0])
    s = D - b
    sm = sm0 - b
    pk = torch.where(torch.isnan(sm), ninf, sm).reshape(-1, N).max(0).values
    zero = torch.zeros_like(s)
    Wm = W & meas_n
    sw = torch.where(Wm, s, zero)
    S = sw.reshape(-1, N).sum(0)
    grids = torch.meshgrid(*[_t(c, dev, dt) for c in coords], indexing="ij")
    cen = torch.full((N, d), NAN, device=dev, dtype=dt)
    for a in range(d):
        cen[:, a] = _safe_div((sw * grids[a][..., None]).reshape(-1, N).sum(0), S)
    for a in range(d):
        lo, hi = float(np.min(coords[a])), float(np.max(coords[a]))
        cen[:, a] = torch.where((cen[:, a] >= lo) & (cen[:, a] <= hi), cen[:, a], torch.full_like(cen[:, a], NAN))
    swp = sw.clamp(min=0.0)
    Sp = swp.reshape(-1, N).sum(0)
    cov = torch.full((N, d, d), NAN, device=dev, dtype=dt)
    for a in range(d):
        for c in range(a, d):
            da = grids[a][..., None] - cen[:, a]; dc = grids[c][..., None] - cen[:, c]
            v = _safe_div((swp * da * dc).reshape(-1, N).sum(0), Sp)
            cov[:, a, c] = v; cov[:, c, a] = v
    fw = torch.full((N, d), NAN, device=dev, dtype=dt)
    swm = torch.where(Wm, sm, zero).clamp(min=0.0)
    for a in range(d):
        other = tuple(i for i in range(d) if i != a)
        marg = swm.sum(dim=other) if other else swm
        if shape[a] >= 2:
            fw[:, a] = _fwhm(marg, _t(coords[a], dev, dt), marg.argmax(0))
    swr = torch.where(reg & meas_n, s, zero).clamp(min=0.0)
    Sr = swr.reshape(-1, N).sum(0)
    rms_w = torch.full((N, d), NAN, device=dev, dtype=dt)
    for a in range(d):
        ca = _safe_div((swr * grids[a][..., None]).reshape(-1, N).sum(0), Sr)
        va = _safe_div((swr * (grids[a][..., None] - ca) ** 2).reshape(-1, N).sum(0), Sr)
        rms_w[:, a] = torch.where(torch.isfinite(va), torch.sqrt(va.clamp(min=0)), torch.sqrt(cov[:, a, a].clamp(min=0)))
    cut_thr = torch.maximum(end_frac * pkw, 3.0 * sigw_sm)
    cut = torch.zeros((N, d, 2), dtype=torch.bool, device=dev)
    endf = torch.zeros(N, device=dev, dtype=dt)
    for ax, side, sl in _faces(shape):
        if shape[ax] < 2:
            continue
        face_s = torch.where(reg[sl], smw[sl], torch.full_like(smw[sl], -float("inf"))).reshape(-1, N).max(0).values
        cut[:, ax, side] = face_s > cut_thr
        endf = torch.fmax(endf, torch.where(pkw > 0, face_s / torch.where(pkw > 0, pkw, torch.ones_like(pkw)), torch.full_like(pkw, NAN)))
    endf = torch.where(torch.isfinite(endf), endf.clamp(min=0), torch.full_like(endf, NAN))
    truncated = cut.reshape(N, -1).any(1)
    all_above = (torch.where(torch.isnan(smw), ninf, smw) >= thr) & meas_n
    tot_above = torch.where(all_above, s.clamp(min=0), zero).reshape(-1, N).sum(0)
    in_reg = torch.where(reg, s.clamp(min=0), zero).reshape(-1, N).sum(0)
    main_share = _safe_div(in_reg, tot_above)
    steps = [float(np.median(np.diff(c))) if len(c) > 1 else 1.0 for c in coords]
    reg_cov = cov + torch.diag(_t([(st ** 2) / 12.0 for st in steps], dev, dt))[None]
    Pm = torch.linalg.inv(torch.where(torch.isfinite(reg_cov), reg_cov, torch.eye(d, device=dev, dtype=dt)[None].expand_as(reg_cov)))
    q = torch.zeros_like(D)
    for a in range(d):
        for c in range(d):
            da = grids[a][..., None] - cen[:, a]; dc = grids[c][..., None] - cen[:, c]
            q = q + da * Pm[:, a, c] * dc
    g = torch.where(Wm, torch.exp(-0.5 * q), zero)
    gg = (g * g).reshape(-1, N).sum(0)
    A = _safe_div((g * sw).reshape(-1, N).sum(0), gg)
    r = torch.where(Wm, s - A * g, zero)
    ss = (sw * sw).reshape(-1, N).sum(0)
    shape_resid = torch.sqrt(_safe_div((r * r).reshape(-1, N).sum(0), ss))
    n_sup = Wm.reshape(-1, N).sum(0)
    v_b = torch.where(ok, math.pi / (2.0 * n_out.clamp(min=1).to(dt)), 1.0 / n_floor)
    n1 = n_sup.clamp(min=1).to(dt)
    snr = torch.where(sig > 0, S / (sig * torch.sqrt(n1 + n1 ** 2 * v_b)), torch.full_like(S, NAN))
    if same_w:
        snr_w = snr
    else:
        Sw_ = torch.where(Wm, Dw - bw, zero).reshape(-1, N).sum(0)
        okw = n_out >= min_out
        v_bw = torch.where(okw, math.pi / (2.0 * n_out.clamp(min=1).to(dt)), 1.0 / n_floor)
        snr_w = torch.where(sigw > 0, Sw_ / (sigw * torch.sqrt(n1 + n1 ** 2 * v_bw)), torch.full_like(S, NAN))
    out = dict(centre=cen, cov=cov, fwhm=2.3548 * rms_w, fwhm_halfmax=fw, rms_width=rms_w, snr_w=snr_w, intensity=S,
               baseline=b, baseline_ok=ok, truncated=truncated, cut=cut, end_fraction=endf, snr=snr, sigma_frame=sig,
               n_support=n_sup, main_share=main_share, shape_resid=shape_resid, peak=pk, argmax=k)
    if halves is not None:
        cs = []
        for J in halves:
            bj0 = _ends_floor(J, shape, peak_idx)[0]
            med = _median0(torch.where(W | ~meas_n, torch.full_like(J, NAN), J).reshape(-1, N))
            bj = torch.where(ok, med, bj0)
            sj = torch.where(W & torch.isfinite(J), J - bj, torch.zeros_like(J))
            Sj = sj.reshape(-1, N).sum(0)
            cj = torch.full((N, d), NAN, device=dev, dtype=dt)
            for a in range(d):
                cj[:, a] = _safe_div((sj * grids[a][..., None]).reshape(-1, N).sum(0), Sj)
                lo, hi = float(np.min(coords[a])), float(np.max(coords[a]))
                cj[:, a] = torch.where((cj[:, a] >= lo) & (cj[:, a] <= hi), cj[:, a], torch.full_like(cj[:, a], NAN))
            cs.append(cj)
        out["diff"] = cs[0] - cs[1]
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def box_filter_rows(slab, k):
    """uniform_filter(size=(1,k,k), mode='nearest') of a (M, rows, W) slab, on its device."""
    h = k // 2
    x = Fnn.pad(slab[:, None], (h, h, h, h), mode="replicate")
    return Fnn.avg_pool2d(x, k, stride=1)[:, 0]
