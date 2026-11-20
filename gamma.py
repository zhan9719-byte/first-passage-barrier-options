import time

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

from first_passage import bs_down_out


def conditional_doc(S, K, B, T, r, q, variance):
    """Killed lognormal continuation over the final frozen-variance step."""
    S, v = np.broadcast_arrays(np.asarray(S, float), np.asarray(variance, float))
    if B > K:
        raise ValueError('Requires barrier<=strike')
    out = np.zeros(S.shape)
    alive = S > B
    deterministic = alive & (v * T < 1e-12)
    end = S * np.exp((r - q) * T)
    out[deterministic] = np.exp(-r * T) * np.maximum(end[deterministic] - K,
        0) * (end[deterministic] > B)
    keep = alive & ~deterministic
    if not np.any(keep):
        return out
    s = S[keep]
    v = v[keep]
    sd = np.sqrt(v * T)
    m = r - q - 0.5 * v
    x = np.log(s / B)
    a = np.log(K / B)
    d2 = (x + m * T - a) / sd
    d1 = d2 + sd
    direct = s * np.exp(-q * T) * ndtr(d1) - K * np.exp(-r * T) * ndtr(d2)
    y2 = (-x + m * T - a) / sd
    y1 = y2 + sd
    logw = -2 * m * x / v
    image1 = np.exp(logw + np.log(B) - x - q * T + log_ndtr(y1))
    image2 = np.exp(logw + np.log(K) - r * T + log_ndtr(y2))
    out[keep] = np.maximum(direct - image1 + image2, 0.0)
    return out


def _pairs(a):
    n = a.shape[-1] // 2
    return 0.5 * (a[..., :n] + a[..., n:])


def _greek_samples(payoff, h):
    return np.stack([payoff[1], (payoff[2] - payoff[0]) / (2 * h),
        (payoff[2] - 2 * payoff[1] + payoff[0]) / h ** 2])


def coupled_samples(S=100.0, K=100.0, B=95.0, T=0.5, r=0.03, q=0.015, params=None, n_paths=20000,
    n_steps=128, seed=1729, spot_bump=0.5, include_control=True, include_raw=True,
    include_tail=True, model='heston'):
    """Return antithetic-pair samples of the price and finite-bump Greeks."""
    start = time.perf_counter()
    if n_paths % 2 or n_paths < 4 or n_steps < 1:
        raise ValueError('invalid path/step count')
    p = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    if params:
        p.update(params)
    if model == 'bs':
        p['xi'] = 0
        p['theta'] = p['v0']
    dt = T / n_steps
    N = n_paths // 2
    rng = np.random.default_rng(seed)
    spots = np.array([S - spot_bump, S, S + spot_bump])
    dist = np.log(spots / B)[:, None]
    relative = np.zeros(n_paths)
    v = np.full(n_paths, p['v0'])
    weights = np.ones((3, n_paths)) * (spots[:, None] > B)
    meanvar = p['theta'] + (p['v0'] - p['theta']) * -np.expm1(-p['kappa'] * T) / (p['kappa'] * T) if p['kappa'] > 0 else p['v0']
    sigmac = np.sqrt(max(meanvar, 1e-12))
    crel = np.zeros(n_paths)
    cw = weights.copy()
    tail = None
    ctail = None
    negative = 0
    for step in range(n_steps):
        vp = np.maximum(v, 0.0)
        if step == n_steps - 1 and include_tail:
            tail = weights * conditional_doc(spots[:, None] * np.exp(relative), K, B, dt, r, q,
                vp[None, :]) * np.exp(-r * (T - dt))
            if include_control:
                ctail = cw * conditional_doc(spots[:, None] * np.exp(crel), K, B, dt, r, q,
                    np.full((1, n_paths), sigmac ** 2)) * np.exp(-r * (T - dt))
            if not include_raw:
                break
        z = rng.standard_normal(N)
        z1 = np.r_[z, -z]
        zi = rng.standard_normal(N)
        z2 = p['rho'] * z1 + np.sqrt(1 - p['rho'] ** 2) * np.r_[zi, -zi]
        iv = vp * dt
        rn = relative + (r - q - 0.5 * vp) * dt + np.sqrt(iv) * z1
        a = dist + relative
        b = dist + rn
        prob = np.where((a > 0) & (b > 0), -np.expm1(-2 * np.maximum(a, 0) * np.maximum(b,
            0) / np.maximum(iv, 1e-300)), 0.0)
        weights *= prob
        v += p['kappa'] * (p['theta'] - vp) * dt + p['xi'] * np.sqrt(iv) * z2
        negative += np.count_nonzero(v < 0)
        relative = rn
        if include_control:
            cn = crel + (r - q - 0.5 * sigmac ** 2) * dt + sigmac * np.sqrt(dt) * z1
            a = dist + crel
            b = dist + cn
            cw *= np.where((a > 0) & (b > 0), -np.expm1(-2 * np.maximum(a, 0) * np.maximum(b,
                0) / (sigmac ** 2 * dt)), 0.0)
            crel = cn
    result = {'tail': _pairs(_greek_samples(tail, spot_bump))} if include_tail else {}
    if include_raw:
        raw = np.exp(-r * T) * np.maximum(spots[:, None] * np.exp(relative) - K, 0) * weights
        result['raw'] = _pairs(_greek_samples(raw, spot_bump))
    if include_control and include_tail:
        result['control'] = _pairs(_greek_samples(ctail, spot_bump))
        vals = np.array([bs_down_out(float(s), K, B, T, r, q, sigmac) for s in spots])
        result['control_expectation'] = _greek_samples(vals, spot_bump)
    result['elapsed'] = time.perf_counter() - start
    result['sigma_control'] = sigmac
    result['negative_variance_fraction'] = negative / (n_paths * n_steps)
    return result


def survival_samples(S=100.0, K=100.0, B=95.0, T=0.5, r=0.03, q=0.015, params=None, n_paths=20000,
    n_steps=128, seed=1729, spot_bump=0.5, include_control=True, model='heston'):
    """Condition on each surviving endpoint and integrate the final step."""
    start = time.perf_counter()
    p = dict(v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7)
    if params:
        p.update(params)
    if model == 'bs':
        p['xi'] = 0
        p['theta'] = p['v0']
    if n_paths % 2 or n_paths < 4 or n_steps < 1:
        raise ValueError('invalid path/step count')
    dt = T / n_steps
    N = n_paths // 2
    rng = np.random.default_rng(seed)
    spots = np.array([S - spot_bump, S, S + spot_bump])
    dist = np.log(spots / B)
    x = np.broadcast_to(dist[:, None], (3, n_paths)).copy()
    v = np.full(x.shape, p['v0'])
    weights = (x > 0).astype(float)
    cx = x.copy()
    cw = weights.copy()
    meanvar = p['theta'] + (p['v0'] - p['theta']) * -np.expm1(-p['kappa'] * T) / (p['kappa'] * T) if p['kappa'] > 0 else p['v0']
    vc = max(meanvar, 1e-12)
    sigmac = np.sqrt(vc)
    negative = 0
    for step in range(n_steps - 1):
        zz = rng.standard_normal(N)
        u = ndtr(np.r_[zz, -zz])[None, :]
        zz = rng.standard_normal(N)
        zi = np.r_[zz, -zz][None, :]
        vp = np.maximum(v, 0.0)
        iv = vp * dt
        sd = np.sqrt(iv)
        mean = x + (r - q - 0.5 * vp) * dt
        active = iv > 1e-14
        threshold = -mean / np.maximum(sd, 1e-300)
        surv = ndtr(-threshold)
        tailprob = np.clip((1 - u) * surv, 1e-15, 1 - 1e-15)
        z = -ndtri(tailprob)
        nxt = np.where(active, mean + sd * z, mean)
        probability = np.where(active, surv, (mean > 0).astype(float))
        bridge = np.where(active, -np.expm1(-2 * np.maximum(x, 0) * np.maximum(nxt,
            0) / np.maximum(iv, 1e-300)), (nxt > 0).astype(float))
        weights *= probability * bridge
        zvar = p['rho'] * z + np.sqrt(1 - p['rho'] ** 2) * zi
        v += p['kappa'] * (p['theta'] - vp) * dt + p['xi'] * sd * zvar
        negative += np.count_nonzero(v < 0)
        x = nxt
        if include_control:
            cm = cx + (r - q - 0.5 * vc) * dt
            cs = np.sqrt(vc * dt)
            ps = ndtr(cm / cs)
            cz = -ndtri(np.clip((1 - u) * ps, 1e-15, 1 - 1e-15))
            cn = cm + cs * cz
            cw *= ps * -np.expm1(-2 * np.maximum(cx, 0) * np.maximum(cn, 0) / (vc * dt))
            cx = cn
    y = weights * conditional_doc(B * np.exp(x), K, B, dt, r, q, np.maximum(v,
        0.0)) * np.exp(-r * (T - dt))
    out = {'survival': _pairs(_greek_samples(y, spot_bump)), 'sigma_control': sigmac,
        'negative_variance_fraction': negative / (3 * n_paths * max(1, n_steps - 1))}
    if include_control:
        cy = cw * conditional_doc(B * np.exp(cx), K, B, dt, r, q, np.full(cx.shape,
            vc)) * np.exp(-r * (T - dt))
        out['control'] = _pairs(_greek_samples(cy, spot_bump))
        vals = np.array([bs_down_out(float(s), K, B, T, r, q, sigmac) for s in spots])
        out['control_expectation'] = _greek_samples(vals, spot_bump)
    out['elapsed'] = time.perf_counter() - start
    return out
