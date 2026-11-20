from functools import lru_cache

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr


PARAMS = ('v0', 'kappa', 'theta', 'xi', 'rho')


LOWER = np.array([0.0025, 0.05, 0.0025, 0.05, -0.99])


UPPER = np.array([0.5, 15.0, 0.5, 3.0, -0.02])


@lru_cache(maxsize=12)
def quadrature(n=128, u_max=200.0):
    x, w = np.polynomial.legendre.leggauss(n)
    return (0.5 * u_max * (x + 1), 0.5 * u_max * w)


def heston_cf(u, T, params):
    """Characteristic function of log(S_T/F) under the pricing measure."""
    v0, kappa, theta, xi, rho = [params[k] for k in PARAMS]
    u = np.asarray(u, dtype=complex)
    T = np.asarray(T)
    b = kappa - rho * xi * 1j * u
    d = np.sqrt(b * b + xi * xi * (u * u + 1j * u))
    d = np.where(d.real < 0, -d, d)
    g = (b - d) / (b + d)
    ed = np.exp(-d * T)
    C = kappa * theta / (xi * xi) * ((b - d) * T - 2 * np.log((1 - g * ed) / (1 - g)))
    D = (b - d) / (xi * xi) * (1 - ed) / (1 - g * ed)
    return np.exp(C + D * v0)


def heston_forward(F, K, T, discount, params, n=128, u_max=200.0, option_type='call'):
    F, K, T, discount = np.broadcast_arrays(F, K, T, discount)
    shape = F.shape
    F, K, T, discount = [x.ravel() for x in (F, K, T, discount)]
    u, w = quadrature(n, float(u_max))
    unique, inv = np.unique(T, return_inverse=True)
    cf = heston_cf(u[None, :] - 0.5j, unique[:, None], params)[inv]
    integral = np.real(np.exp(1j * np.log(F / K)[:, None] * u) * cf) @ (w / (u * u + 0.25))
    call = discount * (F - np.sqrt(F * K) * integral / np.pi)
    typ = np.broadcast_to(np.asarray(option_type), shape).ravel()
    out = np.where(typ == 'put', call - discount * (F - K), call).reshape(shape)
    return float(out) if out.ndim == 0 else out


def heston_call(S, K, T, r, q, params, n=128, u_max=200.0):
    return heston_forward(np.asarray(S) * np.exp((np.asarray(r) - q) * T), K, T,
        np.exp(-np.asarray(r) * T), params, n, u_max)


def bs_forward(F, K, T, discount, sigma, option_type='call'):
    F, K, T, discount, sigma = np.broadcast_arrays(F, K, T, discount, sigma)
    vol = np.maximum(sigma * np.sqrt(T), 1e-14)
    d1 = np.log(F / K) / vol + 0.5 * vol
    d2 = d1 - vol
    call = discount * (F * ndtr(d1) - K * ndtr(d2))
    out = np.where(np.asarray(option_type) == 'put', call - discount * (F - K), call)
    return float(out) if out.ndim == 0 else out


def implied_vol(price, F, K, T, discount=1.0, option_type='call'):
    intrinsic = discount * max(F - K, 0) if option_type == 'call' else discount * max(K - F, 0)
    upper = discount * (F if option_type == 'call' else K)
    if T <= 0 or not intrinsic + 1e-09 < price < upper - 1e-09:
        return np.nan
    try:
        return brentq(lambda s: bs_forward(F, K, T, discount, s, option_type) - price, 0.0001, 5.0,
            xtol=1e-09)
    except ValueError:
        return np.nan


class PanelPricer:

    def __init__(self, panel, n=128, u_max=200.0):
        self.F = np.asarray(panel.F, float)
        self.K = np.asarray(panel.K, float)
        self.T = np.asarray(panel.T if not hasattr(panel, 'columns') else panel['T'], float)
        self.discount = np.asarray(panel.discount, float)
        self.put = np.asarray(panel['type']) == 'put'
        self.u, self.w = quadrature(n, float(u_max))
        self.unique, self.inv = np.unique(self.T, return_inverse=True)
        self.kernel = np.exp(1j * np.log(self.F / self.K)[:,
            None] * self.u) * (self.w / (self.u * self.u + 0.25))
        self.prefactor = self.discount * np.sqrt(self.F * self.K) / np.pi
        self.constant = self.discount * np.where(self.put, self.K, self.F)

    def __call__(self, p):
        params = dict(zip(PARAMS, p)) if not isinstance(p, dict) else p
        cf = heston_cf(self.u[None, :] - 0.5j, self.unique[:, None], params)[self.inv]
        return self.constant - self.prefactor * np.real(np.sum(self.kernel * cf, axis=1))


def scales(panel):
    return np.maximum(np.asarray(panel.half_spread, float), 0.0001 * np.asarray(panel.S, float))


def fit_heston(panel, start, method='midpoint', anchor=None, lam=0.05, target=None, max_nfev=200):
    """Fit four log-positive parameters and rho by spread-weighted least squares."""
    pricer = PanelPricer(panel)
    weights = scales(panel)
    mid = np.asarray(panel.mid, float) if target is None else np.asarray(target, float)
    x0 = np.r_[np.log([start[k] for k in PARAMS[:4]]), start['rho']]
    lo = np.r_[np.log(LOWER[:4]), LOWER[4]]
    hi = np.r_[np.log(UPPER[:4]), UPPER[4]]
    x0 = np.clip(x0, lo + 1e-07, hi - 1e-07)
    if anchor is None:
        anchor = start
    a = np.r_[np.log([anchor[k] for k in PARAMS[:4]]), anchor['rho']]
    regscale = np.array([1.0, 2.0, 1.0, 1.0, 0.5])

    def residual(x):
        p = np.r_[np.exp(x[:4]), x[4]]
        z = (pricer(p) - mid) / weights
        if method in ('deadzone', 'regularized'):
            z = np.sign(z) * np.maximum(np.abs(z) - np.asarray(panel.half_spread) / weights, 0)
        if method == 'regularized':
            z = np.r_[z, np.sqrt(len(panel) * lam) * (x - a) / regscale]
        return z
    opt = least_squares(residual, x0, bounds=(lo, hi), max_nfev=max_nfev, ftol=2e-07, xtol=2e-07,
        gtol=2e-07)
    p = np.r_[np.exp(opt.x[:4]), opt.x[4]]
    return (dict(zip(PARAMS, p)), dict(success=bool(opt.success), nfev=opt.nfev,
        cost=float(2 * opt.cost / len(panel)), optimality=float(opt.optimality)))
