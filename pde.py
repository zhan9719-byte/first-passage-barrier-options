import time

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.sparse import diags, eye, kron, lil_matrix
from scipy.sparse.linalg import splu


def _march(A, initial, bc, T, nt):
    """Crank–Nicolson with four backward Euler half-steps at startup."""
    if nt < 2:
        raise ValueError('nt must be at least 2 for Rannacher startup')
    if T <= 0:
        raise ValueError('PDE maturity must be positive')
    h = T / nt
    I = eye(A.shape[0], format='csc')
    u = initial.copy()
    t = 0.0
    be = splu((I - 0.5 * h * A).tocsc())
    for j in range(4):
        t += 0.5 * h
        u = be.solve(u + 0.5 * h * bc(t))
    if nt > 2:
        lhs = splu((I - 0.5 * h * A).tocsc())
        rhs = (I + 0.5 * h * A).tocsr()
        for j in range(2, nt):
            u = lhs.solve(rhs @ u + 0.5 * h * (bc(t) + bc(t + h)))
            t += h
    return u


def _variance_matrices(v, kappa, theta, xi):
    n = len(v)
    D1 = lil_matrix((n, n))
    D2 = lil_matrix((n, n))
    for j in range(1, n - 1):
        hm = v[j] - v[j - 1]
        hp = v[j + 1] - v[j]
        D1[j, j - 1] = -hp / (hm * (hm + hp))
        D1[j, j] = (hp - hm) / (hm * hp)
        D1[j, j + 1] = hm / (hp * (hm + hp))
        D2[j, j - 1] = 2 / (hm * (hm + hp))
        D2[j, j] = -2 / (hm * hp)
        D2[j, j + 1] = 2 / (hp * (hm + hp))
    D1[0, 0] = -1 / (v[1] - v[0])
    D1[0, 1] = 1 / (v[1] - v[0])
    D1[-1, -2] = -1 / (v[-1] - v[-2])
    D1[-1, -1] = 1 / (v[-1] - v[-2])
    D1 = D1.tocsr()
    D2 = D2.tocsr()
    Av = diags(kappa * (theta - v)) @ D1 + diags(0.5 * xi * xi * v) @ D2
    return (D1, D2, Av)


def focused_log_grid(lower, upper, K, nx, focus_width=0.1, grid='sinh'):
    if nx < 7 or not 0 < lower < upper or focus_width <= 0:
        raise ValueError('Invalid grid settings')
    a, b = (np.log(lower), np.log(upper))
    if grid == 'uniform':
        return np.linspace(a, b, nx)
    if grid != 'sinh':
        raise ValueError('grid must be uniform or sinh')
    c = np.log(K)
    if not a < c < b:
        c = 0.5 * (a + b)
    left = np.arcsinh((a - c) / focus_width)
    right = np.arcsinh((b - c) / focus_width)
    nl = int(np.clip(round((nx - 1) * -left / (right - left)), 2, nx - 3))
    nr = nx - 1 - nl
    y = np.r_[np.linspace(left, 0.0, nl + 1), np.linspace(0.0, right, nr + 1)[1:]]
    x = c + focus_width * np.sinh(y)
    x[0], x[nl], x[-1] = (a, c, b)
    return x


def log_derivative_matrices(x):
    n = len(x) - 2
    D1 = lil_matrix((n, n))
    D2 = lil_matrix((n, n))
    left = []
    right = []
    for i in range(n):
        j = i + 1
        hm = x[j] - x[j - 1]
        hp = x[j + 1] - x[j]
        d1 = np.array([-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp))])
        d2 = np.array([2 / (hm * (hm + hp)), -2 / (hm * hp), 2 / (hp * (hm + hp))])
        for k, offset in enumerate([-1, 0, 1]):
            if 0 <= i + offset < n:
                D1[i, i + offset] = d1[k]
                D2[i, i + offset] = d2[k]
        if i == 0:
            left = [d1[0], d2[0]]
        if i == n - 1:
            right = [d1[2], d2[2]]
    return (D1.tocsr(), D2.tocsr(), np.asarray(left), np.asarray(right))


def call_cell_average(x, K, local_only=True):
    a = 0.5 * (x[:-2] + x[1:-1])
    b = 0.5 * (x[1:-1] + x[2:])
    c = np.maximum(a, np.log(K))
    length = np.maximum(b - c, 0.0)
    numer = np.exp(c) * np.expm1(length) - K * length
    averages = np.where(length > 0, numer / (b - a), 0.0)
    if local_only:
        return np.where((a < np.log(K)) & (b > np.log(K)), averages,
            np.maximum(np.exp(x[1:-1]) - K, 0.0))
    return averages


def heston_pde_refined(S, K, T, r, q, params, B=85.0, nx=201, nv=81, nt=200, vmax=1.0, smax=400.0,
    return_grid=False, payoff='call', U=None, zero_upwind=False, smin=None, grid='sinh',
    smoothing=False, focus_width=0.1):
    """Solve the barrier PDE on a log-price × variance grid."""
    started = time.perf_counter()
    v0, kappa, theta, xi, rho = [float(params[k]) for k in ['v0', 'kappa', 'theta', 'xi', 'rho']]
    if v0 >= vmax or min(v0, kappa, theta, xi) < 0 or abs(rho) > 1:
        raise ValueError('Invalid parameters/domain')
    lower = B if B is not None else smin if smin is not None else S * 0.01
    upper = U if U is not None else smax
    if S <= lower:
        return {'price': 0.0, 'delta': 0.0, 'gamma': 0.0} if return_grid else 0.0
    if S >= upper:
        raise ValueError('S must be below upper spatial boundary')
    x = focused_log_grid(lower, upper, K, nx, focus_width, grid)
    n = nx - 2
    v = vmax * (np.arange(nv) / (nv - 1)) ** 2
    Dx, Dxx, leftcoef, rightcoef = log_derivative_matrices(x)
    Dv, Dvv, Av = _variance_matrices(v, kappa, theta, xi)
    A = (kron(diags(v), 0.5 * (Dxx - Dx)) + kron(eye(nv), (r - q) * Dx - r * eye(n)) + kron(Av,
        eye(n)) + kron(diags(rho * xi * v) @ Dv, Dx)).tocsc()
    drift = r - q - 0.5 * v
    right = 0.5 * v * rightcoef[1] + drift * rightcoef[0]
    if zero_upwind:
        d = r - q
        Dup = lil_matrix((n, n))
        for i in range(n):
            if d >= 0:
                step = x[i + 2] - x[i + 1]
                Dup[i, i] = -1 / step
                if i + 1 < n:
                    Dup[i, i + 1] = 1 / step
            else:
                step = x[i + 1] - x[i]
                Dup[i, i] = 1 / step
                if i - 1 >= 0:
                    Dup[i, i - 1] = -1 / step
        select = np.zeros(nv)
        select[0] = 1.0
        A = A + kron(diags(select), d * (Dup.tocsr() - Dx)).tocsc()
        right[0] = max(d, 0) / (x[-1] - x[-2])

    def high(t):
        return 0.0 if payoff == 'dnt' else upper * np.exp(-q * t) - K * np.exp(-r * t)

    def bc(t):
        z = np.zeros((nv, n))
        z[:, -1] = right * high(t)
        return z.ravel()
    if payoff == 'dnt':
        initial = np.ones(n)
    elif payoff == 'call':
        initial = call_cell_average(x, K) if smoothing else np.maximum(np.exp(x[1:-1]) - K, 0.0)
    else:
        raise ValueError('payoff must be call or dnt')
    inner = _march(A, np.tile(initial, nv), bc, T, nt).reshape(nv, n)
    values = np.column_stack([np.zeros(nv), inner, np.full(nv, high(T))])
    spl = RectBivariateSpline(v, x, values, kx=3, ky=3)
    xx = np.log(S)
    price = float(spl.ev(v0, xx))
    if not return_grid:
        return price
    px = float(spl.ev(v0, xx, dy=1))
    pxx = float(spl.ev(v0, xx, dy=2))
    column = CubicSpline(v, values, axis=0)(v0)
    ids = np.sort(np.argsort(abs(x - xx))[:5])
    poly = np.polynomial.polynomial.polyfit(x[ids] - xx, column[ids], 4)
    delta_poly5 = poly[1] / S
    gamma_poly5 = (2 * poly[2] - poly[1]) / S ** 2
    return dict(price=price, delta=px / S, gamma=(pxx - px) / S ** 2, x=x, v=v, values=values,
        delta_poly5=float(delta_poly5), gamma_poly5=float(gamma_poly5),
        elapsed=time.perf_counter() - started, nx=nx, nv=nv, nt=nt, vmax=vmax, smax=upper,
        min_value=float(values.min()), grid=grid, smoothing=bool(smoothing),
        focus_width=focus_width, minimum_dx=float(np.min(np.diff(x))),
        maximum_dx=float(np.max(np.diff(x))), strike_aligned=bool(np.any(x == np.log(K))))
