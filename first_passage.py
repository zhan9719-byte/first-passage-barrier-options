import numpy as np
from scipy.special import ndtr


def eventual_upper_probability(S, L, U, mu, sigma):
    a, b, x = (np.log(L), np.log(U), np.log(S))
    alpha = mu - 0.5 * sigma ** 2
    z, width = (x - a, b - a)
    if abs(alpha) < 1e-10:
        return z / width
    lam = 2 * alpha / sigma ** 2
    return np.expm1(-lam * z) / np.expm1(-lam * width)


def expected_exit_time(S, L, U, mu, sigma):
    a, b, x = (np.log(L), np.log(U), np.log(S))
    alpha = mu - 0.5 * sigma ** 2
    if abs(alpha) < 1e-10:
        return (x - a) * (b - x) / sigma ** 2
    h = eventual_upper_probability(S, L, U, mu, sigma)
    return ((b - a) * h - (x - a)) / alpha


def survival_and_restricted_mean(S, L, U, H, mu, sigma, terms=1200):
    """Return the survival probability and restricted mean exit time for GBM."""
    if not L < S < U:
        return (0.0, 0.0)
    if H <= 0:
        return (1.0, 0.0)
    x, d = (np.log(S / L), np.log(U / L))
    alpha = mu - 0.5 * sigma ** 2
    beta = alpha / sigma ** 2
    n = np.arange(1, terms + 1)
    k = n * np.pi / d
    eigen = 0.5 * sigma ** 2 * k * k + alpha * alpha / (2 * sigma ** 2)
    integ = k * (1 - (-1.0) ** n * np.exp(beta * d)) / (beta * beta + k * k)
    coeff = 2 / d * np.exp(-beta * x) * np.sin(k * x) * integ
    survival = np.sum(coeff * np.exp(-eigen * H))
    restricted = np.sum(coeff * -np.expm1(-eigen * H) / eigen)
    return (float(survival), float(restricted))


def bs_down_out(S, K, B, T, r, q, sigma):
    """Continuous down-and-out call with zero rebate and B <= K."""
    if not 0 < B <= K:
        raise ValueError('Require 0 < B <= K')
    if S <= B:
        return 0.0
    if T <= 0:
        return max(S - K, 0.0)
    x = np.log(S / B)
    a = np.log(K / B)
    variance = sigma * sigma * T
    sd = np.sqrt(variance)
    m = r - q - 0.5 * sigma * sigma

    def integral(mean):
        return B * np.exp(mean + 0.5 * variance) * ndtr((mean + variance - a) / sd) - K * ndtr((mean - a) / sd)
    return float(np.exp(-r * T) * (integral(x + m * T) - np.exp(-2 * m * x / (sigma * sigma)) * integral(-x + m * T)))
