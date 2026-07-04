import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def _log_gamma(x):
    if x <= 0.0:
        return 1e300

    # Lanczos approximation (g=7, n=9 coefficients)
    lanczos_approx_p = (
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )

    if x < 0.5:
        return np.log(np.pi / np.sin(np.pi * x)) - _log_gamma(1.0 - x)

    x -= 1.0
    a = lanczos_approx_p[0]
    t = x + 7.5

    for i in range(1, 9):
        a += lanczos_approx_p[i] / (x + i)

    return (
        0.5 * np.log(2 * np.pi)
        + (x + 0.5) * np.log(t)
        - t
        + np.log(a)
    )


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def _gammainc_series(a, x, max_iter=300, eps=3e-14):
    if x <= 0.0:
        return 0.0

    ap = a
    delta = 1.0 / a
    total = delta

    for _ in range(max_iter):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * np.exp(-x + a * np.log(x) - _log_gamma(a))


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def _gammainc_cf(a, x, max_iter=300, eps=3e-14):
    fp_min = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fp_min
    d = 1.0 / b
    h = d

    for i in range(1, max_iter + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b

        if abs(d) < fp_min:
            d = fp_min

        c = b + an / c
        if abs(c) < fp_min:
            c = fp_min

        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < eps:
            break

    return np.exp(-x + a * np.log(x) - _log_gamma(a)) * h


@njit(cache=True)
def _regularized_gammainc(a, x):
    if x < 0.0:
        return 0.0
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gammainc_series(a, x)

    return 1.0 - _gammainc_cf(a, x)


@njit(cache=True)
def chi2_cdf_scalar(x, df):
    if x <= 0.0:
        return 0.0
    return _regularized_gammainc(df * 0.5, x * 0.5)
