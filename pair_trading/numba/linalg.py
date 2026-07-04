import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def invert_numba(A):
    # for under ~200
    n = A.shape[0]
    A = A.copy()
    I = np.eye(n, dtype=A.dtype)

    # forward elimination
    for i in range(n):
        pivot = A[i, i]

        # normalize row
        for j in range(n):
            A[i, j] /= pivot
            I[i, j] /= pivot

        # eliminate other rows
        for k in range(n):
            if k != i:
                factor = A[k, i]
                for j in range(n):
                    A[k, j] -= factor * A[i, j]
                    I[k, j] -= factor * I[i, j]

    return I


@njit(fastmath=True, boundscheck=False, nogil=True, cache=True)
def roll_numba(x, shift):
    n = x.shape[0]
    y = np.empty(n, dtype=x.dtype)

    for i in range(shift):
        y[i] = 0

    for i in range(shift, n):
        y[i] = x[i - shift]

    return y


@njit(fastmath=True, boundscheck=False, nogil=True, cache=True)
def diff_numba(x):
    n = x.shape[0]

    y = np.empty(n - 1, dtype=x.dtype)

    for i in range(n - 1):
        y[i] = x[i + 1] - x[i]

    return y


@njit(boundscheck=False, cache=True)
def cholesky_numba(A):
    """
    Lower-triangular Cholesky decomposition:
    A = L @ L.T
    A must be symmetric positive definite.
    """

    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            s = 0.0

            # dot product of row i and j
            for k in range(j):
                s += L[i, k] * L[j, k]

            val = A[i, j] - s
            L[i, j] = np.sqrt(max(0.0, val)) if j == i else val / L[j, j]

    return L


@njit(fastmath=True, nogil=True, boundscheck=False, cache=True)
def solve_numba(A, b):
    n = A.shape[0]

    # copy because LU modifies matrix
    LU = A.copy()
    piv = np.arange(n)

    # --- LU decomposition with partial pivoting ---
    for k in range(n):

        max_row = k
        max_val = abs(LU[k, k])

        for i in range(k+1, n):
            v = abs(LU[i, k])
            if v > max_val:
                max_val = v
                max_row = i

        if max_row != k:
            tmp = LU[k].copy()
            LU[k] = LU[max_row]
            LU[max_row] = tmp

            t = piv[k]
            piv[k] = piv[max_row]
            piv[max_row] = t

        for i in range(k+1, n):
            LU[i, k] /= LU[k, k]
            for j in range(k+1, n):
                LU[i, j] -= LU[i, k] * LU[k, j]

    # --- forward substitution (Ly = Pb) ---
    y = np.empty(n)
    for i in range(n):
        y[i] = b[piv[i]]
        for j in range(i):
            y[i] -= LU[i, j] * y[j]

    # --- backward substitution (Ux = y) ---
    x = np.empty(n)
    for i in range(n-1, -1, -1):
        v = y[i]
        for j in range(i+1, n):
            v -= LU[i, j] * x[j]
        x[i] = v / LU[i, i]

    return x


@njit(fastmath=True, cache=True)
def slogdet_numba(A):
    A = A.copy()  # avoid mutating input
    n = A.shape[0]

    sign = 1.0
    logdet = 0.0

    for k in range(n):
        # --- pivot selection ---
        piv = k
        max_val = abs(A[k, k])
        for i in range(k + 1, n):
            val = abs(A[i, k])
            if val > max_val:
                max_val = val
                piv = i

        # singular matrix
        if max_val == 0.0:
            return 0.0, -np.inf

        # row swap if needed
        if piv != k:
            for j in range(n):
                tmp = A[k, j]
                A[k, j] = A[piv, j]
                A[piv, j] = tmp
            sign *= -1.0

        pivot = A[k, k]
        logdet += np.log(abs(pivot))

        # eliminate below
        for i in range(k + 1, n):
            A[i, k] /= pivot
            factor = A[i, k]
            for j in range(k + 1, n):
                A[i, j] -= factor * A[k, j]

    return sign, logdet


@njit(fastmath=True, cache=True)
def lstsq_numba(X, Y):
    """
    Solve min ||X B - Y|| using QR decomposition.

    X: (n, k)
    Y: (n, m)
    Returns:
        B: (k, m)
    """
    n, k = X.shape
    _, m = Y.shape

    # Copy because we overwrite
    R = X.copy()
    QTY = Y.copy()

    # --- Householder QR ---
    for j in range(k):
        # Compute norm of column below diagonal
        norm_x = 0.0
        for i in range(j, n):
            norm_x += R[i, j] * R[i, j]
        norm_x = np.sqrt(norm_x)

        if norm_x == 0.0:
            continue

        # Build Householder vector
        sign = -1.0 if R[j, j] < 0 else 1.0
        u1 = R[j, j] + sign * norm_x

        v = np.zeros(n - j)
        v[0] = u1
        for i in range(j + 1, n):
            v[i - j] = R[i, j]

        # Normalize v
        norm_v = 0.0
        for i in range(v.shape[0]):
            norm_v += v[i] * v[i]
        norm_v = np.sqrt(norm_v)

        if norm_v == 0.0:
            continue

        for i in range(v.shape[0]):
            v[i] /= norm_v

        # Apply reflection to R
        for col in range(j, k):
            dot = 0.0
            for i in range(v.shape[0]):
                dot += v[i] * R[j + i, col]
            for i in range(v.shape[0]):
                R[j + i, col] -= 2.0 * v[i] * dot

        # Apply reflection to Q^T Y
        for col in range(m):
            dot = 0.0
            for i in range(v.shape[0]):
                dot += v[i] * QTY[j + i, col]
            for i in range(v.shape[0]):
                QTY[j + i, col] -= 2.0 * v[i] * dot

    # --- Back substitution: solve R B = Q^T Y ---
    B = np.zeros((k, m))

    for col in range(m):
        for i in range(k - 1, -1, -1):
            s = QTY[i, col]
            for j in range(i + 1, k):
                s -= R[i, j] * B[j, col]
            B[i, col] = s / R[i, i]

    return B
