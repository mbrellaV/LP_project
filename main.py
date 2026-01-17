import numpy as np
from numpy.linalg import solve as np_solve

tolerance = 1e-9

def solve(lp):
    m, n = lp.num_rows, lp.num_columns

    A = np.zeros((m, n))
    b = np.zeros(m)
    c = np.array(lp.objective, dtype=float)

    for i, cons in enumerate(lp.constraints):
        b[i] = cons['rhs']

        for j, coef in cons['coefficients'].items():
            A[i, j] = coef

    basis = list(lp.basis)

    iterations = max(1000, 10 * m * n)

    for _ in range(iterations):
        B = A[:, basis]
        x_B = np_solve(B, b)
        y = np_solve(B.T, c[basis])
        r = c - A.T @ y

        k = -1

        for j in range(n):
            if j not in basis and r[j] < -tolerance:
                k = j
                break

        if k == -1:
            x = np.zeros(n)

            for i, j in enumerate(basis):
                x[j] = x_B[i]

            return {"primal": x.tolist(), "dual": y.tolist()}

        d = np_solve(B, A[:, k])

        theta, l = np.inf, -1

        for i in range(m):
            if d[i] > tolerance and x_B[i] / d[i] < theta:
                theta, l = x_B[i] / d[i], i

        if l == -1:
            ray = np.zeros(n)
            ray[k] = 1

            for i, j in enumerate(basis):
                ray[j] = -d[i]

            return {"status": "unbounded", "ray": ray.tolist()}

        basis[l] = k

    return {"status": "error"}
