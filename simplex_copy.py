import numpy as np
from numpy.linalg import solve as np_solve

tolerance = 1e-9

def simplex(lp):
    m, n = lp.num_rows, lp.num_columns

    A = np.zeros((m, n))
    b = np.zeros(m)
    c = np.array(lp.objective, dtype=float)

    for i, cons in enumerate(lp.constraints):
        b[i] = cons['rhs']

        for j, coef in cons['coefficients'].items():
            A[i, j] = coef

    if lp.has_basis:
        basis = list(lp.basis)
    else:
        return {"status": "no basis"}

    iterations = 10000

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

            return {"primal": x.tolist(), "dual": y.tolist(), "basis": basis}

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

def solve(lp):
    if lp.has_basis:
        return simplex(lp)
    else:
        m, n = lp.num_rows, lp.num_columns

        A = np.zeros((m, n))
        b = np.zeros(m)

        for i, cons in enumerate(lp.constraints):
            b[i] = cons["rhs"]

            for j, coef in cons["coefficients"].items():
                A[i, j] = coef

            if b[i] < 0:
                A[i, :] *= -1
                b[i] *= -1

        A_bar = np.hstack([A, np.eye(m)])
        c_bar = np.zeros(n + m)
        c_bar[n:] = 1.0

        aux_data = {"sense": "minimize", "objective": c_bar.tolist(), "signs": [1] * (n + m), "constraints": []}

        for i in range(m):
            coeffs = {j: A_bar[i, j] for j in range(n + m) if abs(A_bar[i, j]) > tolerance}
            aux_data["constraints"].append({"coefficients": coeffs, "relation": "=", "rhs": b[i]})

        aux_lp = type(lp)(aux_data)
        aux_lp.basis = list(range(n, n + m))
        result = solve(aux_lp)

        if "primal" not in result:
            return {"status": "infeasible"}

        x_bar = np.array(result["primal"])
        if x_bar[n:].sum() > tolerance:
            return {"status": "infeasible", "farkas": result["dual"]}

        basis = []
        rows = []
        for i, j in enumerate(result["basis"]):
            if j < n:
                basis.append(j)
                rows.append(i)

        lp.constraints = [lp.constraints[i] for i in rows]
        lp.basis = basis

        return solve(lp)
