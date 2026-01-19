import numpy as np
from numpy.linalg import solve as np_solve

tolerance = 1e-9

def simplex(lp, A, b, c, basis):
    m, n = A.shape
    iterations = 10000

    basis = list(basis)
    basis_set = set(basis)

    for _ in range(iterations):
        B = A[:, basis]
        try:
            x_B = np_solve(B, b)
        except np.linalg.LinAlgError:
            return {"farkas": np.zeros(m).tolist()}

        y = np_solve(B.T, c[basis])
        r = c - A.T @ y

        entering = -1
        for j in range(n):
            if j not in basis_set and r[j] < -tolerance:
                entering = j
                break

        if entering == -1:
            x = np.zeros(n)
            for i, j in enumerate(basis):
                x[j] = x_B[i]
            return {"primal": x.tolist(), "dual": y.tolist(), "basis": basis}

        d = np_solve(B, A[:, entering])

        theta = np.inf
        leaving = -1
        for i in range(m):
            if d[i] > tolerance:
                if x_B[i] / d[i] < theta:
                    theta = x_B[i] / d[i]

        for i in range(m):
            if d[i] > tolerance:
                if abs(x_B[i] / d[i] - theta) < tolerance:
                    if leaving == -1 or basis[i] < basis[leaving]:
                        leaving = i

        if leaving == -1:
            ray = np.zeros(n)
            ray[entering] = 1
            for i, j in enumerate(basis):
                ray[j] = -d[i]
            return {"status": "unbounded", "ray": ray.tolist()}

        basis_set.remove(basis[leaving])
        basis[leaving] = entering
        basis_set.add(entering)

    return {"status": "error"}


def solve(lp):
    m, n = lp.num_rows, lp.num_columns

    A = np.zeros((m, n))
    b = np.zeros(m)
    c = np.array(lp.objective, dtype=float)

    for i, cons in enumerate(lp.constraints):
        b[i] = cons["rhs"]
        for j, coef in cons["coefficients"].items():
            A[i, j] = coef

    if lp.has_basis:
        return simplex(lp, A, b, c, list(lp.basis))

    for i in range(m):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1

    A_phase1 = np.hstack([A, np.eye(m)])
    c_phase1 = np.concatenate([np.zeros(n), np.ones(m)])
    basis_phase1 = list(range(n, n + m))

    result_phase1 = simplex(lp, A_phase1, b, c_phase1, basis_phase1)

    if "primal" not in result_phase1:
        return {"status": "infeasible", "farkas": result_phase1.get("dual", np.zeros(m).tolist())}

    if "basis" not in result_phase1:
        return {"status": "error"}

    x_bar = np.array(result_phase1["primal"])
    if x_bar[n:].sum() > tolerance:
        return {"status": "infeasible", "farkas": result_phase1["dual"]}

    A_fixed = A
    b_fixed = b
    A1 = A_phase1
    basis_full = list(result_phase1["basis"])
    basis_set = set(basis_full)

    i = 0
    while i < m:
        if basis_full[i] >= n:
            B = A1[:, basis_full]

            try:
                T = np_solve(B, A1[:, :n])
            except np.linalg.LinAlgError:
                return {"status": "error"}

            entering = -1

            for j in range(n):
                if j not in basis_set and abs(T[i, j]) > tolerance:
                    entering = j
                    break

            if entering != -1:
                basis_set.remove(basis_full[i])
                basis_full[i] = entering
                basis_set.add(entering)
                i += 1
            else:
                A_fixed = np.delete(A_fixed, i, axis=0)
                b_fixed = np.delete(b_fixed, i)
                A1 = np.delete(A1, i, axis=0)
                basis_set.remove(basis_full[i])
                basis_full.pop(i)
                m -= 1
        else:
            i += 1

    basis = list(basis_full)

    if len(basis) < m:
        basis_set2 = set(basis)

        while len(basis) < m:
            added_basis = False

            for j in range(n):
                if j not in basis_set2:
                    B_check = A_fixed[:, basis + [j]]

                    if np.linalg.matrix_rank(B_check) == len(basis) + 1:
                        basis.append(j)
                        basis_set2.add(j)
                        added_basis = True
                        break

            if not added_basis:
                break

    lp.basis = basis

    return simplex(lp, A_fixed, b_fixed, c, basis)
