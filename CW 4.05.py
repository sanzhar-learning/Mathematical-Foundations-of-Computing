"""
Gauss-Seidel solver for sparse SLE.
Sparse representation: list of dicts — A[i] = {col: value}
"""

import random
import math
import time


# ─────────────────────────────────────────────
#  1. Matrix generation
# ─────────────────────────────────────────────

def build_sparse_matrix(n, nnz_per_row=10, seed=42):
    """
    Random sparse diagonally-dominant matrix.
    Each row has `nnz_per_row` random off-diagonal entries in [-1, 1].
    Diagonal = sum|off-diag| + 1  →  GS is guaranteed to converge.

    Returns list of dicts: A[i][j] = a_ij
    """
    random.seed(seed)
    A = [{} for _ in range(n)]

    for i in range(n):
        # pick random off-diagonal columns (sample extra to handle i)
        cols = random.sample(range(n), min(nnz_per_row + 1, n))
        if i in cols:
            cols.remove(i)
        cols = cols[:nnz_per_row]

        off_sum = 0.0
        for j in cols:
            v = random.uniform(-1.0, 1.0)
            A[i][j] = v
            off_sum += abs(v)

        A[i][i] = off_sum + 1.0   # diagonal dominance

    return A


# ─────────────────────────────────────────────
#  2. Sparse matvec  b = A @ x
# ─────────────────────────────────────────────

def matvec(A, x):
    return [sum(v * x[j] for j, v in row.items()) for row in A]


# ─────────────────────────────────────────────
#  3. Gauss-Seidel
# ─────────────────────────────────────────────

def gauss_seidel(A, b, tol=1e-10, max_iter=200):
    """
    Classical Gauss-Seidel, in-place x update.

    For each row i:
        x[i] = (b[i] - sum_{j != i} A[i][j] * x[j]) / A[i][i]

    Because x is updated in place, rows j < i already use the new values
    from this sweep — that is what distinguishes GS from Jacobi.
    """
    n = len(b)
    x = [0.0] * n

    for iteration in range(1, max_iter + 1):
        diff_sq = 0.0

        for i in range(n):
            row = A[i]
            a_ii = row[i]

            s = sum(a_ij * x[j] for j, a_ij in row.items() if j != i)

            x_new = (b[i] - s) / a_ii
            diff_sq += (x_new - x[i]) ** 2
            x[i] = x_new

        diff = math.sqrt(diff_sq)
        if diff < tol:
            return x, iteration, diff

    return x, max_iter, diff


# ─────────────────────────────────────────────
#  4. Error metric
# ─────────────────────────────────────────────

def max_abs_error(x, x_exact):
    return max(abs(a - b) for a, b in zip(x, x_exact))


# ─────────────────────────────────────────────
#  5. Single experiment
# ─────────────────────────────────────────────

def run(n, nnz_per_row=10, tol=1e-10, max_iter=200, seed=42):
    print(f"\n{'─'*55}")
    print(f"  N = {n:,}   nnz/row = {nnz_per_row}   tol = {tol:.0e}")
    print(f"{'─'*55}")

    # build matrix
    t0 = time.perf_counter()
    A = build_sparse_matrix(n, nnz_per_row=nnz_per_row, seed=seed)
    t_build = time.perf_counter() - t0
    nnz = sum(len(row) for row in A)
    print(f"  build   : {t_build:.3f} s   nnz = {nnz:,}  ({nnz/n:.1f}/row)")

    # known solution: 1, 2, 3, ..., N
    x_exact = list(range(1, n + 1))

    # right-hand side
    b = matvec(A, x_exact)

    # solve
    t0 = time.perf_counter()
    x_sol, iters, last_diff = gauss_seidel(A, b, tol=tol, max_iter=max_iter)
    t_solve = time.perf_counter() - t0

    err = max_abs_error(x_sol, x_exact)
    status = "converged" if iters < max_iter else "NOT CONVERGED"
    print(f"  solve   : {t_solve:.3f} s   iters = {iters}   [{status}]")
    print(f"  |dx|    : {last_diff:.2e}")
    print(f"  max|err|: {err:.2e}")


# ─────────────────────────────────────────────
#  6. Experiments
# ─────────────────────────────────────────────

# correctness check on small N
run(n=1_000,   nnz_per_row=10)

# medium scale
run(n=10_000,  nnz_per_row=10)

# denser sparsity pattern
run(n=10_000,  nnz_per_row=50)

# large scale — the main target
run(n=100_000, nnz_per_row=10)

# sparser — fewer neighbours, easier problem
run(n=100_000, nnz_per_row=3)
