import time
import numpy as np


def generate_diagonally_dominant(n, seed=42):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, (n, n))
    np.fill_diagonal(A, np.abs(A).sum(axis=1) - np.abs(np.diag(A)) + rng.uniform(1, 2, n))
    b = rng.uniform(-10, 10, n)
    return A, b


# --- Naive---
def jacobi_naive(A, b, num_iterations=500, tol=1e-10):
    n = len(b)
    x = [0.0] * n
    for k in range(num_iterations):
        x_new = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]
            x_new[i] = (b[i] - s) / A[i][i]
        diff = sum((x_new[i] - x[i]) ** 2 for i in range(n)) ** 0.5
        x = x_new
        if diff < tol:
            print(f"  [naive]  converged at iteration {k + 1}")
            break
    return x


# --- Matrix-vector---
# x^(k+1) = D^{-1} (b - (L+U) x^(k))
def jacobi_matrix(A, b, num_iterations=500, tol=1e-10):
    d_inv = 1.0 / np.diag(A)
    LU = A - np.diag(np.diag(A))
    x = np.zeros(len(b))
    for k in range(num_iterations):
        x_new = d_inv * (b - LU @ x)
        if np.linalg.norm(x_new - x) < tol:
            print(f"  [matrix] converged at iteration {k + 1}")
            x = x_new
            break
        x = x_new
    return x


N, ITERS = 100, 500
A_np, b_np = generate_diagonally_dominant(N)

print(f"System size: {N}x{N},  iterations: {ITERS}\n")

t0 = time.perf_counter()
x_naive = jacobi_naive(A_np.tolist(), b_np.tolist(), ITERS)
t_naive = time.perf_counter() - t0
print(f"  [naive]  time: {t_naive:.4f} s\n")

t0 = time.perf_counter()
x_matrix = jacobi_matrix(A_np, b_np, ITERS)
t_matrix = time.perf_counter() - t0
print(f"  [matrix] time: {t_matrix:.4f} s\n")

x_ref = np.linalg.solve(A_np, b_np)
print(f"Residual naive  vs exact: {np.max(np.abs(np.array(x_naive) - x_ref)):.2e}")
print(f"Residual matrix vs exact: {np.max(np.abs(x_matrix - x_ref)):.2e}")
print(f"\nSpeedup (naive / matrix): {t_naive / t_matrix:.1f}x")
