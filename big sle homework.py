import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Matrix generators
# -----------------------------

def generate_random_matrix(n, low=-5, high=5):
    A = np.random.uniform(low, high, size=(n, n))

    # Avoid zero diagonal, because it can break Gaussian elimination
    for i in range(n):
        if abs(A[i, i]) < 1e-12:
            A[i, i] = 1.0

    return A


def generate_diagonally_dominant_matrix(n, low=-5, high=5):
    A = np.random.uniform(low, high, size=(n, n))

    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(1, 5)

    return A


def generate_hilbert_matrix(n):
    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i, j] = 1 / (i + j + 1)

    return A


def generate_test_data(A):
    n = A.shape[0]
    x_test = np.ones(n)
    b = A @ x_test
    return b, x_test


# -----------------------------
# Gaussian elimination
# -----------------------------

def gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    n = len(b)

    # Forward elimination
    for col in range(n):
        # Partial pivoting
        pivot_row = col
        max_value = abs(A[col, col])

        for row in range(col + 1, n):
            if abs(A[row, col]) > max_value:
                max_value = abs(A[row, col])
                pivot_row = row

        if max_value < 1e-15:
            raise ValueError("Matrix is singular or almost singular")

        if pivot_row != col:
            A[[col, pivot_row]] = A[[pivot_row, col]]
            b[col], b[pivot_row] = b[pivot_row], b[col]

        for row in range(col + 1, n):
            factor = A[row, col] / A[col, col]

            for k in range(col, n):
                A[row, k] -= factor * A[col, k]

            b[row] -= factor * b[col]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        s = 0

        for j in range(i + 1, n):
            s += A[i, j] * x[j]

        x[i] = (b[i] - s) / A[i, i]

    return x


# -----------------------------
# Jacobi method
# -----------------------------

def jacobi_method(A, b, epsilon=1e-8, max_iterations=10000):
    A = A.astype(float)
    b = b.astype(float)

    n = len(b)
    x_old = np.zeros(n)

    for iteration in range(1, max_iterations + 1):
        x_new = np.zeros(n)

        for i in range(n):
            s = 0

            for j in range(n):
                if j != i:
                    s += A[i, j] * x_old[j]

            x_new[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x_new - x_old) < epsilon:
            return x_new, iteration, True

        x_old = x_new.copy()

    return x_old, max_iterations, False


# -----------------------------
# Gauss-Seidel method
# -----------------------------

def gauss_seidel_method(A, b, epsilon=1e-8, max_iterations=10000):
    A = A.astype(float)
    b = b.astype(float)

    n = len(b)
    x = np.zeros(n)

    for iteration in range(1, max_iterations + 1):
        x_old = x.copy()

        for i in range(n):
            s1 = 0
            s2 = 0

            for j in range(i):
                s1 += A[i, j] * x[j]

            for j in range(i + 1, n):
                s2 += A[i, j] * x_old[j]

            x[i] = (b[i] - s1 - s2) / A[i, i]

        if np.linalg.norm(x - x_old) < epsilon:
            return x, iteration, True

    return x, max_iterations, False


# -----------------------------
# Error calculation
# -----------------------------

def calculate_residue(A, x, b):
    return np.linalg.norm(A @ x - b)


def calculate_absolute_error(x, x_test):
    return np.linalg.norm(x - x_test)


# -----------------------------
# Run one algorithm
# -----------------------------

def run_gauss(problem_type, A):
    b, x_test = generate_test_data(A)

    try:
        x = gaussian_elimination(A, b)
        residue = calculate_residue(A, x, b)
        abs_error = calculate_absolute_error(x, x_test)

        return {
            "Problem type": problem_type,
            "Algorithm": "Gauss",
            "Dimension": A.shape[0],
            "Iterations": 1,
            "Converged": True,
            "Residue": residue,
            "Abs error": abs_error
        }

    except ValueError:
        return {
            "Problem type": problem_type,
            "Algorithm": "Gauss",
            "Dimension": A.shape[0],
            "Iterations": 1,
            "Converged": False,
            "Residue": np.nan,
            "Abs error": np.nan
        }


def run_jacobi(problem_type, A, epsilon=1e-8, max_iterations=10000):
    b, x_test = generate_test_data(A)

    x, iterations, converged = jacobi_method(A, b, epsilon, max_iterations)

    residue = calculate_residue(A, x, b)
    abs_error = calculate_absolute_error(x, x_test)

    return {
        "Problem type": problem_type,
        "Algorithm": "Jacobi",
        "Dimension": A.shape[0],
        "Iterations": iterations,
        "Converged": converged,
        "Residue": residue,
        "Abs error": abs_error
    }


def run_seidel(problem_type, A, epsilon=1e-8, max_iterations=10000):
    b, x_test = generate_test_data(A)

    x, iterations, converged = gauss_seidel_method(A, b, epsilon, max_iterations)

    residue = calculate_residue(A, x, b)
    abs_error = calculate_absolute_error(x, x_test)

    return {
        "Problem type": problem_type,
        "Algorithm": "Seidel",
        "Dimension": A.shape[0],
        "Iterations": iterations,
        "Converged": converged,
        "Residue": residue,
        "Abs error": abs_error
    }


# -----------------------------
# Plotting
# -----------------------------

def plot_errors(df, problem_type):
    data = df[df["Problem type"] == problem_type]

    # Residue plot
    plt.figure(figsize=(10, 6))

    for algorithm in data["Algorithm"].unique():
        part = data[data["Algorithm"] == algorithm]
        plt.plot(part["Dimension"], part["Residue"], marker="o", label=algorithm)

    plt.xlabel("Dimension")
    plt.ylabel("Residue ||Ax - b||")
    plt.title(f"{problem_type}: Residue error")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{problem_type}_residue.png")
    plt.close()

    # Absolute error plot
    plt.figure(figsize=(10, 6))

    for algorithm in data["Algorithm"].unique():
        part = data[data["Algorithm"] == algorithm]
        plt.plot(part["Dimension"], part["Abs error"], marker="o", label=algorithm)

    plt.xlabel("Dimension")
    plt.ylabel("Absolute error ||x - x_test||")
    plt.title(f"{problem_type}: Absolute error")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{problem_type}_absolute_error.png")
    plt.close()


# -----------------------------
# Main experiment
# -----------------------------

def main():
    np.random.seed(42)

    results = []

    epsilon = 1e-8
    max_iterations = 10000

    for n in range(3, 51):
        print(f"Running dimension n = {n}")

        # Experiment A:
        # Gaussian elimination for random matrix
        A_random = generate_random_matrix(n)
        results.append(run_gauss("Random", A_random))

        # Experiment B:
        # Gaussian elimination and Seidel for Hilbert matrix
        A_hilbert = generate_hilbert_matrix(n)
        results.append(run_gauss("Hilbert", A_hilbert))
        results.append(run_seidel("Hilbert", A_hilbert, epsilon, max_iterations))

        # Experiment C:
        # Gaussian, Jacobi, Seidel for diagonally dominant matrix
        A_diag = generate_diagonally_dominant_matrix(n)
        results.append(run_gauss("DiagDominant", A_diag))
        results.append(run_jacobi("DiagDominant", A_diag, epsilon, max_iterations))
        results.append(run_seidel("DiagDominant", A_diag, epsilon, max_iterations))

    df = pd.DataFrame(results)

    print("\nFinal results:")
    print(df)

    df.to_csv("results.csv", index=False)
    df.to_excel("results.xlsx", index=False)

    plot_errors(df, "Random")
    plot_errors(df, "Hilbert")
    plot_errors(df, "DiagDominant")

    print("\nDone.")
    print("Saved:")
    print("results.csv")
    print("results.xlsx")
    print("Random_residue.png")
    print("Random_absolute_error.png")
    print("Hilbert_residue.png")
    print("Hilbert_absolute_error.png")
    print("DiagDominant_residue.png")
    print("DiagDominant_absolute_error.png")


if __name__ == "__main__":
    main()