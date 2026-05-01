def gaussian_elimination(A, b):
    n = len(b)

    # Forward elimination
    for col in range(n):
        for row in range(col + 1, n):
            factor = A[row][col] / A[col][col]
            for k in range(col, n):
                A[row][k] -= factor * A[col][k]
            b[row] -= factor * b[col]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i][j] * x[j]
        x[i] /= A[i][i]

    return x


A = [
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2],
]
b = [8, -11, -3]

x = gaussian_elimination(A, b)
print("Solution:", x)
