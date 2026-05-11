import numpy as np
import matplotlib.pyplot as plt

# 1. Linear function: y = 2.5x + 1.0
true_slope = 2.5
true_intercept = 1.0

# 2. Generate x points uniformly on [0, 5]
np.random.seed(42)
n = 50
x = np.random.uniform(0, 5, n)

# 3. Generate y using the linear function
y_clean = true_slope * x + true_intercept

# 4. Add small zero-centered Gaussian noise
noise = np.random.normal(0, 0.8, n)
y = y_clean + noise

# 5. Solve linear regression via normal equations: A^T A c = A^T y
#    A is the design matrix with columns [x, 1] for fitting y = c0*x + c1
A = np.column_stack([x, np.ones(n)])

# Normal equations: (A^T A) c = A^T y
ATA = A.T @ A
ATy = A.T @ y
c = np.linalg.solve(ATA, ATy)

fitted_slope, fitted_intercept = c
print(f"True function:   y = {true_slope}x + {true_intercept}")
print(f"Fitted function: y = {fitted_slope:.4f}x + {fitted_intercept:.4f}")

# 6. Plot
x_line = np.linspace(0, 5, 200)
y_line = fitted_slope * x_line + fitted_intercept

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color="steelblue", alpha=0.7, label="Noisy data")
plt.plot(x_line, y_line, color="crimson", linewidth=2,
         label=f"Fit: y = {fitted_slope:.2f}x + {fitted_intercept:.2f}")
plt.plot(x_line, true_slope * x_line + true_intercept, color="gray",
         linewidth=1.5, linestyle="--", label=f"True: y = {true_slope}x + {true_intercept}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression via Normal Equations")
plt.legend()
plt.tight_layout()
plt.savefig("linear_regression.png", dpi=150)
plt.show()
print("Plot saved to linear_regression.png")
