import numpy as np
import pandas as pd

df = pd.read_csv("points_lin.csv", header=None, names=["x", "y"])

x = df["x"].to_numpy()
y = df["y"].to_numpy()

n = len(x)

A = np.vstack([np.ones(n), x]).T

c_vect = np.linalg.solve(A.T @ A, A.T @ y)

a = c_vect[0]
b = c_vect[1]

print("a =", a)
print("b =", b)

print("Rounded coefficients:")
print("a =", round(a))
print("b =", round(b))

print(f"Line: y = {round(a)} + ({round(b)})x")