import math

def f(x):
    return x**3 - 27

def g(x, alpha):
    return x + alpha * f(x)

def fixed_point(alpha, x0=2.0, eps=1e-10, max_iter=1000):
    x = x0
    history = [x]

    for iteration in range(1, max_iter + 1):
        x_next = g(x, alpha)
        history.append(x_next)

        if abs(x_next - x) < eps:
            return x_next, iteration, history

        x = x_next

    raise RuntimeError("Fixed-point iteration did not converge")

if __name__ == "__main__":
    alphas = [-1/100, -1/40, -1/30, -1/27, -1/25, -1/20, -1/10]

    for alpha in alphas:
        print("\n" + "=" * 60)
        print(f"alpha = {alpha}")
        try:
            root, iterations, history = fixed_point(alpha)
            print(f"Final root: {root:.12f}")
            print(f"Iterations: {iterations}")
            print(f"f(root): {f(root):.3e}")
        except RuntimeError as error:
            print(error)
