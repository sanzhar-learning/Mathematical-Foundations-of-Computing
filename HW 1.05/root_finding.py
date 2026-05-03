import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import math, os

# ──────────────────────────────────────────────
#  EASY-CHANGE INPUTS  (edit only this section)
# ──────────────────────────────────────────────

# Example set — change EXAMPLE to 0,1,2,3 to switch
EXAMPLE = 2

EXAMPLES = {
    0: {
        "name": "x² − 4",
        "f":    lambda x: x**2 - 4,
        "df":   lambda x: 2*x,
        "g":    lambda x, a: x + a * (x**2 - 4),
        "a": 0.1, "b": 3.0,
        "x0": 2.5,
        "eps": 1e-8,
        "alpha": -0.1,
    },
    1: {
        "name": "sin(x) − x/2",
        "f":    lambda x: math.sin(x) - x/2,
        "df":   lambda x: math.cos(x) - 0.5,
        "g":    lambda x, a: x + a * (math.sin(x) - x/2),
        "a": 0.5, "b": 2.5,
        "x0": 2.0,
        "eps": 1e-8,
        "alpha": -2.0,
    },
    2: {
        "name": "x³ − 2x − 5",
        "f":    lambda x: x**3 - 2*x - 5,
        "df":   lambda x: 3*x**2 - 2,
        "g":    lambda x, a: x + a * (x**3 - 2*x - 5),
        "a": 1.5, "b": 3.0,
        "x0": 2.5,
        "eps": 1e-8,
        "alpha": -0.05,
    },
    3: {
        "name": "cos(x) − x",
        "f":    lambda x: math.cos(x) - x,
        "df":   lambda x: -math.sin(x) - 1,
        "g":    lambda x, a: x + a * (math.cos(x) - x),
        "a": 0.0, "b": 1.5,
        "x0": 0.5,
        "eps": 1e-8,
        "alpha": 0.5,
    },
}

cfg = EXAMPLES[EXAMPLE]
f      = cfg["f"]
df     = cfg["df"]
g      = cfg["g"]
A, B   = cfg["a"], cfg["b"]
X0     = cfg["x0"]
EPS    = cfg["eps"]
ALPHA  = cfg["alpha"]
NAME   = cfg["name"]
MAX_ITER = 200

# ──────────────────────────────────────────────
#  ALGORITHMS
# ──────────────────────────────────────────────

def bisection(f, a, b, eps=EPS, max_iter=MAX_ITER):
    if f(a) * f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    errors, midpoints = [], []
    for _ in range(max_iter):
        m = (a + b) / 2
        midpoints.append(m)
        err = (b - a) / 2
        errors.append(err)
        if err < eps:
            break
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return midpoints[-1], midpoints, errors


def fixed_point(f, g, alpha, x0=X0, eps=EPS, max_iter=MAX_ITER):
    x = x0
    xs, errors = [x], []
    for _ in range(max_iter):
        xn = g(x, alpha)
        errors.append(abs(xn - x))
        xs.append(xn)
        if abs(xn - x) < eps:
            break
        x = xn
    return xs[-1], xs, errors


def newton(f, df, x0=X0, eps=EPS, max_iter=MAX_ITER):
    x = x0
    xs, errors = [x], []
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-14:
            break
        xn = x - fx / dfx
        errors.append(abs(xn - x))
        xs.append(xn)
        if abs(xn - x) < eps:
            break
        x = xn
    return xs[-1], xs, errors


# ──────────────────────────────────────────────
#  MULTI-ROOT SCAN
# ──────────────────────────────────────────────

def find_all_roots(f, df, g, alpha, a, b, eps=EPS, n_scan=500):
    """Scan [a,b] for sign changes, then refine with all three methods."""
    xs = np.linspace(a, b, n_scan)
    fvals = [f(x) for x in xs]
    roots = {"bisection": [], "fixed_point": [], "newton": []}

    brackets = []
    for i in range(len(xs) - 1):
        if fvals[i] * fvals[i+1] <= 0:
            brackets.append((xs[i], xs[i+1]))

    for la, lb in brackets:
        try:
            r, _, _ = bisection(f, la, lb, eps)
            if not any(abs(r - prev) < eps*100 for prev in roots["bisection"]):
                roots["bisection"].append(r)
        except Exception:
            pass

        mid = (la + lb) / 2
        try:
            r, _, _ = fixed_point(f, g, alpha, x0=mid, eps=eps)
            if abs(f(r)) < eps*100:
                if not any(abs(r - prev) < eps*100 for prev in roots["fixed_point"]):
                    roots["fixed_point"].append(r)
        except Exception:
            pass

        try:
            r, _, _ = newton(f, df, x0=mid, eps=eps)
            if abs(f(r)) < eps*100:
                if not any(abs(r - prev) < eps*100 for prev in roots["newton"]):
                    roots["newton"].append(r)
        except Exception:
            pass

    return roots


# ──────────────────────────────────────────────
#  CONVERGENCE PLOT
# ──────────────────────────────────────────────

def plot_convergence(errors_bis, errors_fp, errors_newt, name, ex_id, save=True):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.semilogy(range(1, len(errors_bis)+1),  errors_bis,  'b-o',  ms=4, label='Bisection')
    ax.semilogy(range(1, len(errors_fp)+1),   errors_fp,   'r-s',  ms=4, label='Fixed-point')
    ax.semilogy(range(1, len(errors_newt)+1), errors_newt, 'g-^',  ms=4, label="Newton")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Error  (log scale)", fontsize=12)
    ax.set_title(f"Convergence — {name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if save:
        path = f"convergence_{ex_id}.png"
        plt.savefig(path, dpi=120)
        print(f"  Saved: {path}")
    plt.close()


# ──────────────────────────────────────────────
#  DEMO ANIMATIONS
# ──────────────────────────────────────────────

def _curve(a, b, f, n=400):
    xs = np.linspace(a - 0.3*(b-a), b + 0.3*(b-a), n)
    return xs, np.array([f(x) for x in xs])


def animate_bisection(f, a, b, xs_hist, name, ex_id, fps=3):
    xc, yc = _curve(a, b, f)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xc, yc, 'k-', lw=2, label=f'f(x) = {name}')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title("Bisection method")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")

    vline   = ax.axvline(xs_hist[0], color='blue', lw=1.5, ls='--')
    lo_line = ax.axvline(a,          color='orange', lw=1, ls=':')
    hi_line = ax.axvline(b,          color='orange', lw=1, ls=':')
    pt,     = ax.plot([], [], 'bo', ms=8, zorder=5)
    info    = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top', fontsize=10,
                      bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax.legend()

    lo, hi = a, b
    def update(frame):
        nonlocal lo, hi
        m = xs_hist[frame]
        if frame > 0:
            if f(lo) * f(xs_hist[frame-1]) < 0:
                hi = xs_hist[frame-1]
            else:
                lo = xs_hist[frame-1]
        vline.set_xdata([m, m])
        lo_line.set_xdata([lo, lo])
        hi_line.set_xdata([hi, hi])
        pt.set_data([m], [f(m)])
        info.set_text(f"Iter {frame+1}\nm = {m:.6f}\nf(m) = {f(m):.2e}\nwidth = {hi-lo:.2e}")

    ani = animation.FuncAnimation(fig, update, frames=len(xs_hist), interval=1000//fps, repeat=False)
    path = f"bisection_{ex_id}.gif"
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  Saved: {path}")


def animate_fixed_point(f, g, alpha, xs_hist, name, ex_id, fps=3):
    all_x = [x for x in xs_hist if math.isfinite(x)]
    margin = max(0.5, 0.2*(max(all_x)-min(all_x)))
    xmin, xmax = min(all_x)-margin, max(all_x)+margin
    xc = np.linspace(xmin, xmax, 400)
    yf  = np.array([f(x) for x in xc])
    yg  = np.array([g(x, alpha) - x for x in xc])   # cobweb uses g(x)-x = 0

    # cobweb diagram: y = x and y = g(x)
    yg2 = np.array([g(x, alpha) for x in xc])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xc, yf,  'k-',  lw=2, label=f'f(x) = {name}')
    ax.plot(xc, yg2, 'b--', lw=1.5, label='g(x) = x + α·f(x)')
    ax.plot(xc, xc,  'r:',  lw=1, label='y = x')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title("Fixed-point iteration")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(fontsize=9)

    cobweb_lines, = ax.plot([], [], 'm-', lw=1, alpha=0.6)
    pt,           = ax.plot([], [], 'ro', ms=8, zorder=5)
    info          = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top', fontsize=10,
                            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    cob_xs, cob_ys = [], []

    def update(frame):
        x = xs_hist[frame]
        if not math.isfinite(x):
            return
        gx = g(x, alpha)
        cob_xs.extend([x, x, gx])
        cob_ys.extend([0, gx, gx])
        cobweb_lines.set_data(cob_xs, cob_ys)
        pt.set_data([x], [f(x)])
        err = abs(xs_hist[frame] - xs_hist[frame-1]) if frame > 0 else float('nan')
        info.set_text(f"Iter {frame+1}\nx = {x:.6f}\nf(x) = {f(x):.2e}\nerr = {err:.2e}")

    ani = animation.FuncAnimation(fig, update, frames=len(xs_hist), interval=1000//fps, repeat=False)
    path = f"fixed_point_{ex_id}.gif"
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  Saved: {path}")


def animate_newton(f, df, xs_hist, name, ex_id, fps=3):
    all_x = [x for x in xs_hist if math.isfinite(x)]
    margin = max(0.5, 0.3*(max(all_x)-min(all_x)))
    xmin, xmax = min(all_x)-margin, max(all_x)+margin
    xc = np.linspace(xmin, xmax, 400)
    yc = np.array([f(x) for x in xc])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xc, yc, 'k-', lw=2, label=f'f(x) = {name}')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title("Newton's method")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.legend()

    tangent_line, = ax.plot([], [], 'b--', lw=1.5, alpha=0.8)
    pt,           = ax.plot([], [], 'ro', ms=9, zorder=5)
    zero_pt,      = ax.plot([], [], 'g^', ms=9, zorder=5)
    info          = ax.text(0.02, 0.95, '', transform=ax.transAxes, va='top', fontsize=10,
                            bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    def update(frame):
        x = xs_hist[frame]
        if not math.isfinite(x):
            return
        fx  = f(x)
        dfx = df(x)
        # tangent: y - fx = dfx*(t - x)  =>  y=0 at t = x - fx/dfx
        t_range = np.array([xmin, xmax])
        tang_y  = fx + dfx * (t_range - x)
        tangent_line.set_data(t_range, tang_y)
        pt.set_data([x], [fx])
        xn = x - fx/dfx if abs(dfx) > 1e-14 else x
        zero_pt.set_data([xn], [0])
        err = abs(xs_hist[frame] - xs_hist[frame-1]) if frame > 0 else float('nan')
        info.set_text(f"Iter {frame+1}\nx = {x:.6f}\nf(x) = {fx:.2e}\nf'(x) = {dfx:.2e}\nerr = {err:.2e}")

    ani = animation.FuncAnimation(fig, update, frames=len(xs_hist), interval=1000//fps, repeat=False)
    path = f"newton_{ex_id}.gif"
    ani.save(path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def run_example(ex_id):
    c = EXAMPLES[ex_id]
    f_, df_, g_ = c["f"], c["df"], c["g"]
    a_, b_, x0_ = c["a"], c["b"], c["x0"]
    eps_, alpha_, name_ = c["eps"], c["alpha"], c["name"]

    print(f"\n{'='*55}")
    print(f"  Example {ex_id}  |  f(x) = {name_}")
    print(f"  Interval [{a_}, {b_}],  x0={x0_},  eps={eps_},  alpha={alpha_}")
    print(f"{'='*55}")

    print("Bisection:")
    root_b, xs_b, err_b = bisection(f_, a_, b_, eps_)
    print(f"  root = {root_b:.10f},  iters = {len(xs_b)}")

    print("Fixed-point:")
    root_fp, xs_fp, err_fp = fixed_point(f_, g_, alpha_, x0=x0_, eps=eps_)
    print(f"  root = {root_fp:.10f},  iters = {len(xs_fp)}")

    print("Newton:")
    root_n, xs_n, err_n = newton(f_, df_, x0=x0_, eps=eps_)
    print(f"  root = {root_n:.10f},  iters = {len(xs_n)}")

    print("\nMulti-root scan:")
    all_roots = find_all_roots(f_, df_, g_, alpha_, a_, b_, eps_)
    for method, roots in all_roots.items():
        print(f"  {method:12s}: {[round(r,8) for r in roots]}")

    print("\nGenerating convergence plot...")
    plot_convergence(err_b, err_fp, err_n, name_, ex_id)

    print("Generating animations...")
    animate_bisection(f_, a_, b_, xs_b, name_, ex_id)
    animate_fixed_point(f_, g_, alpha_, xs_fp, name_, ex_id)
    animate_newton(f_, df_, xs_n, name_, ex_id)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run a single example (set EXAMPLE above) or ALL examples
    RUN_ALL = True

    if RUN_ALL:
        for ex_id in sorted(EXAMPLES.keys()):
            run_example(ex_id)
    else:
        run_example(EXAMPLE)
