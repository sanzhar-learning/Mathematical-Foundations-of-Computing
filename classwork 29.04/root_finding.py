"""
Newton's method vs Bisection method - root finding comparison.
Function: f(x) = x^3 - 2x - 5 on [2, 3], root ~ 2.0945514815...
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ---------- Test function ----------
def f(x):
    return x**3 - 2*x - 5

def df(x):
    return 3*x**2 - 2

A, B = 2.0, 3.0          # interval
EPS = 1e-10              # stop criterion
MAX_ITER = 200


def high_precision_root(a, b, iters=200):
    """
    Reference root computed independently of the methods we benchmark:
    bisection until the bracket is at machine precision, then a few
    Newton steps to polish to ~1e-16. Self-contained, no scipy.
    """
    fa = f(a)
    for _ in range(iters):
        m = 0.5 * (a + b)
        if m == a or m == b:           # bracket cannot be split further
            break
        fm = f(m)
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    x = 0.5 * (a + b)
    for _ in range(20):                # polish
        dx = f(x) / df(x)
        x -= dx
        if abs(dx) < 1e-17:
            break
    return x


TRUE_ROOT = high_precision_root(A, B)
print(f"Reference root: {TRUE_ROOT:.16f}")


# ---------- Newton's method ----------
def newton(f, df, x0, eps=EPS, max_iter=MAX_ITER):
    """
    Stop criterion: |x_{k+1} - x_k| < eps.
    Returns history list of dicts (iter, x, error_to_true_root, step_size).
    """
    history = []
    x = x0
    history.append({"iter": 0, "x": x, "error": abs(x - TRUE_ROOT), "step": np.nan})
    for k in range(1, max_iter + 1):
        fx, dfx = f(x), df(x)
        if dfx == 0:
            print("Newton: zero derivative, abort.")
            break
        x_new = x - fx / dfx
        step = abs(x_new - x)
        history.append({"iter": k, "x": x_new, "error": abs(x_new - TRUE_ROOT), "step": step})
        if step < eps:
            x = x_new
            break
        x = x_new
    return history


# ---------- Bisection ----------
def bisection(f, a, b, eps=EPS, max_iter=MAX_ITER):
    """
    Stop criterion: |b - a| < eps.
    Returns history list of dicts (iter, a, b, mid, error_to_true_root, width).
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a)*f(b) > 0 - bisection cannot start.")
    history = []
    mid = 0.5 * (a + b)
    history.append({"iter": 0, "a": a, "b": b, "mid": mid,
                    "error": abs(mid - TRUE_ROOT), "width": b - a})
    for k in range(1, max_iter + 1):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
        new_mid = 0.5 * (a + b)
        history.append({"iter": k, "a": a, "b": b, "mid": new_mid,
                        "error": abs(new_mid - TRUE_ROOT), "width": b - a})
        if (b - a) < eps:
            break
    return history


# ---------- Run both methods ----------
x0 = 0.5 * (A + B)               # interval center as starting point for Newton
print(f"Newton starting point x0 = {x0}")
print(f"Bisection initial interval = [{A}, {B}]\n")

newton_hist = newton(f, df, x0)
bisect_hist = bisection(f, A, B)


# ---------- Print iterations ----------
print(f"{'='*60}\nNewton's method\n{'='*60}")
print(f"{'iter':>4} {'x_k':>20} {'|x - x*|':>16} {'|x_k+1 - x_k|':>18}")
for h in newton_hist:
    step = f"{h['step']:.3e}" if not np.isnan(h['step']) else "  -"
    print(f"{h['iter']:>4} {h['x']:>20.16f} {h['error']:>16.3e} {step:>18}")

print(f"\n{'='*60}\nBisection method\n{'='*60}")
print(f"{'iter':>4} {'mid':>20} {'|mid - x*|':>16} {'|b-a|':>14}")
for h in bisect_hist[:30]:                 # first 30 to keep printout short
    print(f"{h['iter']:>4} {h['mid']:>20.16f} {h['error']:>16.3e} {h['width']:>14.3e}")
print(f"... total {len(bisect_hist)} iterations, "
      f"final error = {bisect_hist[-1]['error']:.3e}")
print(f"\nNewton converged in {len(newton_hist)-1} iterations")
print(f"Bisection converged in {len(bisect_hist)-1} iterations")


# ---------- (A) Convergence plot ----------
fig, ax = plt.subplots(figsize=(9, 6))
n_err = [max(h["error"], 1e-17) for h in newton_hist]   # clamp for log plot
b_err = [max(h["error"], 1e-17) for h in bisect_hist]
n_it  = [h["iter"] for h in newton_hist]
b_it  = [h["iter"] for h in bisect_hist]

ax.semilogy(n_it, n_err, "o-", lw=2, ms=7, label=f"Newton ({len(newton_hist)-1} iter)")
ax.semilogy(b_it, b_err, "s-", lw=2, ms=5, label=f"Bisection ({len(bisect_hist)-1} iter)")
ax.axhline(EPS, color="gray", ls="--", lw=1, label=f"eps = {EPS:g}")
ax.set_xlabel("iteration k")
ax.set_ylabel(r"absolute error  $|x_k - x^*|$")
ax.set_title(r"Convergence: Newton vs Bisection,  $f(x)=x^3-2x-5$")
ax.grid(True, which="both", alpha=0.4)
ax.legend()
fig.tight_layout()
fig.savefig("convergence.png", dpi=140)
plt.close(fig)
print("\nSaved convergence.png")


# ---------- (B) Animation: Newton ----------
def make_newton_video(history, path):
    xs_plot = np.linspace(A - 0.05, B + 0.05, 400)
    ys_plot = f(xs_plot)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs_plot, ys_plot, "b-", lw=2, label=r"$f(x)=x^3-2x-5$")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(TRUE_ROOT, color="green", ls=":", lw=1.2, label=f"true root  {TRUE_ROOT:.6f}")
    ax.set_xlim(A - 0.05, B + 0.05)
    ax.set_ylim(min(ys_plot) - 1, max(ys_plot) + 1)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)

    point,    = ax.plot([], [], "ro", ms=10, zorder=5)
    vline,    = ax.plot([], [], "r--", lw=1)
    tangent,  = ax.plot([], [], "orange", lw=1.5, label="tangent")
    title     = ax.set_title("")
    ax.legend(loc="upper left")

    def init():
        point.set_data([], []); vline.set_data([], []); tangent.set_data([], [])
        return point, vline, tangent, title

    def update(frame):
        h = history[frame]
        x = h["x"]; y = f(x)
        point.set_data([x], [y])
        vline.set_data([x, x], [0, y])
        # Tangent line: y - f(x) = f'(x)(t - x)  ->  draw across the plot
        if frame < len(history) - 1:
            slope = df(x)
            t = np.linspace(A - 0.05, B + 0.05, 50)
            tangent.set_data(t, y + slope * (t - x))
        else:
            tangent.set_data([], [])
        title.set_text(f"Newton  iter={h['iter']}  "
                       f"x={x:.10f}  |error|={h['error']:.2e}")
        return point, vline, tangent, title

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=len(history), interval=900, blit=False)
    writer = FFMpegWriter(fps=1.2, bitrate=1800)
    anim.save(path, writer=writer, dpi=130)
    plt.close(fig)


# ---------- (B) Animation: Bisection ----------
def make_bisection_video(history, path, max_frames=60):
    history = history[:max_frames]
    xs_plot = np.linspace(A - 0.05, B + 0.05, 400)
    ys_plot = f(xs_plot)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(xs_plot, ys_plot, "b-", lw=2, label=r"$f(x)=x^3-2x-5$")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(TRUE_ROOT, color="green", ls=":", lw=1.2, label=f"true root  {TRUE_ROOT:.6f}")
    ax.set_xlim(A - 0.05, B + 0.05)
    ax.set_ylim(min(ys_plot) - 1, max(ys_plot) + 1)
    ax.set_xlabel("x"); ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)

    interval_patch = ax.axvspan(A, B, color="yellow", alpha=0.25, label="interval [a,b]")
    a_line, = ax.plot([], [], "r-", lw=2)
    b_line, = ax.plot([], [], "r-", lw=2)
    mid_pt, = ax.plot([], [], "ro", ms=10, zorder=5)
    title   = ax.set_title("")
    ax.legend(loc="upper left")
    ymin, ymax = ax.get_ylim()

    def update(frame):
        h = history[frame]
        a, b, m = h["a"], h["b"], h["mid"]
        # update yellow span (Rectangle): set lower-left corner and width
        interval_patch.set_x(a)
        interval_patch.set_width(b - a)
        a_line.set_data([a, a], [ymin, ymax])
        b_line.set_data([b, b], [ymin, ymax])
        mid_pt.set_data([m], [f(m)])
        title.set_text(f"Bisection  iter={h['iter']}  "
                       f"mid={m:.10f}  |b-a|={h['width']:.2e}  "
                       f"|error|={h['error']:.2e}")
        return interval_patch, a_line, b_line, mid_pt, title

    anim = FuncAnimation(fig, update, frames=len(history), interval=400, blit=False)
    writer = FFMpegWriter(fps=2.5, bitrate=1800)
    anim.save(path, writer=writer, dpi=130)
    plt.close(fig)
