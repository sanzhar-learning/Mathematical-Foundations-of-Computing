
"""
Root-finding test suite: Bisection vs Fixed-Point Iteration
f(x) = x^3 - 27,  root at x* = 3

Fixed-point scheme: g(x) = x + alpha * f(x)
Convergence rate:  |g'(x*)| = |1 + 27*alpha|

Convergence region for alpha: -2/27 < alpha < 0
Optimal alpha:  alpha = -1/27  => g'(x*) = 0  (quadratic convergence)
"""

import numpy as np
import matplotlib.pyplot as plt

ROOT = 3.0
EPS = 1e-10


def f(x):
    return x**3 - 27


def bisection(a, b, eps=EPS, max_iter=2000):
    if f(a) * f(b) > 0:
        raise ValueError(f"f({a:.4g}) and f({b:.4g}) have the same sign")
    errors = []
    for i in range(max_iter):
        mid = (a + b) / 2
        errors.append(abs(mid - ROOT))
        if errors[-1] == 0 or (b - a) / 2 < eps:
            return mid, i + 1, errors
        if f(a) * f(mid) < 0:
            b = mid
        else:
            a = mid
    return (a + b) / 2, max_iter, errors


def fixed_point(alpha, x0, eps=EPS, max_iter=2000):
    x = x0
    errors = [abs(x - ROOT)]
    for i in range(max_iter):
        x_new = x + alpha * f(x)
        err = abs(x_new - ROOT) if np.isfinite(x_new) else np.inf
        # cap stored error so diverging runs don't break log-scale plots
        errors.append(min(err, 1e15))
        if not np.isfinite(x_new) or err > 1e12:
            return x_new, i + 1, errors, False   # diverged
        if abs(x_new - x) < eps:
            return x_new, i + 1, errors, True
        x = x_new
    return x, max_iter, errors, False


# ── Experiment definitions ─────────────────────────────────────────────────────
#
#  Five experiments chosen to cover three qualitative regimes:
#
#  Exp 1  alpha = -1/54   |g'(x*)| = 0.50  →  FP ≈ bisection  (same linear rate)
#  Exp 2  alpha = -0.001  |g'(x*)| = 0.97  →  FP much SLOWER  (rate near 1)
#  Exp 3  alpha = -1/27   g'(x*) = 0       →  FP much FASTER  (quadratic!)
#  Exp 4  alpha = -1/27,  x0=10.0          →  FP DIVERGES from far start; bisection reliable
#  Exp 5  alpha = -1/27,  interval=[0,1e9] →  very wide interval: bisection slow, FP still fast

experiments = [
    dict(
        title="Exp 1: FP  ≈  Bisection",
        subtitle=f"alpha = -1/54 ≈ {-1/54:.5f}\n|g'(x*)| = 0.50  (same rate as bisection)",
        alpha=-1/54, x0=2.0, a=0.0, b=5.0,
    ),
    dict(
        title="Exp 2: FP  MUCH SLOWER",
        subtitle="alpha = -0.001\n|g'(x*)| = 0.973  (rate near 1 -> very slow)",
        alpha=-0.001, x0=2.0, a=0.0, b=5.0,
    ),
    dict(
        title="Exp 3: FP  MUCH FASTER",
        subtitle=f"alpha = -1/27 ≈ {-1/27:.5f}\ng'(x*) = 0  -> quadratic convergence!",
        alpha=-1/27, x0=2.0, a=0.0, b=5.0,
    ),
    dict(
        title="Exp 4: FP diverges from x0=10",
        subtitle="alpha = -1/27,  x0 = 10.0\nbisection interval [0, 50]",
        alpha=-1/27, x0=10.0, a=0.0, b=50.0,
    ),
    dict(
        title="Exp 5: Very wide bisection interval",
        subtitle="alpha = -1/27,  x0 = 2.0\nbisection interval [0, 10^9]",
        alpha=-1/27, x0=2.0, a=0.0, b=1e9,
    ),
]


# ── Run experiments & print summary table ─────────────────────────────────────
print(f"\n{'='*74}")
print("ROOT-FINDING TEST SUITE      f(x) = x^3 - 27      (root x* = 3)")
print(f"Tolerance: eps = {EPS:.0e}")
print(f"{'='*74}")
print(f"  {'Experiment':<34}  {'Bisection':>10}  {'Fixed Pt':>14}  {'Winner':>10}")
print(f"  {'-'*70}")

all_results = []
for exp in experiments:
    bis_root, bis_iters, bis_errors = bisection(exp['a'], exp['b'])
    fp_root,  fp_iters,  fp_errors, fp_conv = fixed_point(exp['alpha'], exp['x0'])
    all_results.append((bis_iters, bis_errors, fp_iters, fp_errors, fp_conv))

    fp_str = str(fp_iters) if fp_conv else f"{fp_iters} DIVERGED"
    if fp_conv:
        winner = ("bisection" if bis_iters < fp_iters
                  else "FP" if fp_iters < bis_iters else "TIE")
    else:
        winner = "bisection"

    short = exp['title'].replace('\n', ' ')
    print(f"  {short:<34}  {bis_iters:>10}  {fp_str:>14}  {'[' + winner + ']':>12}")

print(f"  {'='*70}\n")


# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, (exp, res) in enumerate(zip(experiments, all_results)):
    bis_iters, bis_errors, fp_iters, fp_errors, fp_conv = res
    ax = axes[idx]

    ax.semilogy(range(len(bis_errors)), bis_errors,
                'b-o', ms=3, lw=1.5,
                label=f'Bisection  ({bis_iters} iter)')

    fp_label = f'Fixed Point  ({fp_iters} iter)' + ('' if fp_conv else '  — DIVERGED')
    fp_arr = np.array(fp_errors, dtype=float)
    ax.semilogy(range(len(fp_arr)), fp_arr,
                'r-s', ms=3, lw=1.5, label=fp_label)

    if not fp_conv:
        ax.text(0.97, 0.95, 'FP DIVERGED', transform=ax.transAxes,
                ha='right', va='top', color='red', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.9))

    ax.axhline(EPS, color='gray', ls='--', lw=0.8, alpha=0.7,
               label=f'eps = {EPS:.0e}')

    ax.set_xlabel('Iteration', fontsize=9)
    ax.set_ylabel('|x_n - x*|', fontsize=9)
    ax.set_title(f"{exp['title']}\n{exp['subtitle']}", fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_visible(False)   # 5 experiments -> hide 6th panel

fig.suptitle("Bisection vs Fixed-Point Iteration:  f(x) = x^3 - 27  (x* = 3)",
             fontsize=13, fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig('convergence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved to: convergence_comparison.png")
