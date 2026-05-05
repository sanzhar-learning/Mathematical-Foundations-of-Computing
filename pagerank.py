import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# ── 1. Graph generation ───────────────────────────────────────────────────────

def random_web_graph(n: int, avg_out: int = 3, seed: int = 42) -> nx.DiGraph:
    """Random directed graph where every node has at least one outgoing edge."""
    rng = random.Random(seed)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for src in range(n):
        candidates = [j for j in range(n) if j != src]
        k = rng.randint(1, min(avg_out * 2, n - 1))
        for dst in rng.sample(candidates, k):
            G.add_edge(src, dst)

    assert all(G.out_degree(v) >= 1 for v in G.nodes()), "dead end detected"
    return G


# ── 2. Build B = alpha*A + (1-alpha)*M ───────────────────────────────────────

def link_matrix(G: nx.DiGraph) -> np.ndarray:
    """Column-stochastic link matrix A."""
    n = G.number_of_nodes()
    A = np.zeros((n, n))
    for j in range(n):
        out_nbrs = list(G.successors(j))
        for i in out_nbrs:
            A[i, j] = 1.0 / len(out_nbrs)
    return A


def google_matrix(A: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """B = alpha*A + (1-alpha)*M,  M[i,j] = 1/n."""
    n = A.shape[0]
    M = np.ones((n, n)) / n
    return alpha * A + (1 - alpha) * M


# ── 3. Power iteration to solve Bx = x ───────────────────────────────────────

def pagerank(B: np.ndarray, tol: float = 1e-12, max_iter: int = 500) -> np.ndarray:
    """
    Power iteration: x_{k+1} = B x_k / ||B x_k||_1
    Converges to the unique positive eigenvector with eigenvalue 1.
    """
    n = B.shape[0]
    x = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        x_new = B @ x
        x_new /= x_new.sum()
        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new
        x = x_new
    return x


# ── 4. Plotting ───────────────────────────────────────────────────────────────

def plot_web(G: nx.DiGraph, ranks: np.ndarray, title: str, ax: plt.Axes) -> None:
    n = G.number_of_nodes()

    norm = Normalize(vmin=ranks.min(), vmax=ranks.max())
    colors = cm.plasma(norm(ranks))

    # Node sizes: linearly scaled between 400 and 3500
    sizes = 400 + (ranks / ranks.max()) * 3100

    pos = nx.spring_layout(G, seed=42, k=2.5 / max(1, np.sqrt(n)))

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="gray", alpha=0.4,
        arrows=True, arrowsize=12,
        connectionstyle="arc3,rad=0.12",
        width=0.8,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=sizes, node_color=colors,
    )
    labels = {i: f"{i}\n{ranks[i]:.3f}" for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_color="white")

    sm = cm.ScalarMappable(cmap="plasma", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.75, label="PageRank score")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")


# ── Pipeline helper ───────────────────────────────────────────────────────────

def run(n: int, alpha: float = 0.85, avg_out: int = 3, seed: int = 42,
        label: str = "") -> tuple:
    G = random_web_graph(n, avg_out=avg_out, seed=seed)
    A = link_matrix(G)
    B = google_matrix(A, alpha=alpha)
    ranks = pagerank(B)

    print(f"\n{'═'*52}")
    print(f"  {label or f'n={n}'}   (alpha={alpha}, avg_out={avg_out})")
    print(f"{'═'*52}")
    print(f"  col-sums of B all ≈ 1: {np.allclose(B.sum(axis=0), 1)}")
    print(f"  ||Bx - x||_1 = {np.linalg.norm(B @ ranks - ranks, 1):.2e}")
    print()
    order = np.argsort(ranks)[::-1]
    for rank_pos, page in enumerate(order, 1):
        print(f"  #{rank_pos:2d}  page {page:2d}  score={ranks[page]:.5f}"
              f"  in-deg={G.in_degree(page)}  out-deg={G.out_degree(page)}")

    if n <= 6:
        print("\n  Link matrix A:")
        print(np.round(A, 4))
        print("\n  Google matrix B = alpha*A + (1-alpha)*M:")
        print(np.round(B, 4))

    return G, ranks


# ── Main ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    'PageRank  —  "The $25,000,000,000 Eigenvector"\n'
    r"B = $\alpha A + (1-\alpha)M$,   power iteration for $Bx = x$",
    fontsize=13, y=1.02,
)

# Small graph: 5 nodes
G5, r5 = run(5, alpha=0.85, avg_out=2, seed=7, label="Small web  (n=5)")
plot_web(G5, r5, "Small web graph  (n=5, α=0.85)\nnode size ∝ PageRank", axes[0])

# Larger graph: 15 nodes
G15, r15 = run(15, alpha=0.85, avg_out=3, seed=42, label="Larger web  (n=15)")
plot_web(G15, r15, "Larger web graph  (n=15, α=0.85)\nnode size ∝ PageRank", axes[1])

plt.tight_layout()
out_path = "/Users/developer/Documents/project/pagerank_graph.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nGraph saved → {out_path}")
