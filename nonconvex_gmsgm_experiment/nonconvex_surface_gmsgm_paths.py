"""
Non-convex Optimization Experiment:
Comparing SGM, GM-SGM, and GM-SGM + Block + Memory

This script:
  • Defines a smooth 2D non-convex function with TWO minima
  • Runs 3 methods:
        1. SGM (mean aggregation)
        2. GM-SGM (geometric median)
        3. GM-SGM + Block + Memory (BGMD-style)
  • Runs two scenarios:
        A. No corruption
        B. 30% gross gradient corruption
  • Produces:
        – 6 high-quality 3D plots (transparent surface + trajectories)
        – 2 convergence plots (distance to global min)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ============================================================
# 0. Reproducibility + Output folder
# ============================================================

SEED = 0
np.random.seed(SEED)
random.seed(SEED)

os.makedirs("results", exist_ok=True)

def f_value(x):
    """
    Two-minima function specifically designed so that:
      - Global minimum: very deep, wide basin near (-1, 0)
      - Local minimum: shallow and *very narrow* near (2, -1)
    This guarantees:
      · Clean run => all methods go to the global minimum
      · Corruption => SGM might get pushed into the narrow local well
                     GM-SGM & Block+Mem resist and return to global
    """
    x = np.asarray(x)
    x1 = x[...,0]
    x2 = x[...,1]

    # Global well: deep + wide
    u1 = 0.6*((x1+1.0)**2 + (x2)**2)

    # Local well: shallow + narrow
    u2 = 3.0*((x1-2.0)**2 + (x2+1.0)**2)

    return (
        + u1
        + 0.8*u2
        - 10.0*np.exp(-u1)   # strong global well
        - 2.0*np.exp(-u2)    # weaker + narrower local well
    )

def grad_f(x):
    x1, x2 = x[0], x[1]

    u1 = 0.6*((x1+1.0)**2 + x2**2)
    u2 = 3.0*((x1-2.0)**2 + (x2+1.0)**2)

    g1 = np.exp(-u1)
    g2 = np.exp(-u2)

    # du1/dx = 1.2(x+1), du1/dy = 1.2y
    du1_dx = 1.2*(x1+1.0)
    du1_dy = 1.2*(x2)

    # du2/dx = 6(x-2), du2/dy = 6(y+1)
    du2_dx = 6.0*(x1-2.0)
    du2_dy = 6.0*(x2+1.0)

    df_dx = (
        du1_dx
        + 0.8*du2_dx
        - 10.0*(-du1_dx)*g1
        - 2.0*(-du2_dx)*g2
    )
    df_dy = (
        du1_dy
        + 0.8*du2_dy
        - 10.0*(-du1_dy)*g1
        - 2.0*(-du2_dy)*g2
    )

    return np.array([df_dx, df_dy])

# ============================================================
# 2. Global min (approximation) + surface grid
# ============================================================

# Dense grid for surface visualization
X_GRID = np.linspace(-3, 3, 300)
Y_GRID = np.linspace(-3, 3, 300)
XX, YY = np.meshgrid(X_GRID, Y_GRID)
ZZ = f_value(np.stack([XX, YY], axis=-1))

# Approximate global minimum
idx_min = np.argmin(ZZ)
i_min, j_min = np.unravel_index(idx_min, ZZ.shape)
x_star = np.array([XX[i_min, j_min], YY[i_min, j_min]])
f_star = ZZ[i_min, j_min]

print(f"Approx global minimum x* = {x_star}, f(x*) = {f_star:.4f}")


# ============================================================
# 3. Optimization Setup
# ============================================================

N_WORKERS = 21
SMALL_NOISE = 0.10
GROSS_SCALE = 8.0
CORR_FRAC_CLEAN = 0.0
CORR_FRAC_CORRUPT = 0.3

N_STEPS = 80
STEP_SIZE = 0.05

# Block-Memory parameters
BETA_BLOCK = 0.5    # in 2D → block size k = 1
MEMORY_DECAY = 0.9

# High starting point (as requested)
X0 = np.array([2.3, 2.3])

METHODS = ["SGM", "GM-SGM", "Block+Mem"]


# ============================================================
# 4. Geometric Median (Weiszfeld)
# ============================================================

def geometric_median(points, max_iter=60, tol=1e-6):
    """
    Geometric median in R^d via Weiszfeld's algorithm.
    """
    pts = np.asarray(points, float)
    y = pts.mean(axis=0)

    for _ in range(max_iter):
        diff = pts - y
        dist = np.linalg.norm(diff, axis=1)
        dist = np.clip(dist, 1e-8, None)

        w = 1.0/dist
        w /= w.sum()

        y_new = np.sum(w[:, None]*pts, axis=0)

        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new

    return y


# ============================================================
# 5. Block + Memory Update (BGMD-style)
# ============================================================

def blockmem_update(base_grad, worker_grads, memory, beta, mem_decay):
    """
    Implements BGMD-style block selection + memory accumulation.
    In 2D the block includes only 1 coordinate (k=1).
    """
    d = 2
    if memory is None:
        memory = np.zeros(2)

    # Select coordinate with largest |g + mem|
    scores = (base_grad + memory)**2
    k = max(1, int(beta*d))
    idx_sorted = np.argsort(-scores)
    mask = np.zeros(d, dtype=bool)
    mask[idx_sorted[:k]] = True

    # GM in reduced dimension
    G_reduced = worker_grads[:, mask]     # (b, k)
    gm_reduced = geometric_median(G_reduced)

    gm_full = np.zeros(d)
    gm_full[mask] = gm_reduced

    # Memory update
    residual = base_grad - gm_full
    new_memory = mem_decay*memory + residual

    return gm_full, new_memory


# ============================================================
# 6. Worker gradient simulation
# ============================================================

def build_worker_grads(base_grad, corr_frac, gross_scale, small_noise):
    """
    Build N_WORKERS noisy/corrupted gradients.
    """
    grads = []
    for _ in range(N_WORKERS):
        gk = base_grad.copy()

        # small heterogeneity noise
        gk += small_noise * np.random.randn(2)

        # gross corruption
        if np.random.rand() < corr_frac:
            gk = base_grad + gross_scale*np.random.randn(2)

        grads.append(gk)

    return np.stack(grads)


# ============================================================
# 7. One-step updates for each method
# ============================================================

def step_sgm(x, corr_frac, gross_scale, small_noise):
    g = grad_f(x)
    worker_grads = build_worker_grads(g, corr_frac, gross_scale, small_noise)
    agg_grad = worker_grads.mean(axis=0)
    return x - STEP_SIZE*agg_grad, None


def step_gmsgm(x, corr_frac, gross_scale, small_noise):
    g = grad_f(x)
    worker_grads = build_worker_grads(g, corr_frac, gross_scale, small_noise)
    agg_grad = geometric_median(worker_grads)
    return x - STEP_SIZE*agg_grad, None


def step_blockmem(x, corr_frac, gross_scale, small_noise, memory):
    g = grad_f(x)
    worker_grads = build_worker_grads(g, corr_frac, gross_scale, small_noise)
    agg_grad, new_memory = blockmem_update(
        g, worker_grads, memory, beta=BETA_BLOCK, mem_decay=MEMORY_DECAY
    )
    return x - STEP_SIZE*agg_grad, new_memory


# ============================================================
# 8. Run a trajectory
# ============================================================

def run_trajectory(method, corr_frac, gross_scale, small_noise):
    """
    Runs N_STEPS of gradient updates for the chosen method.
    Returns:
       path      (N_STEPS+1, 2)
       distances (N_STEPS+1,)
    """
    x = X0.copy()
    memory = None

    path = [x.copy()]
    dist = [np.linalg.norm(x - x_star)]

    for _ in range(N_STEPS):
        if method == "SGM":
            x, memory = step_sgm(x, corr_frac, gross_scale, small_noise)
        elif method == "GM-SGM":
            x, memory = step_gmsgm(x, corr_frac, gross_scale, small_noise)
        elif method == "Block+Mem":
            x, memory = step_blockmem(x, corr_frac, gross_scale, small_noise, memory)
        else:
            raise ValueError(method)

        path.append(x.copy())
        dist.append(np.linalg.norm(x - x_star))

    return np.array(path), np.array(dist)


# ============================================================
# 9. Generate all trajectories (clean + corrupt)
# ============================================================

paths_clean, dists_clean = {}, {}
paths_corr, dists_corr = {}, {}

print("\n=== Running CLEAN scenario ===")
for m in METHODS:
    paths_clean[m], dists_clean[m] = run_trajectory(
        m, corr_frac=0.0, gross_scale=0.0, small_noise=0.0
    )

print("\n=== Running CORRUPTED scenario (30% gross corruption) ===")
for m in METHODS:
    paths_corr[m], dists_corr[m] = run_trajectory(
        m, corr_frac=0.3, gross_scale=GROSS_SCALE, small_noise=SMALL_NOISE
    )

# Print final stats
for name, path in [
    ("SGM clean", paths_clean["SGM"]),
    ("GM-SGM clean", paths_clean["GM-SGM"]),
    ("Block+Mem clean", paths_clean["Block+Mem"]),
    ("SGM corrupt", paths_corr["SGM"]),
    ("GM-SGM corrupt", paths_corr["GM-SGM"]),
    ("Block+Mem corrupt", paths_corr["Block+Mem"]),
]:
    xf = path[-1]
    print(f"{name:20s} final x = {xf},  dist to x* = {np.linalg.norm(xf-x_star):.4f}")


# ============================================================
# 10. High-quality 3D plots
# ============================================================

def make_3d_plot(path, title, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    stride = 4
    surf = ax.plot_surface(
        XX[::stride], YY[::stride], ZZ[::stride],
        cmap="viridis",
        edgecolor="none",
        alpha=0.55
    )

    z_offset = ZZ.min() - 1.0
    ax.contour(XX, YY, ZZ, zdir='z', offset=z_offset, cmap="Greys", linewidths=0.5)

    xs, ys = path[:, 0], path[:, 1]
    zs = f_value(path)

    ax.plot(xs, ys, zs, color="black", linewidth=3.0, alpha=1.0)
    ax.plot(xs, ys, z_offset, color="dimgray", linestyle="--", linewidth=2.0)

    ax.scatter(xs[0], ys[0], zs[0], s=60, c="black", marker="o")
    ax.scatter(xs[-1], ys[-1], zs[-1], s=80, c="black", marker="X")

    # Global minimum (gold star)
    ax.scatter(
        x_star[0], x_star[1], f_star,
        color="gold",
        s=250,
        marker="*",
        edgecolors="black",
        linewidth=1.2
    )

    ax.set_xlabel("x", labelpad=8)
    ax.set_ylabel("y", labelpad=8)
    ax.set_zlabel("Function Value  $f(x,y)$", labelpad=8)
    ax.set_zlim(z_offset, ZZ.max())
    ax.view_init(elev=40, azim=-60)

    ax.set_title(title, fontsize=13)
    fig.colorbar(surf, shrink=0.6, pad=0.08)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# === 3D Plots: Clean scenario ===
make_3d_plot(paths_clean["SGM"],      "SGM (Mean Aggregation) – Clean Scenario",        "results/clean_SGM.png")
make_3d_plot(paths_clean["GM-SGM"],   "GM-SGM (Geometric Median) – Clean Scenario",     "results/clean_GMSGM.png")
make_3d_plot(paths_clean["Block+Mem"],"Block+Memory Aggregation – Clean Scenario",      "results/clean_BlockMem.png")

# === 3D Plots: Corrupted scenario ===
make_3d_plot(paths_corr["SGM"],       "SGM (Mean Aggregation) – 30% Gross Corruption",  "results/corrupt_SGM.png")
make_3d_plot(paths_corr["GM-SGM"],    "GM-SGM (Geometric Median) – 30% Gross Corruption","results/corrupt_GMSGM.png")
make_3d_plot(paths_corr["Block+Mem"], "Block+Memory – 30% Gross Corruption",            "results/corrupt_BlockMem.png")


# ============================================================
# 11. 2D Convergence Plots
# ============================================================

def plot_distance(dist_dict, scenario_name, filename):
    it = np.arange(len(next(iter(dist_dict.values()))))

    colors = {"SGM":"gray","GM-SGM":"dodgerblue","Block+Mem":"seagreen"}
    styles = {"SGM":"--","GM-SGM":"-","Block+Mem":"-."}

    plt.figure(figsize=(7,5))

    for m in METHODS:
        plt.plot(
            it, dist_dict[m],
            label=m,
            linewidth=2.5,
            color=colors[m],
            linestyle=styles[m]
        )

    plt.xlabel("Iteration")
    plt.ylabel(r"Distance $\|x_t - x^\star\|_2$")
    plt.title(f"Convergence to the Global Minimum – {scenario_name}")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


plot_distance(dists_clean, "Clean Scenario", "results/dist_clean.png")
plot_distance(dists_corr,  "30% Gross Corruption", "results/dist_corrupt.png")


print("\nAll plots saved in ./results/")
