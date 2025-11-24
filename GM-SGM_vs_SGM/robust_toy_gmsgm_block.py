import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from time import time

# ============================================================
# 0. Global setup
# ============================================================

dtype = torch.float64
Path("results").mkdir(exist_ok=True)

# ============================================================
# 1. 2D Toy Problem: objective and constraint
# ============================================================

def f(w0, w1):
    """Convex quadratic objective (works with numpy or torch)."""
    return (w0 - 1) ** 2 + (w1 - 2) ** 2


def grad_f(w: torch.Tensor) -> torch.Tensor:
    """Gradient of f in 2D."""
    return torch.tensor(
        [2 * (w[0] - 1), 2 * (w[1] - 2)],
        dtype=dtype,
    )


def g_value(w: torch.Tensor) -> torch.Tensor:
    """Constraint value g(w) = w0 + w1 - 2."""
    return w[0] + w[1] - 2.0


def grad_g() -> torch.Tensor:
    """Gradient of g in 2D."""
    return torch.tensor([1.0, 1.0], dtype=dtype)


# ============================================================
# 2. Geometric Median (Weiszfeld)
# ============================================================

def geometric_median(vectors, eps: float = 1e-6, max_iter: int = 100):
    """
    Weiszfeld's algorithm for the geometric median.
    vectors: list of 1D tensors with same shape.
    """
    v = torch.stack(vectors)  # (b, d)
    guess = v.mean(dim=0)
    for _ in range(max_iter):
        distances = torch.norm(v - guess, dim=1).clamp_min(eps)
        weights = 1.0 / distances
        weights = weights / weights.sum()
        new_guess = (v * weights.unsqueeze(1)).sum(dim=0)
        if torch.norm(new_guess - guess) < eps:
            break
        guess = new_guess
    return guess


# ============================================================
# 3. Gross corruption model
# ============================================================

def gross_corruption(grad: torch.Tensor, psi: float = 0.3, magnitude: float = 8.0):
    """
    Gross Corruption Model:
      with prob psi, replace gradient with large arbitrary vector.
    """
    if np.random.rand() < psi:
        corrupt_direction = torch.randn_like(grad, dtype=dtype)
        corrupt_direction = corrupt_direction / corrupt_direction.norm().clamp_min(1e-12)
        grad = magnitude * corrupt_direction
    return grad


# ============================================================
# 4. Single SGM step (2D)
# ============================================================

def sgm_step(
    w: torch.Tensor,
    eta: float = 0.1,
    eps: float = 0.0,
    corruption: bool = False,
    psi: float = 0.4,
    magnitude: float = 8.0,
):
    """Single SGM update with optional gross corruption."""
    g_val = g_value(w)
    grad = grad_f(w) if g_val <= eps else grad_g()
    if corruption:
        grad = gross_corruption(grad, psi=psi, magnitude=magnitude)
    return w - eta * grad


# ============================================================
# 5. Single GM-SGM step (full geometric median in R^2)
# ============================================================

def gm_sgm_step(
    w: torch.Tensor,
    eta: float = 0.1,
    eps: float = 0.0,
    corruption: bool = True,
    batch_size: int = 10,
    psi: float = 0.4,
    magnitude: float = 8.0,
):
    """GM-SGM step with geometric median aggregation in full dimension."""
    grads = []
    for _ in range(batch_size):
        g_val = g_value(w)
        grad = grad_f(w) if g_val <= eps else grad_g()
        if corruption:
            grad = gross_corruption(grad, psi=psi, magnitude=magnitude)
        grads.append(grad)

    gm_grad = geometric_median(grads)
    return w - eta * gm_grad


# ============================================================
# 6. GM-SGM + Block Coordinate Selection + Memory (2D)
# ============================================================

def gm_sgm_block_memory_step(
    w: torch.Tensor,
    memory: torch.Tensor,
    eta: float = 0.1,
    eps: float = 0.0,
    n_workers: int = 10,
    psi: float = 0.4,
    magnitude: float = 8.0,
    block_k: int = 2,
):
    """
    2D version of GM-SGM with:
      - block coordinate selection (subset of {0,1})
      - memory mechanism (error feedback).

    In 2D, block_k can be 1 or 2; we usually take 2 here
    just to keep the direction accurate while still showcasing
    the mechanism.
    """
    d = 2
    block_k = min(block_k, d)

    # Clean gradient according to switching rule
    g_val = g_value(w)
    base_grad = grad_f(w) if g_val <= eps else grad_g()

    # Simulate worker gradients (corrupted versions of base_grad)
    workers = []
    for _ in range(n_workers):
        g = base_grad.clone()
        g = gross_corruption(g, psi=psi, magnitude=magnitude)
        workers.append(g)

    # Pick block coordinates (here just for illustration; in high-d this is crucial)
    idx = np.random.choice(d, size=block_k, replace=False)
    mask = torch.zeros(d, dtype=dtype)
    mask[idx] = 1.0

    # Project worker grads to block
    workers_block = [w_i * mask for w_i in workers]

    # Geometric median on the restricted block
    gm_block = geometric_median(workers_block)

    # Effective step direction with memory
    step_dir = gm_block + 0.5 * memory

    w_new = w - eta * step_dir
    memory_new = step_dir.detach().clone()

    return w_new, memory_new


# ============================================================
# 7. Run methods on the 2D toy
# ============================================================

def run_method_2d(
    method: str = "sgm_clean",
    steps: int = 40,
    eta: float = 0.15,
    psi: float = 0.4,
    magnitude: float = 8.0,
    batch_size: int = 15,
    block_k: int = 2,
):
    """
    method in:
      - 'sgm_clean'
      - 'sgm_corrupt'
      - 'gm_sgm'
      - 'block_memory'
    """
    w = torch.tensor([-2.0, 3.0], dtype=dtype)
    memory = torch.zeros(2, dtype=dtype)

    traj = [w.clone()]
    losses = [f(w[0], w[1])]
    gvals = [g_value(w)]

    for _ in range(steps):
        if method == "sgm_clean":
            w = sgm_step(w, eta=eta, corruption=False,
                         psi=psi, magnitude=magnitude)

        elif method == "sgm_corrupt":
            w = sgm_step(w, eta=eta, corruption=True,
                         psi=psi, magnitude=magnitude)

        elif method == "gm_sgm":
            w = gm_sgm_step(
                w,
                eta=eta,
                eps=0.0,
                corruption=True,
                batch_size=batch_size,
                psi=psi,
                magnitude=magnitude,
            )

        elif method == "block_memory":
            w, memory = gm_sgm_block_memory_step(
                w,
                memory,
                eta=eta,
                eps=0.0,
                n_workers=batch_size,
                psi=psi,
                magnitude=magnitude,
                block_k=block_k,
            )

        traj.append(w.clone())
        losses.append(f(w[0], w[1]))
        gvals.append(g_value(w))

    return torch.stack(traj), np.array(losses), np.array(gvals)


# ============================================================
# 8. Run 2D experiments
# ============================================================

psi = 0.4
magnitude = 8.0
batch_size = 15

traj_clean, loss_clean, g_clean = run_method_2d(
    "sgm_clean", steps=40, eta=0.15, psi=psi, magnitude=magnitude,
    batch_size=batch_size,
)

traj_corrupted, loss_corrupted, g_corrupted = run_method_2d(
    "sgm_corrupt", steps=40, eta=0.15, psi=psi, magnitude=magnitude,
    batch_size=batch_size,
)

traj_gm, loss_gm, g_gm = run_method_2d(
    "gm_sgm", steps=40, eta=0.15, psi=psi, magnitude=magnitude,
    batch_size=batch_size,
)

traj_block, loss_block, g_block = run_method_2d(
    "block_memory", steps=40, eta=0.15, psi=psi, magnitude=magnitude,
    batch_size=batch_size, block_k=2,
)


# ============================================================
# 9. Surface mesh for 3D plots
# ============================================================

w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

# ============================================================
# 10. 3D plotting helper
# ============================================================

def plot_3d(traj, title, filename, color, cmap):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        W0, W1, Z, cmap=cmap, alpha=0.8,
        rstride=1, cstride=1, edgecolor="none",
    )
    ax.plot(traj[:, 0], traj[:, 1], f(traj[:, 0], traj[:, 1]),
            "-o", color=color, linewidth=2)
    ax.scatter(1, 2, f(1, 2), c="black", s=60, marker="*", label="True Minimum")
    ax.set_xlabel("w₀")
    ax.set_ylabel("w₁")
    ax.set_zlabel("f(w)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()


# 3D plots
plot_3d(traj_clean.numpy(), "Clean SGM — Stable Convergence",
        "sgm_clean_3d.png", "blue", "viridis")
plot_3d(traj_corrupted.numpy(), "SGM under Gross Corruption — Divergent",
        "sgm_gross_corruption_3d.png", "red", "plasma")
plot_3d(traj_gm.numpy(), "GM-SGM — Robust under Gross Corruption",
        "gm_sgm_gross_corruption_3d.png", "green", "cividis")
plot_3d(traj_block.numpy(), "GM-SGM Block+Memory — Robust & Fast",
        "gm_sgm_block_gross_corruption_3d.png", "purple", "magma")

# ============================================================
# 11. 2D trajectory comparison (all four)
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(traj_clean[:, 0], traj_clean[:, 1], "-o",
         label="Clean SGM", color="blue", linewidth=2, markersize=5)
plt.plot(traj_corrupted[:, 0], traj_corrupted[:, 1], "-o",
         label="Corrupted SGM", color="red", linewidth=2, markersize=5, alpha=0.8)
plt.plot(traj_gm[:, 0], traj_gm[:, 1], "-o",
         label="GM-SGM", color="green", linewidth=2, markersize=5, alpha=0.9)
plt.plot(traj_block[:, 0], traj_block[:, 1], "-o",
         label="GM-SGM Block+Memory", color="purple", linewidth=2, markersize=5, alpha=0.9)

plt.scatter(1, 2, marker="*", color="black", s=150, label="True Minimum (1,2)")
plt.xlabel("w₀")
plt.ylabel("w₁")
plt.title("2D Trajectories — SGM vs GM-SGM vs Block+Memory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_gmsgm_block_comparison_2d.png", dpi=300)
plt.close()


# ============================================================
# 12. Save trajectories to CSV
# ============================================================

def to_df(traj, loss, gvals, corrupted, method):
    return pd.DataFrame({
        "step": np.arange(len(traj)),
        "w0": traj[:, 0].numpy(),
        "w1": traj[:, 1].numpy(),
        "f(w)": loss,
        "g(w)": gvals,
        "corrupted": corrupted,
        "method": method,
    })


df = pd.concat([
    to_df(traj_clean, loss_clean, g_clean, False, "SGM_clean"),
    to_df(traj_corrupted, loss_corrupted, g_corrupted, True, "SGM_corrupt"),
    to_df(traj_gm, loss_gm, g_gm, True, "GM_SGM_full"),
    to_df(traj_block, loss_block, g_block, True, "GM_SGM_block_memory"),
], ignore_index=True)

df.to_csv("results/sgm_gmsgm_block_gross_corruption_trajectories.csv", index=False)


# ============================================================
# 13. High-dimensional timing benchmark
#      (Correctly separates GM-SGM vs Block+Memory)
# ============================================================

def random_grad(d, psi: float = 0.3, mag: float = 10.0):
    g = torch.randn(d, dtype=dtype)
    if np.random.rand() < psi:
        v = torch.randn(d, dtype=dtype)
        v = v / v.norm().clamp_min(1e-12)
        return mag * v
    return g


def warmup(d: int):
    """Warm-up for fair timing (especially on GPU)."""
    _ = random_grad(d)
    _ = geometric_median([random_grad(d) for _ in range(5)])


def benchmark_sgm(d: int, iters: int = 40):
    warmup(d)
    t0 = time()
    for _ in range(iters):
        _ = (random_grad(d) + random_grad(d) + random_grad(d)) / 3.0
    return time() - t0


def benchmark_gm_sgm(d: int, iters: int = 40, batch: int = 40):
    """Full GM-SGM: geometric median in R^d over 'batch' workers."""
    warmup(d)
    t0 = time()
    for _ in range(iters):
        grads = [random_grad(d) for _ in range(batch)]  # each in R^d
        _ = geometric_median(grads)                      # GM in R^d
    return time() - t0


def benchmark_block(d: int, iters: int = 40, batch: int = 40, block_k: int | None = None):
    """
    Block+Memory GM-SGM:
      - gradients in R^d
      - choose a block I_k of size k << d
      - compute GM only in R^k and embed back
      - update memory (error feedback style)
    """
    warmup(d)

    if block_k is None:
        block_k = max(10, d // 200)   # ~0.5% of dimensions

    # fixed block for this run (cheaper, still captures idea)
    idx = torch.randperm(d)[:block_k]
    mask = torch.zeros(d, dtype=dtype)
    mask[idx] = 1.0

    memory = torch.zeros(d, dtype=dtype)

    t0 = time()
    for _ in range(iters):
        # simulate batch gradients
        grads = [random_grad(d) for _ in range(batch)]

        # approximate "true" aggregate gradient
        g_bar = sum(grads) / batch

        # add memory (error feedback)
        v = g_bar + memory

        # project each worker gradient + memory to block
        block_vectors = [(g_i + memory)[idx] for g_i in grads]  # each in R^k

        # geometric median in R^k
        gm_block = geometric_median(block_vectors)

        # embed back to R^d
        update = torch.zeros(d, dtype=dtype)
        update[idx] = gm_block

        # memory update: m_{t+1} = m_t + g_bar - update
        memory = memory + g_bar - update

    return time() - t0


def run_timing_experiment():
    dims = [10_000, 100_000, 250_000, 500_000]
    sgm_t, gm_t, block_t = [], [], []

    print("\n==================== TIMING EXPERIMENT ====================")
    print("d\tSGM (s)\tGM-SGM (s)\tBlock+Memory (s)")
    print("-----------------------------------------------------------")

    for d in dims:
        t1 = benchmark_sgm(d)
        t2 = benchmark_gm_sgm(d)
        t3 = benchmark_block(d)

        sgm_t.append(t1)
        gm_t.append(t2)
        block_t.append(t3)

        print(f"{d}\t{t1:.4f}\t{t2:.4f}\t\t{t3:.4f}")

    # Save CSV
    timing_df = pd.DataFrame({
        "dimension": dims,
        "sgm_time": sgm_t,
        "gm_sgm_time": gm_t,
        "block_memory_time": block_t,
    })
    timing_df.to_csv("results/timing_vs_dimension.csv", index=False)

    # Plot
    plt.figure(figsize=(9, 6))
    plt.plot(dims, sgm_t, "-o", label="SGM (mean, fast, non-robust)")
    plt.plot(dims, gm_t, "-o", label="GM-SGM (full median, slow)")
    plt.plot(dims, block_t, "-o", label="GM-SGM Block+Memory (fast & robust)")
    plt.xlabel("Dimension d")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Dimension — SGM vs GM-SGM vs Block+Memory")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/timing_vs_dimension.png", dpi=300)
    plt.close()

    print("\nTiming results saved:")
    print(" - results/timing_vs_dimension.png")
    print(" - results/timing_vs_dimension.csv")
    print("===========================================================\n")


run_timing_experiment()

# ============================================================
# 14. Summary
# ============================================================

print("Done. Results saved in 'results/' folder:")
print(" - sgm_clean_3d.png")
print(" - sgm_gross_corruption_3d.png")
print(" - gm_sgm_gross_corruption_3d.png")
print(" - gm_sgm_block_gross_corruption_3d.png")
print(" - sgm_gmsgm_block_comparison_2d.png")
print(" - sgm_gmsgm_block_gross_corruption_trajectories.csv")
print(" - timing_vs_dimension.png")
print(" - timing_vs_dimension.csv")
