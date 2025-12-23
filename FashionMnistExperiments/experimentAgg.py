import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# ============================================================
# Setup
# ============================================================
os.makedirs("results_block_bench", exist_ok=True)

# Larger dimensions for clearer scaling
D_FULL = 500_000       # full dimension d (very large)
N_POINTS = 60          # workers for GM-SGM
BETAS = [1.0, 0.6, 0.3, 0.1]
REPEATS = 3            # average over several trials
MAX_ITER = 15          # GM iterations

device = torch.device("cpu")
torch.set_num_threads(1)   # reduce nondeterministic noise
print("Using device:", device)
print(f"Full dimension d = {D_FULL}, workers = {N_POINTS}\n")


# ============================================================
# Geometric Median (Weiszfeld)
# ============================================================

def geometric_median(points, max_iter=10, eps=1e-5):
    """
    Geometric median of N points in R^k using Weiszfeld’s algorithm.
    Complexity: O(N * k * max_iter)
    """
    X = torch.stack(points, dim=0)   # (N, k)
    y = X.mean(dim=0)

    for _ in range(max_iter):
        diff = X - y
        dist = torch.norm(diff, dim=1) + 1e-8
        w = 1.0 / dist
        w = w / w.sum()
        y_new = (w.unsqueeze(1) * X).sum(dim=0)

        if torch.norm(y_new - y) < eps:
            break
        y = y_new

    return y


# ============================================================
# Generate Worker Gradients
# ============================================================

def generate_gradients(n, dim):
    """Generate n random vectors in R^dim."""
    return [torch.randn(dim, device=device) for _ in range(n)]


# ============================================================
# Benchmark Pure Geometric Median in dimension k
# ============================================================

def benchmark_gm(dimension, n_workers, repeats=3, max_iter=10):
    """
    Benchmark geometric median in R^dimension.
    """
    times = []
    for _ in range(repeats):
        points = generate_gradients(n_workers, dimension)
        start = time.time()
        geometric_median(points, max_iter=max_iter)
        times.append(time.time() - start)
    return float(np.mean(times))


# ============================================================
# Run Experiments: Normal GM-SGM + Block GM-SGM
# ============================================================

dims = {}
times = {}

print("=== Running Geometric Median Benchmarks ===\n")

for beta in BETAS:
    k = int(beta * D_FULL)
    k = max(k, 1)
    dims[beta] = k

    print(f"[β = {beta:.1f}] dimension k = {k:,}")

    t = benchmark_gm(
        dimension=k,
        n_workers=N_POINTS,
        repeats=REPEATS,
        max_iter=MAX_ITER,
    )

    times[beta] = t
    print(f"  GM time = {t:.4f} s\n")


print("\n=== Final Results ===")
for b in sorted(BETAS, reverse=True):
    print(f"β={b:.1f}, k={dims[b]:8d}, time={times[b]:.4f} s")


# ============================================================
# Plot 1 — Bar chart of runtimes
# ============================================================
plt.figure(figsize=(10, 6))
labels = [f"β={b:.1f}\n(k={dims[b]})" for b in sorted(BETAS, reverse=True)]
bar_vals = [times[b] for b in sorted(BETAS, reverse=True)]
plt.bar(labels, bar_vals, color="royalblue")
plt.ylabel("Runtime (seconds)")
plt.title("Geometric Median Runtime for Different Block Fractions β")
plt.tight_layout()
plt.savefig("results_block_bench/plot_gm_runtime_bar.png")
plt.close()


# ============================================================
# Plot 2 — Runtime vs Dimension
# ============================================================
plt.figure(figsize=(10, 6))
dim_vals = np.array([dims[b] for b in sorted(BETAS, reverse=True)])
time_vals = np.array([times[b] for b in sorted(BETAS, reverse=True)])
plt.plot(dim_vals, time_vals, "-o", linewidth=2)
plt.xlabel("Dimension k")
plt.ylabel("Runtime (seconds)")
plt.title("Geometric Median Runtime vs Dimension")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results_block_bench/plot_gm_runtime_vs_dimension.png")
plt.close()


# ============================================================
# Plot 3 — Runtime vs β
# ============================================================
plt.figure(figsize=(10, 6))
beta_vals = np.array(sorted(BETAS, reverse=True))
plt.plot(beta_vals, time_vals, "-o", linewidth=2)
plt.xlabel("β (fraction of coordinates kept)")
plt.ylabel("Runtime (seconds)")
plt.title("Geometric Median Runtime vs Block Fraction β")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results_block_bench/plot_gm_runtime_vs_beta.png")
plt.close()


print("\nSaved plots to results_block_bench/:")
print("  - plot_gm_runtime_bar.png")
print("  - plot_gm_runtime_vs_dimension.png")
print("  - plot_gm_runtime_vs_beta.png\n")
