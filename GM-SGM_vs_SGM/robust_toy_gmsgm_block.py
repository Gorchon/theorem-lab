import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. Setup: Objective and Constraint (2D Quadratic)
# ============================================================

def f(w0, w1):
    """Convex quadratic objective."""
    return (w0 - 1)**2 + (w1 - 2)**2

def grad_f(w):
    return 2 * (w - torch.tensor([1.0, 2.0]))

def g_value(w):
    return w[0] + w[1] - 2.0

def grad_g():
    return torch.tensor([1.0, 1.0])

# ============================================================
# 2. Geometric Median (Weiszfeld)
# ============================================================

def geometric_median(vectors, eps=1e-6, max_iter=100):
    """Weiszfeld's algorithm for geometric median."""
    v = torch.stack(vectors)
    guess = v.mean(dim=0)
    for _ in range(max_iter):
        diffs = v - guess
        dists = torch.norm(diffs, dim=1).clamp_min(eps)
        weights = 1.0 / dists
        new_guess = (weights.unsqueeze(1) * v).sum(dim=0) / weights.sum()
        if torch.norm(new_guess - guess) < eps:
            break
        guess = new_guess
    return guess

# ============================================================
# 3. Gross Corruption
# ============================================================

def gross_corruption(grad, psi=0.3, magnitude=8.0):
    """Apply Gross Corruption Model (GCM)."""
    if np.random.rand() < psi:
        noise = torch.randn_like(grad)
        noise = noise / torch.norm(noise)
        return magnitude * noise
    return grad

# ============================================================
# 4. Block Coordinate Selection + Memory Mechanism
# ============================================================

def block_coordinate_selection(worker_grads, k):
    """
    worker_grads: (n_workers, d)
    Returns indices of top-k "energetic" coordinates.
    """
    energy = (worker_grads**2).sum(dim=0)
    k = min(k, energy.numel())
    _, idx = torch.topk(energy, k)
    return idx

def compress_to_block(grad, idx, dim):
    """
    Zero all coordinates except the top-k indices.
    """
    out = torch.zeros(dim)
    out[idx] = grad[idx]
    return out

# ============================================================
# 5. SGM step (vanilla)
# ============================================================

def sgm_step(w, eta=0.1, corruption=False, psi=0.3, magnitude=8.0):
    gval = g_value(w)
    grad = grad_f(w) if gval <= 0.0 else grad_g()
    if corruption:
        grad = gross_corruption(grad, psi=psi, magnitude=magnitude)
    return w - eta * grad

# ============================================================
# 6. GM-SGM step (full geometric median)
# ============================================================

def gm_sgm_step(w, eta=0.1, batch_size=10, corruption=True, psi=0.3, magnitude=8.0):
    grads = []
    for _ in range(batch_size):
        gval = g_value(w)
        grad = grad_f(w) if gval <= 0.0 else grad_g()
        if corruption:
            grad = gross_corruption(grad, psi=psi, magnitude=magnitude)
        grads.append(grad)
    gm_grad = geometric_median(grads)
    return w - eta * gm_grad

# ============================================================
# 7. GM-SGM + Block Coordinate Selection + Memory
# ============================================================

def gm_sgm_block_memory_step(w, memory, eta=0.1, n_workers=10,
                              psi=0.3, magnitude=8.0, block_k=1):
    """
    w: current point
    memory: momentum-like memory vector
    block_k: number of coordinates to keep (block)
    """
    d = 2  # dimension is 2D here

    # True clean gradient
    gval = g_value(w)
    base_grad = grad_f(w) if gval <= 0 else grad_g()

    # Combine with memory (NO in-place)
    combined = base_grad + memory

    # Simulate n workers
    workers = []
    for _ in range(n_workers):
        g = combined.clone()
        if np.random.rand() < psi:
            noise = torch.randn(d)
            noise = noise / torch.norm(noise)
            g = magnitude * noise
        workers.append(g)
    W = torch.stack(workers, dim=0)  # (n_workers, d)

    # Select block coordinates (k)
    idx = block_coordinate_selection(W, block_k)

    # Compress workers to block
    W_block = torch.stack([w_i[idx] for w_i in workers], dim=0)

    # GM over compressed vectors
    gm_block = geometric_median([v for v in W_block])

    # Full update vector (dim=2)
    update = torch.zeros(d)
    update[idx] = gm_block

    # Update memory (OUT-OF-PLACE)
    new_memory = combined - update

    # Actual step
    w_new = w - eta * update
    return w_new, new_memory

# ============================================================
# 8. Unified Runner
# ============================================================

def run_method(method="sgm", steps=40, eta=0.1, psi=0.3, magnitude=8.0,
               block_k=1):
    w = torch.tensor([-2.0, 3.0], dtype=torch.float32)
    memory = torch.zeros(2)

    traj = [w.clone()]
    loss = [f(w[0], w[1])]
    gvals = [g_value(w)]

    for _ in range(steps):
        if method == "sgm_clean":
            w = sgm_step(w, eta=eta, corruption=False)

        elif method == "sgm_corrupt":
            w = sgm_step(w, eta=eta, corruption=True, psi=psi, magnitude=magnitude)

        elif method == "gm_sgm":
            w = gm_sgm_step(w, eta=eta, corruption=True,
                            psi=psi, magnitude=magnitude)

        elif method == "block_memory":
            w, memory = gm_sgm_block_memory_step(
                w, memory, eta=eta, psi=psi, magnitude=magnitude,
                n_workers=10, block_k=block_k
            )

        traj.append(w.clone())
        loss.append(f(w[0], w[1]))
        gvals.append(g_value(w))

    return torch.stack(traj), np.array(loss), np.array(gvals)

# ============================================================
# 9. Run All Methods
# ============================================================

psi = 0.4
magnitude = 8.0

traj_clean, loss_clean, g_clean = run_method("sgm_clean")
traj_corrupt, loss_corrupt, g_corrupt = run_method("sgm_corrupt", psi=psi, magnitude=magnitude)
traj_gm, loss_gm, g_gm = run_method("gm_sgm", psi=psi, magnitude=magnitude)

# NEW:
traj_block, loss_block, g_block = run_method("block_memory",
                                             psi=psi, magnitude=magnitude,
                                             block_k=1)

# ============================================================
# 10. Create results folder
# ============================================================

Path("results").mkdir(exist_ok=True)

# ============================================================
# 11. Meshgrid for plotting
# ============================================================

w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

# ============================================================
# 12. 3D Plot function
# ============================================================

def plot_3d(traj, title, filename, color, cmap):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W0, W1, Z, cmap=cmap, alpha=0.8)
    ax.plot(traj[:,0], traj[:,1], f(traj[:,0], traj[:,1]),
            '-o', color=color, linewidth=2)
    ax.scatter(1, 2, f(1,2), s=80, c='black', marker='*')
    ax.set_xlabel("w0"); ax.set_ylabel("w1"); ax.set_zlabel("f(w)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

# Save all 3D plots
plot_3d(traj_clean.numpy(), "Clean SGM", "sgm_clean_3d.png", "blue", "viridis")
plot_3d(traj_corrupt.numpy(), "Corrupted SGM (Divergent)", "sgm_corrupt_3d.png", "red", "plasma")
plot_3d(traj_gm.numpy(), "GM-SGM (Robust)", "gm_sgm_3d.png", "green", "cividis")
plot_3d(traj_block.numpy(), "GM-SGM Block+Memory (Fast+Robust)", "gm_sgm_block_3d.png", "purple", "magma")

# ============================================================
# 13. 2D Trajectories
# ============================================================

plt.figure(figsize=(8,6))
plt.plot(traj_clean[:,0], traj_clean[:,1], '-o', label='Clean SGM')
plt.plot(traj_corrupt[:,0], traj_corrupt[:,1], '-o', label='Corrupted SGM')
plt.plot(traj_gm[:,0], traj_gm[:,1], '-o', label='GM-SGM')
plt.plot(traj_block[:,0], traj_block[:,1], '-o', label='GM-SGM Block+Memory')
plt.scatter(1, 2, c='black', marker='*', s=140)
plt.xlabel("w0"); plt.ylabel("w1")
plt.legend()
plt.title("2D Trajectories")
plt.tight_layout()
plt.savefig("results/sgm_all_methods_2d.png", dpi=300)
plt.close()

# ============================================================
# 14. Save CSV
# ============================================================

def to_df(traj, loss, gvals, label):
    return pd.DataFrame({
        "step": np.arange(len(traj)),
        "w0": traj[:,0].numpy(),
        "w1": traj[:,1].numpy(),
        "f(w)": loss,
        "g(w)": gvals,
        "method": label
    })

df = pd.concat([
    to_df(traj_clean, loss_clean, g_clean, "SGM_clean"),
    to_df(traj_corrupt, loss_corrupt, g_corrupt, "SGM_corrupted"),
    to_df(traj_gm, loss_gm, g_gm, "GM_SGM"),
    to_df(traj_block, loss_block, g_block, "GM_SGM_block_memory")
])

df.to_csv("results/toy_sgm_all_methods.csv", index=False)

# ============================================================
# 15. Summary
# ============================================================

print("\nDone! Results saved to results/:")
print(" - sgm_clean_3d.png")
print(" - sgm_corrupt_3d.png")
print(" - gm_sgm_3d.png")
print(" - gm_sgm_block_3d.png")
print(" - sgm_all_methods_2d.png")
print(" - toy_sgm_all_methods.csv")
