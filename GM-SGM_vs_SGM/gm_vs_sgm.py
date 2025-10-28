import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# ====================================================
# 1. Setup: Objective and Constraint
# ====================================================

def f(w0, w1):
    """Convex quadratic objective."""
    return (w0 - 1)**2 + (w1 - 2)**2

def grad_f(w):
    return 2 * (w - torch.tensor([1.0, 2.0]))

def g_value(w):
    return w[0] + w[1] - 2.0

def grad_g():
    return torch.tensor([1.0, 1.0])

# ====================================================
# 2. Helper: Geometric Median (Weiszfeld)
# ====================================================

def geometric_median(vectors, eps=1e-6, max_iter=100):
    v = torch.stack(vectors)
    guess = v.mean(dim=0)
    for _ in range(max_iter):
        distances = torch.norm(v - guess, dim=1).clamp_min(eps)
        weights = 1.0 / distances
        new_guess = (v * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        if torch.norm(new_guess - guess) < eps:
            break
        guess = new_guess
    return guess

# ====================================================
# 3. Single SGM step
# ====================================================

def sgm_step(w, eta=0.1, eps=0.0, corruption=False):
    g_val = g_value(w)
    grad = grad_f(w) if g_val <= eps else grad_g()
    if corruption:
        noise = 2.0 * torch.randn_like(grad)
        if np.random.rand() < 0.3:
            grad = grad + noise * 3.0  # 30% corrupted spikes
    return w - eta * grad

# ====================================================
# 4. Single GM-SGM step
# ====================================================

def gm_sgm_step(w, eta=0.1, eps=0.0, corruption=True, batch_size=10):
    grads = []
    for _ in range(batch_size):
        g_val = g_value(w)
        grad = grad_f(w) if g_val <= eps else grad_g()
        if corruption and np.random.rand() < 0.3:
            noise = 2.0 * torch.randn_like(grad)
            grad = grad + noise * 3.0
        grads.append(grad)
    gm_grad = geometric_median(grads)
    return w - eta * gm_grad

# ====================================================
# 5. Run a method
# ====================================================

def run_method(method="sgm", clean=True, steps=40, eta=0.15):
    w = torch.tensor([-2.0, 3.0])
    traj, losses, gvals = [w.clone()], [f(w[0], w[1])], [g_value(w)]
    for _ in range(steps):
        if method == "gm-sgm":
            w = gm_sgm_step(w, eta=eta, corruption=(not clean))
        else:
            w = sgm_step(w, eta=eta, corruption=(not clean))
        traj.append(w.clone())
        losses.append(f(w[0], w[1]))
        gvals.append(g_value(w))
    return torch.stack(traj), np.array(losses), np.array(gvals)

# ====================================================
# 6. Run all three cases
# ====================================================

traj_clean, loss_clean, g_clean = run_method("sgm", clean=True)
traj_corrupted, loss_corrupted, g_corrupted = run_method("sgm", clean=False)
traj_gm, loss_gm, g_gm = run_method("gm-sgm", clean=False)

# ====================================================
# 7. Create folder
# ====================================================

Path("results").mkdir(exist_ok=True)

# ====================================================
# 8. Prepare mesh for plotting
# ====================================================

w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

# ====================================================
# 9. 3D Plots
# ====================================================

def plot_3d(traj, title, filename, color, cmap):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W0, W1, Z + 0.5*np.random.randn(*Z.shape)*0.3, cmap=cmap, alpha=0.7)
    ax.plot(traj[:,0], traj[:,1], f(traj[:,0], traj[:,1]), '-o', color=color)
    ax.scatter(1, 2, f(1,2), c='black', s=60, marker='*', label='True Minimum')
    ax.set_xlabel('w₀'); ax.set_ylabel('w₁'); ax.set_zlabel('f(w)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

plot_3d(traj_clean.numpy(), "Clean SGM — Stable Convergence", "sgm_clean_3d.png", "blue", "viridis")
plot_3d(traj_corrupted.numpy(), "Corrupted SGM — Divergent / Unstable", "sgm_corrupted_3d.png", "red", "plasma")
plot_3d(traj_gm.numpy(), "GM-SGM — Robust Convergence under Corruption", "gm_sgm_corrupted_3d.png", "green", "cividis")

# ====================================================
# 10. 2D Trajectory Plot (Comparison)
# ====================================================

plt.figure(figsize=(7,5))
plt.plot(traj_clean[:,0], traj_clean[:,1], '-o', label='Clean SGM', color='blue')
plt.plot(traj_corrupted[:,0], traj_corrupted[:,1], '-o', label='Corrupted SGM', color='red', alpha=0.7)
plt.plot(traj_gm[:,0], traj_gm[:,1], '-o', label='GM-SGM (Robust)', color='green', alpha=0.9)
plt.scatter(1, 2, marker='*', color='black', s=150, label='True Minimum (1,2)')
plt.xlabel("w₀"); plt.ylabel("w₁")
plt.title("2D Trajectories — Clean SGM vs Corrupted SGM vs GM-SGM")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_gmsgm_2d_comparison.png", dpi=300)
plt.close()

# ====================================================
# 11. Save CSV with all trajectories
# ====================================================

def to_df(traj, loss, gvals, corrupted, method):
    return pd.DataFrame({
        "step": np.arange(len(traj)),
        "w0": traj[:,0].numpy(),
        "w1": traj[:,1].numpy(),
        "f(w)": loss,
        "g(w)": gvals,
        "corrupted": corrupted,
        "method": method
    })

df = pd.concat([
    to_df(traj_clean, loss_clean, g_clean, False, "SGM_clean"),
    to_df(traj_corrupted, loss_corrupted, g_corrupted, True, "SGM_corrupted"),
    to_df(traj_gm, loss_gm, g_gm, True, "GM_SGM_corrupted")
], ignore_index=True)

df.to_csv("results/sgm_gmsgm_trajectories.csv", index=False)

# ====================================================
# 12. Summary
# ====================================================

print("Done. Results saved in 'results/' folder:")
print(" - sgm_clean_3d.png")
print(" - sgm_corrupted_3d.png")
print(" - gm_sgm_corrupted_3d.png")
print(" - sgm_gmsgm_2d_comparison.png")
print(" - sgm_gmsgm_trajectories.csv")
