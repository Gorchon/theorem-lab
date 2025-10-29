import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# ====================================================
# 1. Setup simple quadratic + constraint
# ====================================================

def f(w0, w1):
    """Objective function: convex paraboloid."""
    return (w0 - 1)**2 + (w1 - 2)**2

def grad_f(w):
    """Gradient of f."""
    return 2 * (w - torch.tensor([1.0, 2.0]))

def g_value(w):
    """Constraint: feasible region when g(w) <= 0."""
    return w[0] + w[1] - 2.0

def grad_g():
    """Gradient of constraint g."""
    return torch.tensor([1.0, 1.0])

# ====================================================
# 2. SGM update rule
# ====================================================

def sgm_step(w, eta=0.1, eps=0.0, corruption=False):
    """Single SGM step with optional corruption."""
    g_val = g_value(w)
    if g_val <= eps:
        grad = grad_f(w)
    else:
        grad = grad_g()

    # Add gradient corruption (simulate noisy gradients or outliers)
    if corruption:
        noise = 2.0 * torch.randn_like(grad)
        if np.random.rand() < 0.3:
            grad = grad + noise * 3.0  # large random spike
    
    return w - eta * grad

# ====================================================
# 3. Run SGM trajectories
# ====================================================

def run_sgm(clean=True, steps=40, eta=0.15):
    """Run SGM trajectory and record progress."""
    w = torch.tensor([-2.0, 3.0])
    traj = [w.clone()]
    losses = [f(w[0], w[1])]
    gvals = [g_value(w)]
    for t in range(steps):
        w = sgm_step(w, eta=eta, corruption=(not clean))
        traj.append(w.clone())
        losses.append(f(w[0], w[1]))
        gvals.append(g_value(w))
    return torch.stack(traj), np.array(losses), np.array(gvals)

traj_clean, loss_clean, g_clean = run_sgm(clean=True)
traj_noisy, loss_noisy, g_noisy = run_sgm(clean=False)

# ====================================================
# 4. Create visualization folder
# ====================================================
Path("results").mkdir(exist_ok=True)

# ====================================================
# 5. 3D plots (same as before)
# ====================================================

w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

# ----- Clean -----
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, Z, cmap='viridis', alpha=0.7)
traj = traj_clean.numpy()
ax.plot(traj[:,0], traj[:,1], f(traj[:,0], traj[:,1]), '-o', color='blue', label='Clean SGM trajectory')
ax.set_xlabel('w₀'); ax.set_ylabel('w₁'); ax.set_zlabel('f(w)')
ax.set_title(' SGM on Clean Data — Smooth Convergence to Minimum')
ax.legend()
plt.tight_layout()
plt.savefig("results/sgm_clean_3d.png", dpi=300)
plt.close()

# ----- Corrupted -----
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, Z + 0.5*np.random.randn(*Z.shape), cmap='plasma', alpha=0.7)
traj = traj_noisy.numpy()
ax.plot(traj[:,0], traj[:,1], f(traj[:,0], traj[:,1]), '-o', color='red', label='Corrupted SGM trajectory')
ax.set_xlabel('w₀'); ax.set_ylabel('w₁'); ax.set_zlabel('f(w)')
ax.set_title('❌ SGM on Corrupted Data — Unstable / Divergent Behavior')
ax.legend()
plt.tight_layout()
plt.savefig("results/sgm_corrupted_3d.png", dpi=300)
plt.close()

# ====================================================
# 6. 2D Trajectory Plots
# ====================================================

plt.figure(figsize=(7,5))
plt.plot(traj_clean[:,0], traj_clean[:,1], '-o', label='Clean (Converges)', color='blue')
plt.plot(traj_noisy[:,0], traj_noisy[:,1], '-o', label='Corrupted (Fails)', color='red', alpha=0.7)
plt.scatter(1, 2, marker='*', color='green', s=150, label='True minimum (1,2)')
plt.xlabel("w₀"); plt.ylabel("w₁")
plt.title("2D Trajectories — Clean vs Corrupted SGM")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_2d_trajectories.png", dpi=300)
plt.close()

# ====================================================
# 7. Loss evolution plots
# ====================================================

plt.figure(figsize=(7,5))
plt.plot(loss_clean, '-o', color='blue', label='Clean loss f(w)')
plt.plot(loss_noisy, '-o', color='red', alpha=0.7, label='Corrupted loss f(w)')
plt.xlabel("Iteration t"); plt.ylabel("f(wₜ)")
plt.title("Loss evolution — Convergence vs Divergence")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_loss_evolution.png", dpi=300)
plt.close()

# ====================================================
# 8. Save CSV with trajectories
# ====================================================

data_clean = pd.DataFrame({
    "step": np.arange(len(traj_clean)),
    "w0": traj_clean[:,0].numpy(),
    "w1": traj_clean[:,1].numpy(),
    "f(w)": loss_clean,
    "g(w)": g_clean,
    "corrupted": False
})
data_noisy = pd.DataFrame({
    "step": np.arange(len(traj_noisy)),
    "w0": traj_noisy[:,0].numpy(),
    "w1": traj_noisy[:,1].numpy(),
    "f(w)": loss_noisy,
    "g(w)": g_noisy,
    "corrupted": True
})

df = pd.concat([data_clean, data_noisy], ignore_index=True)
df.to_csv("results/sgm_trajectories.csv", index=False)

# ====================================================
# 9. Summary
# ====================================================

print(" Done. Results saved in 'results/' folder:")
print(" - sgm_clean_3d.png")
print(" - sgm_corrupted_3d.png")
print(" - sgm_2d_trajectories.png")
print(" - sgm_loss_evolution.png")
print(" - sgm_trajectories.csv")
