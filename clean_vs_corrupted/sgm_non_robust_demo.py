import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# -----------------------------
# Setup simple quadratic + constraint
# -----------------------------
def f(w0, w1):
    return (w0 - 1)**2 + (w1 - 2)**2

def grad_f(w):
    return 2 * (w - torch.tensor([1.0, 2.0]))

def g_value(w):
    return w[0] + w[1] - 2.0  # feasible when <= 0

def grad_g():
    return torch.tensor([1.0, 1.0])

# -----------------------------
# SGM update rule
# -----------------------------
def sgm_step(w, eta=0.1, eps=0.0, corruption=False):
    g_val = g_value(w)
    if g_val <= eps:
        grad = grad_f(w)
    else:
        grad = grad_g()

    # Apply corruption (simulate noisy gradients or outliers)
    if corruption:
        noise = 2.0 * torch.randn_like(grad)
        if np.random.rand() < 0.3:
            grad = grad + noise * 3.0  # random spikes
    return w - eta * grad

# -----------------------------
# Run SGM trajectories
# -----------------------------
def run_sgm(clean=True, steps=40):
    w = torch.tensor([-2.0, 3.0])
    traj = [w.clone()]
    for t in range(steps):
        w = sgm_step(w, eta=0.15, corruption=(not clean))
        traj.append(w.clone())
    return torch.stack(traj)

traj_clean = run_sgm(clean=True)
traj_noisy = run_sgm(clean=False)

# -----------------------------
# Create 3D visualization grid
# -----------------------------
w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

Path("results").mkdir(exist_ok=True)

# -----------------------------
# PLOT 1: Clean SGM converges
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, Z, cmap='viridis', alpha=0.7)

traj = traj_clean.numpy()
ax.plot(traj[:,0], traj[:,1], f(traj[:,0], traj[:,1]), '-o', color='blue', label='SGM trajectory (clean)')
ax.set_xlabel('w₀'); ax.set_ylabel('w₁'); ax.set_zlabel('f(w)')
ax.set_title(' SGM on Clean Data — Smooth Convergence to Minimum')
ax.legend()
plt.tight_layout()
plt.savefig("results/sgm_clean_3d.png", dpi=300)
plt.close()

# -----------------------------
# PLOT 2: Corrupted SGM fails
# -----------------------------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, Z + 0.5*np.random.randn(*Z.shape), cmap='plasma', alpha=0.7)  # noisy landscape
traj = traj_noisy.numpy()
ax.plot(traj[:,0], traj[:,1], f(traj[:,0], traj[:,1]), '-o', color='red', label='SGM trajectory (corrupted)')
ax.set_xlabel('w₀'); ax.set_ylabel('w₁'); ax.set_zlabel('f(w)')
ax.set_title('❌ SGM on Corrupted Data — Unstable / Divergent Behavior')
ax.legend()
plt.tight_layout()
plt.savefig("results/sgm_corrupted_3d.png", dpi=300)
plt.close()

print(" Done. Saved to results/:")
print(" - sgm_clean_3d.png")
print(" - sgm_corrupted_3d.png")
