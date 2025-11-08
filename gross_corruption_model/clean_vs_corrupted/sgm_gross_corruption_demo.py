import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

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
# 2. Gross Corruption Model
# ====================================================

def gross_corruption(grad, psi=0.4, magnitude=8.0):
    """
    Apply the Gross Corruption Model (GCM).
    With probability psi, replace the true gradient with an arbitrary (corrupted) vector.
    """
    if np.random.rand() < psi:
        # Adversarial replacement: completely overwrite gradient
        corrupt_direction = torch.randn_like(grad)
        corrupt_direction = corrupt_direction / torch.norm(corrupt_direction)  # normalize direction
        grad = magnitude * corrupt_direction  # replace by a large arbitrary vector
    return grad

# ====================================================
# 3. SGM update rule (with optional corruption)
# ====================================================

def sgm_step(w, eta=0.1, eps=0.0, corruption=False, psi=0.4, magnitude=8.0):
    """Single SGM step with optional Gross Corruption Model."""
    g_val = g_value(w)
    grad = grad_f(w) if g_val <= eps else grad_g()

    # Apply Gross Corruption Model if enabled
    if corruption:
        grad = gross_corruption(grad, psi=psi, magnitude=magnitude)

    return w - eta * grad

# ====================================================
# 4. Run SGM trajectories
# ====================================================

def run_sgm(clean=True, steps=40, eta=0.15, psi=0.4, magnitude=8.0):
    """Run SGM trajectory and record progress."""
    w = torch.tensor([-2.0, 3.0])
    traj = [w.clone()]
    losses = [f(w[0], w[1])]
    gvals = [g_value(w)]

    for t in range(steps):
        w = sgm_step(w, eta=eta, corruption=(not clean), psi=psi, magnitude=magnitude)
        traj.append(w.clone())
        losses.append(f(w[0], w[1]))
        gvals.append(g_value(w))
    
    return torch.stack(traj), np.array(losses), np.array(gvals)

# ====================================================
# 5. Run clean and corrupted cases
# ====================================================

psi = 0.4          # fraction of corrupted gradients
magnitude = 8.0    # intensity of adversarial corruption

traj_clean, loss_clean, g_clean = run_sgm(clean=True)
traj_corrupted, loss_corrupted, g_corrupted = run_sgm(clean=False, psi=psi, magnitude=magnitude)

# ====================================================
# 6. Create visualization folder
# ====================================================

Path("results").mkdir(exist_ok=True)

# ====================================================
# 7. 3D plots
# ====================================================

w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

def plot_3d(traj, title, filename, color, cmap):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W0, W1, Z, cmap=cmap, alpha=0.7)
    ax.plot(traj[:,0], traj[:,1], f(traj[:,0], traj[:,1]), '-o', color=color)
    ax.scatter(1, 2, f(1,2), c='black', s=60, marker='*', label='True Minimum')
    ax.set_xlabel('w₀'); ax.set_ylabel('w₁'); ax.set_zlabel('f(w)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()

plot_3d(traj_clean.numpy(), "Clean SGM — Stable Convergence", "sgm_clean_3d.png", "blue", "viridis")
plot_3d(traj_corrupted.numpy(), "SGM under Gross Corruption Model — Divergent Behavior", "sgm_gcm_corrupted_3d.png", "red", "plasma")

# ====================================================
# 8. 2D Trajectory Plot
# ====================================================

plt.figure(figsize=(7,5))
plt.plot(traj_clean[:,0], traj_clean[:,1], '-o', label='Clean (Converges)', color='blue')
plt.plot(traj_corrupted[:,0], traj_corrupted[:,1], '-o', label='Gross Corruption (Fails)', color='red', alpha=0.7)
plt.scatter(1, 2, marker='*', color='black', s=150, label='True minimum (1,2)')
plt.xlabel("w₀"); plt.ylabel("w₁")
plt.title("2D Trajectories — Clean SGM vs Gross Corruption Model")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_gcm_2d_trajectories.png", dpi=300)
plt.close()



plt.figure(figsize=(7,5))
plt.plot(loss_clean, '-o', color='blue', label='Clean loss f(w)')
plt.plot(loss_corrupted, '-o', color='red', alpha=0.7, label='Gross Corruption loss f(w)')
plt.xlabel("Iteration t"); plt.ylabel("f(wₜ)")
plt.title("Loss evolution — Convergence vs Gross Corruption Model")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_gcm_loss_evolution.png", dpi=300)
plt.close()


data_clean = pd.DataFrame({
    "step": np.arange(len(traj_clean)),
    "w0": traj_clean[:,0].numpy(),
    "w1": traj_clean[:,1].numpy(),
    "f(w)": loss_clean,
    "g(w)": g_clean,
    "corrupted": False
})

data_corrupted = pd.DataFrame({
    "step": np.arange(len(traj_corrupted)),
    "w0": traj_corrupted[:,0].numpy(),
    "w1": traj_corrupted[:,1].numpy(),
    "f(w)": loss_corrupted,
    "g(w)": g_corrupted,
    "corrupted": True
})

df = pd.concat([data_clean, data_corrupted], ignore_index=True)
df.to_csv("results/sgm_gross_corruption_trajectories.csv", index=False)


print("Done. Results saved in 'results/' folder:")
print(" - sgm_clean_3d.png")
print(" - sgm_gcm_corrupted_3d.png")
print(" - sgm_gcm_2d_trajectories.png")
print(" - sgm_gcm_loss_evolution.png")
print(" - sgm_gross_corruption_trajectories.csv")
