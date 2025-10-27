import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# PROBLEM SETUP
# ==========================================================
@dataclass
class Problem:
    X: torch.Tensor
    y: torch.Tensor
    c: torch.Tensor
    eps: float
    d: int

def make_problem(N=512, d=2, noise_std=0.05, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(N, d)
    theta_star = torch.tensor([1.0, 0.0])
    y = (X @ theta_star) + noise_std * torch.randn(N)
    c = torch.tensor([1.0, 1.0])
    return Problem(X, y, c, eps=0.0, d=d)

def f_batch(w, Xb, yb): return 0.5 * torch.mean((Xb @ w - yb)**2)
def grad_f_samples(w, Xb, yb): return (Xb @ w - yb).unsqueeze(1) * Xb
def g_value(w, c): return torch.dot(c, w) - 1.0

# ==========================================================
# SGM IMPLEMENTATION
# ==========================================================
def run_sgm(problem, steps=100, eta=0.2, batch_size=64, corruption=0.0, seed=42):
    torch.manual_seed(seed)
    N, d = problem.X.shape
    w = torch.zeros(d)
    traj, f_hist, g_hist = [w.clone()], [], []

    for t in range(steps):
        idx = torch.randint(0, N, (batch_size,))
        Xb, yb = problem.X[idx], problem.y[idx]
        grads = grad_f_samples(w, Xb, yb)

        # Corrupt a fraction of the gradients (simulate outliers)
        if corruption > 0.0:
            B = grads.shape[0]
            num_bad = int(corruption * B)
            idx_bad = torch.randperm(B)[:num_bad]
            grads[idx_bad] = grads[idx_bad] * 3.0 + 5.0 * torch.randn_like(grads[idx_bad])

        grad_mean = grads.mean(0)
        g_pred = g_value(w, problem.c).item()
        step_dir = grad_mean if g_pred <= problem.eps else problem.c
        w = w - eta * step_dir

        traj.append(w.clone())
        f_hist.append(f_batch(w, Xb, yb).item())
        g_hist.append(g_value(w, problem.c).item())

    return {
        "traj": torch.stack(traj),
        "f_hist": torch.tensor(f_hist),
        "g_hist": torch.tensor(g_hist),
    }

# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================
problem = make_problem()
corruption_levels = [0.0, 0.1, 0.2, 0.4]
results = {}

for c in corruption_levels:
    results[c] = run_sgm(problem, steps=120, eta=0.2, batch_size=64, corruption=c)

# Create results folder
Path("results").mkdir(exist_ok=True)

# ==========================================================
# SAVE METRICS TO CSV
# ==========================================================
rows = []
for c in corruption_levels:
    hist = results[c]["f_hist"].numpy()
    for t, val in enumerate(hist):
        rows.append({"iteration": t, "corruption_level": c, "objective": val})
df = pd.DataFrame(rows)
df.to_csv("results/sgm_corruption_experiment.csv", index=False)
print("✅ Saved metrics to results/sgm_corruption_experiment.csv")

# ==========================================================
# PLOT: Objective vs Iterations
# ==========================================================
plt.figure(figsize=(7,5))
colors = {0.0: "#1f77b4", 0.1: "#2ca02c", 0.2: "#ff7f0e", 0.4: "#d62728"}
for c in corruption_levels:
    plt.plot(results[c]["f_hist"].numpy(), label=f"{int(c*100)}% corrupted", color=colors[c])
plt.xlabel("Iteration")
plt.ylabel("Objective f(w)")
plt.title("SGM Objective under Increasing Data Corruption")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("results/sgm_objective_vs_corruption.png", dpi=300)
plt.close()
print("✅ Saved: results/sgm_objective_vs_corruption.png")

# ==========================================================
# PLOT: Constraint Values
# ==========================================================
plt.figure(figsize=(7,5))
for c in corruption_levels:
    plt.plot(results[c]["g_hist"].numpy(), label=f"{int(c*100)}% corrupted", color=colors[c])
plt.axhline(0.0, linestyle="--", color="k", label="Constraint boundary g=0")
plt.xlabel("Iteration")
plt.ylabel("Constraint g(w)")
plt.title("Constraint Behavior — Clean vs Corrupted")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("results/sgm_constraint_vs_corruption.png", dpi=300)
plt.close()
print(" Saved: results/sgm_constraint_vs_corruption.png")

# ==========================================================
# PLOT: 3D visualization for intuition
# ==========================================================
def f_surface(w0, w1):
    return (w0 - 1)**2 + (w1 - 2)**2

w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f_surface(W0, W1)

for c in [0.0, 0.4]:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    surface_noise = 0.5*np.random.randn(*Z.shape) if c > 0 else 0
    ax.plot_surface(W0, W1, Z + surface_noise, cmap="plasma" if c > 0 else "viridis", alpha=0.7)

    traj = results[c]["traj"].numpy()
    ax.plot(traj[:,0], traj[:,1], f_surface(traj[:,0], traj[:,1]), '-o',
            color="#d62728" if c > 0 else "#1f77b4",
            label=f"{int(c*100)}% corrupted" if c > 0 else "Clean data")
    ax.set_xlabel("w₀"); ax.set_ylabel("w₁"); ax.set_zlabel("f(w)")
    ax.set_title(f"SGM Trajectory ({int(c*100)}% Corruption)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/sgm_{int(c*100)}corruption_3d.png", dpi=300)
    plt.close()

print("✅ Saved 3D surfaces for clean and 40% corrupted scenarios.")
print("All done!")
