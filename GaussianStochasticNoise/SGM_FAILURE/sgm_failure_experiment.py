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
def run_sgm(problem, steps=100, eta=0.15, batch_size=64, corruption=0.0, seed=42):
    torch.manual_seed(seed)
    N, d = problem.X.shape

    # Start far from the minimum to make the trajectory visible
    w = torch.tensor([-3.0, 3.0])
    traj, f_hist = [w.clone()], []

    for _ in range(steps):
        idx = torch.randint(0, N, (batch_size,))
        Xb, yb = problem.X[idx], problem.y[idx]
        grads = grad_f_samples(w, Xb, yb)

        # Corrupt a fraction of gradients (simulate outliers)
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

    return {"traj": torch.stack(traj), "f_hist": torch.tensor(f_hist)}

# ==========================================================
# EXPERIMENT CONFIGURATION
# ==========================================================
problem = make_problem()
corruption_levels = [0.0, 0.1, 0.2, 0.5]
results = {c: run_sgm(problem, steps=50, eta=0.15, batch_size=64, corruption=c)
            for c in corruption_levels}

Path("results").mkdir(exist_ok=True)

# ==========================================================
# SURFACE FUNCTION FOR VISUALIZATION
# ==========================================================
def f_surface(w0, w1):
    return (w0 - 1)**2 + (w1 - 2)**2

# Define grid for surface
w0 = np.linspace(-4, 4, 150)
w1 = np.linspace(-2, 5, 150)
W0, W1 = np.meshgrid(w0, w1)
Z = f_surface(W0, W1)

# ==========================================================
# METRIC CALCULATION (for CSV)
# ==========================================================
target = torch.tensor([1.0, 2.0])
rows = []
for c in corruption_levels:
    traj = results[c]["traj"]
    final_w = traj[-1]
    dist_to_opt = torch.norm(final_w - target).item()
    final_loss = results[c]["f_hist"][-1].item()
    rows.append({
        "corruption_level": f"{int(c*100)}%",
        "final_distance_to_optimum": round(dist_to_opt, 4),
        "final_objective": round(final_loss, 4)
    })

df = pd.DataFrame(rows)
df.to_csv("results/sgm_failure_summary.csv", index=False)
print("✅ Saved results/sgm_failure_summary.csv")

# ==========================================================
# PLOT: 3D TRAJECTORY FOR EACH CORRUPTION LEVEL
# ==========================================================
for c in corruption_levels:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    # Add visual surface (noisy when corrupted)
    surface_noise = 0.7 * np.random.randn(*Z.shape) if c > 0 else 0
    ax.plot_surface(W0, W1, Z + surface_noise,
                    cmap="plasma" if c > 0 else "viridis",
                    alpha=0.75, linewidth=0, antialiased=True)

    traj = results[c]["traj"].numpy()
    f_values = f_surface(traj[:,0], traj[:,1])

    # Plot trajectory
    ax.plot(traj[:,0], traj[:,1], f_values, '-o',
            color="#1f77b4" if c == 0 else "#d62728",
            linewidth=2, markersize=4,
            label=f"SGM path ({int(c*100)}% corrupted)")

    # Add arrows to highlight the path direction
    for i in range(0, len(traj)-1, 3):
        ax.quiver(traj[i,0], traj[i,1], f_values[i],
                  traj[i+1,0]-traj[i,0],
                  traj[i+1,1]-traj[i,1],
                  f_values[i+1]-f_values[i],
                  color="black", arrow_length_ratio=0.3, linewidth=0.5)

    # Labels, legend, style
    ax.set_xlabel("w₀"); ax.set_ylabel("w₁"); ax.set_zlabel("f(w)")
    ax.set_title(f"SGM Trajectory with {int(c*100)}% Corrupted Gradients")
    ax.legend()
    ax.view_init(elev=35, azim=-45)
    plt.tight_layout()
    plt.savefig(f"results/sgm_3d_{int(c*100)}corruption.png", dpi=300)
    plt.close()

print(" Saved 3D plots for 0%, 10%, 20%, 50% corruption levels.")


print("\n Experiment Summary:")
print(df.to_string(index=False))
print("\n All results saved in ./results/")
