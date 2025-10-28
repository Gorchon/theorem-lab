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
# 2. Helper: Geometric Median (Weiszfeld algorithm)
# ====================================================

def geometric_median(vectors, eps=1e-6, max_iter=100):
    """Compute geometric median of a list of vectors."""
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
# 3. SGM step (original)
# ====================================================

def sgm_step(w, eta=0.1, eps=0.0, corruption=False):
    """Single SGM step with optional corruption."""
    g_val = g_value(w)
    grad = grad_f(w) if g_val <= eps else grad_g()

    # Add gradient corruption (simulate noisy gradients or outliers)
    if corruption:
        noise = 2.0 * torch.randn_like(grad)
        if np.random.rand() < 0.3:
            grad = grad + noise * 3.0  # large random spike

    return w - eta * grad

# ====================================================
# 4. GM-SGM step (robust geometric median aggregation)
# ====================================================

def gm_sgm_step(w, eta=0.1, eps=0.0, corruption=True, batch_size=10):
    """Single GM-SGM step using geometric median aggregation."""
    # Generate noisy batch of gradients (simulating stochastic gradients)
    grads = []
    for _ in range(batch_size):
        g_val = g_value(w)
        grad = grad_f(w) if g_val <= eps else grad_g()

        # Add corruption to ~30% of gradients
        if corruption and np.random.rand() < 0.3:
            noise = 2.0 * torch.randn_like(grad)
            grad = grad + noise * 3.0
        grads.append(grad)

    # Robust aggregation: geometric median instead of mean
    gm_grad = geometric_median(grads)
    return w - eta * gm_grad

# ====================================================
# 5. Run methods
# ====================================================

def run_method(method="sgm", clean=True, steps=40, eta=0.15):
    """Run SGM or GM-SGM trajectory."""
    w = torch.tensor([-2.0, 3.0])
    traj = [w.clone()]
    losses = [f(w[0], w[1])]
    gvals = [g_value(w)]

    for _ in range(steps):
        if method == "gm-sgm":
            w = gm_sgm_step(w, eta=eta, corruption=(not clean))
        else:
            w = sgm_step(w, eta=eta, corruption=(not clean))
        traj.append(w.clone())
        losses.append(f(w[0], w[1]))
        gvals.append(g_value(w))

    return torch.stack(traj), np.array(losses), np.array(gvals)

# Run all three cases
traj_clean, loss_clean, g_clean = run_method("sgm", clean=True)
traj_noisy, loss_noisy, g_noisy = run_method("sgm", clean=False)
traj_gm, loss_gm, g_gm = run_method("gm-sgm", clean=False)

# ====================================================
# 6. Create visualization folder
# ====================================================

Path("results").mkdir(exist_ok=True)

# ====================================================
# 7. 3D Plots
# ====================================================

w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

# ----- All trajectories in one plot -----
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W0, W1, Z, cmap='viridis', alpha=0.6)
ax.plot(traj_clean[:,0], traj_clean[:,1], f(traj_clean[:,0], traj_clean[:,1]), '-o', color='blue', label='Clean SGM')
ax.plot(traj_noisy[:,0], traj_noisy[:,1], f(traj_noisy[:,0], traj_noisy[:,1]), '-o', color='red', label='Corrupted SGM')
ax.plot(traj_gm[:,0], traj_gm[:,1], f(traj_gm[:,0], traj_gm[:,1]), '-o', color='green', label='GM-SGM (Robust)')
ax.set_xlabel('w₀'); ax.set_ylabel('w₁'); ax.set_zlabel('f(w)')
ax.set_title('3D Trajectories — Clean, Corrupted, and GM-SGM')
ax.legend()
plt.tight_layout()
plt.savefig("results/sgm_gmsgm_3d.png", dpi=300)
plt.close()

# ====================================================
# 8. 2D Trajectory Plot
# ====================================================

plt.figure(figsize=(7,5))
plt.plot(traj_clean[:,0], traj_clean[:,1], '-o', label='Clean SGM', color='blue')
plt.plot(traj_noisy[:,0], traj_noisy[:,1], '-o', label='Corrupted SGM', color='red', alpha=0.7)
plt.plot(traj_gm[:,0], traj_gm[:,1], '-o', label='GM-SGM (Robust)', color='green', alpha=0.9)
plt.scatter(1, 2, marker='*', color='black', s=150, label='True minimum (1,2)')
plt.xlabel("w₀"); plt.ylabel("w₁")
plt.title("2D Trajectories — SGM vs GM-SGM")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_gmsgm_2d_trajectories.png", dpi=300)
plt.close()

# ====================================================
# 9. Loss evolution plot
# ====================================================

plt.figure(figsize=(7,5))
plt.plot(loss_clean, '-o', color='blue', label='Clean SGM')
plt.plot(loss_noisy, '-o', color='red', alpha=0.7, label='Corrupted SGM')
plt.plot(loss_gm, '-o', color='green', alpha=0.9, label='GM-SGM (Robust)')
plt.xlabel("Iteration t"); plt.ylabel("f(wₜ)")
plt.title("Loss Evolution — SGM vs GM-SGM (with noisy gradients)")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_gmsgm_loss_evolution.png", dpi=300)
plt.close()

# ====================================================
# 10. Save CSV with trajectories
# ====================================================

def to_df(traj, loss, gvals, corrupted, label):
    return pd.DataFrame({
        "step": np.arange(len(traj)),
        "w0": traj[:,0].numpy(),
        "w1": traj[:,1].numpy(),
        "f(w)": loss,
        "g(w)": gvals,
        "corrupted": corrupted,
        "method": label
    })

df = pd.concat([
    to_df(traj_clean, loss_clean, g_clean, False, "SGM_clean"),
    to_df(traj_noisy, loss_noisy, g_noisy, True, "SGM_corrupted"),
    to_df(traj_gm, loss_gm, g_gm, True, "GM_SGM_corrupted")
], ignore_index=True)

df.to_csv("results/sgm_gmsgm_trajectories.csv", index=False)

# ====================================================
# 11. Summary
# ====================================================

print(" Done. Results saved in 'results/' folder:")
print(" - sgm_gmsgm_3d.png")
print(" - sgm_gmsgm_2d_trajectories.png")
print(" - sgm_gmsgm_loss_evolution.png")
print(" - sgm_gmsgm_trajectories.csv")
