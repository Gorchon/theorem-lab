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
# 2. Gross Corruption Model + SGM step
# ====================================================

def sgm_step(w, eta=0.1, eps=0.0, corruption=False, psi=0.3, b=20):
    """
    Single SGM step using the Gross Corruption Model.
    ψ (psi): fraction of corrupted gradients (0 ≤ ψ < 0.5)
    b: mini-batch size.
    """
    g_val = g_value(w)
    true_grad = grad_f(w) if g_val <= eps else grad_g()

    grads = []
    for i in range(b):
        if corruption and np.random.rand() < psi:
            # Corrupted gradient: arbitrary large or flipped vector
            bad_grad = torch.randn_like(true_grad) * 5.0
            if np.random.rand() < 0.5:
                bad_grad = -bad_grad  # opposite direction
            grads.append(bad_grad)
        else:
            # Slightly noisy correct gradient
            grads.append(true_grad + 0.1 * torch.randn_like(true_grad))

    grads = torch.stack(grads)
    grad = grads.mean(dim=0)  # mean aggregation (SGM baseline)

    return w - eta * grad

# ====================================================
# 3. Run SGM trajectories
# ====================================================

def run_sgm(clean=True, steps=40, eta=0.15, psi=0.3):
    """Run SGM trajectory under given corruption level."""
    w = torch.tensor([-2.0, 3.0])
    traj = [w.clone()]
    losses = [f(w[0], w[1])]
    gvals = [g_value(w)]
    for t in range(steps):
        w = sgm_step(w, eta=eta, corruption=(not clean), psi=psi)
        traj.append(w.clone())
        losses.append(f(w[0], w[1]))
        gvals.append(g_value(w))
    return torch.stack(traj), np.array(losses), np.array(gvals)

# ====================================================
# 4. Run experiments
# ====================================================

# Different corruption levels
psi_levels = [0.0, 0.1, 0.3, 0.4]

results = []
for psi in psi_levels:
    traj, loss, gval = run_sgm(clean=(psi == 0.0), psi=psi)
    results.append((psi, traj, loss, gval))

# ====================================================
# 5. Visualization setup
# ====================================================

Path("results").mkdir(exist_ok=True)

# Landscape
w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

# ====================================================
# 6. 3D plots
# ====================================================

for psi, traj, loss, gval in results:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    cmap = 'viridis' if psi == 0 else 'plasma'
    ax.plot_surface(W0, W1, Z, cmap=cmap, alpha=0.7)

    traj_np = traj.numpy()
    color = 'blue' if psi == 0 else 'red'
    label = f"ψ = {psi*100:.0f}% corruption" if psi > 0 else "Clean data"
    ax.plot(traj_np[:,0], traj_np[:,1], f(traj_np[:,0], traj_np[:,1]),
            '-o', color=color, label=label)
    ax.set_xlabel('w₀'); ax.set_ylabel('w₁'); ax.set_zlabel('f(w)')
    ax.set_title(f'SGM under Gross Corruption Model (ψ={psi:.2f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/sgm_3d_psi_{int(psi*100)}.png", dpi=300)
    plt.close()

# ====================================================
# 7. 2D Trajectories
# ====================================================

plt.figure(figsize=(7,5))
for psi, traj, loss, gval in results:
    color = 'blue' if psi == 0 else plt.cm.plasma(psi*2)
    plt.plot(traj[:,0], traj[:,1], '-o', label=f'ψ={psi:.2f}', color=color)
plt.scatter(1, 2, marker='*', color='green', s=150, label='True minimum (1,2)')
plt.xlabel("w₀"); plt.ylabel("w₁")
plt.title("2D Trajectories — Effect of Increasing Corruption")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_2d_corruption_levels.png", dpi=300)
plt.close()

# ====================================================
# 8. Loss evolution
# ====================================================

plt.figure(figsize=(7,5))
for psi, traj, loss, gval in results:
    color = 'blue' if psi == 0 else plt.cm.plasma(psi*2)
    plt.plot(loss, '-o', color=color, label=f'ψ={psi:.2f}')
plt.xlabel("Iteration t"); plt.ylabel("f(wₜ)")
plt.title("Loss evolution — Gross Corruption Model")
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_loss_vs_corruption.png", dpi=300)
plt.close()

# ====================================================
# 9. Save CSV with all trajectories
# ====================================================

rows = []
for psi, traj, loss, gval in results:
    for t in range(len(traj)):
        rows.append({
            "psi": psi,
            "step": t,
            "w0": traj[t,0].item(),
            "w1": traj[t,1].item(),
            "f(w)": loss[t],
            "g(w)": gval[t]
        })

df = pd.DataFrame(rows)
df.to_csv("results/sgm_gross_corruption_results.csv", index=False)

# ====================================================
# 10. Summary
# ====================================================

print(" Done! Results saved in 'results/' folder:")
print(" - sgm_3d_psi_0.png, sgm_3d_psi_10.png, sgm_3d_psi_30.png, sgm_3d_psi_40.png")
print(" - sgm_2d_corruption_levels.png")
print(" - sgm_loss_vs_corruption.png")
print(" - sgm_gross_corruption_results.csv")
print("Each ψ = corruption fraction (0 ≤ ψ < 0.5).")
