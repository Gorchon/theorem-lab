import math
import random
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------#
#   PROBLEM DEFINITION
# -----------------------------#
@dataclass
class Problem:
    X: torch.Tensor
    y: torch.Tensor
    c: torch.Tensor
    eps: float
    d: int

def make_problem(N=512, d=2, noise_std=0.05, seed=0) -> Problem:
    torch.manual_seed(seed)
    X = torch.randn(N, d)
    theta_star = torch.tensor([1.0, 0.0] + [0.0]*(d-2))
    y = (X @ theta_star) + noise_std * torch.randn(N)
    c = torch.zeros(d); c[:2] = torch.tensor([1.0, 1.0])
    return Problem(X=X, y=y, c=c, eps=0.0, d=d)

def f_batch(w, Xb, yb):
    return 0.5 * torch.mean((Xb @ w - yb)**2)

def grad_f_samples(w, Xb, yb):
    residual = (Xb @ w - yb).unsqueeze(1)
    return residual * Xb

def g_value(w, c):  # scalar
    return torch.dot(c, w) - 1.0

def g_samples(w, c, B, noise_std=0.0):
    base = g_value(w, c).item()
    return base + noise_std * torch.randn(B) if noise_std > 0.0 else torch.full((B,), base)

# -----------------------------#
#   CORRUPTION MODELS
# -----------------------------#
@dataclass
class CorruptionCfg:
    alpha: float = 0.0
    grad_noise_scale: float = 0.0
    negate_grad: bool = False
    amplify_grad: float = 1.0
    g_measure_noise: float = 0.0
    corrupt_g_measure: bool = False

def corrupt_gradients(grads, cfg: CorruptionCfg):
    B, _ = grads.shape
    if cfg.alpha <= 0.0:
        return grads
    num_bad = int(cfg.alpha * B)
    idx = torch.randperm(B)[:num_bad]
    g_cor = grads.clone()
    if cfg.amplify_grad != 1.0:
        g_cor[idx] = cfg.amplify_grad * g_cor[idx]
    if cfg.negate_grad:
        g_cor[idx] = -g_cor[idx]
    if cfg.grad_noise_scale > 0.0:
        g_cor[idx] += cfg.grad_noise_scale * torch.randn_like(g_cor[idx])
    return g_cor

def corrupt_g_measure(gs, cfg: CorruptionCfg):
    out = gs.clone()
    if cfg.g_measure_noise > 0.0:
        out += cfg.g_measure_noise * torch.randn_like(out)
    if cfg.corrupt_g_measure and cfg.alpha > 0.0:
        num_bad = int(cfg.alpha * gs.shape[0])
        idx = torch.randperm(gs.shape[0])[:num_bad]
        out[idx] += 5.0 * torch.randn_like(out[idx]) + 10.0  # large infeasible spikes
    return out

# -----------------------------#
#   SGM TRAINING LOOP
# -----------------------------#
@dataclass
class RunCfg:
    steps: int = 100
    batch_size: int = 64
    eta: float = 0.2
    seed: int = 0

def run_sgm(problem: Problem, run: RunCfg, corruption: CorruptionCfg):
    torch.manual_seed(run.seed)
    N, d = problem.X.shape
    w = torch.zeros(d)
    traj, f_hist, g_hist = [w.clone()], [], []

    for _ in range(run.steps):
        idx = torch.randint(0, N, (run.batch_size,))
        Xb, yb = problem.X[idx], problem.y[idx]

        grads = grad_f_samples(w, Xb, yb)
        grads = corrupt_gradients(grads, corruption)
        grad_mean = grads.mean(0)

        g_meas = g_samples(w, problem.c, B=run.batch_size, noise_std=0.0)
        g_meas = corrupt_g_measure(g_meas, corruption)
        g_predicate = g_meas.mean().item()

        if g_predicate <= problem.eps:
            step_dir = grad_mean
        else:
            step_dir = problem.c

        w = w - run.eta * step_dir
        traj.append(w.clone())
        f_hist.append(f_batch(w, Xb, yb).item())
        g_hist.append(g_value(w, problem.c).item())

    return {"traj": torch.stack(traj), "f_hist": torch.tensor(f_hist), "g_hist": torch.tensor(g_hist)}

# -----------------------------#
#   EXPERIMENT SETUP
# -----------------------------#
problem = make_problem(N=1000, d=2, noise_std=0.05, seed=123)
run_cfg = RunCfg(steps=120, batch_size=64, eta=0.2, seed=7)

clean_cfg = CorruptionCfg()
corrupt_grad_cfg = CorruptionCfg(alpha=0.4, grad_noise_scale=5.0, negate_grad=True, amplify_grad=3.0)
corrupt_both_cfg = CorruptionCfg(alpha=0.4, grad_noise_scale=5.0, negate_grad=True, amplify_grad=3.0, corrupt_g_measure=True)

out_clean = run_sgm(problem, run_cfg, clean_cfg)
out_bad_grad = run_sgm(problem, run_cfg, corrupt_grad_cfg)
out_bad_both = run_sgm(problem, run_cfg, corrupt_both_cfg)

# -----------------------------#
#   PLOTTING UTILITIES
# -----------------------------#
Path("results").mkdir(exist_ok=True)

def save_fig(name: str):
    plt.tight_layout()
    plt.savefig(f"results/{name}.png", dpi=300)
    print(f" Saved: results/{name}.png")

# ---- 1. Trajectories ----
plt.figure(figsize=(6,6))
xx = np.linspace(-1.5, 2.0, 200)
yy = 1.0 - xx
plt.plot(xx, yy, 'k--', linewidth=1.2, label='Constraint boundary g(w)=0')

colors = {
    "Clean SGM": "#1f77b4",  # blue
    "SGM - corrupted grads (40%)": "#ff7f0e",  # orange
    "SGM - corrupted grads+predicate (40%)": "#d62728",  # red
}

for label, out in [
    ("Clean SGM", out_clean),
    ("SGM - corrupted grads (40%)", out_bad_grad),
    ("SGM - corrupted grads+predicate (40%)", out_bad_both),
]:
    traj = out["traj"].numpy()
    plt.plot(traj[:,0], traj[:,1], marker='o', label=label, color=colors[label], alpha=0.85)

plt.xlabel('w₀'); plt.ylabel('w₁')
plt.grid(True, alpha=0.4)
plt.axis('equal')
plt.legend(loc='best', fontsize=9)
plt.title('SGM trajectories: clean vs corrupted')
save_fig("sgm_trajectory")
plt.close()

# ---- 2. Objective value ----
plt.figure(figsize=(6,4))
for label, out in [
    ("Clean SGM", out_clean),
    ("SGM - corrupted grads (40%)", out_bad_grad),
    ("SGM - corrupted grads+predicate (40%)", out_bad_both),
]:
    plt.plot(out["f_hist"].numpy(), label=label, color=colors[label])

plt.xlabel('Iteration'); plt.ylabel('Mini-batch objective f(w)')
plt.grid(True, alpha=0.4); plt.legend()
plt.title('Objective value vs iterations')
save_fig("sgm_objective")
plt.close()

# ---- 3. Constraint value ----
plt.figure(figsize=(6,4))
for label, out in [
    ("Clean SGM", out_clean),
    ("SGM - corrupted grads (40%)", out_bad_grad),
    ("SGM - corrupted grads+predicate (40%)", out_bad_both),
]:
    plt.plot(out["g_hist"].numpy(), label=label, color=colors[label])
plt.axhline(0.0, linestyle='--', linewidth=1.0, color='k', label='g=0 boundary')
plt.xlabel('Iteration'); plt.ylabel('Constraint g(w)')
plt.grid(True, alpha=0.4); plt.legend()
plt.title('Constraint value vs iterations')
save_fig("sgm_constraint")
plt.close()

print(" Done! Plots saved in ./results/")
