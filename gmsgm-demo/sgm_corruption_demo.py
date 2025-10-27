
import math
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

@dataclass
class Problem:
    X: torch.Tensor
    y: torch.Tensor
    c: torch.Tensor
    eps: float
    d: int

def make_problem(N: int = 512, d: int = 2, noise_std: float = 0.05, seed: int = 0) -> Problem:
    torch.manual_seed(seed)
    X = torch.randn(N, d)
    theta_star = torch.tensor([1.0, 0.0] + [0.0]*(d-2))
    y = (X @ theta_star) + noise_std * torch.randn(N)
    c = torch.zeros(d); c[:2] = torch.tensor([1.0, 1.0])
    return Problem(X=X, y=y, c=c, eps=0.0, d=d)

def f_batch(w: torch.Tensor, Xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    pred = Xb @ w
    return 0.5 * torch.mean((pred - yb)**2)

def grad_f_samples(w: torch.Tensor, Xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    residual = (Xb @ w - yb).unsqueeze(1)
    return residual * Xb

def g_value(w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.dot(c, w) - 1.0

def g_samples(w: torch.Tensor, c: torch.Tensor, B: int, noise_std: float = 0.0) -> torch.Tensor:
    base = g_value(w, c).item()
    if noise_std <= 0.0:
        return torch.full((B,), base)
    return base + noise_std * torch.randn(B)

@dataclass
class CorruptionCfg:
    alpha: float = 0.0
    grad_noise_scale: float = 0.0
    negate_grad: bool = False
    amplify_grad: float = 1.0
    g_measure_noise: float = 0.0
    corrupt_g_measure: bool = False

def corrupt_gradients(grads: torch.Tensor, cfg: CorruptionCfg) -> torch.Tensor:
    B, d = grads.shape
    if cfg.alpha <= 0.0:
        return grads
    num_bad = max(0, int(cfg.alpha * B))
    if num_bad == 0:
        return grads
    idx = torch.randperm(B)[:num_bad]
    g_cor = grads.clone()

    if cfg.amplify_grad != 1.0:
        g_cor[idx] = cfg.amplify_grad * g_cor[idx]

    if cfg.negate_grad:
        g_cor[idx] = -g_cor[idx]

    if cfg.grad_noise_scale > 0.0:
        g_cor[idx] = g_cor[idx] + cfg.grad_noise_scale * torch.randn_like(g_cor[idx])

    return g_cor

def corrupt_g_measure(gs: torch.Tensor, cfg: CorruptionCfg) -> torch.Tensor:
    out = gs.clone()
    if cfg.g_measure_noise > 0.0:
        out = out + cfg.g_measure_noise * torch.randn_like(out)
    if cfg.corrupt_g_measure and cfg.alpha > 0.0:
        B = gs.shape[0]
        num_bad = max(0, int(cfg.alpha * B))
        if num_bad > 0:
            idx = torch.randperm(B)[:num_bad]
            out[idx] = out[idx] + 5.0 * torch.randn_like(out[idx]) + 10.0
    return out

@dataclass
class RunCfg:
    steps: int = 100
    batch_size: int = 64
    eta: float = 0.2
    seed: int = 0

def run_sgm(problem: Problem, run: RunCfg, corruption: CorruptionCfg) -> Dict[str, torch.Tensor]:
    random.seed(run.seed)
    torch.manual_seed(run.seed)
    N, d = problem.X.shape
    w = torch.zeros(d)
    traj = [w.clone()]
    f_hist = []
    g_hist = []

    for t in range(run.steps):
        idx = torch.randint(0, N, (run.batch_size,))
        Xb = problem.X[idx]
        yb = problem.y[idx]

        grads = grad_f_samples(w, Xb, yb)
        grads_cor = corrupt_gradients(grads, corruption)
        grad_mean = grads_cor.mean(0)

        g_meas = g_samples(w, problem.c, B=run.batch_size, noise_std=0.0)
        g_meas_cor = corrupt_g_measure(g_meas, corruption)
        g_predicate = g_meas_cor.mean().item()

        if g_predicate <= problem.eps:
            step_dir = grad_mean
        else:
            step_dir = problem.c

        w = w - run.eta * step_dir
        traj.append(w.clone())

        f_hist.append(f_batch(w, Xb, yb).item())
        g_hist.append(g_value(w, problem.c).item())

    return {"traj": torch.stack(traj), "f_hist": torch.tensor(f_hist), "g_hist": torch.tensor(g_hist)}

# Build problem and configs
problem = make_problem(N=1000, d=2, noise_std=0.05, seed=123)

clean_cfg = CorruptionCfg()

corrupt_cfg = CorruptionCfg(
    alpha=0.4,
    grad_noise_scale=5.0,
    negate_grad=True,
    amplify_grad=3.0,
    g_measure_noise=0.0,
    corrupt_g_measure=False,
)

run_cfg = RunCfg(steps=120, batch_size=64, eta=0.2, seed=7)

out_clean = run_sgm(problem, run_cfg, clean_cfg)
out_bad_grad = run_sgm(problem, run_cfg, corrupt_cfg)

corrupt_predicate_cfg = CorruptionCfg(
    alpha=0.4,
    grad_noise_scale=5.0,
    negate_grad=True,
    amplify_grad=3.0,
    g_measure_noise=0.0,
    corrupt_g_measure=True,
)
out_bad_both = run_sgm(problem, run_cfg, corrupt_predicate_cfg)

# ---- Plot 1: Trajectories in (w0,w1) ----
plt.figure(figsize=(6,6))
xx = np.linspace(-1.5, 2.0, 200)
yy = 1.0 - xx
plt.plot(xx, yy, linestyle='--', linewidth=1.0, label='g(w)=0 boundary')

for label, out in [
    ("Clean SGM", out_clean),
    ("SGM - corrupted grads (40%)", out_bad_grad),
    ("SGM - corrupted grads+predicate (40%)", out_bad_both),
]:
    traj = out["traj"].numpy()
    plt.plot(traj[:,0], traj[:,1], marker='o', label=label)

plt.xlabel('w0'); plt.ylabel('w1'); plt.grid(True); plt.axis('equal')
plt.legend()
plt.title('SGM trajectories: clean vs corrupted')
plt.tight_layout()
plt.show()

# ---- Plot 2: Objective over iterations ----
plt.figure(figsize=(6,4))
plt.plot(out_clean["f_hist"].numpy(), label='Clean SGM')
plt.plot(out_bad_grad["f_hist"].numpy(), label='Corrupted grads (40%)')
plt.plot(out_bad_both["f_hist"].numpy(), label='Corrupted grads+predicate (40%)')
plt.xlabel('Iteration'); plt.ylabel('Mini-batch f(w)')
plt.grid(True); plt.legend()
plt.title('Objective vs iterations')
plt.tight_layout()
plt.show()

# ---- Plot 3: Constraint value over iterations ----
plt.figure(figsize=(6,4))
plt.plot(out_clean["g_hist"].numpy(), label='Clean SGM')
plt.plot(out_bad_grad["g_hist"].numpy(), label='Corrupted grads (40%)')
plt.plot(out_bad_both["g_hist"].numpy(), label='Corrupted grads+predicate (40%)')
plt.axhline(0.0, linestyle='--', linewidth=1.0, label='g=0')
plt.xlabel('Iteration'); plt.ylabel('g(w)')
plt.grid(True); plt.legend()
plt.title('Constraint value vs iterations')
plt.tight_layout()
plt.show()

print("Done. Three figures generated.")
