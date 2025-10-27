import argparse
import random
import torch
import matplotlib.pyplot as plt

# ---- Problem definition: objective and constraint ----
# f(w) = ||w - [1, 0]||^2,  g(w) = w0 + w1 - 1  (feasible when <= 0)
def f(w: torch.Tensor) -> torch.Tensor:
    target = torch.tensor([1.0, 0.0], dtype=w.dtype)
    return torch.sum((w - target) ** 2)

def g(w: torch.Tensor) -> torch.Tensor:
    return w[0] + w[1] - 1.0


# ---- Plain SGM step (no robustness) ----
def sgm_step(w: torch.Tensor, eta: float = 0.1, eps: float = 0.0) -> torch.Tensor:
    w = w.clone().detach().requires_grad_(True)
    if g(w) <= eps:
        grad = torch.autograd.grad(f(w), w)[0]
    else:
        grad = torch.autograd.grad(g(w), w)[0]
    return (w - eta * grad).detach()

# ---- SGM step with gradient corruption (simulate bad workers/noisy grads) ----
def sgm_step_corrupted(
    w: torch.Tensor,
    eta: float = 0.1,
    eps: float = 0.0,
    corruption_prob: float = 0.3,
    noise_scale: float = 3.0,
) -> torch.Tensor:
    w = w.clone().detach().requires_grad_(True)
    # choose regime
    grad = torch.autograd.grad(f(w), w)[0] if g(w) <= eps else torch.autograd.grad(g(w), w)[0]
    # corrupt the gradient with some probability
    if random.random() < corruption_prob:
        grad = grad + noise_scale * torch.randn_like(grad)
    return (w - eta * grad).detach()
