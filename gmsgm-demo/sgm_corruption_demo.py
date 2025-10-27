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