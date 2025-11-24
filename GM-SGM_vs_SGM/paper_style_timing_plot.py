import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 1. Setup: Objective and Constraint
# ============================================================


def f(w0, w1):
    """Convex quadratic objective."""
    return (w0 - 1) ** 2 + (w1 - 2) ** 2


def grad_f(w: torch.Tensor) -> torch.Tensor:
    """
    Gradient of the objective f.
    For our toy example: ∇f(w) = 2 (w - [1, 2]).
    """
    return 2 * (w - torch.tensor([1.0, 2.0]))


def g_value(w: torch.Tensor) -> torch.Tensor:
    """
    Scalar constraint value g(w) = w0 + w1 - 2.
    """
    return w[0] + w[1] - 2.0


def grad_g() -> torch.Tensor:
    """
    Gradient of the constraint g.
    For our toy example: ∇g(w) = [1, 1].
    """
    return torch.tensor([1.0, 1.0])


# ============================================================
# 2. Helper: Geometric Median (Weiszfeld)
# ============================================================


def geometric_median(vectors, eps: float = 1e-6, max_iter: int = 100) -> torch.Tensor:
    """
    Weiszfeld's algorithm for the geometric median.

    Given vectors {v_i} in R^d, we compute:
        y* = argmin_y sum_i ||v_i - y||.

    Parameters
    ----------
    vectors : list[torch.Tensor] or torch.Tensor of shape (n, d)
    eps : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    torch.Tensor of shape (d,)
    """
    v = torch.stack(vectors)  # (n, d)
    guess = v.mean(dim=0)
    for _ in range(max_iter):
        distances = torch.norm(v - guess, dim=1).clamp_min(eps)
        weights = 1.0 / distances
        new_guess = (v * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        if torch.norm(new_guess - guess) < eps:
            break
        guess = new_guess
    return guess


# ============================================================
# 3. Gross Corruption Model
# ============================================================


def gross_corruption(grad: torch.Tensor, psi: float = 0.3, magnitude: float = 8.0) -> torch.Tensor:
    """
    Apply the Gross Corruption Model (GCM).

    With probability psi, replace the true gradient with an arbitrary (corrupted) vector.

    Parameters
    ----------
    grad : torch.Tensor (d,)
        True gradient.
    psi : float
        Probability that the gradient is corrupted.
    magnitude : float
        Norm of the adversarial gradient when corrupted.

    Returns
    -------
    torch.Tensor
        Possibly corrupted gradient.
    """
    if np.random.rand() < psi:
        corrupt_direction = torch.randn_like(grad)
        corrupt_direction = corrupt_direction / torch.norm(corrupt_direction)
        grad = magnitude * corrupt_direction
    return grad


# ============================================================
# 4. Single SGM step
# ============================================================


def sgm_step(
    w: torch.Tensor,
    eta: float = 0.1,
    eps: float = 0.0,
    corruption: bool = False,
    psi: float = 0.4,
    magnitude: float = 8.0,
) -> torch.Tensor:
    """
    Single SGM step with optional Gross Corruption on the gradient.

    This implements the continuous-time SGM idea:
        x_{t+1} = x_t - η * (∇f(x_t) or ∇g(x_t))

    depending on whether the constraint is active or not.
    """
    g_val = g_value(w)
    grad = grad_f(w) if g_val <= eps else grad_g()

    if corruption:
        grad = gross_corruption(grad, psi=psi, magnitude=magnitude)

    return w - eta * grad


# ============================================================
# 5. Standard GM-SGM step (no block, no memory)
# ============================================================


def gm_sgm_step(
    w: torch.Tensor,
    eta: float = 0.1,
    eps: float = 0.0,
    corruption: bool = True,
    batch_size: int = 10,
    psi: float = 0.4,
    magnitude: float = 8.0,
) -> torch.Tensor:
    """
    Single GM-SGM step with geometric median aggregation under Gross Corruption.

    We simulate a batch of size 'batch_size' of stochastic gradients, possibly corrupted,
    and aggregate them using the geometric median:
        g̃_t = GM({g_i^t}_{i=1}^b)
        x_{t+1} = x_t - η g̃_t
    """
    grads = []
    for _ in range(batch_size):
        g_val = g_value(w)
        grad = grad_f(w) if g_val <= eps else grad_g()
        if corruption:
            grad = gross_corruption(grad, psi=psi, magnitude=magnitude)
        grads.append(grad)
    gm_grad = geometric_median(grads)
    return w - eta * gm_grad


# ============================================================
# 6. Block Coordinate Selection and Memory Mechanism (BGMD-style)
# ============================================================


def block_coordinate_selection(G: torch.Tensor, beta: float) -> (torch.Tensor, torch.Tensor):
    """
    Block coordinate selection operator C_k(·) from the BGMD paper.

    Given G ∈ R^{b×d} (each row is a stochastic gradient), we compute coordinate
    importances:
        s_j = ||G[:, j]||_2^2  for j = 1,...,d

    Then we select k = max(1, floor(beta * d)) coordinates with largest s_j, and
    zero-out the rest.

    Parameters
    ----------
    G : torch.Tensor, shape (b, d)
        Matrix of gradients.
    beta : float in (0, 1]
        Fraction of coordinates to keep. k = beta * d.

    Returns
    -------
    G_block : torch.Tensor, shape (b, d)
        Matrix where only the top-k coordinates are kept (others are 0).
    mask : torch.Tensor, shape (d,)
        Boolean mask indicating which coordinates are selected.
    """
    b, d = G.shape
    k = max(1, int(beta * d))  # ensure at least 1 coordinate
    # Importance of each coordinate: squared ℓ2 norm across the batch
    importances = (G ** 2).sum(dim=0)  # (d,)
    # Indices of top-k coordinates
    topk_vals, topk_idx = torch.topk(importances, k=k, largest=True)
    mask = torch.zeros(d, dtype=torch.bool)
    mask[topk_idx] = True
    G_block = G.clone()
    G_block[:, ~mask] = 0.0
    return G_block, mask


def gm_sgm_block_memory_step(
    w: torch.Tensor,
    m: torch.Tensor,
    eta: float = 0.1,
    beta: float = 0.5,
    eps: float = 0.0,
    corruption: bool = True,
    batch_size: int = 10,
    psi: float = 0.4,
    magnitude: float = 8.0,
) -> (torch.Tensor, torch.Tensor):
    """
    Single step of GM-SGM with Block Coordinate Selection + Memory Mechanism (BGMD-style).

    This mirrors Algorithm 1 (BGMD) conceptually, adapted to our low-dimensional toy problem.

    Notation (matching the paper loosely):

        • G_t ∈ R^{b×d}  : matrix of stochastic gradients (each row g_i^t).
        • m_t ∈ R^d      : memory vector at iteration t.
        • C_k(·)         : block coordinate selection operator (keeps k = β d coordinates).
        • Δ_t            : selected block, Δ_t = C_k(G_t + m_t).
        • M_{t+1}        : residual matrix, M_{t+1} = (G_t + m_t) - Δ_t.
        • m_{t+1}        : updated memory, average residual across samples.
        • g̃_t           : robust aggregate gradient, geometric median of rows of Δ_t.

    Update rule:
        1. Sample gradients, optionally corrupted → G_t.
        2. Add memory: G̃_t = G_t + m_t.
        3. Block selection: Δ_t = C_k(G̃_t).
        4. Residuals: M_{t+1} = G̃_t - Δ_t.
        5. Memory update: m_{t+1} = (1/b) Σ_i M_{t+1}[i, :].
        6. Robust aggregation: g̃_t = GM({Δ_t[i, :]}_i).
        7. Parameter update: x_{t+1} = x_t - η g̃_t.

    Parameters
    ----------
    w : torch.Tensor, shape (d,)
        Current iterate x_t.
    m : torch.Tensor, shape (d,)
        Current memory vector m_t.
    eta : float
        Step size η.
    beta : float
        Block fraction β ∈ (0,1]; k = β d coordinates are kept.
    eps, corruption, batch_size, psi, magnitude:
        Same roles as in gm_sgm_step.

    Returns
    -------
    w_new : torch.Tensor
        Updated parameter.
    m_new : torch.Tensor
        Updated memory vector.
    """
    grads = []
    for _ in range(batch_size):
        g_val = g_value(w)
        grad = grad_f(w) if g_val <= eps else grad_g()
        if corruption:
            grad = gross_corruption(grad, psi=psi, magnitude=magnitude)
        grads.append(grad)

    G = torch.stack(grads)  # (b, d)

    # 1) Add memory to each gradient (broadcasted)
    G_tilde = G + m  # shape (b, d)

    # 2) Block coordinate selection: keep only k = β d coordinates
    Delta, mask = block_coordinate_selection(G_tilde, beta=beta)

    # 3) Residuals (error of dimensionality reduction)
    residuals = G_tilde - Delta  # (b, d)

    # 4) Update memory as average residual across batch
    m_new = residuals.mean(dim=0)  # (d,)

    # 5) Robust aggregation (geometric median) on the selected block
    #    (Rows of Delta; unselected coordinates are exactly zero.)
    gm_grad = geometric_median([Delta[i, :] for i in range(Delta.shape[0])])

    # 6) Parameter update
    w_new = w - eta * gm_grad

    return w_new, m_new


# ============================================================
# 7. Run a method (SGM, GM-SGM, GM-SGM + Block + Memory)
# ============================================================


def run_method(
    method: str = "sgm",
    clean: bool = True,
    steps: int = 40,
    eta: float = 0.15,
    psi: float = 0.4,
    magnitude: float = 8.0,
    batch_size: int = 10,
    beta: float = 1.0,
):
    """
    Run one of the methods for a fixed number of steps.

    Parameters
    ----------
    method : {"sgm", "gm-sgm", "gm-sgm-block-mem"}
    clean : bool
        If True, no corruption is applied.
    steps : int
        Number of iterations.
    eta : float
        Step size.
    psi, magnitude, batch_size, beta :
        Corruption parameters and block size for block-mem method.

    Returns
    -------
    traj : torch.Tensor, shape (steps+1, 2)
        Trajectory of iterates.
    losses : np.ndarray
        f(w_t) values.
    gvals : np.ndarray
        g(w_t) values.
    """
    w = torch.tensor([-2.0, 3.0], dtype=torch.float32)
    m = torch.zeros_like(w)  # memory (used only for block-mem method)

    traj = [w.clone()]
    losses = [f(w[0], w[1])]
    gvals = [g_value(w)]

    for _ in range(steps):
        if method == "gm-sgm":
            w = gm_sgm_step(
                w,
                eta=eta,
                eps=0.0,
                corruption=(not clean),
                batch_size=batch_size,
                psi=psi,
                magnitude=magnitude,
            )
        elif method == "gm-sgm-block-mem":
            w, m = gm_sgm_block_memory_step(
                w,
                m,
                eta=eta,
                beta=beta,
                eps=0.0,
                corruption=(not clean),
                batch_size=batch_size,
                psi=psi,
                magnitude=magnitude,
            )
        else:  # "sgm"
            w = sgm_step(
                w,
                eta=eta,
                eps=0.0,
                corruption=(not clean),
                psi=psi,
                magnitude=magnitude,
            )

        traj.append(w.clone())
        losses.append(f(w[0], w[1]))
        gvals.append(g_value(w))

    return torch.stack(traj), np.array(losses), np.array(gvals)


# ============================================================
# 8. Run baseline methods and BGMD-style variants
# ============================================================

psi = 0.4          # fraction of corrupted gradients
magnitude = 8.0    # corruption strength
steps = 40
eta = 0.15
batch_size = 10

# Baselines
traj_clean, loss_clean, g_clean = run_method("sgm", clean=True, steps=steps, eta=eta)
traj_corrupted, loss_corrupted, g_corrupted = run_method(
    "sgm", clean=False, steps=steps, eta=eta, psi=psi, magnitude=magnitude
)
traj_gm, loss_gm, g_gm = run_method(
    "gm-sgm", clean=False, steps=steps, eta=eta, psi=psi, magnitude=magnitude, batch_size=batch_size
)

# GM-SGM + Block + Memory for different β
betas = [1.0, 0.6, 0.3]  # In 2D, β=1.0 ⇒ k=2; β<0.5 ⇒ k=1 (top-1 coordinate)
block_results = []
for beta in betas:
    traj_b, loss_b, g_b = run_method(
        "gm-sgm-block-mem",
        clean=False,
        steps=steps,
        eta=eta,
        psi=psi,
        magnitude=magnitude,
        batch_size=batch_size,
        beta=beta,
    )
    block_results.append((beta, traj_b, loss_b, g_b))

# ============================================================
# 9. Create folder
# ============================================================

Path("results").mkdir(exist_ok=True)

# ============================================================
# 10. Prepare mesh for plotting
# ============================================================

w0 = np.linspace(-3, 4, 100)
w1 = np.linspace(-2, 5, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = f(W0, W1)

# ============================================================
# 11. 3D Plots for baselines
# ============================================================


def plot_3d(traj, title, filename, color, cmap):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(W0, W1, Z, cmap=cmap, alpha=0.8, rstride=1, cstride=1, edgecolor="none")
    ax.plot(traj[:, 0], traj[:, 1], f(traj[:, 0], traj[:, 1]), "-o", color=color, linewidth=2)
    ax.scatter(1, 2, f(1, 2), c="black", s=60, marker="*", label="True Minimum")
    ax.set_xlabel("w₀")
    ax.set_ylabel("w₁")
    ax.set_zlabel("f(w)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=300)
    plt.close()


plot_3d(traj_clean.numpy(), "Clean SGM — Stable Convergence", "sgm_clean_3d.png", "blue", "viridis")
plot_3d(
    traj_corrupted.numpy(),
    "SGM under Gross Corruption — Divergent Behavior",
    "sgm_gross_corruption_3d.png",
    "red",
    "plasma",
)
plot_3d(
    traj_gm.numpy(),
    "GM-SGM — Robust Convergence under Gross Corruption",
    "gm_sgm_gross_corruption_3d.png",
    "green",
    "cividis",
)

# Optionally, 3D plots for block+memory variants (one example for β=0.3)
for beta, traj_b, _, _ in block_results:
    plot_3d(
        traj_b.numpy(),
        f"GM-SGM + Block+Memory (β={beta:.2f})",
        f"gm_sgm_blockmem_beta_{beta:.2f}_3d.png",
        "purple",
        "magma",
    )

# ============================================================
# 12. 2D Trajectory Plot (Baseline Comparison)
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(
    traj_clean[:, 0],
    traj_clean[:, 1],
    "-o",
    label="Clean SGM",
    color="blue",
    linewidth=2,
    markersize=5,
)
plt.plot(
    traj_corrupted[:, 0],
    traj_corrupted[:, 1],
    "-o",
    label="Gross Corrupted SGM",
    color="red",
    alpha=0.7,
    linewidth=2,
    markersize=5,
)
plt.plot(
    traj_gm[:, 0],
    traj_gm[:, 1],
    "-o",
    label="GM-SGM (Robust)",
    color="green",
    alpha=0.9,
    linewidth=2,
    markersize=5,
)
plt.scatter(1, 2, marker="*", color="black", s=150, label="True Minimum (1,2)")
plt.xlabel("w₀")
plt.ylabel("w₁")
plt.title("2D Trajectories — SGM vs Corrupted SGM vs GM-SGM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/sgm_gmsgm_gross_comparison_2d.png", dpi=300)
plt.close()

# ============================================================
# 13. 2D Trajectory Plot (GM-SGM vs Block+Memory for different β)
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot(
    traj_gm[:, 0],
    traj_gm[:, 1],
    "-o",
    label="GM-SGM (full dim)",
    color="green",
    linewidth=2,
    markersize=5,
)

colors = ["purple", "orange", "brown"]
for (beta, traj_b, _, _), col in zip(block_results, colors):
    plt.plot(
        traj_b[:, 0],
        traj_b[:, 1],
        "-o",
        label=f"GM-SGM + Block+Mem (β={beta:.2f})",
        color=col,
        linewidth=2,
        markersize=5,
        alpha=0.9,
    )

plt.scatter(1, 2, marker="*", color="black", s=150, label="True Minimum (1,2)")
plt.xlabel("w₀")
plt.ylabel("w₁")
plt.title("2D Trajectories — GM-SGM vs Block+Memory (various β)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/gmsgm_blockmem_beta_comparison_2d.png", dpi=300)
plt.close()

# ============================================================
# 14. Save CSV with all trajectories (including block+memory)
# ============================================================


def to_df(traj, loss, gvals, corrupted, method, beta_value=None):
    beta_col = np.full(len(traj), np.nan if beta_value is None else beta_value)
    return pd.DataFrame(
        {
            "step": np.arange(len(traj)),
            "w0": traj[:, 0].numpy(),
            "w1": traj[:, 1].numpy(),
            "f(w)": loss,
            "g(w)": gvals,
            "corrupted": corrupted,
            "method": method,
            "beta": beta_col,
        }
    )


dfs = [
    to_df(traj_clean, loss_clean, g_clean, False, "SGM_clean"),
    to_df(traj_corrupted, loss_corrupted, g_corrupted, True, "SGM_gross_corrupted"),
    to_df(traj_gm, loss_gm, g_gm, True, "GM_SGM_gross_corrupted"),
]

for beta, traj_b, loss_b, g_b in block_results:
    dfs.append(
        to_df(
            traj_b,
            loss_b,
            g_b,
            True,
            f"GM_SGM_BlockMem_beta_{beta:.2f}",
            beta_value=beta,
        )
    )

df = pd.concat(dfs, ignore_index=True)
df.to_csv("results/sgm_gmsgm_blockmem_gross_corruption_trajectories.csv", index=False)

# ============================================================
# 15. Summary
# ============================================================

print("Done. Results saved in 'results/' folder:")
print(" - sgm_clean_3d.png")
print(" - sgm_gross_corruption_3d.png")
print(" - gm_sgm_gross_corruption_3d.png")
print(" - gm_sgm_blockmem_beta_XX_3d.png (for each β)")
print(" - sgm_gmsgm_gross_comparison_2d.png")
print(" - gmsgm_blockmem_beta_comparison_2d.png")
print(" - sgm_gmsgm_blockmem_gross_corruption_trajectories.csv")
