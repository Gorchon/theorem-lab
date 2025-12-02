import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
# 1. General config
# ============================================================
os.makedirs("results", exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 20        
LR = 1e-3
LAMBDA_PENALTY = 4.0  # kept for reference (not used in hard switching)
KAPPA = 0.3
TARGET_CLASS = 7      # Sneaker

# ---- Gradient corruption settings (for SGM / GM-SGM) ----
GRAD_CORR_ELEM_PROB = 0.3      # prob of corrupting each grad element (for sgm_corrupt)
GRAD_CORR_SCALE = 10.0         # magnitude of corruption

N_WORKERS = 11                 # number of "workers" for GM-SGM gradient aggregation
WORKER_CORR_FRAC = 0.4         # fraction of workers corrupted in GM-SGM
WORKER_CORR_SCALE = 10.0       # magnitude of worker corruption

# ---- Robust constraint evaluation settings (Section 3.2) ----
N_CONSTRAINT_WORKERS = 11      # number of workers for constraint evaluations
CONSTRAINT_CORR_FRAC = 0.4     # fraction of corrupted constraint workers
CONSTRAINT_CORR_SCALE = 5.0    # magnitude of corruption on constraint values

# how to aggregate constraint evaluations:
# "median" (paper), "mean", "max", "trimmed", "single"
CONSTRAINT_MODE = "median"

# ============================================================
# 2. Data
# ============================================================
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True
)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# 3. Helpers: model, grads, geometric median, constraints
# ============================================================
def make_model():
    """Small 3-layer MLP classifier."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    return model.to(device)


def flatten_grads(params):
    """Flatten all parameter gradients into a single 1D tensor."""
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


def set_grads_from_flat(params, flat):
    """Set parameter gradients from a flattened gradient tensor."""
    offset = 0
    for p in params:
        numel = p.numel()
        grad_slice = flat[offset:offset + numel].view_as(p)
        if p.grad is None:
            p.grad = grad_slice.clone()
        else:
            p.grad.copy_(grad_slice)
        offset += numel


def geometric_median(points, max_iter=10, eps=1e-5):
    """
    Compute geometric median of a list of 1D tensors using Weiszfeld's algorithm.
    All tensors must have same shape.
    """
    # stack -> (n_points, dim)
    stack = torch.stack(points, dim=0)
    median = stack.mean(dim=0)

    for _ in range(max_iter):
        diffs = stack - median
        distances = torch.norm(diffs, dim=1) + 1e-8  # avoid div by zero
        weights = 1.0 / distances
        weights = weights / weights.sum()
        new_median = (weights.unsqueeze(1) * stack).sum(dim=0)

        if torch.norm(new_median - median) < eps:
            break
        median = new_median

    return median


def apply_gross_corruption_to_grads(params, p_elem=0.3, scale=10.0):
    """
    Grossly corrupt gradient entries for standard SGM baseline.
    Each element has prob p_elem of receiving a big noisy perturbation.
    """
    for p in params:
        if p.grad is None:
            continue
        mask = (torch.rand_like(p.grad) < p_elem)
        noise = scale * torch.randn_like(p.grad)
        p.grad.add_(mask * noise)


def apply_gm_sgm_aggregation(params, n_workers=11,
                             worker_corr_frac=0.4, scale=10.0):
    """
    Simulate n_workers stochastic gradients, corrupt a fraction of them, and
    aggregate using geometric median. This replaces the standard grad.
    """
    base_grad = flatten_grads(params)

    grads = []
    for _ in range(n_workers):
        gk = base_grad.clone()
        # corrupt some workers completely (gross corruption)
        if torch.rand(()) < worker_corr_frac:
            gk = scale * torch.randn_like(gk)
        grads.append(gk)

    gm_grad = geometric_median(grads)
    set_grads_from_flat(params, gm_grad)

def robust_constraint_eval(base_violation,
                           avg_violation=None,
                           n_workers=N_CONSTRAINT_WORKERS,
                           corr_frac=CONSTRAINT_CORR_FRAC,
                           scale=CONSTRAINT_CORR_SCALE,
                           mode=CONSTRAINT_MODE):
    """
    Robust feasibility test (Section 3.2) + custom constraint modes.
    """

    # === NEW: average violation across all classes ===
    if mode == "class_avg":
        return avg_violation

    # === Worker-based modes (median, mean, max, trimmed, single) ===
    g_vals = []
    for _ in range(n_workers):
        gk = base_violation.clone()
        if torch.rand(()) < corr_frac:
            gk = gk + scale * torch.randn_like(gk)
        g_vals.append(gk)

    g_stack = torch.stack(g_vals)

    if mode == "median":
        return torch.median(g_stack)
    elif mode == "mean":
        return torch.mean(g_stack)
    elif mode == "max":
        return torch.max(g_stack)
    elif mode == "trimmed":
        if n_workers <= 2:
            return torch.mean(g_stack)
        sorted_vals, _ = torch.sort(g_stack)
        trimmed = sorted_vals[1:-1]
        return torch.mean(trimmed)
    elif mode == "single":
        return base_violation

    else:
        raise ValueError(f"Unknown constraint mode '{mode}'")

# ============================================================
# 4. Training loop (SGM / SGM+corruption / GM-SGM + robust constraints)
# ============================================================
def train(mode="sgm_clean"):
    """
    mode:
      - 'sgm_clean'      : no gradient corruption
      - 'sgm_corrupt'    : SGM with gross gradient corruption
      - 'gmsgm_corrupt'  : GM-SGM with gross worker corruption

    Constraints are handled with a robust feasibility test and
    hard switching (Section 3.2 + 3.3).
    """
    assert mode in ["sgm_clean", "sgm_corrupt", "gmsgm_corrupt"]

    model = make_model()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_target_hist = []
    penalty_hist = []
    g_hat_hist = []
    per_class_hist = []

    for epoch in range(EPOCHS):
        per_class_loss_epoch = torch.zeros(10, device=device)
        total_batches = 0

        penalty_epoch = 0.0
        g_hat_epoch = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            losses = criterion(logits, labels)

            # Per-class loss on this batch
            per_class_loss = []
            for c in range(10):
                mask = (labels == c)
                if mask.any():
                    per_class_loss.append(losses[mask].mean())
                else:
                    per_class_loss.append(torch.tensor(0.0, device=device))
            per_class_loss = torch.stack(per_class_loss)

            # Objective: target class (Sneaker)
            obj = per_class_loss[TARGET_CLASS]

            # --- Constraint violation values ---
            diff = per_class_loss - KAPPA
            diff[TARGET_CLASS] = 0.0

            base_violation = torch.max(diff)
            avg_violation  = torch.mean(diff)

            # --- Robust feasibility test (Section 3.2, soft version) ---
            g_hat = robust_constraint_eval(
                base_violation,
                avg_violation=avg_violation,
            )

            # === SOFT GM-SGM ===
            soft_penalty = torch.relu(g_hat)
            total_loss = obj + LAMBDA_PENALTY * soft_penalty

            # Backward
            total_loss.backward()

            # Different gradient handling depending on mode
            if mode == "sgm_corrupt":
                apply_gross_corruption_to_grads(
                    model.parameters(),
                    p_elem=GRAD_CORR_ELEM_PROB,
                    scale=GRAD_CORR_SCALE,
                )
            elif mode == "gmsgm_corrupt":
                apply_gm_sgm_aggregation(
                    model.parameters(),
                    n_workers=N_WORKERS,
                    worker_corr_frac=WORKER_CORR_FRAC,
                    scale=WORKER_CORR_SCALE,
                )

            optimizer.step()

            # accumulate stats
            per_class_loss_epoch += per_class_loss.detach()
            penalty_epoch += float(soft_penalty.item())
            g_hat_epoch += float(g_hat.item())
            total_batches += 1

        # epoch averages
        per_class_loss_epoch /= total_batches
        per_class_hist.append(per_class_loss_epoch.cpu())
        loss_target_hist.append(
            float(per_class_loss_epoch[TARGET_CLASS].item())
        )
        penalty_hist.append(penalty_epoch / total_batches)
        g_hat_hist.append(g_hat_epoch / total_batches)

        print(
            f"[{mode}] Epoch {epoch+1}/{EPOCHS} | "
            f"Target(Sneaker) Loss={loss_target_hist[-1]:.3f} | "
            f"Penalty={penalty_hist[-1]:.3f} | "
            f"g_hat={g_hat_hist[-1]:.3f}"
        )

    per_class_hist = torch.stack(per_class_hist)
    return {
        "model": model,
        "loss_target_hist": loss_target_hist,
        "penalty_hist": penalty_hist,
        "g_hat_hist": g_hat_hist,
        "per_class_hist": per_class_hist,
    }


# ============================================================
# 5. Run experiments
# ============================================================
print("\n=== Running SGM with gross gradient corruption ===")
sgm_corrupt_logs = train(mode="sgm_corrupt")

print("\n=== Running GM-SGM with gross gradient corruption ===")
gmsgm_logs = train(mode="gmsgm_corrupt")

# Save logs
torch.save(
    {
        "sgm_corrupt": sgm_corrupt_logs,
        "gmsgm_corrupt": gmsgm_logs,
    },
    "results/fashion_mnpc_gm_sgm_logs.pt",
)

# ============================================================
# 6. Enhanced Plotting
# ============================================================
def plot_per_class_evolution(logs, name):
    """Plot per-class curves with target highlighted (SGM-style visualization)."""

    per_class = logs["per_class_hist"].numpy()
    plt.figure(figsize=(12, 6))

    # plot all non-target classes in gray
    for c in range(10):
        if c == TARGET_CLASS:
            continue
        plt.plot(per_class[:, c], color="gray", alpha=0.4)

    # highlight target class
    plt.plot(
        per_class[:, TARGET_CLASS],
        color="red",
        linewidth=3,
        label=f"Target ({classes[TARGET_CLASS]})",
    )

    plt.axhline(KAPPA, color="black", linestyle="--", label="κ_i threshold")

    plt.title(f"Per-Class Loss Evolution – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/per_class_evolution_{name}.png")
    plt.close()


def plot_target_vs_others_comparison(logA, logB, labelA, labelB):
    """Compare target class loss and penalty between two methods."""
    plt.figure(figsize=(12, 8))

    # Upper subplot: Target class loss
    plt.subplot(2, 1, 1)
    plt.plot(logA["loss_target_hist"], label=f"{labelA} – Target Loss")
    plt.plot(logB["loss_target_hist"], label=f"{labelB} – Target Loss")
    plt.axhline(KAPPA, linestyle="--", color="red", label="κ_i")
    plt.ylabel("Loss")
    plt.title("Target Class Loss Comparison")
    plt.legend()

    # Lower subplot: Penalty
    plt.subplot(2, 1, 2)
    plt.plot(logA["penalty_hist"], label=f"{labelA} – Penalty")
    plt.plot(logB["penalty_hist"], label=f"{labelB} – Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Penalty")
    plt.title("Constraint Violation Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/target_vs_penalty_comparison.png")
    plt.close()


def plot_classwise_last_epoch(logA, logB, labelA, labelB):
    """Better per-class comparison at last epoch."""

    A = logA["per_class_hist"][-1].numpy()
    B = logB["per_class_hist"][-1].numpy()

    x = range(10)
    plt.figure(figsize=(12, 6))

    plt.bar([i - 0.15 for i in x], A, width=0.3, label=labelA)
    plt.bar([i + 0.15 for i in x], B, width=0.3, label=labelB)

    plt.axhline(KAPPA, linestyle="--", color="red", label="κ_i")

    plt.xticks(x, classes, rotation=40)
    plt.ylabel("Loss (Last Epoch)")
    plt.title("Per-Class Last Epoch Loss Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/last_epoch_bar_comparison.png")
    plt.close()


def plot_deviation_from_kappa(logs, name):
    """Plot (Loss_i - kappa) across epochs for all classes."""
    per_class = logs["per_class_hist"].numpy()
    deviation = per_class - KAPPA  # If >0 → violating

    plt.figure(figsize=(12, 6))
    for c in range(10):
        if c == TARGET_CLASS:
            continue
        plt.plot(deviation[:, c], alpha=0.5)

    plt.axhline(0, color="black", linestyle="--", label="κ boundary")
    plt.plot(
        deviation[:, TARGET_CLASS],
        color="red",
        linewidth=3,
        label=f"Target ({classes[TARGET_CLASS]})",
    )

    plt.title(f"Loss – κ Deviation per Class – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss_i - κ_i")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/deviation_from_kappa_{name}.png")
    plt.close()


def plot_last_epoch_box(logA, logB, labelA, labelB):
    """Boxplot of per-class losses at last epoch."""
    A = logA["per_class_hist"][-1].numpy()
    B = logB["per_class_hist"][-1].numpy()

    plt.figure(figsize=(8, 6))
    plt.boxplot([A, B], labels=[labelA, labelB])
    plt.axhline(KAPPA, linestyle="--", color="red", label="κ")
    plt.ylabel("Last Epoch Class Loss")
    plt.title("Class Loss Distribution at Last Epoch")
    plt.tight_layout()
    plt.savefig("results/last_epoch_boxplot.png")
    plt.close()


# ============================================================
# Generate all enhanced comparison plots
# ============================================================
print("Generating enhanced comparison plots...")

plot_per_class_evolution(sgm_corrupt_logs, "SGM-corrupt")
plot_per_class_evolution(gmsgm_logs, "GM-SGM-corrupt")

plot_target_vs_others_comparison(
    sgm_corrupt_logs,
    gmsgm_logs,
    labelA="SGM-corrupt",
    labelB="GM-SGM-corrupt",
)

plot_classwise_last_epoch(
    sgm_corrupt_logs,
    gmsgm_logs,
    labelA="SGM-corrupt",
    labelB="GM-SGM-corrupt",
)

plot_deviation_from_kappa(sgm_corrupt_logs, "SGM-corrupt")
plot_deviation_from_kappa(gmsgm_logs, "GM-SGM-corrupt")

plot_last_epoch_box(
    sgm_corrupt_logs,
    gmsgm_logs,
    labelA="SGM-corrupt",
    labelB="GM-SGM-corrupt",
)

print("All enhanced plots saved into ./results/")
