import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
# 1. General config
# ============================================================
# ---- Switching mode flag ----
HARD_SWITCHING = True  # True = hard switching, False = soft switching

# ---- Results directory based on mode ----
RESULTS_DIR = "results_hard_switching" if HARD_SWITCHING else "results_soft_switching"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3

# This will scale the constraint direction (λ in front of ∇g)
LAMBDA_PENALTY = 4.0

# κ in the constraints: require other-class losses ≤ κ
KAPPA = 0.3
TARGET_CLASS = 7  # Sneaker

# ---- Switching hyperparameters (paper-style) ----
EPS_SWITCH = 0.0       # ε in the paper (threshold on g_hat)
BETA_SWITCH = 5.0      # β in the soft switching rule

# ---- Gradient corruption settings ----
GRAD_CORR_ELEM_PROB = 0.3
GRAD_CORR_SCALE = 10.0

N_WORKERS = 11
WORKER_CORR_FRAC = 0.4
WORKER_CORR_SCALE = 10.0

# ---- Robust constraint evaluation settings ----
N_CONSTRAINT_WORKERS = 11
CONSTRAINT_CORR_FRAC = 0.4
CONSTRAINT_CORR_SCALE = 5.0

# modes: "median", "mean", "max", "trimmed", "single", "class_avg"
# For your current experiment, you can keep "class_avg" (no corruption)
CONSTRAINT_MODE = "mean"

# ============================================================
# 2. Data
# ============================================================
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
print(f"HARD_SWITCHING = {HARD_SWITCHING}")
print(f"Results will be saved in: {RESULTS_DIR}")

# ============================================================
# 3. Helpers
# ============================================================
def make_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model.to(device)


def flatten_grads(params):
    grads = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p).view(-1))
        else:
            grads.append(p.grad.view(-1))
    return torch.cat(grads)


def set_grads_from_flat(params, flat):
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
    stack = torch.stack(points, dim=0)
    median = stack.mean(dim=0)

    for _ in range(max_iter):
        diffs = stack - median
        distances = torch.norm(diffs, dim=1) + 1e-8
        weights = 1.0 / distances
        weights /= weights.sum()
        new_median = (weights.unsqueeze(1) * stack).sum(dim=0)
        if torch.norm(new_median - median) < eps:
            break
        median = new_median

    return median


def apply_gross_corruption_to_grads(params, p_elem=0.3, scale=10.0):
    for p in params:
        if p.grad is None:
            continue
        mask = (torch.rand_like(p.grad) < p_elem)
        noise = scale * torch.randn_like(p.grad)
        p.grad.add_(mask * noise)


def apply_gm_sgm_aggregation(params, n_workers=11, worker_corr_frac=0.4, scale=10.0):
    base_grad = flatten_grads(params)
    grads = []

    for _ in range(n_workers):
        gk = base_grad.clone()
        if torch.rand(()) < worker_corr_frac:
            gk = scale * torch.randn_like(gk)
        grads.append(gk)

    gm_grad = geometric_median(grads)
    set_grads_from_flat(params, gm_grad)


def robust_constraint_eval(
    base_violation,
    avg_violation=None,
    n_workers=N_CONSTRAINT_WORKERS,
    corr_frac=CONSTRAINT_CORR_FRAC,
    scale=CONSTRAINT_CORR_SCALE,
    mode=CONSTRAINT_MODE
):
    # === "class_avg" mode: just use avg_violation directly ===
    if mode == "class_avg":
        return avg_violation

    # === Worker-based modes ===
    g_vals = []
    for _ in range(n_workers):
        gk = base_violation.clone()
        if torch.rand(()) < corr_frac:
            gk += scale * torch.randn_like(gk)
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
        return torch.mean(sorted_vals[1:-1])
    elif mode == "single":
        return base_violation
    else:
        raise ValueError(f"Unknown constraint mode '{mode}'")


# ============================================================
# 4. Training loop with hard / soft switching (paper-style)
# ============================================================
def train(mode="sgm_clean"):
    assert mode in ["sgm_clean", "sgm_corrupt", "gmsgm_corrupt"]

    model = make_model()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_target_hist = []
    penalty_hist = []
    g_hat_hist = []
    per_class_hist = []

    params = list(model.parameters())

    for epoch in range(EPOCHS):
        per_class_loss_epoch = torch.zeros(10, device=device)
        total_batches = 0
        penalty_epoch = 0.0
        g_hat_epoch = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # -------------------------------------------------
            # Forward pass, per-class losses
            # -------------------------------------------------
            optimizer.zero_grad()
            logits = model(imgs)
            losses = criterion(logits, labels)

            # per-class mean loss on this batch
            per_class_loss = []
            for c in range(10):
                mask = (labels == c)
                if mask.any():
                    per_class_loss.append(losses[mask].mean())
                else:
                    per_class_loss.append(torch.tensor(0.0, device=device))
            per_class_loss = torch.stack(per_class_loss)

            # Objective: loss of target class
            obj = per_class_loss[TARGET_CLASS]

            # Constraint: other classes should be ≤ κ
            diff = per_class_loss - KAPPA
            diff[TARGET_CLASS] = 0.0  # don't constrain target class itself

            # Base (true) constraint function g(w) = max_c diff_c
            base_violation = torch.max(diff)
            avg_violation = torch.mean(diff)

            # Robust scalar estimate g_hat used for switching (paper)
            g_hat = robust_constraint_eval(
                base_violation,
                avg_violation=avg_violation,
                mode=CONSTRAINT_MODE
            )

            # -------------------------------------------------
            # Compute gradients ∇f(w) and ∇g(w) (paper-style)
            # -------------------------------------------------
            grad_f = torch.autograd.grad(obj, params, retain_graph=True)
            # Gradient of the true constraint function g(w) = max(diff)
            grad_g = torch.autograd.grad(base_violation, params)

            # Optionally scale constraint direction by λ
            grad_g = tuple(LAMBDA_PENALTY * g for g in grad_g)

            g_val = float(g_hat.detach().item())

            # -------------------------------------------------
            # Switching dynamics
            # Hard switching:
            #   u_t = ∇f(w_t) if g_hat <= ε
            #       = ∇g(w_t) if g_hat > ε
            #
            # Soft switching:
            #   p_t = min{1, [1 + β (g_hat - ε)]_+}
            #   u_t = p_t ∇g(w_t) + (1-p_t) ∇f(w_t)
            # -------------------------------------------------
            if HARD_SWITCHING:
                if g_val <= EPS_SWITCH:
                    # use only objective gradient
                    chosen_grads = grad_f
                else:
                    # use only constraint gradient
                    chosen_grads = grad_g
                # For logging we can still use relu(g_hat) as "penalty-like"
                penalty_val = max(g_val, 0.0)
            else:
                # Soft switching
                # p_t = min{1, [1 + β (g_hat - ε)]_+}
                p_t = torch.clamp(1.0 + BETA_SWITCH * (g_hat - EPS_SWITCH),
                                  min=0.0, max=1.0)
                p_val = float(p_t.detach().item())

                chosen_grads = tuple(
                    p_val * gg + (1.0 - p_val) * gf
                    for gf, gg in zip(grad_f, grad_g)
                )
                penalty_val = max(g_val, 0.0)

            # -------------------------------------------------
            # Apply chosen gradient u_t to model parameters
            # -------------------------------------------------
            optimizer.zero_grad()
            for p, g in zip(params, chosen_grads):
                p.grad = g.detach().clone()

            # Corrupt / aggregate gradients as requested
            if mode == "sgm_corrupt":
                apply_gross_corruption_to_grads(
                    model.parameters(),
                    p_elem=GRAD_CORR_ELEM_PROB,
                    scale=GRAD_CORR_SCALE
                )
            elif mode == "gmsgm_corrupt":
                apply_gm_sgm_aggregation(
                    model.parameters(),
                    n_workers=N_WORKERS,
                    worker_corr_frac=WORKER_CORR_FRAC,
                    scale=WORKER_CORR_SCALE
                )

            optimizer.step()

            # Logging accumulators
            per_class_loss_epoch += per_class_loss.detach()
            penalty_epoch += penalty_val
            g_hat_epoch += g_val
            total_batches += 1

        # Epoch-level stats
        per_class_loss_epoch /= total_batches
        per_class_hist.append(per_class_loss_epoch.cpu())
        loss_target_hist.append(float(per_class_loss_epoch[TARGET_CLASS]))
        penalty_hist.append(penalty_epoch / total_batches)
        g_hat_hist.append(g_hat_epoch / total_batches)

        mode_name = f"{mode} | {'HARD' if HARD_SWITCHING else 'SOFT'}"
        print(f"[{mode_name}] Epoch {epoch+1}/{EPOCHS} | "
              f"Target Loss={loss_target_hist[-1]:.3f} | "
              f'Avg relu(g_hat)={penalty_hist[-1]:.3f} | '
              f"g_hat={g_hat_hist[-1]:.3f}")

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
print("\n=== Running SGM-corrupt ===")
sgm_corrupt_logs = train(mode="sgm_corrupt")

print("\n=== Running GM-SGM-corrupt ===")
gmsgm_logs = train(mode="gmsgm_corrupt")

torch.save(
    {"sgm_corrupt": sgm_corrupt_logs, "gmsgm_corrupt": gmsgm_logs},
    os.path.join(RESULTS_DIR, "fashion_mnpc_gm_sgm_logs.pt"),
)

# ============================================================
# 6. Plotting
# ============================================================
def plot_per_class_evolution(logs, name):
    per_class = logs["per_class_hist"].numpy()
    plt.figure(figsize=(12, 6))

    for c in range(10):
        if c != TARGET_CLASS:
            plt.plot(per_class[:, c], color="gray", alpha=0.4)
    plt.plot(per_class[:, TARGET_CLASS], color="red", linewidth=3,
             label=f"Target ({classes[TARGET_CLASS]})")

    plt.axhline(KAPPA, color="black", linestyle="--")
    plt.title(f"Per-Class Loss Evolution – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"per_class_evolution_{name}.png"))
    plt.close()


def plot_target_vs_others_comparison(logA, logB, labelA, labelB):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(logA["loss_target_hist"], label=f"{labelA} – Target Loss")
    plt.plot(logB["loss_target_hist"], label=f"{labelB} – Target Loss")
    plt.axhline(KAPPA, linestyle="--", color="red")
    plt.ylabel("Loss")
    plt.title("Target Class Loss Comparison")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(logA["penalty_hist"], label=f"{labelA} – relu(g_hat)")
    plt.plot(logB["penalty_hist"], label=f"{labelB} – relu(g_hat)")
    plt.xlabel("Epoch")
    plt.ylabel("relu(g_hat)")
    plt.title("Constraint Violation Proxy Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "target_vs_penalty_comparison.png"))
    plt.close()


def plot_classwise_last_epoch(logA, logB, labelA, labelB):
    A = logA["per_class_hist"][-1].numpy()
    B = logB["per_class_hist"][-1].numpy()
    x = range(10)

    plt.figure(figsize=(12, 6))
    plt.bar([i - 0.15 for i in x], A, width=0.3, label=labelA)
    plt.bar([i + 0.15 for i in x], B, width=0.3, label=labelB)
    plt.axhline(KAPPA, linestyle="--", color="red")
    plt.xticks(x, classes, rotation=40)
    plt.ylabel("Loss")
    plt.title("Last Epoch Per-Class Loss Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "last_epoch_bar_comparison.png"))
    plt.close()


def plot_deviation_from_kappa(logs, name):
    per_class = logs["per_class_hist"].numpy()
    deviation = per_class - KAPPA

    plt.figure(figsize=(12, 6))
    for c in range(10):
        if c != TARGET_CLASS:
            plt.plot(deviation[:, c], alpha=0.5)
    plt.axhline(0, color="black", linestyle="--")
    plt.plot(deviation[:, TARGET_CLASS], color="red", linewidth=3)
    plt.title(f"Loss – κ Deviation – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss_i - κ")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"deviation_from_kappa_{name}.png"))
    plt.close()


def plot_last_epoch_box(logA, logB, labelA, labelB):
    A = logA["per_class_hist"][-1].numpy()
    B = logB["per_class_hist"][-1].numpy()

    plt.figure(figsize=(8, 6))
    plt.boxplot([A, B], labels=[labelA, labelB])
    plt.axhline(KAPPA, linestyle="--", color="red")
    plt.ylabel("Loss")
    plt.title("Class Loss Distribution Last Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "last_epoch_boxplot.png"))
    plt.close()


print("Generating enhanced plots...")

name_suffix = "HARD" if HARD_SWITCHING else "SOFT"

plot_per_class_evolution(sgm_corrupt_logs, f"SGM-corrupt-{name_suffix}")
plot_per_class_evolution(gmsgm_logs, f"GM-SGM-corrupt-{name_suffix}")

plot_target_vs_others_comparison(
    sgm_corrupt_logs, gmsgm_logs,
    f"SGM-corrupt-{name_suffix}", f"GM-SGM-corrupt-{name_suffix}"
)

plot_classwise_last_epoch(
    sgm_corrupt_logs, gmsgm_logs,
    f"SGM-corrupt-{name_suffix}", f"GM-SGM-corrupt-{name_suffix}"
)

plot_deviation_from_kappa(sgm_corrupt_logs, f"SGM-corrupt-{name_suffix}")
plot_deviation_from_kappa(gmsgm_logs, f"GM-SGM-corrupt-{name_suffix}")

plot_last_epoch_box(
    sgm_corrupt_logs, gmsgm_logs,
    f"SGM-corrupt-{name_suffix}", f"GM-SGM-corrupt-{name_suffix}"
)

print(f"All enhanced plots and logs saved to ./{RESULTS_DIR}/")
