import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ============================================================
# 1. Config general
# ============================================================
BATCH_SIZE = 256
EPOCHS = 100           # puedes aumentar si quieres
LR = 1e-3
LAMBDA_PENALTY = 4.0
KAPPA = 0.3
TARGET_CLASS = 7

# corrupción
GRAD_CORR_ELEM_PROB = 0.3
GRAD_CORR_SCALE = 10.0

N_WORKERS = 11
WORKER_CORR_FRAC = 0.4
WORKER_CORR_SCALE = 10.0

BETAS = [1.0, 0.6, 0.3]    # fracciones de bloque (k = β d)

# ============================================================
# 2. Data
# ============================================================
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(root="./data", train=True,
                              download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train, batch_size=BATCH_SIZE, shuffle=True
)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Si quieres timings más “honestos” tipo paper, fuerza CPU:
# device = torch.device("cpu")
device = torch.device("cpu")
print(f"Using device: {device}")

# ============================================================
# 3. Helpers: model + gradient ops
# ============================================================

def make_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
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

# ===========================
# 3A. Geometric median (GM)
# ===========================

def geometric_median(points, max_iter=10, eps=1e-5):
    stack = torch.stack(points, dim=0)
    median = stack.mean(dim=0)

    for _ in range(max_iter):
        diffs = stack - median
        distances = torch.norm(diffs, dim=1) + 1e-8
        weights = 1.0 / distances
        weights = weights / weights.sum()
        new_median = (weights.unsqueeze(1) * stack).sum(dim=0)

        if torch.norm(new_median - median) < eps:
            break
        median = new_median
    return median

# ============================================================
# 4. Corruption Models
# ============================================================

def apply_gross_corruption_to_grads(params, p_elem=0.3, scale=10.0):
    """
    SGM corruption: each grad element corrupted independently.
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
    GM-SGM: some workers fully corrupted, GM aggregation over workers
    in full dimension d.
    """
    base_grad = flatten_grads(params)
    grads = []
    for _ in range(n_workers):
        gk = base_grad.clone()
        if torch.rand(()) < worker_corr_frac:
            gk = scale * torch.randn_like(gk)
        grads.append(gk)
    gm_grad = geometric_median(grads)
    set_grads_from_flat(params, gm_grad)

# ============================================================
# 5. BLOCK COORDINATE + MEMORY (BGMD-style, CORREGIDO)
# ============================================================

def block_select(G, beta):
    """
    BGMD's C_k operator:
        G ∈ R^{b × d}
        s_j = ||G[:,j]||^2
        keep top-k coordinates
    Returns:
        G_block (b, d) with zeros outside block
        mask (d,) boolean indicating selected coords
    """
    b, d = G.shape
    k = max(1, int(beta * d))
    scores = (G * G).sum(dim=0)          # (d,)
    _, idx = torch.topk(scores, k)
    mask = torch.zeros(d, dtype=torch.bool, device=G.device)
    mask[idx] = True
    G_out = G.clone()
    G_out[:, ~mask] = 0
    return G_out, mask

def apply_gm_sgm_block_memory(params, memory, beta,
                              n_workers=11, worker_corr_frac=0.4, scale=10.0):
    """
    Full BGMD-style GM-SGM + Block + Memory aggregation step.

    Corrección importante:
      - GM se computa en dimensión reducida k = β d (no en d).
      - Luego se re-expande a un gradiente de dimensión d,
        poniendo cero fuera del bloque.
    """
    # Gradiente base (d,)
    base_grad = flatten_grads(params)          # (d,)
    d = base_grad.numel()

    # Construimos gradientes de "workers"
    grads = []
    for _ in range(n_workers):
        gk = base_grad.clone()
        if torch.rand(()) < worker_corr_frac:
            gk = scale * torch.randn_like(gk)
        grads.append(gk)

    # G ∈ R^{b × d}
    G = torch.stack(grads)                     # (b, d)

    # Añadir memoria a cada worker (broadcast)
    # memory ∈ R^d
    G_tilde = G + memory                       # (b, d)

    # Selección de bloque: C_k(G_tilde)
    Delta, mask = block_select(G_tilde, beta=beta)  # Delta: (b, d), mask: (d,)

    # Reducir realmente a dimensión k: nos quedamos solo con coordenadas seleccionadas
    Delta_reduced = Delta[:, mask]            # (b, k)

    # Residuales para memoria en dimensión completa (d)
    residuals = G_tilde - Delta               # (b, d)
    new_memory = residuals.mean(dim=0)        # (d,)

    # GM en dimensión reducida (k)
    gm_grad_reduced = geometric_median(
        [Delta_reduced[i] for i in range(Delta_reduced.shape[0])]
    )                                         # (k,)

    # Re-expandimos a dimensión completa (d)
    gm_full = torch.zeros(d, device=base_grad.device)
    gm_full[mask] = gm_grad_reduced           # (d,)

    # Guardamos el gradiente como si fuera el "grad" de los params
    set_grads_from_flat(params, gm_full)

    return new_memory

# ============================================================
# 6. Training loop with multiple MODES
# ============================================================

def train(mode="sgm_corrupt", beta=None):
    """
    Valid modes:
        "sgm_corrupt"
        "gmsgm_corrupt"
        "blockmem_beta"
    """
    model = make_model()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_hist = []       # target class
    penalty_hist = []
    per_class_hist = []

    # memory for block+mem mode
    memory = None
    if mode == "blockmem_beta":
        memory = torch.zeros(
            sum(p.numel() for p in model.parameters()),
            device=device
        )

    start = time.time()

    for epoch in range(EPOCHS):
        per_class_epoch = torch.zeros(10, device=device)
        total_batches = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            losses = criterion(logits, labels)

            cls_losses = []
            for c in range(10):
                mask = (labels == c)
                cls_losses.append(
                    losses[mask].mean() if mask.any()
                    else torch.tensor(0.0, device=device)
                )
            cls_losses = torch.stack(cls_losses)

            obj = cls_losses[TARGET_CLASS]

            eps_tensor = torch.tensor([KAPPA] * 10, device=device)
            violations = cls_losses - eps_tensor
            violations[TARGET_CLASS] = 0.0

            g = torch.max(violations)
            penalty = torch.relu(g)
            total_loss = obj + LAMBDA_PENALTY * penalty

            total_loss.backward()

            # gradient manipulation depending on mode
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
            elif mode == "blockmem_beta":
                memory = apply_gm_sgm_block_memory(
                    model.parameters(),
                    memory,
                    beta=beta,
                    n_workers=N_WORKERS,
                    worker_corr_frac=WORKER_CORR_FRAC,
                    scale=WORKER_CORR_SCALE,
                )

            optimizer.step()

            per_class_epoch += cls_losses.detach()
            total_batches += 1

        per_class_epoch /= total_batches
        per_class_hist.append(per_class_epoch.cpu())
        loss_hist.append(float(per_class_epoch[TARGET_CLASS]))
        penalty_hist.append(float(penalty.item()))

        print(
            f"[{mode}] Epoch {epoch+1}/{EPOCHS} | "
            f"Loss={loss_hist[-1]:.3f} | Penalty={penalty_hist[-1]:.3f}"
        )

    elapsed = time.time() - start

    return {
        "loss": loss_hist,
        "penalty": penalty_hist,
        "per_class": torch.stack(per_class_hist),
        "time": elapsed,
    }

# ============================================================
# 7. Run experiments
# ============================================================

print("\n=== Running SGM (corrupt) ===")
sgm_logs = train("sgm_corrupt")

print("\n=== Running GM-SGM (corrupt) ===")
gmsgm_logs = train("gmsgm_corrupt")

block_logs = {}
for beta in BETAS:
    print(f"\n=== Running GM-SGM + Block + Memory (β={beta}) ===")
    block_logs[beta] = train("blockmem_beta", beta=beta)

# save everything
torch.save({
    "sgm": sgm_logs,
    "gmsgm": gmsgm_logs,
    "block": block_logs,
}, "results/all_methods_logs.pt")

# ============================================================
# 8. Runtime summary
# ============================================================

print("\n=== Runtime Summary ===")
print(f"SGM-corrupt:       {sgm_logs['time']:.2f}s")
print(f"GM-SGM-corrupt:    {gmsgm_logs['time']:.2f}s")
for beta in BETAS:
    print(f"Block+Mem β={beta}: {block_logs[beta]['time']:.2f}s")

# ============================================================
# 9. Plotting utilities (per-method + global)
# ============================================================

def plot_per_class_evolution(logs, name):
    """
    Plot per-class curves with target highlighted.
    """
    per_class = logs["per_class"].numpy()
    plt.figure(figsize=(12, 6))

    for c in range(10):
        if c == TARGET_CLASS:
            continue
        plt.plot(per_class[:, c], color="gray", alpha=0.4)

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

def plot_deviation_from_kappa(logs, name):
    """
    Plot (Loss_i - κ_i) across epochs for all classes.
    """
    per_class = logs["per_class"].numpy()
    deviation = per_class - KAPPA

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

def plot_target_and_penalty_multi(method_logs, labels, filename):
    """
    Compare target loss and penalty across multiple methods.
    """
    plt.figure(figsize=(12, 8))

    # Target loss
    plt.subplot(2, 1, 1)
    for logs, label in zip(method_logs, labels):
        plt.plot(logs["loss"], label=f"{label} – Target Loss")
    plt.axhline(KAPPA, linestyle="--", color="red", label="κ_i")
    plt.ylabel("Loss")
    plt.title("Target Class Loss Comparison")
    plt.legend()

    # Penalty
    plt.subplot(2, 1, 2)
    for logs, label in zip(method_logs, labels):
        plt.plot(logs["penalty"], label=f"{label} – Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Penalty")
    plt.title("Constraint Violation Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/{filename}.png")
    plt.close()

def plot_last_epoch_box_multi(method_logs, labels, filename):
    """
    Boxplot de pérdidas por clase en el último epoch para múltiples métodos.
    """
    data = []
    for logs in method_logs:
        last = logs["per_class"][-1].numpy()
        data.append(last)

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.axhline(KAPPA, linestyle="--", color="red", label="κ")
    plt.ylabel("Last Epoch Class Loss")
    plt.title("Class Loss Distribution at Last Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{filename}.png")
    plt.close()

def plot_time_vs_beta(sgm_logs, gmsgm_logs, block_logs, betas):
    """
    Plot tiempo vs β para los métodos Block+Memory.
    Añade líneas horizontales para SGM y GM-SGM.
    """
    times_block = [block_logs[b]["time"] for b in betas]

    plt.figure(figsize=(8, 6))
    plt.plot(betas, times_block, "-o", label="Block+Mem (β)", linewidth=2, markersize=6)
    plt.axhline(sgm_logs["time"], linestyle="--", color="gray", label="SGM-corrupt time")
    plt.axhline(gmsgm_logs["time"], linestyle="--", color="green", label="GM-SGM-corrupt time")

    plt.gca().invert_xaxis()  # opcional: β grande a la izquierda, como BGMD
    plt.xlabel("β (k / d)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Block Fraction β (GM-SGM + Block + Memory)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/runtime_vs_beta.png")
    plt.close()

# ============================================================
# 10. Generate plots for each method + global
# ============================================================

print("Generating per-method plots...")

# Diccionario con todos los métodos
all_logs = {
    "SGM-corrupt": sgm_logs,
    "GM-SGM-corrupt": gmsgm_logs,
}
for beta in BETAS:
    all_logs[f"BlockMem-β={beta}"] = block_logs[beta]

# Plots por método (per-class & deviation)
for name, logs in all_logs.items():
    safe_name = name.replace(" ", "_").replace("=", "")
    plot_per_class_evolution(logs, safe_name)
    plot_deviation_from_kappa(logs, safe_name)

print("Generating multi-method comparison plots...")

# Multi-método: target loss + penalty
method_logs = [sgm_logs, gmsgm_logs] + [block_logs[b] for b in BETAS]
labels = ["SGM-corrupt", "GM-SGM-corrupt"] + [f"Block+Mem β={b}" for b in BETAS]
plot_target_and_penalty_multi(method_logs, labels, "target_penalty_all_methods")

# Multi-método: boxplot último epoch
plot_last_epoch_box_multi(method_logs, labels, "last_epoch_box_all_methods")

# Tiempo vs β
plot_time_vs_beta(sgm_logs, gmsgm_logs, block_logs, BETAS)

print("All plots saved into ./results/")
