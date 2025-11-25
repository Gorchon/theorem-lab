import os
import time
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
EPOCHS = 120
LR = 1e-3
LAMBDA_PENALTY = 4.0
KAPPA = 0.3
TARGET_CLASS = 7     # Sneaker

# Corruption (for SGM baseline)
GRAD_CORR_ELEM_PROB = 0.3
GRAD_CORR_SCALE = 10.0

# GM-SGM: number of synthetic "workers"
N_WORKERS = 11
WORKER_CORR_FRAC = 0.4
WORKER_CORR_SCALE = 10.0

# Block + Memory: we will compare several betas
BETA_BLOCK_LIST = [0.1, 0.3]

# Benchmark
AGG_BENCH_REPEATS = 50

# ============================================================
# 2. Data
# ============================================================
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
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
# 3. Model + gradient helpers + geometric median
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


def geometric_median_stack(stack, max_iter=10, eps=1e-5):
    """
    Compute geometric median of a (b, d) tensor using Weiszfeld's algorithm.
    """
    # stack: (b, d)
    median = stack.mean(dim=0)

    for _ in range(max_iter):
        diffs = stack - median  # (b, d)
        distances = torch.norm(diffs, dim=1) + 1e-8  # (b,)
        weights = 1.0 / distances
        weights = weights / weights.sum()
        new_median = (weights.unsqueeze(1) * stack).sum(dim=0)

        if torch.norm(new_median - median) < eps:
            break
        median = new_median

    return median


# ============================================================
# 4. Corruption for SGM baseline
# ============================================================
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


# ============================================================
# 5. GM-SGM aggregators (full and block+memory) on flat gradient vectors
# ============================================================
def gm_agg_full_flat(base_grad,
                     n_workers=N_WORKERS,
                     worker_corr_frac=WORKER_CORR_FRAC,
                     scale=WORKER_CORR_SCALE):
    """
    Full GM-SGM aggregation in R^d:
      - build n_workers "worker" gradients as corrupted copies of base_grad
      - aggregate with geometric median in full dimension.
    """
    b = n_workers
    grads = []

    for _ in range(b):
        gk = base_grad.clone()
        # fully corrupted worker
        if torch.rand(()) < worker_corr_frac:
            gk = scale * torch.randn_like(gk)
        grads.append(gk)

    G_t = torch.stack(grads, dim=0)  # (b, d)
    gm_grad = geometric_median_stack(G_t)
    return gm_grad


def gm_agg_block_memory_flat(base_grad,
                             memory,
                             beta,
                             n_workers=N_WORKERS,
                             worker_corr_frac=WORKER_CORR_FRAC,
                             scale=WORKER_CORR_SCALE):
    """
    GM-SGM with Block Coordinate Selection + Memory on the flat gradient base_grad.

    Implements the GM-SGM+Block+Memory pseudocode (in simplified form):
      - Build worker gradient matrix G_t in R^{b × d}.
      - Compute importance scores s_j from G_t (raw, no memory).
      - Select top-k coordinates (block).
      - Residual Δ_t = G_t - C_k(G_t), update full-dimensional memory m_t.
      - Aggregate only in the k-dimensional block using GM(G_t_block + m_t[block]).
      - Reconstruct a full-dimensional aggregated gradient with zeros outside block.
    """
    d = base_grad.numel()
    b = n_workers

    # ---- compute block size k ----
    k_dim = max(1, int(beta * d))
    k_dim = min(k_dim, d)

    # ---- full-dimensional memory m_t in R^d ----
    if memory is None or memory.numel() != d:
        memory = torch.zeros(d, device=base_grad.device)

    # ---- build G_t (b × d) with worker corruption ----
    workers = []
    for _ in range(b):
        gk = base_grad.clone()
        if torch.rand(()) < worker_corr_frac:
            gk = scale * torch.randn_like(gk)
        workers.append(gk)
    G_t = torch.stack(workers, dim=0)  # (b, d)

    # ---- importance scores on G_t (raw) ----
    s = (G_t ** 2).sum(dim=0)  # (d,)
    _, top_idx = torch.topk(s, k_dim)
    mask = torch.zeros(d, dtype=torch.bool, device=base_grad.device)
    mask[top_idx] = True

    # ---- block projection: G_t_block in R^{b × k} ----
    G_t_block = G_t[:, mask]  # (b, k_dim)

    # ---- residual and memory update in full dimension ----
    # C_k(G_t): same as G_t on mask coords, zero elsewhere
    Delta_t = G_t.clone()
    Delta_t[:, mask] = 0.0
    delta_mean = Delta_t.mean(dim=0)          # (d,)
    memory_new = memory + delta_mean          # full-dimensional m_{t+1}

    # ---- augmentation with current memory m_t (not m_{t+1}) ----
    G_t_aug_block = G_t_block + memory[mask].unsqueeze(0)  # (b, k_dim)

    # ---- GM in k-dimensional block ----
    gm_block = geometric_median_stack(G_t_aug_block)  # (k_dim,)

    # ---- reconstruct a full-dim gradient ----
    gm_full = torch.zeros(d, device=base_grad.device)
    gm_full[mask] = gm_block

    return gm_full, memory_new


# ============================================================
# 6. Wrappers at parameter level
# ============================================================
def apply_gm_sgm_full_to_params(params):
    base_grad = flatten_grads(params)
    gm_grad = gm_agg_full_flat(base_grad)
    set_grads_from_flat(params, gm_grad)


def apply_gm_sgm_block_memory_to_params(params, memory, beta):
    base_grad = flatten_grads(params)
    gm_grad, memory_new = gm_agg_block_memory_flat(
        base_grad,
        memory,
        beta=beta
    )
    set_grads_from_flat(params, gm_grad)
    return memory_new


# ============================================================
# 7. Training loop (SGM baseline + GM-SGM variants)
# ============================================================
def train(mode="sgm_corrupt"):
    """
    mode:
      - 'sgm_corrupt'           : SGM with elementwise gradient corruption
      - 'gmsgm_full'            : GM-SGM full aggregation (no block/memory)
      - 'gmsgm_blockmem_<beta>' : GM-SGM with block coordinate selection + memory
                                  e.g. 'gmsgm_blockmem_0.1'
    """
    assert (
        mode == "sgm_corrupt"
        or mode == "gmsgm_full"
        or mode.startswith("gmsgm_blockmem_")
    )

    model = make_model()
    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_target_hist = []
    penalty_hist = []
    per_class_hist = []

    # memory only for block+memory
    memory = None

    for epoch in range(EPOCHS):
        per_class_loss_epoch = torch.zeros(10, device=device)
        total_batches = 0
        last_penalty = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            losses = criterion(logits, labels)

            # Per-class loss for this batch
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

            # Constraints: max violation over non-target classes
            eps_tensor = torch.tensor(
                [KAPPA for _ in range(10)], device=device
            )
            violations = per_class_loss - eps_tensor
            violations[TARGET_CLASS] = 0.0  # target unconstrained

            g_value = torch.max(violations)
            penalty = torch.relu(g_value)
            last_penalty = penalty.item()

            total_loss = obj + LAMBDA_PENALTY * penalty

            # Backward
            total_loss.backward()

            # Gradient handling depending on mode
            if mode == "sgm_corrupt":
                apply_gross_corruption_to_grads(
                    model.parameters(),
                    p_elem=GRAD_CORR_ELEM_PROB,
                    scale=GRAD_CORR_SCALE,
                )
            elif mode == "gmsgm_full":
                apply_gm_sgm_full_to_params(model.parameters())
            elif mode.startswith("gmsgm_blockmem_"):
                # mode format: gmsgm_blockmem_<beta>
                beta = float(mode.split("_")[2])
                memory = apply_gm_sgm_block_memory_to_params(
                    model.parameters(),
                    memory,
                    beta=beta,
                )

            optimizer.step()

            per_class_loss_epoch += per_class_loss.detach()
            total_batches += 1

        # epoch averages
        per_class_loss_epoch /= total_batches
        per_class_hist.append(per_class_loss_epoch.cpu())
        loss_target_hist.append(
            float(per_class_loss_epoch[TARGET_CLASS].item())
        )
        penalty_hist.append(float(last_penalty))

        print(
            f"[{mode}] Epoch {epoch+1}/{EPOCHS} | "
            f"Target(Sneaker) Loss={loss_target_hist[-1]:.3f} | "
            f"Penalty={penalty_hist[-1]:.3f}"
        )

    per_class_hist = torch.stack(per_class_hist)
    return {
        "model": model,
        "loss_target_hist": loss_target_hist,
        "penalty_hist": penalty_hist,
        "per_class_hist": per_class_hist,
    }


# ============================================================
# 8. Aggregation runtime benchmark
# ============================================================
def benchmark_aggregation_runtime():
    """
    Compare aggregation time between:
      - GM-SGM full (beta = 1.0 conceptually)
      - GM-SGM Block + Memory for each beta in BETA_BLOCK_LIST
    using synthetic gradients with dimension equal to the model's gradient.
    """
    # dummy model to get gradient dimension d
    model = make_model()
    dummy_input = torch.randn(4, 1, 28, 28, device=device)
    dummy_target = torch.randint(0, 10, (4,), device=device)
    criterion = nn.CrossEntropyLoss()

    model.zero_grad()
    out = model(dummy_input)
    loss = criterion(out, dummy_target)
    loss.backward()

    base_grad = flatten_grads(model.parameters()).detach()
    d = base_grad.numel()
    print(f"[Benchmark] Gradient dimension d = {d}")

    results = {}

    # full GM-SGM
    start = time.perf_counter()
    for _ in range(AGG_BENCH_REPEATS):
        _ = gm_agg_full_flat(base_grad)
    elapsed_full = time.perf_counter() - start
    results["full"] = elapsed_full / AGG_BENCH_REPEATS

    # block + memory GM-SGM for each beta
    for beta in BETA_BLOCK_LIST:
        memory = torch.zeros(d, device=base_grad.device)
        start = time.perf_counter()
        for _ in range(AGG_BENCH_REPEATS):
            _, memory = gm_agg_block_memory_flat(
                base_grad,
                memory,
                beta=beta,
            )
        elapsed_block = time.perf_counter() - start
        results[beta] = elapsed_block / AGG_BENCH_REPEATS

    print(f"[Benchmark] Full GM-SGM aggregation avg time   : {results['full']:.6f} s")
    for beta in BETA_BLOCK_LIST:
        print(f"[Benchmark] Block+Memory (beta={beta}) avg time: {results[beta]:.6f} s")

    # save CSV-style table
    with open("results/agg_runtime_table_multi_beta.txt", "w") as f:
        f.write("Method,AvgTimePerCall\n")
        f.write(f"GM-SGM-full,{results['full']:.6f}\n")
        for beta in BETA_BLOCK_LIST:
            f.write(f"GM-SGM-blockmem-{beta},{results[beta]:.6f}\n")

    # bar plot
    methods = ["full"] + [f"β={beta}" for beta in BETA_BLOCK_LIST]
    times = [results["full"]] + [results[beta] for beta in BETA_BLOCK_LIST]

    plt.figure(figsize=(7, 5))
    plt.bar(methods, times)
    plt.ylabel("Avg aggregation time (s)")
    plt.title("Aggregation runtime: full vs block+memory (multi-β)")
    plt.tight_layout()
    plt.savefig("results/agg_runtime_bar_multi_beta.png")
    plt.close()


# ============================================================
# 9. Plot helpers for mNPC-style results
# ============================================================
def plot_per_class_evolution(logs, name):
    """Plot per-class curves with target highlighted."""
    per_class = logs["per_class_hist"].numpy()
    plt.figure(figsize=(12, 6))

    for c in range(10):
        if c == TARGET_CLASS:
            continue
        plt.plot(per_class[:, c], alpha=0.4)

    plt.plot(per_class[:, TARGET_CLASS], linewidth=3,
             label=f"Target ({classes[TARGET_CLASS]})")

    plt.axhline(KAPPA, linestyle="--", label="κ_i threshold")

    plt.title(f"Per-Class Loss Evolution – {name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/per_class_evolution_{name}.png")
    plt.close()


def plot_target_vs_others_comparison(logA, logB, labelA, labelB, outfile):
    plt.figure(figsize=(12, 8))

    # Target loss
    plt.subplot(2, 1, 1)
    plt.plot(logA["loss_target_hist"], label=f"{labelA} – Target Loss")
    plt.plot(logB["loss_target_hist"], label=f"{labelB} – Target Loss")
    plt.axhline(KAPPA, linestyle="--", label="κ_i")
    plt.ylabel("Loss")
    plt.title("Target Class Loss Comparison")
    plt.legend()

    # Penalty
    plt.subplot(2, 1, 2)
    plt.plot(logA["penalty_hist"], label=f"{labelA} – Penalty")
    plt.plot(logB["penalty_hist"], label=f"{labelB} – Penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Penalty")
    plt.title("Constraint Violation Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_classwise_last_epoch_multi_beta(full_logs, blockmem_logs_dict):
    """
    One single plot comparing Full GM-SGM vs all Block+Mem betas
    at the last epoch, class-by-class.
    """

    A_full = full_logs["per_class_hist"][-1].numpy()   # (10,)

    plt.figure(figsize=(14, 6))
    x = range(10)
    width = 0.18

    # Full baseline
    plt.bar(
        [i - width*1.5 for i in x],
        A_full,
        width=width,
        label="GM-SGM Full",
        alpha=0.85
    )

    # Each beta
    colors = ["#ff7f0e", "#2ca02c", "#1f77b4"]
    for (beta, logs), color in zip(blockmem_logs_dict.items(), colors):
        B = logs["per_class_hist"][-1].numpy()
        shift = {"0.1": -0.5, "0.3": 0, "0.5": 0.5}[str(beta)]
        shift = float(shift)

        plt.bar(
            [i + width * shift for i in x],
            B,
            width=width,
            label=f"Block+Mem β={beta}",
            alpha=0.85,
            color=color
        )

    plt.axhline(KAPPA, linestyle="--", color="gray", label="κ_i threshold")
    plt.xticks(x, classes, rotation=40)
    plt.ylabel("Loss (Last Epoch)")
    plt.title("Last Epoch Per-Class Loss — GM-SGM Full vs Block+Mem (β=0.1,0.3,0.5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/last_epoch_bar_multi_beta.png")
    plt.close()



def plot_last_epoch_box(logA, logB, labelA, labelB, outfile):
    A = logA["per_class_hist"][-1].numpy()
    B = logB["per_class_hist"][-1].numpy()

    plt.figure(figsize=(8, 6))
    plt.boxplot([A, B], tick_labels=[labelA, labelB])
    plt.axhline(KAPPA, linestyle="--", label="κ")
    plt.ylabel("Last Epoch Class Loss")
    plt.title("Class Loss Distribution at Last Epoch")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


def plot_multi_beta_curves(full_logs, blockmem_logs_dict):
    """
    Plot:
      - target loss vs epochs: full vs each beta
      - penalty vs epochs: full vs each beta
    """
    epochs = range(len(full_logs["loss_target_hist"]))

    # Target loss
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        full_logs["loss_target_hist"],
        label="Full GM-SGM",
        linewidth=2,
    )
    for beta, logs in blockmem_logs_dict.items():
        plt.plot(
            epochs,
            logs["loss_target_hist"],
            label=f"Block+Mem β={beta}",
            linestyle="--",
        )
    plt.axhline(KAPPA, linestyle="--", color="gray", label="κ_i")
    plt.xlabel("Epoch")
    plt.ylabel("Target Loss")
    plt.title("Target class loss – Full vs Block+Mem (multi-β)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/multi_beta_target_loss.png")
    plt.close()

    # Penalty
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs,
        full_logs["penalty_hist"],
        label="Full GM-SGM",
        linewidth=2,
    )
    for beta, logs in blockmem_logs_dict.items():
        plt.plot(
            epochs,
            logs["penalty_hist"],
            label=f"Block+Mem β={beta}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Penalty")
    plt.title("Constraint violation – Full vs Block+Mem (multi-β)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/multi_beta_penalty.png")
    plt.close()

# ============================================================
# 10. Run experiments
# ============================================================
if __name__ == "__main__":
   
    print("\n=== Running GM-SGM (full) with gross corruption ===")
    gmsgm_full_logs = train(mode="gmsgm_full")

    # Block+Memory for multiple betas
    blockmem_logs = {}
    print("\n=== Running GM-SGM (block+memory) with gross corruption for multiple β ===")
    for beta in BETA_BLOCK_LIST:
        mode = f"gmsgm_blockmem_{beta}"
        print(f"\n>>> Running GM-SGM Block+Memory (beta={beta})")
        blockmem_logs[beta] = train(mode=mode)

    # save logs
    torch.save(
        {
            "gmsgm_full": gmsgm_full_logs,
            "gmsgm_blockmem_multi": blockmem_logs,
        },
        "results/fashion_mnpc_gm_sgm_blockmem_logs.pt",
    )

    # mNPC-style plots: full vs each beta separately
    print("Generating mNPC comparison plots (full vs each β)...")

    plot_per_class_evolution(gmsgm_full_logs, "GM-SGM-full")

    for beta, logs in blockmem_logs.items():
        name = f"GM-SGM-blockmem-beta{beta}"
        plot_per_class_evolution(logs, name)

        # Pairwise plots full vs this beta
        plot_target_vs_others_comparison(
            gmsgm_full_logs,
            logs,
            labelA="GM-SGM-full",
            labelB=f"GM-SGM-blockmem-β={beta}",
            outfile=f"results/target_vs_penalty_full_vs_blockmem_beta{beta}.png",
        )


        plot_last_epoch_box(
            gmsgm_full_logs,
            logs,
            labelA="GM-SGM-full",
            labelB=f"GM-SGM-blockmem-β={beta}",
            outfile=f"results/last_epoch_box_full_vs_blockmem_beta{beta}.png",
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ONE SINGLE multi-β plot (full vs all betas together)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    plot_classwise_last_epoch_multi_beta(
        gmsgm_full_logs,
        blockmem_logs
    )

    # Multi-β curves (full vs all betas together)
    print("Generating multi-β comparison plots (target + penalty)...")
    plot_multi_beta_curves(gmsgm_full_logs, blockmem_logs)

    # aggregation runtime benchmark
    print("\n=== Benchmarking aggregation runtimes (multi-β) ===")
    benchmark_aggregation_runtime()

    print("All logs and plots saved into ./results/")
