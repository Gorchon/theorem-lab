import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================================================
# 0. Reproducibility and device
# ============================================================

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cpu")  # fair timing on CPU
print(f"Using device: {device}")

os.makedirs("results", exist_ok=True)

# ============================================================
# 1. Hyperparameters and configuration
# ============================================================

BATCH_SIZE = 128
EPOCHS =  5
LR = 0.01

# Number of "workers" in the corruption model
N_WORKERS = 11
WORKER_CORR_FRAC = 0.4
WORKER_NOISE_SCALE = 20.0
SMALL_NOISE_SCALE = 0.05

# Memory decay is not used in the BGMD-style update below (kept for compatibility)
MEMORY_DECAY = 0.9
BETA_LIST = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

# To keep things fast, we train on a subset of Fashion-MNIST
SUBSET_TRAIN_SIZE = 10000  # from 60k

# ============================================================
# 2. Dataset: Fashion-MNIST (with train subset)
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset_full = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

# Take a random subset of the train set for faster experimentation
perm_indices = torch.randperm(len(train_dataset_full))[:SUBSET_TRAIN_SIZE]
train_dataset = torch.utils.data.Subset(train_dataset_full, perm_indices)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)

NUM_CLASSES = 10

# ============================================================
# 3. Small CNN Model (keeps gradient dimension manageable)
# ============================================================

class SmallCNN(nn.Module):
    """
    A small CNN for Fashion-MNIST.
    Total parameters ~ 100k, so gradient-based
    robust aggregation is feasible on CPU.
    """
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 28 -> 14

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),     # 14 -> 7
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),   # 32*49 = 1568
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


def make_model():
    return SmallCNN().to(device)

# ============================================================
# 4. Gradient utilities
# ============================================================

def get_flat_grad(model):
    """
    Flatten all parameter gradients into a single 1D tensor.
    """
    grads = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).flatten())
        else:
            grads.append(p.grad.flatten())
    return torch.cat(grads)


def set_flat_grad(model, flat):
    """
    Write a flat gradient vector back into model.parameters().
    """
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.grad = flat[offset:offset+numel].view_as(p).clone()
        offset += numel

# ============================================================
# 5. Geometric median + block selection
# ============================================================

def geometric_median(points, max_iter=15, tol=1e-6):
    """
    Weiszfeld algorithm for geometric median in R^d.

    points: tensor of shape (b, d), b = number of workers.
    """
    median = points.mean(dim=0)
    for _ in range(max_iter):
        diff = points - median
        dist = torch.norm(diff, dim=1) + 1e-8
        w = 1.0 / dist
        w = w / w.sum()
        new = (w.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(new - median) < tol:
            return new
        median = new
    return median


def select_block(scores, beta):
    """
    Select the top-k coordinates based on scores.

    scores: tensor (d,)
    beta: fraction 0 < beta <= 1
    """
    d = scores.numel()
    k = max(1, int(beta * d))
    _, idx = torch.topk(scores, k)
    mask = torch.zeros(d, dtype=torch.bool)
    mask[idx] = True
    return mask

# ============================================================
# 6. Aggregation methods
# ============================================================

def sgm_aggregator(base_grad, memory, cfg, timing):
    """
    Non-robust SGM-style plain mean of worker gradients.
    Implemented in a memory-efficient way (no large stack).
    """
    start = time.perf_counter()
    n = cfg["n_workers"]
    avg = torch.zeros_like(base_grad)

    for _ in range(n):
        g = base_grad.clone()
        # small noise (heterogeneity)
        g = g + cfg["small_noise"] * torch.randn_like(base_grad)
        # heavy corruption for a fraction of workers
        if torch.rand(()) < cfg["corr_frac"]:
            g = g + cfg["noise_scale"] * torch.randn_like(base_grad)
        avg += g

    avg = avg / n
    timing["agg"] += time.perf_counter() - start
    return avg, memory


def gmsgm_aggregator(base_grad, memory, cfg, timing):
    """
    GM-SGM full: geometric median over all coordinates in R^d.
    Uses worker-level corruption like SGM, but aggregates robustly.
    """
    start = time.perf_counter()
    n = cfg["n_workers"]
    grads = []

    for _ in range(n):
        g = base_grad.clone()
        g = g + cfg["small_noise"] * torch.randn_like(base_grad)
        if torch.rand(()) < cfg["corr_frac"]:
            g = g + cfg["noise_scale"] * torch.randn_like(base_grad)
        grads.append(g)

    G = torch.stack(grads, dim=0)  # (n, d)
    gm = geometric_median(G)
    timing["agg"] += time.perf_counter() - start
    return gm, memory


def blockmem_aggregator(base_grad, memory, cfg, timing):
    """
    GM-SGM + Block + Memory (BGMD-style inside our corruption model):

    1. Build worker gradient matrix G in R^{b x d} from the base gradient
       with noise + corruption.
    2. Compute importance scores s_j = sum_i G[i,j]^2 on raw G.
    3. Select block I_k = TopK(s).
    4. Project all workers onto block: G_block = G[:, I_k].
    5. Compute full-dimensional residual Δ_t = G - G_block_full.
    6. Update memory m_{t+1} = m_t + (1/b) sum_i Δ_t[i].
    7. Form augmented block gradients G_block_aug = G_block + m_t[I_k].
    8. Compute geometric median in R^k.
    9. Lift back to R^d by placing the block GM into coordinates I_k.
    """
    start = time.perf_counter()
    n = cfg["n_workers"]

    # Initialize memory
    if memory is None:
        memory = torch.zeros_like(base_grad)

    # ===== 1. Build worker gradient matrix G (raw, no memory) =====
    grads = []
    for _ in range(n):
        g = base_grad.clone()
        # worker-level small noise
        g = g + cfg["small_noise"] * torch.randn_like(base_grad)
        # heavy corruption with probability corr_frac
        if torch.rand(()) < cfg["corr_frac"]:
            g = g + cfg["noise_scale"] * torch.randn_like(base_grad)
        grads.append(g)

    G = torch.stack(grads, dim=0)  # (n, d)

    # ===== 2. Importance scores on raw gradients (BGMD-style) =====
    scores = (G ** 2).sum(dim=0)  # (d,)
    mask = select_block(scores, cfg["beta"])  # boolean mask of size d
    # effective dimension k
    # k = mask.sum().item()

    # ===== 3. Project onto selected block =====
    G_block = G[:, mask]  # (n, k)

    # ===== 4. Residuals in full dimension =====
    G_block_full = torch.zeros_like(G)
    G_block_full[:, mask] = G_block
    Delta = G - G_block_full  # (n, d)

    # ===== 5. Memory update (BGMD-style, no decay) =====
    # m_{t+1} = m_t + (1/b) sum_i Δ_t[i]
    Delta_mean = Delta.mean(dim=0)  # (d,)
    memory = memory + Delta_mean

    # ===== 6. Augment block gradients with memory =====
    # G_block_aug[i] = G_block[i] + m_t[mask]
    G_block_aug = G_block + memory[mask].unsqueeze(0)  # broadcast (1,k) over n

    # ===== 7. Geometric median in the k-dimensional block =====
    gm_block = geometric_median(G_block_aug)  # (k,)

    # ===== 8. Lift block GM back to full dimension =====
    full = torch.zeros_like(base_grad)
    full[mask] = gm_block

    timing["agg"] += time.perf_counter() - start
    return full, memory

# ============================================================
# 7. Evaluation
# ============================================================

def evaluate(model, loader):
    """
    Compute classification accuracy on a given DataLoader.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ============================================================
# 8. Training loop
# ============================================================

def train_method(name, aggregator, cfg):
    model = make_model()
    opt = optim.SGD(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    hist = {"loss": [], "acc": []}
    timing = {"agg": 0.0}

    memory = None
    start_total = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()

            base_grad = get_flat_grad(model)
            out_grad, memory = aggregator(base_grad, memory, cfg, timing)
            set_flat_grad(model, out_grad)
            opt.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        acc = evaluate(model, test_loader)
        hist["loss"].append(avg_loss)
        hist["acc"].append(acc)

        print(f"[{name}] Epoch {epoch}/{EPOCHS} | Loss={avg_loss:.4f} | Acc={acc:.4f}")

    total_time = time.perf_counter() - start_total
    print(f"[{name}] Total wall-clock time: {total_time:.2f} s "
          f"(aggregation: {timing['agg']:.2f} s)")

    return {"hist": hist, "timing": timing, "total_time": total_time}

# ============================================================
# 9. Run all experiments
# ============================================================

cfg_base = {
    "n_workers": N_WORKERS,
    "corr_frac": WORKER_CORR_FRAC,
    "noise_scale": WORKER_NOISE_SCALE,
    "small_noise": SMALL_NOISE_SCALE,
    "mem_decay": MEMORY_DECAY,   # not used by blockmem_aggregator now
}

print("\n=== SGM ===")
sgm_r = train_method("SGM", sgm_aggregator, dict(cfg_base))

print("\n=== GM-SGM (full) ===")
gm_r = train_method("GM-SGM", gmsgm_aggregator, dict(cfg_base))

block_r = {}
for beta in BETA_LIST:
    print(f"\n=== Block+Mem β={beta} ===")
    cfg_b = dict(cfg_base)
    cfg_b["beta"] = beta
    r = train_method(f"Block+Mem β={beta}", blockmem_aggregator, cfg_b)
    block_r[beta] = r

# ============================================================
# 10. Plots for paper (accuracy, loss, runtime)
# ============================================================

epochs = np.arange(1, EPOCHS + 1)
cmap = cm.get_cmap("viridis", len(BETA_LIST))

# Accuracy plot
plt.figure(figsize=(8, 6))
plt.plot(epochs, sgm_r["hist"]["acc"], label="SGM", color="gray",
         linestyle="--", linewidth=2.5)
plt.plot(epochs, gm_r["hist"]["acc"], label="GM-SGM", color="blue",
         linewidth=2.8)

for i, beta in enumerate(BETA_LIST):
    plt.plot(
        epochs,
        block_r[beta]["hist"]["acc"],
        label=f"Block+Mem β={beta}",
        color=cmap(i),
        linewidth=2,
    )

plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Epochs\nCNN + Gradient Corruption")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/acc_multi_beta.png", dpi=300)
plt.close()

# Loss plot
plt.figure(figsize=(8, 6))
plt.plot(epochs, sgm_r["hist"]["loss"], label="SGM", color="gray",
         linestyle="--", linewidth=2.5)
plt.plot(epochs, gm_r["hist"]["loss"], label="GM-SGM", color="blue",
         linewidth=2.8)

for i, beta in enumerate(BETA_LIST):
    plt.plot(
        epochs,
        block_r[beta]["hist"]["loss"],
        label=f"Block+Mem β={beta}",
        color=cmap(i),
        linewidth=2,
    )

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epochs")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/loss_multi_beta.png", dpi=300)
plt.close()

# Runtime vs beta
agg_times = [block_r[b]["timing"]["agg"] for b in BETA_LIST]
plt.figure(figsize=(7, 5))
plt.plot(
    BETA_LIST,
    agg_times,
    marker="o",
    linewidth=2.5,
    label="Block+Mem aggregation time",
)
plt.axhline(
    gm_r["timing"]["agg"],
    label="GM-SGM aggregation time",
    linestyle="--",
)

plt.xlabel("β")
plt.ylabel("Aggregation Time (s)")
plt.title("Aggregation Runtime vs Block Fraction β")
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/agg_time_vs_beta.png", dpi=300)
plt.close()

print("\nAll plots saved in ./results/")
