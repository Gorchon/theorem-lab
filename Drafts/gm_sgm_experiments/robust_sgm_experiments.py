import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ============================================================
#  GLOBAL CONFIG
# ============================================================

os.makedirs("results", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Using device:", device)

# ============================================================
#  SECTION 1 — UTILITIES
# ============================================================

def flatten_params(params):
    """Flatten all gradient tensors of a model into a single vector."""
    flat_list = []
    for p in params:
        if p.grad is None:
            flat_list.append(torch.zeros_like(p).view(-1))
        else:
            flat_list.append(p.grad.view(-1))
    return torch.cat(flat_list)


def assign_flat_to_params(params, flat):
    """Assign a flat gradient vector back into model parameters (NO IN-PLACE OPS)."""
    idx = 0
    for p in params:
        num = p.numel()
        new_grad = flat[idx: idx + num].view_as(p)
        p.grad = new_grad.clone()        # NOT in-place
        idx += num


# ============================================================
#  SECTION 2 — GEOMETRIC MEDIAN (Weiszfeld)
# ============================================================

def geometric_median(vectors, eps=1e-5, max_iter=20):
    """
    Robust geometric median of a list of 1D tensors.
    Uses Weiszfeld's algorithm.
    """
    V = torch.stack(vectors, dim=0)
    y = V.mean(dim=0)

    for _ in range(max_iter):
        diffs = V - y
        dist = diffs.norm(dim=1).clamp(min=1e-8)
        w = 1.0 / dist
        w = w / w.sum()
        y_new = (w.unsqueeze(1) * V).sum(dim=0)

        if torch.norm(y_new - y) < eps:
            break
        y = y_new
    return y


# ============================================================
#  SECTION 3 — BLOCK COORDINATE SELECTION + MEMORY
# ============================================================

def topk_block_selection(worker_matrix, k):
    """
    worker_matrix: tensor (n_workers, d)
    Selects top-k "energetic" coordinates by sum of squares.
    """
    scores = (worker_matrix ** 2).sum(dim=0)
    _, idx = torch.topk(scores, k)
    return idx


def compress_block(g, idx, d):
    """
    Zero all coordinates except those in idx.
    """
    out = torch.zeros(d, device=g.device)
    out[idx] = g[idx]
    return out


# ============================================================
#  SECTION 4 — CORRUPTION MODELS (no in-place ops)
# ============================================================

def corrupt_sgm_elementwise(params, p=0., scale=10.0):
    """Gross element-wise corruption for SGM baseline."""
    for p_param in params:
        if p_param.grad is None:
            continue
        mask = (torch.rand_like(p_param.grad) < p)
        noise = scale * torch.randn_like(p_param.grad)
        p_param.grad = p_param.grad + mask * noise   # no in-place


def gm_sgm_worker_aggregation(params, n_workers=11, frac_corr=0.3, scale=10.0):
    """Full GM-SGM: simulate corrupted workers + GM aggregation."""
    base = flatten_params(params)
    d = base.numel()

    workers = []
    for _ in range(n_workers):
        g = base.clone()
        if random.random() < frac_corr:
            g = scale * torch.randn(d, device=g.device)
        workers.append(g)

    gm = geometric_median(workers)
    assign_flat_to_params(params, gm)


def gm_sgm_block_memory(params, memory, n_workers=11,
                        frac_corr=0.3, scale=10.0,
                        block_k=2000):
    """
    GM-SGM but:
    - Use block coordinate selection
    - Use memory mechanism
    """
    base = flatten_params(params)
    combo = base + memory
    d = combo.numel()

    # Create worker gradients
    workers = []
    for _ in range(n_workers):
        g = combo.clone()
        if random.random() < frac_corr:
            g = scale * torch.randn(d, device=g.device)
        workers.append(g)

    W = torch.stack(workers, dim=0)  # (n_workers, d)

    k = min(block_k, d)
    idx = topk_block_selection(W, k)

    # Compress workers
    Wc = torch.stack([w[idx] for w in W], dim=0)

    gm_block = geometric_median([v for v in Wc])

    full_update = torch.zeros(d, device=combo.device)
    full_update[idx] = gm_block

    # Assign back
    assign_flat_to_params(params, full_update)

    # Memory update (OUT OF PLACE)
    memory = combo - full_update

    return memory


# ============================================================
#  SECTION 5 — MODEL (Fashion-MNIST MLP)
# ============================================================

def make_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)


# ============================================================
#  SECTION 6 — FASHION-MNIST EXPERIMENT
# ============================================================

def run_fashion_mnist():
    BATCH = 128
    EPOCHS = 5
    LR = 1e-3

    TARGET = 7        # Sneaker
    KAPPA = 0.3
    LAMBDA = 4.0

    # Corruption settings
    SGM_P = 0.3
    SGM_SCALE = 10.0

    N_WORKERS = 11
    FRAC_CORR = 0.3
    WORKER_SCALE = 10.0
    BLOCK_K = 2000

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST("./data", train=True,
                                  download=True, transform=transform)
    loader = torch.utils.data.DataLoader(train, batch_size=BATCH, shuffle=True)

    logs = {
        "sgm": {"target": [], "penalty": []},
        "gmsgm": {"target": [], "penalty": []},
        "block": {"target": [], "penalty": []},
    }

    def train_method(name):
        model = make_mlp()
        opt = optim.Adam(model.parameters(), lr=LR)
        memory = None

        for epoch in range(EPOCHS):
            total_t = 0.0
            total_p = 0.0
            steps = 0

            for x, y in loader:
                x, y = x.to(device), y.to(device)

                opt.zero_grad()
                logits = model(x)
                loss_all = nn.CrossEntropyLoss(reduction='none')(logits, y)

                # per-class loss
                per_class = torch.zeros(10, device=device)
                for c in range(10):
                    mask = (y == c)
                    if mask.any():
                        per_class[c] = loss_all[mask].mean()

                obj = per_class[TARGET]
                viol = torch.relu(per_class - KAPPA)
                viol[TARGET] = 0.0
                penalty = viol.max()
                total_loss = obj + LAMBDA * penalty

                total_loss.backward()

                # Apply corruption + aggregation
                if name == "sgm":
                    corrupt_sgm_elementwise(model.parameters(), p=SGM_P, scale=SGM_SCALE)
                    opt.step()

                elif name == "gmsgm":
                    gm_sgm_worker_aggregation(model.parameters(),
                                              n_workers=N_WORKERS,
                                              frac_corr=FRAC_CORR,
                                              scale=WORKER_SCALE)
                    opt.step()

                elif name == "block":
                    if memory is None:
                        d = sum(p.numel() for p in model.parameters())
                        memory = torch.zeros(d, device=device)

                    memory = gm_sgm_block_memory(model.parameters(),
                                                 memory,
                                                 n_workers=N_WORKERS,
                                                 frac_corr=FRAC_CORR,
                                                 scale=WORKER_SCALE,
                                                 block_k=BLOCK_K)
                    opt.step()

                total_t += float(obj.item())
                total_p += float(penalty.item())
                steps += 1

            logs[name]["target"].append(total_t / steps)
            logs[name]["penalty"].append(total_p / steps)

            print(f"[MNIST] {name.upper()} Epoch {epoch+1} — "
                  f"Tgt={logs[name]['target'][-1]:.4f}, "
                  f"Pen={logs[name]['penalty'][-1]:.4f}")

    # Train all 3 methods
    train_method("sgm")
    train_method("gmsgm")
    train_method("block")

    # Plot
    for m in ["target", "penalty"]:
        plt.figure(figsize=(9,6))
        for name in logs:
            plt.plot(logs[name][m], label=name)
        plt.title(f"Fashion-MNIST: {m}")
        plt.xlabel("Epoch")
        plt.ylabel(m)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/mnist_{m}.png")
        plt.close()

    return logs


# ============================================================
#  SECTION 7 — SYNTHETIC TIMING BENCHMARK
# ============================================================

def timing_test():
    dims = [1000, 5000, 20000, 50000, 100000]
    n_workers = 11
    frac_corr = 0.3
    scale = 10.0
    block_k = 2000

    t_sgm = []
    t_gm = []
    t_block = []

    for d in dims:
        base = torch.randn(d, device=device)
        workers = [base.clone() for _ in range(n_workers)]
        memory = torch.zeros(d, device=device)

        # SGM (mean)
        t0 = time.time()
        _ = sum(workers) / len(workers)
        t_sgm.append(time.time() - t0)

        # Full GM time
        t0 = time.time()
        geometric_median(workers)
        t_gm.append(time.time() - t0)

        # Block + Memory
        W = torch.stack(workers, dim=0)
        idx = topk_block_selection(W, min(block_k, d))
        Wc = torch.stack([w[idx] for w in W], dim=0)

        t0 = time.time()
        geometric_median([v for v in Wc])
        t_block.append(time.time() - t0)

        print(f"[Timing] d={d}: SGM={t_sgm[-1]:.4f}s, GM={t_gm[-1]:.4f}s, Block={t_block[-1]:.4f}s")

    plt.figure(figsize=(10,6))
    plt.plot(dims, t_sgm, "o-", label="SGM (mean)")
    plt.plot(dims, t_gm, "o-", label="GM-SGM (full GM)")
    plt.plot(dims, t_block, "o-", label="GM-SGM Block+Memory")
    plt.xlabel("Dimension")
    plt.ylabel("Seconds")
    plt.title("Runtime Scaling vs Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/timing_scaling.png")
    plt.close()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    print("=== Running Fashion-MNIST experiment ===")
    run_fashion_mnist()

    print("=== Running Synthetic Timing Benchmark ===")
    timing_test()

    print("All results saved to ./results/")
