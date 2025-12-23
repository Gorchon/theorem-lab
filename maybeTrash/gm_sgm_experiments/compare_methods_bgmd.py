import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 1. PROBLEM DEFINITION
# ============================================================

class QuadraticConstraintProblem:
    """
    Simple high-dimensional quadratic objective with a linear constraint:

        f(w) = 1/2 * ||w - w_star||^2
        g(w) = a^T w - c <= 0

    We choose c so that:
      - w_star is feasible
      - w0 is infeasible
    """

    def __init__(self, dim: int, rng: np.random.Generator):
        self.d = dim
        self.rng = rng

        # True optimum
        self.w_star = rng.normal(size=dim)

        # Initial point (infeasible)
        self.w0 = rng.normal(size=dim)

        # Constraint direction
        a = rng.normal(size=dim)
        a /= np.linalg.norm(a)
        self.a = a

        # Adjust c so that:
        # g(w_star) < 0, g(w0) > 0
        s_star = self.a.dot(self.w_star)
        s0 = self.a.dot(self.w0)

        if s0 <= s_star:
            self.a = -self.a
            s_star = -s_star
            s0 = -s0

        delta = 0.5 * (s0 - s_star)
        self.c = s_star + delta

    def f(self, w):
        return 0.5 * np.sum((w - self.w_star)**2)

    def grad_f(self, w):
        return w - self.w_star

    def g(self, w):
        return float(self.a.dot(w) - self.c)

    def grad_g(self, w):
        return self.a.copy()


# ============================================================
# 2. GEOMETRIC MEDIAN (Weiszfeld)
# ============================================================

def geometric_median(X, eps=1e-6, max_iter=100):
    X = np.asarray(X, float)
    y = X.mean(axis=0)
    for _ in range(max_iter):
        diff = X - y
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, eps)
        w = 1.0 / dist
        y_new = (w[:, None] * X).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            break
        y = y_new
    return y


# ============================================================
# 3. CORRUPTED WORKER GRADIENT GENERATOR
# ============================================================

def generate_worker_gradients(problem, w, rng, n_workers,
                              corrupt_indices, noise_std, corr_scale):
    d = problem.d
    true_grad_f = problem.grad_f(w)
    true_grad_g = problem.grad_g(w)
    g_true = problem.g(w)

    grads_f = np.zeros((n_workers, d))
    grads_g = np.zeros((n_workers, d))
    g_vals = np.zeros(n_workers)

    for i in range(n_workers):
        if i in corrupt_indices:
            grads_f[i] = true_grad_f + corr_scale * rng.normal(size=d)
            grads_g[i] = true_grad_g + corr_scale * rng.normal(size=d)
            g_vals[i] = g_true + corr_scale * rng.normal()
        else:
            grads_f[i] = true_grad_f + noise_std * rng.normal(size=d)
            grads_g[i] = true_grad_g + noise_std * rng.normal(size=d)
            g_vals[i] = g_true + noise_std * rng.normal()

    return grads_f, grads_g, g_vals


# ============================================================
# 4. OPTIMIZATION METHODS
# ============================================================

# -----------------------------
# 4.1 Raw SGM (mean)
# -----------------------------
def run_sgm_mean(problem, steps, lr, n_workers,
                 corruption_frac, noise_std, corr_scale, seed):

    rng = np.random.default_rng(seed)
    w = problem.w0.copy()

    n_corrupt = int(corruption_frac * n_workers)
    corrupt_indices = set(rng.choice(n_workers, size=n_corrupt, replace=False))

    errors, viols, traj2d = [], [], []
    t0 = time.time()

    for _ in range(steps):
        grads_f, grads_g, g_vals = generate_worker_gradients(
            problem, w, rng, n_workers, corrupt_indices, noise_std, corr_scale
        )

        grad_f_hat = grads_f.mean(axis=0)
        grad_g_hat = grads_g.mean(axis=0)
        g_hat = g_vals.mean()

        direction = grad_f_hat if g_hat <= 0 else grad_g_hat
        w = w - lr * direction

        errors.append(np.linalg.norm(w - problem.w_star))
        viols.append(max(problem.g(w), 0.0))
        traj2d.append(w[:2].copy())

    return {
        "name": "SGM (mean)",
        "errors": np.array(errors),
        "viols": np.array(viols),
        "traj2d": np.array(traj2d),
        "runtime": time.time() - t0
    }


# -----------------------------
# 4.2 Full GM-SGM
# -----------------------------
def run_gm_sgm_full(problem, steps, lr, n_workers,
                    corruption_frac, noise_std, corr_scale, seed):

    rng = np.random.default_rng(seed)
    w = problem.w0.copy()

    n_corrupt = int(corruption_frac * n_workers)
    corrupt_indices = set(rng.choice(n_workers, size=n_corrupt, replace=False))

    errors, viols, traj2d = [], [], []
    t0 = time.time()

    for _ in range(steps):
        grads_f, grads_g, g_vals = generate_worker_gradients(
            problem, w, rng, n_workers, corrupt_indices, noise_std, corr_scale
        )

        grad_f_hat = geometric_median(grads_f)
        grad_g_hat = geometric_median(grads_g)
        g_hat = np.median(g_vals)

        direction = grad_f_hat if g_hat <= 0 else grad_g_hat
        w = w - lr * direction

        errors.append(np.linalg.norm(w - problem.w_star))
        viols.append(max(problem.g(w), 0.0))
        traj2d.append(w[:2].copy())

    return {
        "name": "GM-SGM (full GM)",
        "errors": np.array(errors),
        "viols": np.array(viols),
        "traj2d": np.array(traj2d),
        "runtime": time.time() - t0
    }


# -----------------------------
# 4.3 GM-SGM + Block + Memory (Error Feedback)
# -----------------------------
def run_gm_sgm_block_memory(problem, steps, lr, n_workers,
                            corruption_frac, noise_std, corr_scale,
                            block_k, seed):

    rng = np.random.default_rng(seed)
    w = problem.w0.copy()
    d = problem.d
    memory = np.zeros(d)

    n_corrupt = int(corruption_frac * n_workers)
    corrupt_indices = set(rng.choice(n_workers, size=n_corrupt, replace=False))

    errors, viols, traj2d = [], [], []
    t0 = time.time()

    for _ in range(steps):
        grads_f, grads_g, g_vals = generate_worker_gradients(
            problem, w, rng, n_workers, corrupt_indices, noise_std, corr_scale
        )

        # Block coordinate selection
        importance = (grads_f ** 2).sum(axis=0)
        k = min(block_k, d)
        idx_block = np.argpartition(importance, -k)[-k:]

        # GM on block (with memory)
        grads_block = grads_f[:, idx_block] + memory[idx_block]
        gm_block = geometric_median(grads_block)

        grad_f_hat = np.zeros(d)
        grad_f_hat[idx_block] = gm_block

        # Full GM for constraint
        grad_g_hat = geometric_median(grads_g)
        g_hat = np.median(g_vals)

        direction = grad_f_hat if g_hat <= 0 else grad_g_hat
        w_new = w - lr * direction

        # Memory update
        true_grad_f = problem.grad_f(w)
        residual = true_grad_f - direction
        memory = 0.9 * memory + 0.1 * residual

        w = w_new

        errors.append(np.linalg.norm(w - problem.w_star))
        viols.append(max(problem.g(w), 0.0))
        traj2d.append(w[:2].copy())

    return {
        "name": f"GM-SGM (block+memory, k={block_k})",
        "errors": np.array(errors),
        "viols": np.array(viols),
        "traj2d": np.array(traj2d),
        "runtime": time.time() - t0
    }


# ============================================================
# 5. RUNTIME SCALING EXPERIMENT
# ============================================================

def scaling_runtime_experiment(
    dims_list,
    steps,
    lr,
    n_workers,
    corruption_frac,
    noise_std,
    corr_scale,
    block_k_frac,
    seed_base=100,
):
    runtimes_sgm = []
    runtimes_full = []
    runtimes_block = []

    for idx, d in enumerate(dims_list):
        print(f"\nRunning dimension {d}...")

        rng_problem = np.random.default_rng(seed_base + idx)
        problem = QuadraticConstraintProblem(d, rng_problem)

        block_k = max(10, int(block_k_frac * d))

        res_sgm = run_sgm_mean(
            problem, steps, lr, n_workers, corruption_frac,
            noise_std, corr_scale, seed=seed_base + idx + 1
        )
        res_full = run_gm_sgm_full(
            problem, steps, lr, n_workers, corruption_frac,
            noise_std, corr_scale, seed=seed_base + idx + 2
        )
        res_block = run_gm_sgm_block_memory(
            problem, steps, lr, n_workers, corruption_frac,
            noise_std, corr_scale, block_k, seed=seed_base + idx + 3
        )

        runtimes_sgm.append(res_sgm["runtime"])
        runtimes_full.append(res_full["runtime"])
        runtimes_block.append(res_block["runtime"])

    return {
        "dims": dims_list,
        "sgm": runtimes_sgm,
        "full_gm": runtimes_full,
        "block_gm": runtimes_block
    }


def plot_runtime_scaling(runtimes, save_path):
    dims = runtimes["dims"]

    plt.figure(figsize=(7, 5))
    plt.plot(dims, runtimes["sgm"], 'o-', linewidth=2, label="SGM (mean)")
    plt.plot(dims, runtimes["full_gm"], 'o-', linewidth=2, label="GM-SGM (full GM)")
    plt.plot(dims, runtimes["block_gm"], 'o-', linewidth=2, label="GM-SGM (block+memory)")

    plt.xlabel("Dimension d")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Scaling with Dimension")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf")
    plt.close()


# ============================================================
# 6. PLOTTING UTILITIES
# ============================================================

def make_2d_grid(problem, results, n_grid=200):
    all_xy = np.vstack([r["traj2d"] for r in results])
    x_min, x_max = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_min, y_max = all_xy[:, 1].min(), all_xy[:, 1].max()

    x_margin = 0.1 * (x_max - x_min + 1e-8)
    y_margin = 0.1 * (y_max - y_min + 1e-8)

    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    X, Y = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                       np.linspace(y_min, y_max, n_grid))

    Z = np.zeros_like(X)
    fixed = problem.w_star.copy()

    for i in range(n_grid):
        for j in range(n_grid):
            fixed[0] = X[i, j]
            fixed[1] = Y[i, j]
            Z[i, j] = problem.f(fixed)

    return X, Y, Z


def plot_3d_surface(X, Y, Z, problem, save_path):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
    ax.scatter(problem.w_star[0], problem.w_star[1],
               problem.f(problem.w_star), s=150, c='red', marker='*')

    ax.set_title("3D Surface of f(w1, w2)")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_zlabel("f")
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf")
    plt.close()


def plot_contour(X, Y, Z, res, problem, save_path):
    traj = res["traj2d"]

    plt.figure(figsize=(6, 5))
    CS = plt.contour(X, Y, Z, levels=15)
    plt.clabel(CS, inline=True, fontsize=8)

    plt.plot(traj[:, 0], traj[:, 1], linewidth=2)
    plt.scatter(traj[0, 0], traj[0, 1], s=80, marker='o')
    plt.scatter(problem.w_star[0], problem.w_star[1], s=150, marker='*')

    plt.title(res["name"])
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf")
    plt.close()


def plot_all_contours(X, Y, Z, results, problem, save_path):
    plt.figure(figsize=(6, 5))
    CS = plt.contour(X, Y, Z, levels=15)
    plt.clabel(CS, inline=True, fontsize=8)

    for r in results:
        plt.plot(r["traj2d"][:, 0], r["traj2d"][:, 1], linewidth=2, label=r["name"])

    plt.scatter(problem.w_star[0], problem.w_star[1], s=150, marker='*')
    plt.title("Combined Trajectories")
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf")
    plt.close()


def plot_errors(results, save_path):
    T = len(results[0]["errors"])
    iters = np.arange(T)

    plt.figure(figsize=(6, 4))
    for r in results:
        plt.semilogy(iters, r["errors"], linewidth=2, label=r["name"])

    plt.title("Distance to Optimum")
    plt.xlabel("Iteration")
    plt.ylabel("||w - w*||")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf")
    plt.close()


def plot_runtime(results, save_path):
    names = [r["name"] for r in results]
    times = [r["runtime"] for r in results]

    plt.figure(figsize=(6, 4))
    plt.bar(names, times)
    plt.title("Runtime Comparison")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(save_path + ".png", dpi=300)
    plt.savefig(save_path + ".pdf")
    plt.close()


# ============================================================
# 7. MAIN
# ============================================================

def main():
    # ------------------------
    # Experiment settings
    # ------------------------
    dim = 200
    steps = 300
    lr = 0.1
    n_workers = 11
    corruption_frac = 0.3
    noise_std = 0.05
    corr_scale = 10.0
    block_k = 40

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    rng = np.random.default_rng(0)
    problem = QuadraticConstraintProblem(dim, rng)

    # ------------------------
    # Run main three methods
    # ------------------------
    res_sgm = run_sgm_mean(problem, steps, lr, n_workers,
                           corruption_frac, noise_std, corr_scale, seed=1)
    res_full = run_gm_sgm_full(problem, steps, lr, n_workers,
                               corruption_frac, noise_std, corr_scale, seed=2)
    res_block = run_gm_sgm_block_memory(problem, steps, lr, n_workers,
                                        corruption_frac, noise_std, corr_scale,
                                        block_k, seed=3)

    results = [res_sgm, res_full, res_block]

    # ------------------------
    # Create surface/contours
    # ------------------------
    X, Y, Z = make_2d_grid(problem, results)

    plot_3d_surface(X, Y, Z, problem,
                    os.path.join(save_dir, "surface3d"))

    plot_contour(X, Y, Z, res_sgm, problem,
                 os.path.join(save_dir, "traj_sgm"))
    plot_contour(X, Y, Z, res_full, problem,
                 os.path.join(save_dir, "traj_full"))
    plot_contour(X, Y, Z, res_block, problem,
                 os.path.join(save_dir, "traj_block"))

    plot_all_contours(X, Y, Z, results, problem,
                      os.path.join(save_dir, "traj_combined"))

    plot_errors(results, os.path.join(save_dir, "error_vs_iter"))
    plot_runtime(results, os.path.join(save_dir, "runtime_comparison"))

    # ------------------------
    # Print summary in console
    # ------------------------
    print("\nFINAL DISTANCES:")
    for r in results:
        print(f"{r['name']}: {r['errors'][-1]:.4f}")

    print("\nRUNTIMES:")
    for r in results:
        print(f"{r['name']}: {r['runtime']:.4f} s")

    # =========================================================
    # RUNTIME SCALING EXPERIMENT
    # =========================================================
    dims_list = [1000, 2000, 4000, 6000, 8000, 10000, 15000, 20000]

    runtimes = scaling_runtime_experiment(
        dims_list=dims_list,
        steps=40,
        lr=lr,
        n_workers=n_workers,
        corruption_frac=corruption_frac,
        noise_std=noise_std,
        corr_scale=corr_scale,
        block_k_frac=0.1
    )

    plot_runtime_scaling(runtimes, os.path.join(save_dir, "runtime_scaling"))

    print("\nSCALING EXPERIMENT COMPLETE. See results/runtime_scaling.*")


if __name__ == "__main__":
    main()
