import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Geometric Median (Weiszfeld) Algorithm
def geometric_median(X, tol=1e-6, max_iter=1000):
    y = np.median(X, axis=0)  # stable init
    for _ in range(max_iter):
        diff = X - y
        dist = np.linalg.norm(diff, axis=1)
        eps = 1e-12
        w = 1.0 / np.maximum(dist, eps)
        y_new = (X * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


# Data
rng = np.random.default_rng(42)

def make_data(n_inliers=300, n_outliers=0, outlier_shift=(20.0, 20.0), inlier_cov=1.0):
    inliers = rng.normal(0.0, inlier_cov, size=(n_inliers, 2))
    outliers = np.array(outlier_shift)[None, :] + rng.normal(0.0, 1.0, size=(n_outliers, 2))
    return np.vstack([inliers, outliers])

def summarize(X, true_center=np.array([0.0, 0.0])):
    mean = X.mean(axis=0)
    gm = geometric_median(X)
    mean_err = float(np.linalg.norm(mean - true_center))
    gm_err = float(np.linalg.norm(gm - true_center))
    return mean, gm, mean_err, gm_err


# Scenario 1: Clean 
X_clean = make_data(n_inliers=300, n_outliers=0, inlier_cov=1.0)
mean_c, gm_c, err_mean_c, err_gm_c = summarize(X_clean)

plt.figure(figsize=(6,6))
plt.scatter(X_clean[:,0], X_clean[:,1], s=10, alpha=0.5, label="points")
plt.scatter([0],[0], marker="x", s=100, label="true center")
plt.scatter([mean_c[0]], [mean_c[1]], marker="o", s=80, label="mean")
plt.scatter([gm_c[0]], [gm_c[1]], marker="^", s=100, label="geometric median")
plt.title("Clean data (no outliers)")
plt.legend(); plt.xlabel("x1"); plt.ylabel("x2")
plt.tight_layout()
plt.savefig("clean.png", dpi=160)


# Scenario 2: With outliers 
X_out = make_data(n_inliers=300, n_outliers=130, outlier_shift=(25.0, 25.0), inlier_cov=1.0)
mean_o, gm_o, err_mean_o, err_gm_o = summarize(X_out)

plt.figure(figsize=(6,6))
plt.scatter(X_out[:,0], X_out[:,1], s=10, alpha=0.5, label="points")
plt.scatter([0],[0], marker="x", s=100, label="true center")
plt.scatter([mean_o[0]], [mean_o[1]], marker="o", s=80, label="mean")
plt.scatter([gm_o[0]], [gm_o[1]], marker="^", s=100, label="geometric median")
plt.title("Corrupted data (30% outliers far away)")
plt.legend(); plt.xlabel("x1"); plt.ylabel("x2")
plt.tight_layout()
plt.savefig("corrupted.png", dpi=160)

#  Numeric summary 
summary = pd.DataFrame({
    "scenario": ["clean", "corrupted (30% outliers)"],
    "mean (x1,x2)": [tuple(np.round(mean_c,3)), tuple(np.round(mean_o,3))],
    "gm (x1,x2)": [tuple(np.round(gm_c,3)), tuple(np.round(gm_o,3))],
    "mean error vs true center": [round(err_mean_c,4), round(err_mean_o,4)],
    "gm error vs true center": [round(err_gm_c,4), round(err_gm_o,4)],
})
print(summary.to_string(index=False))
summary.to_csv("summary.csv", index=False)
print("\nSaved plots: clean.png, corrupted.png; table: summary.csv")