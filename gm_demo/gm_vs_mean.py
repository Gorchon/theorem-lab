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