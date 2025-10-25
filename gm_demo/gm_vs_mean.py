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