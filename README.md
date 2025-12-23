
# theorem-lab

This repository contains **all experimental code and results** associated with our research on **robust optimization, Switching Gradient Methods (SGM), and Geometric Median–based extensions (GM-SGM)**.

The goal of this repo is to provide **full reproducibility** of the experiments discussed in our LaTeX paper, as well as a structured archive of exploratory and supporting experiments conducted throughout the project.

Each folder corresponds to a **specific set of experiments**, and **every folder contains both the code and the generated results (plots, logs, checkpoints)** for that experiment group.

---

##  Repository Structure

### `FashionMnistExperiments/`

Experiments conducted on the **Fashion-MNIST dataset**, focusing on:

* Constrained optimization using per-class loss constraints
* Soft vs hard switching dynamics
* Robust vs non-robust constraint aggregation (mean, median, etc.)
* Behavior under gradient corruption

This folder includes:

* Training scripts
* Saved logs and metrics
* Plots showing per-class loss evolution, constraint deviation, and convergence behavior

These experiments form a **core empirical section** of the paper.

---

### `GM-SGM_vs_SGM/`

Direct comparison experiments between:

* **SGM (Switching Gradient Method)**
* **GM-SGM (Geometric Median Switching Gradient Method)**

Focus areas:

* Robustness to corrupted gradients
* Failure modes of standard SGM
* Benefits of geometric median aggregation
* Interaction between switching dynamics and robust aggregation

This folder contains the **main comparative results** referenced in the paper.

---

### `GM_Robustness/`

Isolated robustness experiments studying the **geometric median** itself.

Includes experiments analyzing:

* Robustness to adversarial and stochastic corruption
* Effect of number of workers
* Sensitivity to corruption magnitude
* Comparison with mean aggregation

These experiments justify the use of geometric median aggregation in GM-SGM.

---

### `gross_corruption_model/clean_vs_corrupted/`

Early and supporting experiments analyzing **gross corruption models**, including:

* Clean vs corrupted gradient behavior
* Stress-testing optimization under extreme corruption
* Baseline failure cases

This folder mainly contains **preliminary and diagnostic experiments** that motivated later design choices.

---

### `Drafts/`

Work-in-progress experiments, scratch code, and exploratory ideas.

These files are **not part of the final paper results**, but are kept for completeness and future reference (IF NEEDED).

---

### `LatexBackUp/`

Backup copies of LaTeX sources related to the paper.

Includes:

* Older drafts
* Figures and plots used in the paper
* Supporting material for reproducibility

---


##  Reproducibility Notes

* Every experiment folder contains:

  * The exact Python scripts used to generate results
  * Saved plots and logs produced by those scripts
* All figures shown in the LaTeX paper are generated directly from code stored in this repository.
* Switching between experimental configurations (e.g., soft vs hard switching, mean vs median, SGM vs GM-SGM) is done via clearly defined flags inside the scripts.

---

##  Purpose of This Repository

This repository serves as:

* A **complete experimental archive** for the research project
* A **reproducibility companion** to the LaTeX paper
* A reference implementation for robust switching-based optimization methods


