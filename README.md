# BHCF-EPRS: Bayesian Hybrid Collaborative Filtering - Electricity Plan Recommender System

**Final Project - Recommender Systems Course**  
  Muhammed Miraç Taştan  
Based on: Zhang et al., IEEE Trans. Ind. Informat., Vol. 15, No. 8, Aug. 2019

---

## Project Overview

BHCF-EPRS recommends the cheapest residential electricity plan using only appliance usage estimates - no historical bills or smart-meter data required. It implements 10 core concepts: feature extraction, BPMF/ALS imputation, PCC weighting, SVM classification, plan rating extraction, weighted cosine similarity, constrained KNN, memory-based CF, model-based CF, and hybrid CF.

**Key result:** Top-5 Recommendation Precision = **1.00** for alpha >= 0.1 on a 200-user synthetic dataset.

---

## Deliverables

| File | Description |
|------|-------------|
| `BHCF_EPRS_Final.py` | Main Python code (all 10 concepts, all 12 figures) |
| `Report_1.docx` | Full project report (Abstract, Introduction, Related Work, Proposed Approach, Experiment Design, Performance Analysis, Conclusion, References) |
| `Report_2.docx` | One-page summary of differences between midterm and final project |
| `Report_3.docx` | Line-by-line, section-by-section code explanation with output interpretation |
| `README.md` | This file |
| `figures/` | Output directory (created automatically when code runs) |

---

## System Requirements

- **Python:** 3.8 or higher
- **Operating System:** Windows 10/11, macOS 10.15+, or Ubuntu 20.04+
- **RAM:** >= 4 GB recommended (200x200 similarity matrix is computed in memory)
- **Disk:** >= 200 MB free (for figures and dependencies)

---

## Installation

### Step 1 - Install Python (if not already installed)

Download and install Python 3.8+ from https://python.org/downloads/  
Verify installation:
```bash
python --version
# or on some systems:
python3 --version
```

### Step 2 - (Optional but recommended) Create a virtual environment

```bash
# Create a virtual environment named "bhcf_env"
python -m venv bhcf_env

# Activate it:
# Windows:
bhcf_env\Scripts\activate
# macOS/Linux:
source bhcf_env/bin/activate
```

### Step 3 - Install required packages

All dependencies can be installed with a single command:
```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

**Specific versions used during development (also work with newer versions):**
```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
```

To install exact versions:
```bash
pip install numpy==1.24.0 pandas==1.5.3 scipy==1.10.0 scikit-learn==1.2.0 matplotlib==3.6.0
```

### Step 4 - Verify installation

```python
python -c "import numpy, pandas, scipy, sklearn, matplotlib; print('All packages OK')"
```

---

## Running the Code

### Basic run

Navigate to the project directory and run:
```bash
python BHCF_EPRS_Final.py
```

### Expected output

The pipeline prints progress for each concept and the final results table:

```
=================================================================
  BHCF-EPRS - Final Project Pipeline
=================================================================

[1/10] Generating synthetic dataset …
  Users: 200 | Appliances: 12 | Missing: 887 (37.0%)

[C1] Feature extraction …
  Missing after discretisation: 887 (37.0%)
  → saved  figures/fig01_appliance_distributions.png

[C2] BPMF / ALS imputation …
  Train RMSE (final): 0.0151  | Test RMSE (final): 0.0134
  → saved  figures/fig02_bpmf_convergence.png

[C3] PCC weighting …
  Principal appliances (3 selected from top 9): [...]
  → saved  figures/fig03_pcc_weights.png

...

[C10] Hybrid CF evaluation …
   Alpha |     RMSE |  Top-5 Precision
----------------------------------------
  0.0    |  0.6515  |  0.78
  0.1    |  0.5880  |  1.00
  ...
  1.0    |  0.1105  |  0.99

=================================================================
  Pipeline complete.  All figures saved to  ./figures/
=================================================================
```

### Expected runtime

- Total: **60-120 seconds** on a standard laptop CPU (the similarity matrix computation is O(n²) and is the bottleneck)
- Steps C2 (200 ALS iterations) and C9 (300 ALS iterations) each take ~5-15 seconds

---

## Output Files

After running, the `figures/` directory will contain:

| Figure | File | What it shows |
|--------|------|---------------|
| Figure 1 | `fig01_appliance_distributions.png` | Appliance hour distributions with bin boundaries |
| Figure 2 | `fig02_bpmf_convergence.png` | ALS/BPMF train & test RMSE convergence |
| Figure 3 | `fig03_pcc_weights.png` | PCC bar chart with principal/excluded split |
| Figure 4 | `fig04_svm_results.png` | SVM cross-validation ERPS and class distribution |
| Figure 5 | `fig05_rating_heatmap.png` | Rating matrix heatmap and distribution |
| Figure 6 | `fig06_similarity_matrix.png` | Cosine similarity matrix block |
| Figure 7 | `fig07_knn_selection.png` | Neighbour counts and User 0 similarity ranking |
| Figure 8 | `fig08_memory_cf.png` | K sensitivity (K=1..15) and true vs predicted scatter |
| Figure 9 | `fig09_model_cf.png` | Model-based CF convergence and scatter |
| Figure 10 | `fig10_hybrid_cf.png` | Alpha sweep and model comparison |
| Figure 11 | `fig11_feature_class_analysis.png` | Appliance usage by consumption class |
| Figure 12 | `fig12_sensitivity_analysis.png` | Sensitivity to D, lambda, and alpha |

---

## Reproducibility

All results are fully reproducible:
- `numpy.random.seed(42)` is set at the top of the file (governs data generation, ALS initialisation, KFold shuffling)
- `numpy.random.seed(0)` is set inside `simulate_and_extract_ratings()` (governs plan pricing)
- No external data files are required - the dataset is generated from fixed parameters

Running the script twice produces identical outputs.

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'sklearn'"**  
→ Run `pip install scikit-learn` (note: the pip package name is `scikit-learn`, not `sklearn`)

**"cannot connect to X server" or display-related errors**  
→ This is expected in headless environments. The code uses `matplotlib.use("Agg")` to avoid this. If you still see this error, ensure you are not importing matplotlib elsewhere before the `use("Agg")` call.

**Script hangs at [C6] similarity matrix**  
→ The O(n²) loop is the computational bottleneck. On slow machines it can take up to 3-5 minutes. This is normal; the script will complete.

**"RuntimeWarning: invalid value encountered in double_scalars"**  
→ Suppressed by `warnings.filterwarnings("ignore")` in the code. Caused by edge cases in cosine similarity with near-zero vectors; the `+ 1e-9` epsilon handles these numerically.

---

## Code Structure Summary

```
BHCF_EPRS_Final.py
|
|-- Module imports & global constants
|-- generate_dataset()              - synthetic data with 37% missing values
|
|-- class BHCF_EPRS_Recommender
|   |-- __init__()
|   |-- extract_discrete_features() - Concept 1: quintile binning
|   |-- impute_missing_features()   - Concept 2: ALS/BPMF
|   |-- compute_pcc_weights()       - Concept 3: PCC ranking
|   |-- train_svm_classifier()      - Concept 4: RBF-SVM
|   |-- simulate_and_extract_ratings() - Concept 5: cost → ratings
|   |-- compute_similarity_matrix() - Concept 6: weighted cosine
|   |-- get_constrained_knn()       - Concept 7: class-constrained KNN
|   |-- predict_memory_based()      - Concept 8: similarity-weighted avg
|   |-- predict_model_based()       - Concept 9: ALS on rating matrix
|   |-- evaluate_hybrid_alphas()    - Concept 10: alpha sweep
|   |-- get_final_recommendations() - Concept 10: final Top-N output
|   `-- k_sensitivity()             - Extended K=1..15 analysis
|
|-- plot_fig01() ... plot_fig12()   - 12 figure generators
|
`-- main()                          - pipeline orchestrator
```

---

## References

1. Zhang et al., "Bayesian Hybrid Collaborative Filtering-Based Residential Electricity Plan Recommender System," IEEE Trans. Ind. Informat., Vol. 15, No. 8, 2019.
2. Aggarwal, C.C., Recommender Systems: The Textbook, Springer, 2016.


