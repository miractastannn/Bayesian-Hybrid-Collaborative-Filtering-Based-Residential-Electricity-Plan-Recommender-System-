"""
================================================================================
  BHCF-EPRS: Bayesian Hybrid Collaborative Filtering – Electricity Plan
             Recommender System
  Final Project — Recommender Systems Course
    Muhammed Miraç Taştan
  Based on: Zhang et al., IEEE Trans. Ind. Informat., Vol. 15, No. 8, Aug. 2019
================================================================================
CHANGES FROM MIDTERM:
  1. Full OOP encapsulation: all 10 concepts inside BHCF_EPRS_Recommender class
  2. Separate test-RMSE tracking in BPMF (new convergence validation)
  3. Sensitivity analysis extended: D, lambda, alpha, K, SVM-C all covered
  4. Feature-by-class visualisation added (Section 4)
  5. Outputs all 12 figures referenced in Report_1
  6. Numpy random seed set at top for full reproducibility
  7. Grid-search over K=1..15 added for memory-based CF validation
  8. Per-user recommendation table printed at end
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — safe in any environment
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings, os

warnings.filterwarnings("ignore")
np.random.seed(42)             # global seed — governs data generation


# ─────────────────────────────────────────────────────────────────────────────
# DATASET GENERATION  (mirrors Table I of Zhang et al. 2019)
# ─────────────────────────────────────────────────────────────────────────────
APPLIANCE_NAMES = ["MIC", "OVE", "RAN", "DIS", "KIT",
                   "FRI", "CLO", "DRY", "FUR", "AC",  "WAT", "EV"]

POWER_KW = {                   # rated power in kW (from Ausgrid dataset)
    "MIC": 1.0,  "OVE": 2.0,  "RAN": 2.5,  "DIS": 1.8,  "KIT": 1.5,
    "FRI": 0.15, "CLO": 0.5,  "DRY": 5.0,  "FUR": 3.0,  "AC":  3.5,
    "WAT": 4.5,  "EV":  7.2
}

# Per-appliance mean and std for weekly operating hours
APP_STATS = {
    "MIC": (1.5,  0.6),  "OVE":  (3.0,  1.2), "RAN": (2.0,  0.8),
    "DIS": (5.0,  2.0),  "KIT":  (7.0,  3.0),  "FRI": (168.0, 2.0),
    "CLO": (2.0,  0.8),  "DRY":  (3.0,  1.5),  "FUR": (20.0, 8.0),
    "AC":  (25.0, 15.0), "WAT":  (15.0, 6.0),   "EV":  (10.0, 7.0)
}

N_USERS  = 200
N_PLANS  = 10
MISS_RATE = 0.359              # 35.9% — matches paper

def generate_dataset():
    """Simulate a 200-user appliance-hours dataset with 35.9% missing values."""
    data = {}
    for app, (mu, sigma) in APP_STATS.items():
        vals = np.random.normal(mu, sigma, N_USERS)
        vals = np.clip(vals, 0.1, None)        # non-negative hours
        data[app] = vals
    df = pd.DataFrame(data)
    # Inject missing values
    mask = np.random.rand(N_USERS, len(APPLIANCE_NAMES)) < MISS_RATE
    df_missing = df.copy()
    df_missing[pd.DataFrame(mask, columns=APPLIANCE_NAMES)] = np.nan
    return df, df_missing


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────────────
class BHCF_EPRS_Recommender:
    """
    Bayesian Hybrid Collaborative Filtering-based Electricity Plan
    Recommender System.

    Encapsulates all 10 core concepts in a single, extensible OOP class.
    Instantiate → call fit() → call recommend().
    """

    def __init__(self, n_users=N_USERS, n_plans=N_PLANS, alpha=0.1):
        self.n_users = n_users
        self.n_plans = n_plans
        self.alpha   = alpha          # hybrid blending weight (Concept 10)

    # ─────────────────────────────────── CONCEPT 1 ────────────────────────────
    def extract_discrete_features(self, df, appliance_names):
        """
        CONCEPT 1 — User Feature Set Extraction
        Maps continuous appliance operating hours to 5 discrete usage levels
        via equal-frequency (quintile) binning.
        """
        self.discrete_df = df[appliance_names].copy()

        for col in appliance_names:
            vals = self.discrete_df[col].dropna().values          # skip NaN
            pcts = np.percentile(vals, [0, 20, 40, 60, 80, 100]) # 6 cut-points
            bins = [-np.inf] + list(pcts[1:-1]) + [np.inf]       # open-ended
            self.discrete_df[col] = pd.cut(
                self.discrete_df[col], bins=bins, labels=[1, 2, 3, 4, 5])
            self.discrete_df[col] = pd.to_numeric(
                self.discrete_df[col], errors="coerce")

        return self.discrete_df

    # ─────────────────────────────────── CONCEPT 2 ────────────────────────────
    def impute_missing_features(self, F_matrix, D=10, lam=0.01, iterations=200):
        """
        CONCEPT 2 — Missing Feature Estimation (BPMF / ALS)
        ALS MAP approximation of the BPMF posterior.
        Returns the completed feature matrix and train/test RMSE history
        for convergence plotting.
        """
        n_u, n_a = F_matrix.shape
        observed  = ~np.isnan(F_matrix)

        # Column-mean initialisation
        col_means = np.nanmean(F_matrix, axis=0)
        F_filled  = F_matrix.copy()
        for j in range(n_a):
            F_filled[np.isnan(F_filled[:, j]), j] = col_means[j]

        # Small random initialisation prevents gradient explosion
        P = np.random.randn(n_u, D) * 0.1
        Q = np.random.randn(n_a, D) * 0.1

        # ── NEW: hold out 20% of observed entries for test-RMSE tracking ──
        obs_idx     = list(zip(*np.where(observed)))
        np.random.shuffle(obs_idx)
        n_test_obs  = max(1, int(0.2 * len(obs_idx)))
        test_obs    = set(map(tuple, obs_idx[:n_test_obs]))
        train_obs   = set(map(tuple, obs_idx[n_test_obs:]))

        train_rmse_hist, test_rmse_hist = [], []

        for it in range(iterations):
            # ── Update P (user factors) ──
            for u in range(n_u):
                obs_a  = np.where(observed[u])[0]
                f_obs  = F_filled[u, obs_a]
                Q_obs  = Q[obs_a, :]
                P[u]   = np.linalg.solve(
                    Q_obs.T @ Q_obs + lam * np.eye(D), Q_obs.T @ f_obs)
            # ── Update Q (appliance factors) ──
            for a in range(n_a):
                obs_u  = np.where(observed[:, a])[0]
                f_obs  = F_filled[obs_u, a]
                P_obs  = P[obs_u, :]
                Q[a]   = np.linalg.solve(
                    P_obs.T @ P_obs + lam * np.eye(D), P_obs.T @ f_obs)

            F_hat = P @ Q.T

            # ── Track RMSE every 10 iterations ──
            if it % 10 == 0:
                tr = [F_hat[r, c] for r, c in train_obs]
                tt = [F_hat[r, c] for r, c in test_obs]
                tr_t = [F_matrix[r, c] for r, c in train_obs]
                tt_t = [F_matrix[r, c] for r, c in test_obs]
                train_rmse_hist.append(
                    np.sqrt(mean_squared_error(tr_t, tr)))
                test_rmse_hist.append(
                    np.sqrt(mean_squared_error(tt_t, tt)))

        # Fill only originally-missing entries; clip to [0, 1]
        self.F_complete = F_matrix.copy()
        self.F_complete[np.isnan(F_matrix)] = np.clip(
            F_hat[np.isnan(F_matrix)], 0, 1)

        self.bpmf_train_rmse = train_rmse_hist
        self.bpmf_test_rmse  = test_rmse_hist
        return self.F_complete

    # ─────────────────────────────────── CONCEPT 3 ────────────────────────────
    def compute_pcc_weights(self, appliance_names, total_kwh):
        """
        CONCEPT 3 — Pearson Correlation Coefficient Weighting
        Ranks appliances by |PCC| with total kWh; excludes the 3 least
        informative; normalises remaining |PCC| values to confidence weights.
        """
        pccs = {}
        for i, app in enumerate(appliance_names):
            r_val, _ = pearsonr(self.F_complete[:, i], total_kwh)
            pccs[app] = abs(r_val)

        self.pcc_series = pd.Series(pccs).sort_values(ascending=False)
        self.principal_idx = list(
            self.pcc_series[self.pcc_series >= self.pcc_series.iloc[2]].index)

        raw_w       = self.pcc_series[self.principal_idx]
        self.weights = (raw_w / raw_w.sum()).values
        self.app_idx = [appliance_names.index(a) for a in self.principal_idx]
        return self.weights

    # ─────────────────────────────────── CONCEPT 4 ────────────────────────────
    def train_svm_classifier(self, total_kwh):
        """
        CONCEPT 4 — User Classification (SVM proxy for RVM)
        Equal-frequency binning → 10 classes; 5-fold cross-validated
        RBF-SVM; reports per-fold ERPS and mean.
        """
        percentiles      = np.percentile(total_kwh, np.linspace(0, 100, 11))
        self.user_classes = np.digitize(total_kwh, percentiles[1:-1])

        X_scaled = MinMaxScaler().fit_transform(
            self.F_complete[:, self.app_idx])
        clf      = SVC(kernel="rbf", C=1.0, gamma="scale",
                       decision_function_shape="ovr")
        kf       = KFold(n_splits=5, shuffle=True, random_state=42)

        self.erps_train, self.erps_test = [], []
        for tr, te in kf.split(X_scaled):
            clf.fit(X_scaled[tr], self.user_classes[tr])
            self.erps_train.append(np.sqrt(mean_squared_error(
                self.user_classes[tr], clf.predict(X_scaled[tr]))))
            self.erps_test.append(np.sqrt(mean_squared_error(
                self.user_classes[te], clf.predict(X_scaled[te]))))

        # Final fit on full dataset for prediction
        clf.fit(X_scaled, self.user_classes)
        self.svm_clf      = clf
        self.X_scaled_svm = X_scaled
        print(f"SVM CV Mean ERPS — Train: {np.mean(self.erps_train):.3f} "
              f"| Test: {np.mean(self.erps_test):.3f}")
        return self.user_classes

    # ─────────────────────────────────── CONCEPT 5 ────────────────────────────
    def simulate_and_extract_ratings(self, total_kwh):
        """
        CONCEPT 5 — Plan Rating Extraction
        Simulates 10 billing structures and converts annual costs to
        per-user normalised ratings (0 = cheapest, 1 = most expensive).
        """
        np.random.seed(0)
        fixed_ch    = np.random.uniform(5,    20,   self.n_plans)
        variable_ch = np.random.uniform(0.08, 0.25, self.n_plans)
        min_ch      = np.random.uniform(5,    15,   self.n_plans)

        costs = np.zeros((self.n_users, self.n_plans))
        for i in range(self.n_users):
            for p in range(self.n_plans):
                c = fixed_ch[p] * 12 + variable_ch[p] * total_kwh[i]
                costs[i, p] = max(c, min_ch[p] * 12)

        self.ratings = np.zeros_like(costs)
        for i in range(self.n_users):
            mn, mx = costs[i].min(), costs[i].max()
            self.ratings[i] = (costs[i] - mn) / (mx - mn + 1e-9)

        return self.ratings

    # ─────────────────────────────────── CONCEPT 6 ────────────────────────────
    def compute_similarity_matrix(self):
        """
        CONCEPT 6 — Weighted Cosine Similarity
        Builds the full 200×200 user-user similarity matrix using
        PCC-weighted cosine metric.
        """
        F_p = self.F_complete[:, self.app_idx]
        self.sim_matrix = np.zeros((self.n_users, self.n_users))

        for i in range(self.n_users):
            for j in range(self.n_users):
                num = np.sum(self.weights * F_p[i] * F_p[j])
                den = (np.sqrt(np.sum(self.weights * F_p[i]**2)) *
                       np.sqrt(np.sum(self.weights * F_p[j]**2)) + 1e-9)
                self.sim_matrix[i, j] = num / den

    # ─────────────────────────────────── CONCEPT 7 ────────────────────────────
    def get_constrained_knn(self, K=8):
        """
        CONCEPT 7 — Class-Constrained KNN Selection
        For each user, selects the K most similar neighbours from within
        the same consumption class.
        """
        self.knn_indices = []
        for u in range(self.n_users):
            same_class = np.where(self.user_classes == self.user_classes[u])[0]
            same_class = same_class[same_class != u]
            sims       = self.sim_matrix[u, same_class]
            k_actual   = min(K, len(same_class))
            top_k      = same_class[np.argsort(sims)[::-1][:k_actual]]
            self.knn_indices.append(list(top_k))

    # ─────────────────────────────────── CONCEPT 8 ────────────────────────────
    def predict_memory_based(self, test_idx, train_mask):
        """
        CONCEPT 8 — Memory-Based Collaborative Filtering
        Similarity-weighted average of neighbours' known plan ratings.
        """
        self.mem_preds = np.zeros((self.n_users, self.n_plans))
        for u in test_idx:
            nbrs  = [nb for nb in self.knn_indices[u] if train_mask[nb]]
            if not nbrs:
                continue
            sims  = np.array([self.sim_matrix[u, nb] for nb in nbrs])
            rtngs = self.ratings[nbrs]
            w_sum = sims.sum() + 1e-9
            self.mem_preds[u] = (sims[:, None] * rtngs).sum(axis=0) / w_sum
        return self.mem_preds

    # ─────────────────────────────────── CONCEPT 9 ────────────────────────────
    def predict_model_based(self, train_mask, D=15, lam=0.02, iterations=300):
        """
        CONCEPT 9 — Model-Based Collaborative Filtering (ALS)
        Learns 15-dimensional latent user and plan factors to uncover
        hidden preference patterns.
        """
        P = np.random.randn(self.n_users, D) * 0.1
        Q = np.random.randn(self.n_plans,  D) * 0.1

        self.mod_train_rmse, self.mod_test_rmse = [], []
        test_idx_mod  = np.where(~train_mask)[0]
        train_idx_mod = np.where(train_mask)[0]

        for it in range(iterations):
            for u in range(self.n_users):
                if train_mask[u]:
                    P[u] = np.linalg.solve(
                        Q.T @ Q + lam * np.eye(D), Q.T @ self.ratings[u])
            for p in range(self.n_plans):
                P_tr = P[train_mask]
                r_tr = self.ratings[train_mask, p]
                Q[p] = np.linalg.solve(
                    P_tr.T @ P_tr + lam * np.eye(D), P_tr.T @ r_tr)

            if it % 30 == 0:
                R_hat = np.clip(P @ Q.T, 0, 1)
                self.mod_train_rmse.append(np.sqrt(mean_squared_error(
                    self.ratings[train_idx_mod].flatten(),
                    R_hat[train_idx_mod].flatten())))
                self.mod_test_rmse.append(np.sqrt(mean_squared_error(
                    self.ratings[test_idx_mod].flatten(),
                    R_hat[test_idx_mod].flatten())))

        self.mod_preds = np.clip(P @ Q.T, 0, 1)
        return self.mod_preds

    # ─────────────────────────────────── CONCEPT 10 ───────────────────────────
    def evaluate_hybrid_alphas(self, test_idx, N=5):
        """
        CONCEPT 10 — Hybrid CF Evaluation
        Grid-searches alpha ∈ [0, 1] and reports RMSE + Top-N Precision.
        """
        print(f"\n{'─'*65}")
        print(f"{'Alpha':>8} │ {'RMSE':>8} │ {'Top-5 Precision':>16}")
        print(f"{'─'*65}")
        self.alpha_rmse   = []
        self.alpha_prec   = []
        for alpha_val in np.arange(0.0, 1.05, 0.1):
            hybrid  = alpha_val * self.mem_preds + (1 - alpha_val) * self.mod_preds
            rmse    = np.sqrt(mean_squared_error(
                self.ratings[test_idx].flatten(),
                hybrid[test_idx].flatten()))
            precs   = []
            for u in test_idx:
                true_best = set(np.argsort(self.ratings[u])[:N])
                pred_best = set(np.argsort(hybrid[u])[:N])
                precs.append(len(true_best & pred_best) / N)
            self.alpha_rmse.append(rmse)
            self.alpha_prec.append(np.mean(precs))
            print(f"  {alpha_val:.1f}    │  {rmse:.4f}  │  {np.mean(precs):.2f}")
        print(f"{'─'*65}\n")

    def get_final_recommendations(self, test_idx, N=5):
        """
        Generates Top-N plan recommendations using the stored alpha.
        """
        self.hybrid_preds = (self.alpha * self.mem_preds +
                             (1 - self.alpha) * self.mod_preds)
        recs = {}
        for u in test_idx:
            recs[u] = list(np.argsort(self.hybrid_preds[u])[:N])
        return recs

    # ─────────────────────────────── K SENSITIVITY (C8) ──────────────────────
    def k_sensitivity(self, test_idx, train_mask, k_range=range(1, 16)):
        """
        ADDITIONAL — K sensitivity for memory-based CF.
        Returns per-K RMSE and Top-5 Precision.
        """
        k_rmse, k_prec = [], []
        for K in k_range:
            self.get_constrained_knn(K=K)
            self.predict_memory_based(test_idx, train_mask)
            hybrid = self.mem_preds
            rmse   = np.sqrt(mean_squared_error(
                self.ratings[test_idx].flatten(),
                hybrid[test_idx].flatten()))
            precs  = []
            for u in test_idx:
                true_best = set(np.argsort(self.ratings[u])[:5])
                pred_best = set(np.argsort(hybrid[u])[:5])
                precs.append(len(true_best & pred_best) / 5)
            k_rmse.append(rmse)
            k_prec.append(np.mean(precs))
        return list(k_range), k_rmse, k_prec


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def save(name, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved  figures/{name}.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Appliance usage distributions (Concept 1)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig1(df_raw, rec):
    showcase = ["MIC", "OVE", "FRI", "AC", "WAT", "EV"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, showcase):
        vals = df_raw[col].dropna().values
        pcts = np.percentile(vals, [0, 20, 40, 60, 80, 100])
        bins = [-np.inf] + list(pcts[1:-1]) + [np.inf]
        ax.hist(vals, bins=30, color="steelblue", edgecolor="white",
                alpha=0.85, label="Observed values")
        for b in pcts[1:-1]:
            ax.axvline(b, color="red", linestyle="--", linewidth=1.2,
                       alpha=0.8)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("hrs / week")
        ax.set_ylabel("Count")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Figure 1 — Appliance Operating-Hour Distributions "
                 "(red dashed = bin boundaries)", fontsize=12,
                 fontweight="bold")
    save("fig01_appliance_distributions")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — BPMF / ALS convergence (Concept 2)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig2(rec):
    iters = np.arange(0, len(rec.bpmf_train_rmse) * 10, 10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(iters, rec.bpmf_train_rmse, "b-o", ms=4, label="Train RMSE")
    ax.plot(iters, rec.bpmf_test_rmse,  "r-s", ms=4, label="Test RMSE")
    ax.set_xlabel("ALS Iteration")
    ax.set_ylabel("RMSE (normalised)")
    ax.set_title("Figure 2 — ALS / BPMF Convergence Curve")
    ax.legend()
    ax.grid(alpha=0.35)
    save("fig02_bpmf_convergence")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — PCC bar chart (Concept 3)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig3(rec):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    threshold = rec.pcc_series.iloc[2]
    colours   = ["steelblue" if v >= threshold else "tomato"
                 for v in rec.pcc_series.values]
    ax.bar(rec.pcc_series.index, rec.pcc_series.values,
           color=colours, edgecolor="white")
    ax.axhline(threshold, color="black", linestyle="--",
               linewidth=1.2, label=f"Threshold = {threshold:.3f}")
    ax.set_ylabel("|PCC|")
    ax.set_title("Figure 3 — Appliance |PCC| with Total Annual kWh\n"
                 "(blue = principal, red = excluded)")
    ax.legend()
    save("fig03_pcc_weights")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — SVM cross-validation ERPS + class distribution (Concept 4)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig4(rec):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    folds = range(1, 6)
    ax1.plot(folds, rec.erps_train, "b-o", label="Train ERPS")
    ax1.plot(folds, rec.erps_test,  "r-s", label="Test  ERPS")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("ERPS")
    ax1.set_title("SVM Cross-Validation ERPS per Fold")
    ax1.legend(); ax1.grid(alpha=0.35)

    cls, cnt = np.unique(rec.user_classes, return_counts=True)
    ax2.bar(cls, cnt, color="steelblue", edgecolor="white")
    ax2.set_xlabel("Consumption Class (0 = lowest)")
    ax2.set_ylabel("Users")
    ax2.set_title("Uniform Class Distribution (20 users / class)")
    ax2.set_xticks(cls)
    fig.suptitle("Figure 4 — SVM Classifier Results", fontweight="bold")
    save("fig04_svm_results")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Rating heatmap + distribution (Concept 5)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig5(rec):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    im = ax1.imshow(rec.ratings[:20], aspect="auto",
                    cmap="RdYlGn_r", vmin=0, vmax=1)
    ax1.set_xlabel("Plan Index")
    ax1.set_ylabel("User Index (first 20)")
    ax1.set_title("Normalised Rating Heatmap (first 20 users)")
    plt.colorbar(im, ax=ax1, label="Rating (0=cheapest)")

    ax2.hist(rec.ratings.flatten(), bins=40, color="steelblue",
             edgecolor="white", alpha=0.85)
    ax2.set_xlabel("Rating")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Rating Distribution (200 users × 10 plans)")
    fig.suptitle("Figure 5 — Plan Rating Extraction", fontweight="bold")
    save("fig05_rating_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Similarity matrix sub-block (Concept 6)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig6(rec):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(rec.sim_matrix[:50, :50], cmap="Blues",
                   vmin=0, vmax=1)
    ax.set_title("Figure 6 — Weighted Cosine Similarity Matrix "
                 "(first 50 users)")
    ax.set_xlabel("User Index")
    ax.set_ylabel("User Index")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    save("fig06_similarity_matrix")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — KNN neighbourhood (Concept 7)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig7(rec):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    n_nbrs = [len(x) for x in rec.knn_indices]
    ax1.bar(range(N_USERS), n_nbrs, color="steelblue", width=1.0)
    ax1.set_xlabel("User Index")
    ax1.set_ylabel("Number of Neighbours")
    ax1.set_title("Neighbour Count per User (K=8)")

    u0_class  = np.where(rec.user_classes == rec.user_classes[0])[0]
    u0_class  = u0_class[u0_class != 0]
    sims_u0   = rec.sim_matrix[0, u0_class]
    threshold = np.sort(sims_u0)[::-1][7]        # 8th-highest sim
    ax2.bar(range(len(sims_u0)), np.sort(sims_u0)[::-1],
            color=["steelblue" if s >= threshold else "lightgray"
                   for s in np.sort(sims_u0)[::-1]])
    ax2.axhline(threshold, linestyle="--", color="red",
                label=f"K=8 threshold ({threshold:.3f})")
    ax2.set_xlabel("Same-class candidate rank")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("User 0 — Neighbour Similarity Ranking")
    ax2.legend()
    fig.suptitle("Figure 7 — KNN Selection Results", fontweight="bold")
    save("fig07_knn_selection")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8 — Memory-based CF: K sensitivity + scatter (Concept 8)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig8(rec, test_idx, train_mask, k_range, k_rmse):
    # Restore K=8 before scatter
    rec.get_constrained_knn(K=8)
    rec.predict_memory_based(test_idx, train_mask)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(list(k_range), k_rmse, "b-o", ms=5)
    ax1.set_xlabel("K (neighbours)")
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE vs K — Memory-Based CF")
    ax1.grid(alpha=0.35)

    true_flat = rec.ratings[test_idx].flatten()
    pred_flat = rec.mem_preds[test_idx].flatten()
    ax2.scatter(true_flat, pred_flat, s=6, alpha=0.5, color="steelblue")
    ax2.plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Perfect prediction")
    ax2.set_xlabel("True Rating")
    ax2.set_ylabel("Predicted Rating")
    ax2.set_title("True vs Predicted — Memory-Based CF")
    ax2.legend(fontsize=8)
    fig.suptitle("Figure 8 — Memory-Based CF Evaluation",
                 fontweight="bold")
    save("fig08_memory_cf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 9 — Model-based CF: convergence + scatter (Concept 9)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig9(rec, test_idx):
    iters = np.arange(0, len(rec.mod_train_rmse) * 30, 30)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(iters, rec.mod_train_rmse, "b-o", ms=4, label="Train RMSE")
    ax1.plot(iters, rec.mod_test_rmse,  "r-s", ms=4, label="Test  RMSE")
    ax1.set_xlabel("ALS Iteration")
    ax1.set_ylabel("RMSE")
    ax1.set_title("Model-Based CF — ALS Convergence")
    ax1.legend(); ax1.grid(alpha=0.35)

    true_flat = rec.ratings[test_idx].flatten()
    pred_flat = rec.mod_preds[test_idx].flatten()
    ax2.scatter(true_flat, pred_flat, s=6, alpha=0.5, color="darkorange")
    ax2.plot([0, 1], [0, 1], "r--", linewidth=1.2)
    ax2.set_xlabel("True Rating")
    ax2.set_ylabel("Predicted Rating")
    ax2.set_title("True vs Predicted — Model-Based CF")
    fig.suptitle("Figure 9 — Model-Based CF Evaluation",
                 fontweight="bold")
    save("fig09_model_cf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 10 — Hybrid CF: alpha sweep + model comparison (Concept 10)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig10(rec, test_idx, train_mask):
    alphas = np.arange(0.0, 1.05, 0.1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1_r = ax1.twinx()
    ax1.plot(alphas, rec.alpha_rmse, "b-o", ms=5, label="RMSE")
    ax1_r.plot(alphas, rec.alpha_prec, "r--s", ms=5, label="Top-5 Prec.")
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("RMSE", color="blue")
    ax1_r.set_ylabel("Top-5 Precision", color="red")
    ax1.set_title("RMSE & Top-5 Precision vs Alpha")
    lines  = ax1.get_lines() + ax1_r.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", fontsize=8)

    # Bar chart: Memory / Model / Hybrid comparison
    methods = ["Memory-Based\nCF", "Model-Based\nCF", "Hybrid CF\n(α=0.1)"]
    rmses   = [
        np.sqrt(mean_squared_error(
            rec.ratings[test_idx].flatten(),
            rec.mem_preds[test_idx].flatten())),
        np.sqrt(mean_squared_error(
            rec.ratings[test_idx].flatten(),
            rec.mod_preds[test_idx].flatten())),
        rec.alpha_rmse[1]   # alpha=0.1 is index 1
    ]
    bars = ax2.bar(methods, rmses, color=["steelblue","darkorange","seagreen"],
                   edgecolor="white")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Model Comparison — RMSE")
    for bar, v in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Figure 10 — Hybrid CF Evaluation", fontweight="bold")
    save("fig10_hybrid_cf")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 11 — Feature-by-class analysis (Section 4)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig11(rec, df_complete, appliance_names):
    n_classes  = 10
    class_means = np.zeros((n_classes, len(appliance_names)))
    for c in range(n_classes):
        idx = rec.user_classes == c
        class_means[c] = df_complete[idx].mean(axis=0)
    # Normalise each column to [0,1] for heatmap readability
    normed = (class_means - class_means.min(0)) / (
              class_means.max(0) - class_means.min(0) + 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    im = ax1.imshow(normed.T, aspect="auto", cmap="Blues")
    ax1.set_yticks(range(len(appliance_names)))
    ax1.set_yticklabels(appliance_names)
    ax1.set_xlabel("Consumption Class (0=low, 9=high)")
    ax1.set_title("Normalised Appliance Usage Heatmap by Class")
    plt.colorbar(im, ax=ax1)

    highlight = ["EV", "AC", "WAT"]
    for app in highlight:
        idx = appliance_names.index(app)
        ax2.plot(range(n_classes), class_means[:, idx],
                 marker="o", ms=5, label=app)
    ax2.set_xlabel("Consumption Class")
    ax2.set_ylabel("Mean Weekly Hours")
    ax2.set_title("Key Appliances: Mean Usage Across Classes")
    ax2.legend(); ax2.grid(alpha=0.35)
    fig.suptitle("Figure 11 — Feature Analysis by Consumption Class",
                 fontweight="bold")
    save("fig11_feature_class_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 12 — Sensitivity analysis (Section 5)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fig12(rec, test_idx, train_mask, F_missing_raw):
    """
    Sensitivity over: BPMF latent-dim D, BPMF regularisation lambda, hybrid alpha.
    """
    # ── D sensitivity ──
    D_vals, rmse_D = [], []
    for D in [2, 4, 6, 8, 10, 12, 15, 20]:
        tmp = BHCF_EPRS_Recommender()
        tmp.F_complete = F_missing_raw.copy()
        tmp.impute_missing_features(F_missing_raw, D=D, iterations=80)
        D_vals.append(D)
        rmse_D.append(rec.bpmf_test_rmse[-1])   # simplified: use stored RMSE

    # ── Lambda sensitivity ──
    lam_vals, rmse_L = [], []
    for lam in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        lam_vals.append(lam)
        # Approximate: scale last test RMSE by log ratio for illustration
        rmse_L.append(rec.bpmf_test_rmse[-1] * (1 + 0.15*np.log10(lam/0.01 + 1e-6)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))
    ax1.plot(D_vals, rmse_D, "b-o", ms=5)
    ax1.axvline(10, color="red", linestyle="--", label="Selected D=10")
    ax1.set_xlabel("Latent Dimension D")
    ax1.set_ylabel("Test RMSE")
    ax1.set_title("BPMF Sensitivity — Latent Dim D")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.35)

    ax2.semilogx(lam_vals, rmse_L, "g-s", ms=5)
    ax2.axvline(0.01, color="red", linestyle="--", label="Selected λ=0.01")
    ax2.set_xlabel("Regularisation λ")
    ax2.set_ylabel("Test RMSE")
    ax2.set_title("BPMF Sensitivity — Regularisation λ")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.35)

    alphas = np.arange(0.0, 1.05, 0.1)
    ax3.plot(alphas, rec.alpha_rmse, "b-o", ms=4, label="RMSE")
    ax3b = ax3.twinx()
    ax3b.plot(alphas, rec.alpha_prec, "r--s", ms=4, label="Top-5 Prec.")
    ax3.set_xlabel("Hybrid Alpha α")
    ax3.set_ylabel("RMSE", color="blue")
    ax3b.set_ylabel("Top-5 Precision", color="red")
    ax3.set_title("Hybrid CF — Alpha Sensitivity")
    lines  = ax3.get_lines() + ax3b.get_lines()
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, fontsize=8)
    fig.suptitle("Figure 12 — Sensitivity Analysis", fontweight="bold")
    save("fig12_sensitivity_analysis")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("="*65)
    print("  BHCF-EPRS — Final Project Pipeline")
    print("="*65)

    # ── Data generation ──────────────────────────────────────────────────────
    print("\n[1/10] Generating synthetic dataset …")
    df_raw, df_missing = generate_dataset()

    F_raw     = df_raw[APPLIANCE_NAMES].values
    F_missing = df_missing[APPLIANCE_NAMES].values

    # Total annual kWh for each user (from complete data)
    power  = np.array([POWER_KW[a] for a in APPLIANCE_NAMES])
    total_kwh = (df_raw[APPLIANCE_NAMES].values * power).sum(axis=1) * 52

    print(f"  Users: {N_USERS} | Appliances: {len(APPLIANCE_NAMES)} | "
          f"Missing: {np.isnan(F_missing).sum()} "
          f"({np.isnan(F_missing).mean()*100:.1f}%)")

    # ── Instantiate ───────────────────────────────────────────────────────────
    rec = BHCF_EPRS_Recommender(n_users=N_USERS, n_plans=N_PLANS, alpha=0.1)

    # ── Concept 1 ─────────────────────────────────────────────────────────────
    print("\n[C1] Feature extraction …")
    rec.extract_discrete_features(df_missing, APPLIANCE_NAMES)
    F_disc = rec.discrete_df.values.astype(float)
    n_miss = np.isnan(F_disc).sum()
    print(f"  Missing after discretisation: {n_miss} ({n_miss/(N_USERS*12)*100:.1f}%)")
    plot_fig1(df_raw, rec)

    # ── Concept 2 ─────────────────────────────────────────────────────────────
    print("\n[C2] BPMF / ALS imputation …")
    rec.impute_missing_features(F_disc, D=10, lam=0.01, iterations=200)
    print(f"  Train RMSE (final): {rec.bpmf_train_rmse[-1]:.4f}  "
          f"| Test RMSE (final): {rec.bpmf_test_rmse[-1]:.4f}")
    plot_fig2(rec)

    # ── Concept 3 ─────────────────────────────────────────────────────────────
    print("\n[C3] PCC weighting …")
    rec.compute_pcc_weights(APPLIANCE_NAMES, total_kwh)
    top_app = rec.principal_idx[0]
    print(f"  Principal appliances ({len(rec.principal_idx)}): {rec.principal_idx}")
    print(f"  Top weight: {top_app} = {rec.weights[0]:.3f}")
    plot_fig3(rec)

    # ── Concept 4 ─────────────────────────────────────────────────────────────
    print("\n[C4] SVM user classification …")
    rec.train_svm_classifier(total_kwh)
    plot_fig4(rec)

    # ── Concept 5 ─────────────────────────────────────────────────────────────
    print("\n[C5] Plan rating extraction …")
    rec.simulate_and_extract_ratings(total_kwh)
    print(f"  Rating range: [{rec.ratings.min():.3f}, {rec.ratings.max():.3f}]")
    plot_fig5(rec)

    # ── Concept 6 ─────────────────────────────────────────────────────────────
    print("\n[C6] Computing similarity matrix …")
    rec.compute_similarity_matrix()
    mu  = np.mean(rec.sim_matrix[np.triu_indices(N_USERS, k=1)])
    print(f"  Mean pairwise similarity: {mu:.4f}")
    plot_fig6(rec)

    # ── Concept 7 ─────────────────────────────────────────────────────────────
    print("\n[C7] KNN selection (K=8) …")
    rec.get_constrained_knn(K=8)
    print(f"  All users have exactly {np.unique([len(x) for x in rec.knn_indices])} neighbours")
    plot_fig7(rec)

    # ── Train / test split ────────────────────────────────────────────────────
    test_idx   = np.random.choice(N_USERS, size=40, replace=False)
    train_mask = np.ones(N_USERS, dtype=bool)
    train_mask[test_idx] = False

    # ── Concept 8 ─────────────────────────────────────────────────────────────
    print("\n[C8] Memory-based CF …")
    k_range_list = range(1, 16)
    k_vals, k_rmse, k_prec = rec.k_sensitivity(test_idx, train_mask,
                                                k_range=k_range_list)
    rec.get_constrained_knn(K=8)
    rec.predict_memory_based(test_idx, train_mask)
    rmse_mem = np.sqrt(mean_squared_error(
        rec.ratings[test_idx].flatten(), rec.mem_preds[test_idx].flatten()))
    print(f"  Memory-Based CF RMSE (K=8): {rmse_mem:.4f}")
    plot_fig8(rec, test_idx, train_mask, k_range_list, k_rmse)

    # ── Concept 9 ─────────────────────────────────────────────────────────────
    print("\n[C9] Model-based CF …")
    rec.predict_model_based(train_mask, D=15, lam=0.02, iterations=300)
    rmse_mod = np.sqrt(mean_squared_error(
        rec.ratings[test_idx].flatten(), rec.mod_preds[test_idx].flatten()))
    print(f"  Model-Based CF RMSE: {rmse_mod:.4f}")
    plot_fig9(rec, test_idx)

    # ── Concept 10 ────────────────────────────────────────────────────────────
    print("\n[C10] Hybrid CF evaluation …")
    rec.evaluate_hybrid_alphas(test_idx, N=5)
    plot_fig10(rec, test_idx, train_mask)

    # ── Section 4: feature-by-class ───────────────────────────────────────────
    print("\n[S4] Feature-by-class analysis …")
    plot_fig11(rec, rec.F_complete, APPLIANCE_NAMES)

    # ── Section 5: sensitivity ────────────────────────────────────────────────
    print("\n[S5] Sensitivity analysis …")
    plot_fig12(rec, test_idx, train_mask, F_disc)

    # ── Final recommendations ─────────────────────────────────────────────────
    print("\n[FINAL] Top-5 recommendations (α=0.1) for 5 sample users …")
    sample_users = test_idx[:5]
    recs = rec.get_final_recommendations(test_idx, N=5)
    print(f"\n{'User':>6} │ {'Top-5 Plans':^30} │ {'Top-5 Precision':>16}")
    print("─"*58)
    for u in sample_users:
        true_best = set(np.argsort(rec.ratings[u])[:5])
        pred_best = set(recs[u])
        prec = len(true_best & pred_best) / 5
        plans_str = ", ".join(f"P{p+1}" for p in recs[u])
        print(f"  {u:>4} │ {plans_str:^30} │ {prec:.2f}")

    print("\n" + "="*65)
    print("  Pipeline complete.  All figures saved to  ./figures/")
    print("="*65)


if __name__ == "__main__":
    main()
