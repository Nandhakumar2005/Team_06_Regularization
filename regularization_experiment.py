"""
regularization_experiment.py
Team 06 - Investigation Challenge: Regularization
Run from project root:  python src/regularization_experiment.py

Experiment summary:
  - Loads the crop yield dataset and removes 2310 duplicate rows (same as notebook).
  - One-hot encodes categorical features (Area, Item) with drop_first=True.
  - Expands the 4 numeric features to degree-2 polynomial features (14 total).
  - Splits 80/20 train/test with random_state=42.
  - StandardScaler fitted on training data only (no data leakage).
  - Finds best Ridge alpha via efficient leave-one-out CV (RidgeCV, cv=None).
  - Finds best Lasso alpha via 5-fold CV (LassoCV).
  - Fits OLS, Ridge, Lasso at their best alphas.
  - Sweeps 30 alpha values for Ridge and Lasso to produce R^2-vs-alpha curves.
  - Also sweeps Ridge alphas to record L2 norm of coefficients vs alpha.
  - Saves 4 figures and 2 CSVs to the results/ directory.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import (Ridge, Lasso, LinearRegression,
                                   RidgeCV, LassoCV, lasso_path)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

def log(msg):
    print(msg, flush=True)

# -- paths --------------------------------------------------------------
CSV_PATH = "data/yield_df.csv"
OUT      = "results"
os.makedirs(OUT, exist_ok=True)

# -- config ---------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20
DEGREE       = 2
CV_FOLDS     = 5
ALPHAS_RIDGE = np.logspace(-2, 5, 30)
ALPHAS_LASSO = np.logspace( 0, 5, 30)

# -- 1. Load & preprocess ---------------------------------------------------
log("Loading data ...")
df = pd.read_csv(CSV_PATH)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Remove duplicates and missing values (same as notebook preprocessing).
# Duplicates can appear in both train and test after a random split, which
# would leak information. Removing them first prevents this.
n_dupes = df.duplicated().sum()
df = df.drop_duplicates().dropna()
log(f"  Removed {n_dupes} duplicate rows.  Remaining rows: {len(df)}")

TARGET   = "hg/ha_yield"
NUM_COLS = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
CAT_COLS = ["Area", "Item"]

df_enc        = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
cat_feat_cols = [c for c in df_enc.columns if c not in NUM_COLS + [TARGET]]

X_num = df[NUM_COLS].values.astype(float)
X_cat = df_enc[cat_feat_cols].values.astype(float)
y     = df_enc[TARGET].values.astype(float)

poly       = PolynomialFeatures(degree=DEGREE, include_bias=False)
X_num_poly = poly.fit_transform(X_num)

X = np.hstack([X_num_poly, X_cat])
log(f"  Rows: {len(y)}, Features: {X.shape[1]} "
    f"({X_num_poly.shape[1]} poly-numeric + {len(cat_feat_cols)} one-hot)")

# -- 2. Train / test split --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
log(f"  Train: {len(y_train)}, Test: {len(y_test)}")

# StandardScaler fitted on training data only — no test data leaks into
# the scaler parameters. X is already centered by StandardScaler, so
# lasso_path (which has no intercept) also works correctly.
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(X_train)
Xte_sc = scaler.transform(X_test)

# -- 3. Find best alpha with built-in CV (fast) -----------------------------
log("\nFinding best alpha via built-in CV ...")
# cv=None uses efficient leave-one-out via the SVD decomposition — same
# alpha selection as explicit k-fold but ~7x faster for Ridge.
ridgecv = RidgeCV(alphas=np.logspace(-2, 5, 200), cv=None, scoring="r2")
ridgecv.fit(Xtr_sc, y_train)
best_alpha_ridge = ridgecv.alpha_
log(f"  Best Ridge alpha: {best_alpha_ridge:.4f}  (cv=None -> efficient closed-form LOO)")

lassocv = LassoCV(alphas=np.logspace(0, 5, 100), cv=CV_FOLDS, max_iter=5000, n_jobs=-1)
lassocv.fit(Xtr_sc, y_train)
best_alpha_lasso = lassocv.alpha_
log(f"  Best Lasso alpha: {best_alpha_lasso:.4f}")

# -- 4. Fit final models ------------------------------------------------------
log("\nFitting final models ...")
ols   = LinearRegression().fit(Xtr_sc, y_train)
ridge = Ridge(alpha=best_alpha_ridge).fit(Xtr_sc, y_train)
lasso = Lasso(alpha=best_alpha_lasso, max_iter=5000).fit(Xtr_sc, y_train)

results_rows = []
coef_dict    = {}
for name, mdl in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
    tr_r2  = r2_score(y_train, mdl.predict(Xtr_sc))
    te_r2  = r2_score(y_test,  mdl.predict(Xte_sc))
    coef   = mdl.coef_
    coef_dict[name] = coef
    alpha  = (best_alpha_ridge if name == "Ridge"
              else best_alpha_lasso if name == "Lasso" else None)
    nz = int(np.sum(np.abs(coef) > 1e-8))
    l2 = float(np.linalg.norm(coef))
    results_rows.append({"Model": name, "Alpha": alpha,
                         "Train_R2": round(tr_r2, 4), "Test_R2": round(te_r2, 4),
                         "N_nonzero_coefs": nz, "L2_norm_coefs": round(l2, 4)})
    log(f"  {name:6s}  Train R2={tr_r2:.4f}  Test R2={te_r2:.4f}  "
        f"nonzero={nz}/{len(coef)}  L2_norm={l2:.1f}")

summary_df = pd.DataFrame(results_rows)
summary_df.to_csv(os.path.join(OUT, "summary_by_alpha.csv"), index=False)
log("Saved summary_by_alpha.csv")

# -- 5. Alpha-sweep for figures ----------------------------------------------
log("\nAlpha sweep for figures (closed-form Ridge + warm-started Lasso path) ...")

# Ridge sweep: each fit is a closed-form solution, so 30 fits + 5-fold CV
# per alpha is fast.
ridge_rows  = []
ridge_l2    = []  # L2 norm of coefficients at each alpha (for coefficient graph)
for alpha in ALPHAS_RIDGE:
    m = Ridge(alpha=alpha).fit(Xtr_sc, y_train)
    cv_r2 = cross_val_score(Ridge(alpha=alpha), Xtr_sc, y_train,
                            cv=CV_FOLDS, scoring="r2").mean()
    ridge_rows.append({"model": "Ridge", "alpha": alpha,
                       "cv_r2": round(cv_r2, 4),
                       "train_r2": round(r2_score(y_train, m.predict(Xtr_sc)), 4),
                       "test_r2": round(r2_score(y_test, m.predict(Xte_sc)), 4)})
    ridge_l2.append(float(np.linalg.norm(m.coef_)))
log("  Ridge sweep done.")

# Lasso: lasso_path computes the whole regularization path with warm starts
# in one call — much faster than refitting independently at each alpha.
# lasso_path does NOT fit an intercept, so y must be centered first.
y_train_mean = y_train.mean()
alphas_out, coefs_path, _ = lasso_path(Xtr_sc, y_train - y_train_mean,
                                       alphas=ALPHAS_LASSO[::-1], max_iter=5000)
# Build the Lasso CV curve from LassoCV's own internal mse_path_ —
# already computed while selecting best_alpha_lasso, so no second sweep needed.
lassocv_curve = LassoCV(alphas=ALPHAS_LASSO, cv=CV_FOLDS, max_iter=5000, n_jobs=-1).fit(Xtr_sc, y_train)
cv_r2_from_path = 1 - lassocv_curve.mse_path_.mean(axis=1) / np.var(y_train)
cv_lookup = dict(zip(lassocv_curve.alphas_, cv_r2_from_path))

lasso_rows    = []
lasso_nonzero = []
for i, a in enumerate(alphas_out):
    c = coefs_path[:, i]
    pred_tr = Xtr_sc @ c + y_train_mean
    pred_te = Xte_sc @ c + y_train_mean
    lasso_rows.append({"model": "Lasso", "alpha": a,
                       "train_r2": round(r2_score(y_train, pred_tr), 4),
                       "cv_r2": round(float(cv_lookup.get(a, np.nan)), 4),
                       "test_r2": round(r2_score(y_test, pred_te), 4)})
    lasso_nonzero.append(int(np.sum(np.abs(c) > 1e-8)))

all_runs_df = pd.DataFrame(ridge_rows + lasso_rows)
all_runs_df.to_csv(os.path.join(OUT, "all_runs.csv"), index=False)
log(f"Saved all_runs.csv ({len(all_runs_df)} rows)")

# -- 6. Figures ---------------------------------------------------------------
log("\nDrawing figures ...")

# Figure 1: R² vs alpha for Ridge and Lasso
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mname, alphas, label_a in [
    (axes[0], "Ridge", ALPHAS_RIDGE, best_alpha_ridge),
    (axes[1], "Lasso", alphas_out,   best_alpha_lasso),
]:
    sub = all_runs_df[all_runs_df["model"] == mname].sort_values("alpha")
    ax.semilogx(sub["alpha"], sub["train_r2"], color="steelblue", linewidth=2, label="Train R²")
    ax.semilogx(sub["alpha"], sub["cv_r2"],    color="orange",   linewidth=2, label=f"{CV_FOLDS}-fold CV R²")
    ax.semilogx(sub["alpha"], sub["test_r2"],  color="green",    linewidth=2, linestyle="--", label="Test R²")
    ax.axvline(label_a, color="red", linestyle=":", linewidth=1.5, label=f"Best alpha = {label_a:.3g}")
    ax.set_xlabel("Regularization strength alpha (log scale)", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title(f"{mname} Regression: R² vs alpha", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle("Effect of Regularization Strength on Train / CV / Test R²", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_alpha_curves.png"), dpi=150, bbox_inches="tight")
plt.close(); log("Saved fig_alpha_curves.png")

# Figure 2: Model comparison bar chart (OLS vs Ridge vs Lasso at best alpha)
fig, ax = plt.subplots(figsize=(7, 4.5))
x, w = np.arange(3), 0.35
b1 = ax.bar(x-w/2, summary_df["Train_R2"], w, label="Train R²", color="#4e79a7")
b2 = ax.bar(x+w/2, summary_df["Test_R2"],  w, label="Test R²",  color="#f28e2b")
ax.set_xticks(x); ax.set_xticklabels(summary_df["Model"], fontsize=12)
ax.set_ylabel("R²", fontsize=11)
ax.set_title("OLS vs Ridge vs Lasso: Train and Test R²", fontsize=12)
ax.legend(fontsize=10)
ymin = max(0, min(summary_df[["Train_R2","Test_R2"]].min()) - 0.05)
ax.set_ylim(ymin, 1.08)
for bar in list(b1)+list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
ax.grid(axis="y", alpha=0.3); plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_model_comparison.png"), dpi=150)
plt.close(); log("Saved fig_model_comparison.png")

# Figure 3: Top-30 coefficient magnitudes (OLS vs Ridge vs Lasso)
ols_c, ridge_c, lasso_c = [np.abs(coef_dict[n]) for n in ["OLS","Ridge","Lasso"]]
top_idx = np.argsort(ols_c)[-30:][::-1]
fig, ax = plt.subplots(figsize=(11, 5))
xp = np.arange(30)
ax.bar(xp-0.25, ols_c[top_idx],   0.25, label="OLS",   color="#4e79a7")
ax.bar(xp,      ridge_c[top_idx], 0.25, label="Ridge", color="#f28e2b")
ax.bar(xp+0.25, lasso_c[top_idx], 0.25, label="Lasso", color="#59a14f")
ax.set_xticks(xp)
ax.set_xticklabels([f"f{i}" for i in top_idx], rotation=45, ha="right", fontsize=7)
ax.set_ylabel("|Coefficient|", fontsize=11)
ax.set_title("Top-30 Coefficient Magnitudes: OLS vs Ridge vs Lasso", fontsize=12)
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3); plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_coefficients.png"), dpi=150)
plt.close(); log("Saved fig_coefficients.png")

# Figure 4: Lasso sparsity — non-zero coefficients vs alpha
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(alphas_out, lasso_nonzero, color="#59a14f", marker=".", linewidth=1.8)
ax.axvline(best_alpha_lasso, color="red", linestyle=":", linewidth=1.5,
           label=f"Best alpha = {best_alpha_lasso:.3g}")
ax.set_xlabel("alpha (log scale)", fontsize=11)
ax.set_ylabel("# Non-zero Coefficients", fontsize=11)
ax.set_title("Lasso Sparsity: Non-zero Coefficients vs alpha", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3); plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_lasso_sparsity.png"), dpi=150)
plt.close(); log("Saved fig_lasso_sparsity.png")

# Figure 5: Ridge coefficient L2 norm vs alpha
# This shows how strongly Ridge shrinks the overall coefficient vector
# as regularization increases — the core mechanism of L2 regularization.
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(ALPHAS_RIDGE, ridge_l2, color="#f28e2b", marker=".", linewidth=1.8,
            label="L2 norm of Ridge coefficients")
ax.axvline(best_alpha_ridge, color="red", linestyle=":", linewidth=1.5,
           label=f"Best alpha = {best_alpha_ridge:.3g}")
ax.set_xlabel("Regularization strength alpha (log scale)", fontsize=11)
ax.set_ylabel("L2 Norm of Coefficients  ||w||₂", fontsize=11)
ax.set_title("Ridge: Coefficient Magnitude (L2 Norm) vs alpha", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3); plt.tight_layout()
fig.savefig(os.path.join(OUT, "fig_ridge_l2norm.png"), dpi=150)
plt.close(); log("Saved fig_ridge_l2norm.png")

# -- 7. Final summary -----------------------------------------------------------
log("\n" + "="*65)
log("FINAL SUMMARY")
log("="*65)
log(summary_df.to_string(index=False))
log(f"\nBest Ridge alpha (RidgeCV): {best_alpha_ridge:.6f}")
log(f"Best Lasso alpha (LassoCV): {best_alpha_lasso:.6f}")
log(f"Polynomial degree:  {DEGREE}  (numeric features only)")
log(f"Total features:     {X.shape[1]}")
log(f"Train / Test size:  {len(y_train)} / {len(y_test)}")
log("Done.")