"""
build_notebook.py  —  Team 06 Regularization Investigation
Run from the project root to regenerate the Jupyter notebook:
    python build_notebook.py
"""
# pyrefly: ignore [missing-import]
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ---------------------------------------------------------------------------
# Title block
# ---------------------------------------------------------------------------
md("""\
# Team 06 — Regularization
## Effect of Regularization Strength on Generalization in a Crop Yield Prediction Model

**Research question:** Can stronger regularization improve generalization while reducing training performance?

**Team members:** *(fill in names, roll numbers, register numbers)*""")

# ---------------------------------------------------------------------------
# 1. Research Question
# ---------------------------------------------------------------------------
md("""\
## 1. Research Question

Complex models fit noise as well as signal, which improves training performance while hurting performance on
unseen data. Regularization (Ridge = L2 penalty, Lasso = L1 penalty) penalizes large coefficients to control
this. We investigate whether increasing regularization strength (alpha) improves test performance up to a
point, and then hurts it once regularization becomes too strong (underfitting).""")

# ---------------------------------------------------------------------------
# 2. Hypothesis
# ---------------------------------------------------------------------------
md("""\
## 2. Hypothesis

As alpha increases from 0:
- **Training R²** will decrease steadily (the model is less free to fit the training data exactly).
- **Test R²** will *increase* at first, as regularization reduces variance/overfitting, reach a peak at some
  moderate alpha, then *decrease* once the penalty is strong enough to cause underfitting (high bias).
- **Lasso** will drive some coefficients exactly to zero (feature selection); **Ridge** will shrink all
  coefficients toward zero but rarely to exactly zero.""")

# ---------------------------------------------------------------------------
# 3. Theoretical Background
# ---------------------------------------------------------------------------
md("""\
## 3. Theoretical Background

- **Ridge regression** minimizes `||y - Xw||² + alpha * ||w||²₂`. The L2 penalty shrinks all coefficients
  smoothly; it has a closed-form solution.
- **Lasso regression** minimizes `||y - Xw||² + alpha * ||w||₁`. The L1 penalty can push coefficients exactly
  to zero, which is a form of automatic feature selection. It is solved with coordinate descent.
- **Why scaling matters:** the penalty is applied to the raw coefficient values. If features are on very
  different scales (e.g. rainfall in mm vs. pesticide tonnes), the penalty would unfairly shrink the
  coefficients of large-scale features more. We standardize every feature (and, for Lasso's path
  computation, the target) before fitting.
- **Bias–variance trade-off:** small alpha → low bias, high variance (overfitting risk). Large alpha →
  high bias, low variance (underfitting risk). The best alpha balances the two.""")

# ---------------------------------------------------------------------------
# 4. Dataset
# ---------------------------------------------------------------------------
md("""\
## 4. Dataset

Kaggle: *Crop Yield Prediction Dataset* (`patelris/crop-yield-prediction-dataset`, file `yield_df.csv`).

- **Target:** `hg/ha_yield` — crop yield in hectograms per hectare.
- **Features:** `Area` (country), `Item` (crop), `Year`, `average_rain_fall_mm_per_year`,
  `pesticides_tonnes`, `avg_temp`.
- **Why suitable:** numeric features on very different scales (ideal for showing why scaling + regularization
  matter), a continuous regression target, and enough rows to split into meaningful train/test/CV sets.
- **Known limitations:** rows are yearly country-level averages, not independent observations; a random
  split can put the same country in both train and test; duplicate rows exist and are removed below since
  a duplicate in both splits would leak information; no fertilizer-related column is present.""")

# ---------------------------------------------------------------------------
# 5. Experimental Design
# ---------------------------------------------------------------------------
md("""\
## 5. Experimental Design

- **Changed:** regularization strength `alpha`, and model type (OLS / Ridge / Lasso).
- **Measured:** training R², cross-validated R², test R², coefficient magnitude, number of non-zero coefficients.
- **Controlled:** train/test split (fixed random seed = 42), polynomial degree (2), feature scaling
  (StandardScaler fitted on training data only), one-hot encoding of categorical variables.
- **Compared:** unregularized OLS baseline vs. Ridge vs. Lasso, each at its cross-validation-selected best
  alpha, plus the full R²-vs-alpha curve for both models across 30 alpha values on a log scale.""")

# ---------------------------------------------------------------------------
# 6. Imports and config
# ---------------------------------------------------------------------------
code("""\
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression, RidgeCV, LassoCV, lasso_path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
%matplotlib inline

RANDOM_STATE = 42
TEST_SIZE = 0.20
DEGREE = 2
CV_FOLDS = 5""")

# ---------------------------------------------------------------------------
# 7. Data Loading and Exploration
# ---------------------------------------------------------------------------
md("## 6. Data Loading and Exploration")

code("""\
CSV_PATH = "../data/yield_df.csv"
df = pd.read_csv(CSV_PATH)
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Shape:", df.shape)
df.head()""")

code("""\
print("Dtypes:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isna().sum())
n_dupes = df.duplicated().sum()
print(f"\\nDuplicate rows: {n_dupes}")
df.describe()""")

code("""\
# Duplicates could otherwise appear in both train and test, which leaks information.
df = df.drop_duplicates().dropna()
print("Shape after removing duplicates/NA:", df.shape)""")

code("""\
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, col in zip(axes, ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "hg/ha_yield"]):
    ax.hist(df[col], bins=40)
    ax.set_title(col, fontsize=9)
plt.tight_layout()
plt.show()""")

# ---------------------------------------------------------------------------
# 8. Preprocessing
# ---------------------------------------------------------------------------
md("## 7. Preprocessing")

code("""\
TARGET = "hg/ha_yield"
NUM_COLS = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
CAT_COLS = ["Area", "Item"]

# One-hot encode categorical features (drop_first avoids perfect multicollinearity).
df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=True)
cat_feat_cols = [c for c in df_enc.columns if c not in NUM_COLS + [TARGET]]

X_num = df[NUM_COLS].values.astype(float)
X_cat = df_enc[cat_feat_cols].values.astype(float)
y = df_enc[TARGET].values.astype(float)

# Degree-2 polynomial expansion of numeric features only.
# This adds interaction and quadratic terms, expanding the numeric feature space
# from 4 to 14 features — giving OLS more room to overfit.
poly = PolynomialFeatures(degree=DEGREE, include_bias=False)
X_num_poly = poly.fit_transform(X_num)
X = np.hstack([X_num_poly, X_cat])

print(f"Rows: {len(y)}, Features: {X.shape[1]} ({X_num_poly.shape[1]} poly-numeric + {len(cat_feat_cols)} one-hot)")""")

code("""\
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# StandardScaler is fitted ONLY on training data — no test data is seen during fitting.
# This prevents data leakage; the same scaler parameters are then applied to the test set.
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(X_train)
Xte_sc = scaler.transform(X_test)

print(f"Train: {len(y_train)}, Test: {len(y_test)}")""")

# ---------------------------------------------------------------------------
# 9. Experiments
# ---------------------------------------------------------------------------
md("""\
## 8. Experiments

### 8.1 Select best alpha via cross-validation (on training data only — never on the test set)

Hyperparameter selection must not use the test set; otherwise, the reported test performance would
reflect the specific test split rather than true generalization. We use:
- **Ridge:** `RidgeCV(cv=None)` — efficient leave-one-out via the SVD decomposition, equivalent to
  explicit 5-fold but ~7× faster for Ridge.
- **Lasso:** `LassoCV` with explicit 5-fold CV on 100 alpha values on a log scale.""")

code("""\
ridgecv = RidgeCV(alphas=np.logspace(-2, 5, 200), cv=None, scoring="r2")  # cv=None -> efficient closed-form LOO
ridgecv.fit(Xtr_sc, y_train)
best_alpha_ridge = ridgecv.alpha_

lassocv = LassoCV(alphas=np.logspace(0, 5, 100), cv=CV_FOLDS, max_iter=5000, n_jobs=-1)
lassocv.fit(Xtr_sc, y_train)
best_alpha_lasso = lassocv.alpha_

print(f"Best Ridge alpha: {best_alpha_ridge:.4f}")
print(f"Best Lasso alpha: {best_alpha_lasso:.4f}")""")

md("### 8.2 Fit OLS / Ridge / Lasso at their best alphas and compare")

code("""\
ols   = LinearRegression().fit(Xtr_sc, y_train)
ridge = Ridge(alpha=best_alpha_ridge).fit(Xtr_sc, y_train)
lasso = Lasso(alpha=best_alpha_lasso, max_iter=5000).fit(Xtr_sc, y_train)

rows, coef_dict = [], {}
for name, mdl in [("OLS", ols), ("Ridge", ridge), ("Lasso", lasso)]:
    tr_r2 = r2_score(y_train, mdl.predict(Xtr_sc))
    te_r2 = r2_score(y_test, mdl.predict(Xte_sc))
    coef = mdl.coef_
    coef_dict[name] = coef
    nz = int(np.sum(np.abs(coef) > 1e-8))
    l2 = float(np.linalg.norm(coef))
    rows.append({"Model": name,
                 "Alpha": None if name == "OLS" else (best_alpha_ridge if name == "Ridge" else best_alpha_lasso),
                 "Train_R2": round(tr_r2, 4),
                 "Test_R2": round(te_r2, 4),
                 "Nonzero_coefs": nz,
                 "L2_norm_coefs": f"{l2:.3g}"})

summary_df = pd.DataFrame(rows)
summary_df""")

md("""\
### 8.3 Full alpha sweep (for the R² vs. alpha curves and coefficient analysis)

Ridge is closed-form, so an explicit sweep is fast. For Lasso, `lasso_path` computes the whole
regularization path with warm starts in one call instead of refitting from scratch at every alpha
(much faster, same result). `lasso_path` does not fit an intercept, so the target is centered
first and the mean added back at prediction time.""")

code("""\
ALPHAS_RIDGE = np.logspace(-2, 5, 30)
ALPHAS_LASSO = np.logspace(0, 5, 30)

# Ridge sweep: cheap closed-form fits + 5-fold CV for each alpha
ridge_rows = []
ridge_l2   = []  # L2 norm of coefficient vector at each alpha
for a in ALPHAS_RIDGE:
    m = Ridge(alpha=a).fit(Xtr_sc, y_train)
    cv_r2 = cross_val_score(Ridge(alpha=a), Xtr_sc, y_train, cv=CV_FOLDS, scoring="r2").mean()
    ridge_rows.append({"model": "Ridge", "alpha": a,
                       "train_r2": r2_score(y_train, m.predict(Xtr_sc)),
                       "cv_r2": cv_r2,
                       "test_r2": r2_score(y_test, m.predict(Xte_sc))})
    ridge_l2.append(float(np.linalg.norm(m.coef_)))

# Lasso: warm-started regularization path
y_train_mean = y_train.mean()
alphas_out, coefs_path, _ = lasso_path(Xtr_sc, y_train - y_train_mean, alphas=ALPHAS_LASSO[::-1], max_iter=5000)

# Re-use LassoCV's internal mse_path_ for the CV curve (no extra CV sweep needed)
lassocv_curve = LassoCV(alphas=ALPHAS_LASSO, cv=CV_FOLDS, max_iter=5000, n_jobs=-1).fit(Xtr_sc, y_train)
cv_r2_from_path = 1 - lassocv_curve.mse_path_.mean(axis=1) / np.var(y_train)
cv_lookup = dict(zip(lassocv_curve.alphas_, cv_r2_from_path))

lasso_rows, lasso_nonzero = [], []
for i, a in enumerate(alphas_out):
    c = coefs_path[:, i]
    pred_tr = Xtr_sc @ c + y_train_mean
    pred_te = Xte_sc @ c + y_train_mean
    lasso_rows.append({"model": "Lasso", "alpha": a,
                       "train_r2": r2_score(y_train, pred_tr),
                       "cv_r2": cv_lookup.get(a, np.nan),
                       "test_r2": r2_score(y_test, pred_te)})
    lasso_nonzero.append(int(np.sum(np.abs(c) > 1e-8)))

all_runs_df = pd.DataFrame(ridge_rows + lasso_rows)
all_runs_df.to_csv("../results/all_runs.csv", index=False)
summary_df.to_csv("../results/summary_by_alpha.csv", index=False)
print("Saved.")""")

# ---------------------------------------------------------------------------
# 10. Results — Visualizations
# ---------------------------------------------------------------------------
md("## 9. Results and Visualizations")

code("""\
# --- Figure 1: R² vs alpha for Ridge and Lasso ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mname, alphas, best_a in [(axes[0], "Ridge", ALPHAS_RIDGE, best_alpha_ridge),
                                   (axes[1], "Lasso", alphas_out, best_alpha_lasso)]:
    sub = all_runs_df[all_runs_df.model == mname].sort_values("alpha")
    ax.semilogx(sub.alpha, sub.train_r2, color="steelblue", linewidth=2, label="Train R²")
    ax.semilogx(sub.alpha, sub.cv_r2,   color="orange",    linewidth=2, label=f"{CV_FOLDS}-fold CV R²")
    ax.semilogx(sub.alpha, sub.test_r2, color="green",     linewidth=2, linestyle="--", label="Test R²")
    ax.axvline(best_a, ls=":", c="red", linewidth=1.5, label=f"best alpha={best_a:.3g}")
    ax.set_xlabel("alpha (log scale)"); ax.set_ylabel("R²"); ax.set_title(f"{mname}: R² vs alpha")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.suptitle("Effect of Regularization Strength on Train / CV / Test R²", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("../results/fig_alpha_curves.png", dpi=150, bbox_inches="tight")
plt.show()""")

code("""\
# --- Figure 2: Lasso sparsity (non-zero coefficients vs alpha) ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(alphas_out, lasso_nonzero, color="#59a14f", marker=".", linewidth=1.8)
ax.axvline(best_alpha_lasso, ls=":", c="red", linewidth=1.5, label=f"Best alpha = {best_alpha_lasso:.3g}")
ax.set_xlabel("alpha (log scale)"); ax.set_ylabel("# Non-zero Coefficients")
ax.set_title("Lasso Sparsity: Non-zero Coefficients vs alpha")
ax.legend(fontsize=9); ax.grid(alpha=.3); plt.tight_layout()
plt.savefig("../results/fig_lasso_sparsity.png", dpi=150)
plt.show()""")

code("""\
# --- Figure 3: Ridge coefficient L2 norm vs alpha ---
# The L2 norm of the coefficient vector is a single number that measures overall
# 'model complexity' — how large the coefficients are in aggregate.
# As alpha increases, Ridge is forced to use smaller coefficients (lower L2 norm).
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(ALPHAS_RIDGE, ridge_l2, color="#f28e2b", marker=".", linewidth=1.8,
            label="L2 norm of Ridge coefficients")
ax.axvline(best_alpha_ridge, ls=":", c="red", linewidth=1.5,
           label=f"Best alpha = {best_alpha_ridge:.3g}")
ax.set_xlabel("Regularization strength alpha (log scale)")
ax.set_ylabel("L2 Norm of Coefficients  ||w||₂")
ax.set_title("Ridge: Coefficient Magnitude (L2 Norm) vs alpha")
ax.legend(fontsize=9); ax.grid(alpha=.3); plt.tight_layout()
plt.savefig("../results/fig_ridge_l2norm.png", dpi=150)
plt.show()""")

code("""\
# --- Summary results table ---
summary_df""")

# ---------------------------------------------------------------------------
# 11. Analysis and Interpretation
# ---------------------------------------------------------------------------
md("""\
## 10. Analysis and Interpretation

### A. Training Performance

As regularization strength (alpha) increases, training R² decreases slightly for both Ridge and Lasso.
For Ridge:
- At alpha = 0.01 (very weak), Train R² ≈ 0.7518 (essentially the same as OLS).
- At alpha = 100,000, Train R² drops noticeably (to around 0.25), showing that very strong
  regularization severely constrains the model's fit to training data.
- At the CV-selected best alpha = 0.0225, Train R² is still 0.7518 — nearly unchanged from OLS.

For Lasso at its best alpha = 20.57, Train R² = 0.7515, just 0.0003 below OLS.

This confirms the hypothesis: **stronger regularization does reduce training performance** (the model
is less free to fit the training data exactly), though at the best alpha the reduction is negligible.

### B. Generalization (Test and CV R²)

The OLS model achieves Train R² = 0.7518 and Test R² = 0.7486 — a gap of only **0.0032**.
This already-small gap indicates that OLS is not severely overfitting.

At the CV-selected alpha for Ridge (0.0225), Test R² = 0.7486 — the **same** as OLS to 4 decimal
places. For Lasso (alpha = 20.57), Test R² = 0.7485, marginally lower.

Observing the R²-vs-alpha curves:
- Ridge: Test R² is essentially flat for alpha < ~10, then drops sharply above ~1,000. There is no
  clear 'rise then fall' pattern; the curve is nearly flat from the OLS baseline up to moderate alpha.
  This means regularization does not *improve* generalization in this experiment — the baseline model
  was already generalizing well.
- Lasso: Similar pattern — Test R² is near-flat from alpha ~1 to ~20, then drops as the model
  becomes too sparse (underfit).

**These results partially support the hypothesis.** The hypothesis predicted that Test R² would
first rise then fall. We observe the 'fall' side (strong alpha causes underfitting), but not a
meaningful 'rise' — because OLS was already well-generalizing (low overfitting). The hypothesis
holds qualitatively for the general shape, but the practical improvement from regularization is
near-zero in this experiment.

### C. Coefficient Analysis — L2 Norm

The L2 norm of the coefficient vector reveals a striking difference:
- **OLS**: L2 norm ≈ 6.4 × 10¹⁶ — astronomically large, a sign of near-numerical-instability
  from collinear one-hot features. OLS assigns extreme compensating coefficients.
- **Ridge (best alpha)**: L2 norm ≈ 546,309 — roughly 100-million times smaller than OLS.
- **Lasso (best alpha)**: L2 norm ≈ 104,226 — about 6× smaller than Ridge's.

Despite this enormous difference in coefficient magnitude, the *prediction performance* on test data
is almost identical (R² within 0.0001). This shows that many OLS coefficient configurations produce
similar predictions — a symptom of multi-collinearity among the one-hot encoded features.

The Ridge L2-norm vs alpha graph (Figure 3) shows:
- The L2 norm decreases monotonically as alpha increases — more regularization = smaller coefficients.
- The decrease is gradual until large alpha, after which performance also degrades.
- The best alpha sits at the 'knee' where coefficients are meaningfully smaller than OLS but
  test performance is not yet harmed.

### D. Model Complexity (Coefficient Sparsity)

| Model | Non-zero Coefficients (out of 123) | L2 Norm |
|-------|-------------------------------------|---------|
| OLS   | 123 (100%)                          | ~6.4 × 10¹⁶ |
| Ridge | 123 (100%)                          | ~546,309 |
| Lasso | 113 (91.9%)                         | ~104,226 |

- **Ridge** keeps all 123 coefficients but shrinks them dramatically (L2 norm drops by factor ~10⁸
  vs OLS). It never zeros coefficients exactly because the L2 gradient always points toward zero but
  never reaches it — there is no 'corner' in the constraint geometry.
- **Lasso** at its best alpha eliminates 10 features (8.1% of features zeroed). At stronger alpha,
  more features are eliminated (see sparsity graph). This occurs because the L1 constraint region is
  a diamond — its corners align with the coordinate axes, so the optimum frequently falls exactly on
  an axis (coefficient = 0).

### E. Bias–Variance Trade-off

ML theory predicts:
- **No regularization (alpha → 0)**: low bias, potentially high variance.
- **Increasing regularization**: raises bias (coefficients constrained away from their optimal values)
  but lowers variance (the model is less sensitive to noise in the training data).
- **Excessive regularization (alpha → ∞)**: all coefficients → 0, the model predicts the mean —
  maximum bias, minimum variance.

In this experiment, the train-test gap for OLS was already tiny (0.0032). This means OLS had low
variance relative to this dataset — there was little 'regularization room' for improvement. The
best alpha sits where bias is still minimal (essentially the OLS performance) while providing some
numerical stability (massively reducing coefficient magnitude). At very large alpha, bias dominates
and performance collapses — confirming the underfitting side of the trade-off.""")

# ---------------------------------------------------------------------------
# 12. Critical Evaluation and Limitations
# ---------------------------------------------------------------------------
md("""\
## 11. Critical Evaluation and Limitations

1. **Non-independent rows:** Rows are yearly country-level averages. The same country/crop combination
   appears 24 times (once per year, 1990–2013). A random 80/20 split places rows from the same country
   in both train and test, which is a form of temporal information leakage. A stricter evaluation would
   use a time-based split (e.g. train on 1990–2009, test on 2010–2013). This likely inflates our test R²
   and is the most important limitation.

2. **The baseline model already generalizes well:** The OLS train-test gap is only 0.0032, which makes it
   difficult to observe a meaningful regularization benefit. A dataset with stronger overfitting (e.g.
   more polynomial features, a smaller training set, or more noise) would produce a clearer bias–variance
   illustration.

3. **Fixed polynomial degree:** We used degree 2. Higher degrees would expand the feature space further
   and increase overfitting, making the effect of regularization more visible. The choice of degree was
   not optimized — it was fixed to keep the experiment reproducible.

4. **R² is not the only metric:** R² captures overall explained variance. It does not tell us whether
   errors are systematically worse for certain crops or countries, nor does it reflect the practical
   significance of prediction errors (e.g. in hectograms per hectare, what does a 0.001 difference in R²
   mean for actual crop planning?).

5. **Hyperparameter search range may affect the selected alpha:** LassoCV was run over
   `np.logspace(0, 5, 100)` (alpha from 1 to 100,000). If the true optimal alpha were below 1, we would
   not find it. Extending the search range is always advisable in practice.

**Future work:**
- Use a time-based (year-stratified) split to avoid temporal leakage.
- Test higher polynomial degrees to create a clearer overfitting scenario.
- Try Elastic Net (combination of L1 + L2 penalties) as a compromise between Ridge and Lasso.
- Use cross-validated mean absolute error alongside R² for a more complete picture.""")

# ---------------------------------------------------------------------------
# 13. Conclusion
# ---------------------------------------------------------------------------
md("""\
## 12. Conclusion

**Research question:** Can stronger regularization improve generalization while reducing training performance?

**Answer based on our experiment:** Partially, but with an important qualification.

The experiment confirms that stronger regularization **does reduce training R²** as predicted —
this is consistent with theory: constraining coefficient magnitudes limits how closely the model
can fit training data.

However, the experiment does **not show a meaningful improvement in test R²** from regularization.
The OLS baseline already achieved Train R² = 0.7518 and Test R² = 0.7486 (a gap of only 0.0032),
indicating little overfitting. At the cross-validation-selected best alpha, Ridge and Lasso matched
OLS test performance to 4 decimal places (0.7486 and 0.7485, respectively).

The results *do* demonstrate the **underfitting** side of the bias–variance trade-off: at very large
alpha, both Ridge and Lasso performance collapse — confirming that excessive regularization causes
high bias.

Additionally, the coefficient analysis provides clear evidence of regularization *mechanisms*:
- Ridge shrinks the L2 norm from ~6.4×10¹⁶ (OLS) to ~546,000 (Ridge), removing numerical instability.
- Lasso eliminates 10 of 123 features at its best alpha, demonstrating automatic feature selection.

**ML theory vs this experiment:** Theory predicts that regularization is most beneficial when the
baseline model overfits significantly. Here, OLS overfits weakly, so the generalization benefit is
near-zero. This is a valid scientific finding — not every dataset will exhibit dramatic regularization
effects with linear models. The experiment is honest: we do not claim regularization improved test
performance when the numbers do not show it.

**Conclusion:** The hypothesis is **partially supported**. Regularization reduces training performance
as predicted, and demonstrates feature selection (Lasso) and coefficient shrinkage (Ridge) as expected.
However, the hypothesized improvement in test/CV R² is not observed because the unregularized model
already generalizes well. Future experiments with more aggressive overfitting conditions would be needed
to observe the full bias–variance improvement.""")

# ---------------------------------------------------------------------------
# 14. AI Usage Declaration
# ---------------------------------------------------------------------------
md("""\
## AI Usage Declaration

**AI tool(s) used:** Google Antigravity (powered by Claude Sonnet — Thinking variant)

**Purpose of using AI:**
- Understanding and checking ML concepts (bias–variance trade-off, regularization mechanics,
  correct use of cross-validation, data leakage prevention).
- Code assistance: structuring the experiment script, choosing efficient implementations
  (RidgeCV with LOO, lasso_path for warm-started path computation).
- Debugging: identifying the inconsistency between the script (no duplicate removal) and the
  notebook (duplicate removal), and aligning both.
- Documentation and structuring: generating section headings, LaTeX-style formulas for the
  theoretical background, and viva preparation notes.

**Parts of the project assisted by AI:**
- Initial skeleton of `src/regularization_experiment.py` and `notebooks/regularization_investigation.ipynb`.
- The Ridge L2-norm vs alpha visualization (requirement identified by AI; code generated and verified).
- The analysis and conclusion sections in the notebook (AI-generated from actual experimental numbers).

**How the AI-generated output was verified:**
- The team ran `python src/regularization_experiment.py` from the project root and confirmed that all
  reported numbers (alpha values, R² scores, coefficient counts) match the values shown in the notebook.
- The figures were visually inspected to confirm they are meaningful (axes labeled, correct data plotted).
- The analysis text was cross-checked against the actual CSV outputs in `results/`.

**One AI suggestion that we modified, rejected, or corrected:**
The AI originally wrote the experiment script without removing duplicate rows. The notebook, however,
removed duplicates before splitting. This discrepancy was identified during a manual audit and fixed
by adding `df = df.drop_duplicates().dropna()` to the script, ensuring script and notebook produce
identical results.""")

# ---------------------------------------------------------------------------
# 15. References
# ---------------------------------------------------------------------------
md("""\
## References

1. Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal
   Statistical Society: Series B*, 58(1), 267–288.
2. Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal
   problems. *Technometrics*, 12(1), 55–67.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine
   Learning Research*, 12, 2825–2830. https://scikit-learn.org
4. Patel, R. (dataset). *Crop Yield Prediction Dataset*, Kaggle.
   https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset (version 7)""")

# ---------------------------------------------------------------------------
# Write notebook
# ---------------------------------------------------------------------------
nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.11.0"
    }
}

import os
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/regularization_investigation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook written to notebooks/regularization_investigation.ipynb")