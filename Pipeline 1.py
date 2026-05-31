"""
=============================================================================
  VIRAT KOHLI ODI PERFORMANCE ANALYSIS & PREDICTION PIPELINE
  Master's Dissertation: Statistical and Machine Learning Techniques
  Dataset: 285 innings | 2008–2023
=============================================================================

USAGE (Google Colab):
    1. Upload kohli_odi_data.xlsx
    2. Run all cells (Runtime → Run all)

LIBRARY REQUIREMENTS:
    pip install statsmodels  ← run this once if SARIMAX is needed

MODELS TRAINED:
    ├── Ridge Regression
    ├── Lasso Regression
    ├── Random Forest Regressor
    ├── Gradient Boosting Regressor
    ├── SARIMAX (time-series)
    └── Stacking Ensemble (meta: Linear Regression)

OUTPUTS (15+ files):
    Evaluation table, Actual vs Predicted plots, Feature Importance,
    Residual plots, Rolling Forecast, Next-Match Prediction, saved model
=============================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. ENVIRONMENT SETUP
# ──────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # non-interactive backend (works in Colab too)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle, os, sys

from sklearn.linear_model      import Ridge, Lasso, LinearRegression
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection   import GridSearchCV, cross_val_score
from scipy.stats               import pearsonr

# SARIMAX — graceful fallback if not installed
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    print("[WARNING] statsmodels not found. SARIMAX step will be skipped.\n"
          "          Run: !pip install statsmodels  then restart kernel.")

# ── Output directory (works in Colab /content/ and locally) ──────────────────
OUTPUT_DIR = "kohli_pipeline_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved → {path}")

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = "#003366"
GOLD   = "#FFB300"
RED    = "#C0392B"
GREEN  = "#1E8449"
LIGHT  = "#EAF0FB"
sns.set_theme(style="whitegrid", palette="muted")

print("=" * 65)
print("  VIRAT KOHLI ODI PREDICTION PIPELINE  |  Starting …")
print("=" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/14]  LOADING DATA …")

EXCEL_PATH = "kohli_odi_data.xlsx"          # ← change path if needed

# All sentinel values treated as NaN on load
MISSING_VALS = ["-", "--", "DNB", "TDNB", "TDNB*", "DNB*", "", " "]

df_raw = pd.read_excel(
    EXCEL_PATH,
    na_values=MISSING_VALS
)

print(f"\n  Raw shape  : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print("\n  Column dtypes (raw):")
print(df_raw.dtypes.to_string())
print("\n  First 5 rows:")
print(df_raw.head().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/14]  CLEANING DATA …")

df = df_raw.copy()

# 2a. Force numeric columns (coerce non-numeric → NaN)
NUM_COLS = ["Runs", "BF", "Mins", "4s", "6s", "SR", "Pos",
            "Inns", "Z_avg_s_opp", "z_dismissalRateVsOpp",
            "Dismissal_type"]
for col in NUM_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 2b. Drop rows where Runs is still missing (DNB / TDNB innings)
before = len(df)
df.dropna(subset=["Runs"], inplace=True)
print(f"  Dropped {before - len(df)} non-batting rows  (Runs = NaN/DNB/TDNB)")

# 2c. Parse Start Date → datetime; sort chronologically
df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
df.dropna(subset=["Start Date"], inplace=True)
df.sort_values("Start Date", inplace=True)
df.reset_index(drop=True, inplace=True)

# 2d. Compute Rest_Days from 'Rest Day' column
#     (stored as 1900-01-XX Excel serial → extract day-of-month as integer)
if "Rest Day" in df.columns:
    df["Rest Day"] = pd.to_datetime(df["Rest Day"], errors="coerce")
    df["Rest_Days"] = df["Rest Day"].dt.day.fillna(0).astype(int)
else:
    # Fallback: compute from Start Date differences
    df["Rest_Days"] = df["Start Date"].diff().dt.days.fillna(0).clip(lower=0).astype(int)

# 2e. Impute missing numeric → median; categorical → "Unknown"
for col in df.select_dtypes(include=[np.number]).columns:
    med = df[col].median()
    df[col].fillna(med, inplace=True)

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col].fillna("Unknown", inplace=True)

print(f"\n  Clean shape : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Date range  : {df['Start Date'].min().date()}  →  {df['Start Date'].max().date()}")
print(f"  Missing values remaining:\n{df.isnull().sum()[df.isnull().sum() > 0]}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/14]  FEATURE ENGINEERING …")

# ── Lag features (shift(1) avoids leakage) ───────────────────────────────────
df["Runs_lag1"]      = df["Runs"].shift(1)
df["SR_lag1"]        = df["SR"].shift(1)
df["BF_lag1"]        = df["BF"].shift(1)
df["Mins_lag1"]      = df["Mins"].shift(1)

# ── Rolling averages (shift(1) on the series before rolling) ─────────────────
_shifted_runs = df["Runs"].shift(1)
df["Runs_roll3"]  = _shifted_runs.rolling(3,  min_periods=1).mean()
df["Runs_roll5"]  = _shifted_runs.rolling(5,  min_periods=1).mean()
df["Runs_roll10"] = _shifted_runs.rolling(10, min_periods=1).mean()

# ── Strike-rate trend (SR difference from last innings) ───────────────────────
df["SR_trend"] = df["SR"].shift(1) - df["SR"].shift(2)

# ── Cumulative stats ──────────────────────────────────────────────────────────
df["Cum_avg"]    = _shifted_runs.expanding().mean()
df["Innings_No"] = range(1, len(df) + 1)       # sequential innings counter

# ── Drop first row (all lag/roll features will be NaN for row 0) ─────────────
df.dropna(subset=["Runs_lag1", "Runs_roll3"], inplace=True)
df.reset_index(drop=True, inplace=True)

new_features = ["Runs_lag1", "SR_lag1", "BF_lag1", "Mins_lag1",
                "Runs_roll3", "Runs_roll5", "Runs_roll10",
                "SR_trend", "Cum_avg", "Innings_No", "Rest_Days"]
print(f"  Created {len(new_features)} lag/rolling/time features:")
for f in new_features:
    print(f"    • {f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ENCODING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/14]  ONE-HOT ENCODING …")

# Opposition
df = pd.get_dummies(df, columns=["Opposition"], prefix="Opp", drop_first=True)

# Ground: already 0/1 binary in this dataset — no encoding needed
# Dismissal_type: already numeric — keep as-is

cat_encoded = [c for c in df.columns if c.startswith("Opp_")]
print(f"  Opposition dummies created  : {len(cat_encoded)} columns")
print(f"  Final shape after encoding  : {df.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/14]  EXPLORATORY DATA ANALYSIS …")

# ── 5a. Distribution of Runs ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Virat Kohli ODI — Run Distribution", fontsize=15, fontweight="bold", color=NAVY)

axes[0].hist(df["Runs"], bins=30, color=NAVY, edgecolor="white", alpha=0.85)
axes[0].set_title("Histogram of Runs", color=NAVY)
axes[0].set_xlabel("Runs Scored")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df["Runs"].mean(),   color=GOLD, lw=2, ls="--", label=f"Mean  {df['Runs'].mean():.1f}")
axes[0].axvline(df["Runs"].median(), color=RED,  lw=2, ls=":",  label=f"Median {df['Runs'].median():.1f}")
axes[0].legend()

axes[1].boxplot(df["Runs"], vert=True, patch_artist=True,
                boxprops=dict(facecolor=LIGHT, color=NAVY),
                medianprops=dict(color=GOLD, lw=2),
                whiskerprops=dict(color=NAVY),
                capprops=dict(color=NAVY),
                flierprops=dict(marker="o", color=RED, alpha=0.5))
axes[1].set_title("Box-plot of Runs", color=NAVY)
axes[1].set_ylabel("Runs")
plt.tight_layout()
savefig("01_runs_distribution.png")

# ── 5b. Time-series plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.fill_between(df["Start Date"], df["Runs"], alpha=0.15, color=NAVY)
ax.plot(df["Start Date"], df["Runs"],       color=NAVY, lw=0.8,  alpha=0.6, label="Innings Runs")
ax.plot(df["Start Date"], df["Runs_roll10"], color=GOLD, lw=2.5, label="10-innings rolling avg")
ax.set_title("Virat Kohli — ODI Runs Over Time (2008–2023)", fontsize=14, fontweight="bold", color=NAVY)
ax.set_xlabel("Date");  ax.set_ylabel("Runs")
ax.legend()
plt.tight_layout()
savefig("02_runs_timeseries.png")

# ── 5c. Correlation heatmap (numeric features only) ──────────────────────────
NUM_HEAT_COLS = ["Runs","BF","Mins","4s","6s","SR","Pos","Inns",
                 "Z_avg_s_opp","z_dismissalRateVsOpp","Dismissal_type",
                 "Runs_lag1","SR_lag1","Runs_roll3","Runs_roll5","Runs_roll10",
                 "Rest_Days","SR_trend","Cum_avg"]
heat_df = df[[c for c in NUM_HEAT_COLS if c in df.columns]]
corr    = heat_df.corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Kohli ODI Features", fontsize=13, fontweight="bold", color=NAVY)
plt.tight_layout()
savefig("03_correlation_heatmap.png")

print("  Descriptive statistics (key columns):")
print(df[["Runs","BF","Mins","SR","4s","6s"]].describe().round(2).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE SELECTION  (VIF + high-correlation removal)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/14]  FEATURE SELECTION …")

# ── Define modelling feature set ─────────────────────────────────────────────
BASE_FEATURES = [
    "BF", "Mins", "4s", "6s", "SR", "Pos", "Inns",
    "Z_avg_s_opp", "z_dismissalRateVsOpp", "Dismissal_type",
    "Runs_lag1", "SR_lag1", "BF_lag1", "Mins_lag1",
    "Runs_roll3", "Runs_roll5", "Runs_roll10",
    "SR_trend", "Cum_avg", "Innings_No", "Rest_Days", "Ground"
]
# Add one-hot encoded opposition columns
BASE_FEATURES += cat_encoded
# Keep only columns that exist in df
FEATURE_COLS = [c for c in BASE_FEATURES if c in df.columns]
TARGET       = "Runs"

# 6a. Remove highly correlated pairs (|r| > 0.85) ────────────────────────────
feat_df = df[FEATURE_COLS].copy()
corr_mat = feat_df.corr().abs()
upper    = corr_mat.where(np.triu(np.ones(corr_mat.shape, dtype=bool), k=1))
to_drop  = [col for col in upper.columns if any(upper[col] > 0.85)]
print(f"  Dropping {len(to_drop)} highly-correlated features (|r|>0.85): {to_drop}")
FEATURE_COLS = [c for c in FEATURE_COLS if c not in to_drop]

# 6b. VIF — manual computation (no statsmodels dependency) ───────────────────
def compute_vif(X_df):
    """Return VIF series for each column in X_df."""
    from sklearn.linear_model import LinearRegression as LR
    vif_vals = {}
    X = X_df.values
    for i, col in enumerate(X_df.columns):
        y_vif  = X[:, i]
        X_rest = np.delete(X, i, axis=1)
        ss_tot = np.sum((y_vif - y_vif.mean()) ** 2)
        if ss_tot == 0:
            vif_vals[col] = np.inf
            continue
        lr = LR(fit_intercept=True).fit(X_rest, y_vif)
        ss_res = np.sum((y_vif - lr.predict(X_rest)) ** 2)
        r2     = 1 - ss_res / ss_tot
        vif_vals[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(vif_vals).sort_values(ascending=False)

# Only compute VIF on strictly numeric (non-dummy) features to keep it legible
VIF_CANDIDATES = [c for c in FEATURE_COLS
                  if c in df.select_dtypes(include=[np.number]).columns
                  and not c.startswith("Opp_")]

vif_series = compute_vif(df[VIF_CANDIDATES].fillna(0))
print("\n  VIF scores (top 15):")
print(vif_series.head(15).round(2).to_string())

# Drop features with VIF > 10 that are redundant
high_vif = vif_series[vif_series > 10].index.tolist()
# Protect the most predictive features from VIF removal
PROTECTED = {"Runs_lag1", "Runs_roll3", "4s", "6s", "Mins", "BF"}
high_vif_drop = [c for c in high_vif if c not in PROTECTED]
print(f"\n  Dropping {len(high_vif_drop)} high-VIF features: {high_vif_drop}")
FEATURE_COLS = [c for c in FEATURE_COLS if c not in high_vif_drop]
print(f"  Final feature count: {len(FEATURE_COLS)}")

# 6c. Final NaN cleanup on feature columns (median imputation)
for col in FEATURE_COLS:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TRAIN-TEST SPLIT  (time-based, last 30 innings = test)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/14]  TRAIN-TEST SPLIT (time-based) …")

TEST_SIZE = 30

# Final safe extraction — fill any residual NaN with column median, then cast to float
feat_matrix = df[FEATURE_COLS].copy()
feat_matrix = feat_matrix.apply(lambda col: col.fillna(col.median()) if col.isnull().any() else col)
feat_matrix = feat_matrix.astype(float)

X = feat_matrix.values
y = df[TARGET].astype(float).values

X_train, X_test = X[:-TEST_SIZE], X[-TEST_SIZE:]
y_train, y_test = y[:-TEST_SIZE], y[-TEST_SIZE:]

dates_test = df["Start Date"].values[-TEST_SIZE:]

print(f"  Training set : {len(X_train)} innings "
      f"({df['Start Date'].iloc[0].date()} → {df['Start Date'].iloc[-(TEST_SIZE+1)].date()})")
print(f"  Test set     : {len(X_test)}  innings "
      f"({df['Start Date'].iloc[-TEST_SIZE].date()} → {df['Start Date'].iloc[-1].date()})")

# Scale features
scaler  = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MODEL BUILDING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/14]  TRAINING MODELS …")

# ── Helper: evaluate a fitted model ──────────────────────────────────────────
def evaluate(name, model, Xtr, ytr, Xte, yte, scaled=True):
    """Fit model (on training data) and return eval dict + predictions."""
    Xtr_in = Xtr if scaled else X_train
    Xte_in = Xte if scaled else X_test
    model.fit(Xtr_in, ytr)
    preds = model.predict(Xte_in)
    preds = np.clip(preds, 0, None)          # runs cannot be negative

    rmse = np.sqrt(mean_squared_error(yte, preds))
    mae  = mean_absolute_error(yte, preds)
    r2   = r2_score(yte, preds)
    mask = yte > 0                           # guard MAPE division by zero
    mape = (np.abs((yte[mask] - preds[mask]) / yte[mask])).mean() * 100 if mask.sum() > 0 else np.nan

    print(f"  {name:35s}  RMSE={rmse:6.2f}  MAE={mae:6.2f}  R²={r2:6.3f}  MAPE={mape:6.1f}%")
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2, "MAPE(%)": mape}, preds, model

results  = []
all_preds = {}

# ── 8a. Ridge Regression ─────────────────────────────────────────────────────
res, pred, m_ridge = evaluate("Ridge Regression", Ridge(alpha=1.0),
                               Xs_train, y_train, Xs_test, y_test)
results.append(res);  all_preds["Ridge"] = pred

# ── 8b. Lasso Regression ─────────────────────────────────────────────────────
res, pred, m_lasso = evaluate("Lasso Regression", Lasso(alpha=0.5, max_iter=5000),
                               Xs_train, y_train, Xs_test, y_test)
results.append(res);  all_preds["Lasso"] = pred

# ── 8c. Random Forest ────────────────────────────────────────────────────────
res, pred, m_rf = evaluate("Random Forest", RandomForestRegressor(
                                n_estimators=200, max_depth=8,
                                random_state=42, n_jobs=-1),
                            X_train, y_train, X_test, y_test, scaled=False)
results.append(res);  all_preds["Random Forest"] = pred

# ── 8d. Gradient Boosting ────────────────────────────────────────────────────
res, pred, m_gb = evaluate("Gradient Boosting", GradientBoostingRegressor(
                                n_estimators=200, learning_rate=0.05,
                                max_depth=4, random_state=42),
                            X_train, y_train, X_test, y_test, scaled=False)
results.append(res);  all_preds["Gradient Boosting"] = pred


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TIME-SERIES MODEL (SARIMAX)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[9/14]  TIME-SERIES MODEL (SARIMAX) …")

if STATSMODELS_OK:
    try:
        ts_train = df["Runs"].values[:-TEST_SIZE]
        ts_test  = df["Runs"].values[-TEST_SIZE:]

        # Exogenous: lag1 runs + rolling-3 (no leakage — already computed)
        exog_cols  = ["Runs_lag1", "Runs_roll3"]
        exog_train = df[exog_cols].values[:-TEST_SIZE]
        exog_test  = df[exog_cols].values[-TEST_SIZE:]

        sarimax_model = SARIMAX(
            ts_train,
            exog=exog_train,
            order=(1, 0, 1),
            seasonal_order=(0, 0, 0, 0),   # no strong seasonal pattern in ODI
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        sarimax_fit  = sarimax_model.fit(disp=False, maxiter=200)
        sarimax_pred = sarimax_fit.forecast(steps=TEST_SIZE, exog=exog_test)
        sarimax_pred = np.clip(sarimax_pred, 0, None)

        rmse = np.sqrt(mean_squared_error(ts_test, sarimax_pred))
        mae  = mean_absolute_error(ts_test, sarimax_pred)
        r2   = r2_score(ts_test, sarimax_pred)
        mask = ts_test > 0
        mape = (np.abs((ts_test[mask] - sarimax_pred[mask]) / ts_test[mask])).mean() * 100 if mask.sum() > 0 else np.nan

        print(f"  {'SARIMAX(1,0,1) + exog':35s}  RMSE={rmse:6.2f}  MAE={mae:6.2f}  R²={r2:6.3f}  MAPE={mape:6.1f}%")
        results.append({"Model": "SARIMAX(1,0,1)", "RMSE": rmse, "MAE": mae, "R²": r2, "MAPE(%)": mape})
        all_preds["SARIMAX"] = sarimax_pred

    except Exception as e:
        print(f"  [SARIMAX failed] {e}")
        STATSMODELS_OK = False

if not STATSMODELS_OK:
    print("  SARIMAX skipped — statsmodels unavailable.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ENSEMBLE  (Stacking Regressor)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[10/14]  STACKING ENSEMBLE …")

estimators = [
    ("ridge", Ridge(alpha=1.0)),
    ("lasso", Lasso(alpha=0.5, max_iter=5000)),
    ("rf",    RandomForestRegressor(n_estimators=150, max_depth=7, random_state=42, n_jobs=-1)),
    ("gb",    GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42))
]
stacker = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    cv=5,                         # integer cv required for sklearn compatibility
    n_jobs=-1
)

res, pred, m_stack = evaluate("Stacking Ensemble", stacker,
                               Xs_train, y_train, Xs_test, y_test)
results.append(res);  all_preds["Stacking Ensemble"] = pred


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — EVALUATION TABLE  +  BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[11/14]  EVALUATION SUMMARY …")

results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
results_df.index += 1   # rank from 1

print("\n  ┌─────────────────────────────────────────────────────────────────┐")
print("  │               MODEL COMPARISON TABLE                           │")
print("  └─────────────────────────────────────────────────────────────────┘")
print(results_df.to_string(index=True, float_format=lambda x: f"{x:.3f}"))

best_row   = results_df.iloc[0]
BEST_MODEL_NAME = best_row["Model"]
print(f"\n  🏆  Best model  → {BEST_MODEL_NAME}  (RMSE={best_row['RMSE']:.2f}, R²={best_row['R²']:.3f})")

# Map best name to fitted model object
_model_map = {
    "Ridge Regression":     m_ridge,
    "Lasso Regression":     m_lasso,
    "Random Forest":        m_rf,
    "Gradient Boosting":    m_gb,
    "Stacking Ensemble":    m_stack,
}
best_model_obj = _model_map.get(BEST_MODEL_NAME, m_stack)

# Save results CSV
csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
results_df.to_csv(csv_path, index=True)
print(f"  ✔ Results table saved → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[12/14]  GENERATING VISUALISATIONS …")

import matplotlib.dates as mdates

# ── 12a. Actual vs Predicted — all models ────────────────────────────────────
n_models = len(all_preds)
fig, axes = plt.subplots(n_models, 1, figsize=(15, 4.5 * n_models), sharex=True)
if n_models == 1:
    axes = [axes]

fig.suptitle("Virat Kohli — Actual vs Predicted Runs (Test Set)",
             fontsize=15, fontweight="bold", color=NAVY, y=1.01)

for ax, (mname, mpred) in zip(axes, all_preds.items()):
    x_axis = pd.to_datetime(dates_test)
    ax.fill_between(x_axis, y_test, alpha=0.12, color=NAVY)
    ax.plot(x_axis, y_test,  "o-", color=NAVY, lw=1.8, ms=4, label="Actual")
    ax.plot(x_axis, mpred,   "s--", color=GOLD, lw=1.8, ms=4, label=f"Predicted ({mname})")
    row = results_df[results_df["Model"] == mname]
    if not row.empty:
        ax.set_title(f"{mname}  |  RMSE={row['RMSE'].values[0]:.2f}  R²={row['R²'].values[0]:.3f}",
                     color=NAVY, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Runs")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

plt.tight_layout()
savefig("04_actual_vs_predicted_all.png")

# ── 12b. Best model — scatter plot ───────────────────────────────────────────
best_preds = all_preds.get(BEST_MODEL_NAME, list(all_preds.values())[0])
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, best_preds, color=NAVY, edgecolors="white", s=70, alpha=0.8, zorder=3)
lims = [min(y_test.min(), best_preds.min()) - 5, max(y_test.max(), best_preds.max()) + 5]
ax.plot(lims, lims, color=GOLD, lw=2, ls="--", label="Perfect prediction")
ax.set_xlim(lims);  ax.set_ylim(lims)
ax.set_xlabel("Actual Runs", fontsize=12)
ax.set_ylabel("Predicted Runs", fontsize=12)
ax.set_title(f"Actual vs Predicted — {BEST_MODEL_NAME}", fontsize=13, fontweight="bold", color=NAVY)
ax.legend()
plt.tight_layout()
savefig("05_scatter_best_model.png")

# ── 12c. Residual plot (best model) ──────────────────────────────────────────
residuals = y_test - best_preds
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Residual Diagnostics — {BEST_MODEL_NAME}", fontsize=13, fontweight="bold", color=NAVY)

axes[0].scatter(best_preds, residuals, color=NAVY, edgecolors="white", s=60, alpha=0.8)
axes[0].axhline(0, color=GOLD, lw=2, ls="--")
axes[0].set_xlabel("Predicted Runs");  axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted")

axes[1].hist(residuals, bins=15, color=NAVY, edgecolor="white", alpha=0.85)
axes[1].axvline(0, color=GOLD, lw=2, ls="--")
axes[1].set_xlabel("Residual Value");  axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
savefig("06_residuals_best_model.png")

# ── 12d. Feature importance (Random Forest) ──────────────────────────────────
fi = pd.Series(m_rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
top20 = fi.head(20)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top20.index[::-1], top20.values[::-1], color=NAVY, edgecolor="white")
for bar, val in zip(bars, top20.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8, color=NAVY)
ax.set_xlabel("Importance Score")
ax.set_title("Top-20 Feature Importances — Random Forest", fontsize=13, fontweight="bold", color=NAVY)
ax.set_facecolor(LIGHT)
plt.tight_layout()
savefig("07_feature_importance_rf.png")

# ── 12e. Feature importance (Gradient Boosting) ──────────────────────────────
fi_gb = pd.Series(m_gb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
top20_gb = fi_gb.head(20)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(top20_gb.index[::-1], top20_gb.values[::-1], color=GOLD, edgecolor="white")
for bar, val in zip(bars, top20_gb.values[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=8, color=NAVY)
ax.set_xlabel("Importance Score")
ax.set_title("Top-20 Feature Importances — Gradient Boosting", fontsize=13, fontweight="bold", color=NAVY)
ax.set_facecolor(LIGHT)
plt.tight_layout()
savefig("08_feature_importance_gb.png")

# ── 12f. Model comparison bar chart ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Comparison — Evaluation Metrics", fontsize=14, fontweight="bold", color=NAVY)
metrics_plot = [("RMSE", "lower is better"), ("MAE", "lower is better"), ("R²", "higher is better")]
palette = [NAVY, GOLD, GREEN, RED, "#7D3C98", "#2E86C1"]

for ax, (metric, note) in zip(axes, metrics_plot):
    vals  = results_df[metric].values
    names = results_df["Model"].values
    colors = palette[:len(names)]
    bars = ax.bar(names, vals, color=colors, edgecolor="white")
    ax.set_title(f"{metric}  ({note})", fontsize=10, color=NAVY)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * max(vals),
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
plt.tight_layout()
savefig("09_model_comparison_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — ROLLING FORECAST PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[13/14]  ROLLING FORECAST VISUALISATION …")

fig, ax = plt.subplots(figsize=(16, 5))
all_dates = pd.to_datetime(df["Start Date"].values)
all_runs  = df["Runs"].values

ax.plot(all_dates[:-TEST_SIZE], all_runs[:-TEST_SIZE],
        color=NAVY, lw=1, alpha=0.6, label="Training Runs")
ax.plot(all_dates[-TEST_SIZE:], y_test,
        "o-", color=NAVY, lw=2, ms=5, label="Actual (Test)")
ax.plot(pd.to_datetime(dates_test), best_preds,
        "s--", color=GOLD, lw=2, ms=5, label=f"Predicted ({BEST_MODEL_NAME})")

ax.axvline(all_dates[-TEST_SIZE], color=RED, lw=1.5, ls=":", label="Train/Test split")
ax.fill_between(pd.to_datetime(dates_test),
                np.clip(best_preds - 10, 0, None), best_preds + 10,
                alpha=0.15, color=GOLD, label="±10 run band")
ax.set_title("Virat Kohli ODI — Full Career Run Timeline with Rolling Forecast",
             fontsize=13, fontweight="bold", color=NAVY)
ax.set_xlabel("Date");  ax.set_ylabel("Runs")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
savefig("10_rolling_forecast.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — NEXT-MATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[14/14]  NEXT-MATCH PREDICTION …")

# Build next-innings feature vector from last row of df
last_row      = df[FEATURE_COLS].iloc[-1:].copy()
last_row_vals = last_row.values

# For the best model, use appropriate scaling
if BEST_MODEL_NAME in ("Ridge Regression", "Lasso Regression", "Stacking Ensemble"):
    next_input = scaler.transform(last_row_vals)
else:
    next_input = last_row_vals

next_pred = float(np.clip(best_model_obj.predict(next_input)[0], 0, None))
last_date = df["Start Date"].iloc[-1].strftime("%d %b %Y")

print(f"\n  ╔══════════════════════════════════════════════════════════╗")
print(f"  ║  NEXT ODI INNINGS PREDICTION  (after {last_date})     ║")
print(f"  ║  Best Model   : {BEST_MODEL_NAME:<40s}  ║")
print(f"  ║  Predicted    : {next_pred:>5.1f} Runs                           ║")
print(f"  ║  ±Confidence  : ±{results_df.iloc[0]['RMSE']:.1f} runs (RMSE of best model)       ║")
print(f"  ╚══════════════════════════════════════════════════════════╝")

# All-model predictions for next innings
print("\n  Next innings predictions across all models:")
_all_next = {}
for mname, mobj in _model_map.items():
    try:
        if mname in ("Ridge Regression", "Lasso Regression", "Stacking Ensemble"):
            p = float(np.clip(mobj.predict(scaler.transform(last_row_vals))[0], 0, None))
        else:
            p = float(np.clip(mobj.predict(last_row_vals)[0], 0, None))
        _all_next[mname] = p
        print(f"    {mname:<35s}: {p:6.1f} runs")
    except Exception:
        pass

if STATSMODELS_OK:
    try:
        last_exog = df[["Runs_lag1", "Runs_roll3"]].iloc[[-1]].values
        p_sar = float(np.clip(sarimax_fit.forecast(steps=1, exog=last_exog)[0], 0, None))
        _all_next["SARIMAX"] = p_sar
        print(f"    {'SARIMAX(1,0,1)':<35s}: {p_sar:6.1f} runs")
    except Exception:
        pass

# Consensus / ensemble average
avg_next = np.mean(list(_all_next.values()))
print(f"\n  Consensus average across all models: {avg_next:.1f} runs")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — HYPERPARAMETER TUNING (optional, RF as example)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[15/14]  HYPERPARAMETER TUNING (Random Forest, GridSearchCV) …")
print("  [running — may take ~1–2 min …]")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth":    [5, 8],
    "min_samples_split": [2, 5]
}
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring="neg_root_mean_squared_error",
    n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)
tuned_rf = grid_search.best_estimator_
tuned_preds = np.clip(tuned_rf.predict(X_test), 0, None)
tuned_rmse  = np.sqrt(mean_squared_error(y_test, tuned_preds))
tuned_r2    = r2_score(y_test, tuned_preds)
print(f"  Best params  : {grid_search.best_params_}")
print(f"  Tuned RF     RMSE={tuned_rmse:.2f}  R²={tuned_r2:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — SAVE BEST MODEL (pickle)
# ══════════════════════════════════════════════════════════════════════════════
pkl_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump({"model": best_model_obj,
                 "scaler": scaler,
                 "features": FEATURE_COLS,
                 "best_model_name": BEST_MODEL_NAME}, f)
print(f"\n  ✔ Best model saved → {pkl_path}")
print("  Usage: pkl = pickle.load(open('best_model.pkl','rb'))")
print("         pred = pkl['model'].predict(pkl['scaler'].transform(X_new))")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PIPELINE COMPLETE — All outputs saved to:", OUTPUT_DIR)
print("=" * 65)
print(f"\n  FILES GENERATED:")
output_files = sorted(os.listdir(OUTPUT_DIR))
for f in output_files:
    fpath = os.path.join(OUTPUT_DIR, f)
    size  = os.path.getsize(fpath)
    print(f"    {f:<45s}  {size/1024:.1f} KB")

print(f"\n  DATASET   : {len(df)} innings  ·  {len(FEATURE_COLS)} features")
print(f"  TRAIN/TEST: {len(X_train)} / {len(X_test)} innings")
print(f"\n  BEST MODEL       : {BEST_MODEL_NAME}")
print(f"  RMSE             : {best_row['RMSE']:.2f} runs")
print(f"  MAE              : {best_row['MAE']:.2f} runs")
print(f"  R²               : {best_row['R²']:.3f}")
print(f"  MAPE             : {best_row['MAPE(%)']:.1f}%")
print(f"  NEXT MATCH PRED  : {next_pred:.0f} runs")
print(f"\n  Consensus (all models): {avg_next:.0f} runs")
print("\n" + "=" * 65)