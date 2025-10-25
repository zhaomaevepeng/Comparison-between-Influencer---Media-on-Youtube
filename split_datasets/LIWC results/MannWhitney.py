"""
Mann-Whitney U + Welch's t-test comparisons between Category 1 and Category 2

- Compares columns C..end (i.e., all columns except the first two) from:
    Category 1: split_datasets/LIWC results/Category_1 - LIWC Analysis.csv
    Category 2: split_datasets/LIWC results/Category_2 - LIWC Analysis.csv

- For each feature present in both files:
    * median(Category1) and median(Category2)
    * percent difference (median-based), signed (see function pct_diff_median)
    * Mann-Whitney U (two-sided) -> p-value, U
    * Welch's t-test (ttest_ind with equal_var=False) -> t, p
    * Cohen's d (approx) for Welch (unequal variances)
    * significance stars for BOTH tests: *** p<0.001, ** p<0.01, * p<0.05

- Main output CSV (results/mannwhitney_and_welch_cat1_vs_cat2.csv) columns:
    Linguistic Feature, Difference_pct, MannWhitney_p, MannWhitney_sig, MannWhitney_U,
    Welch_t, Welch_p, Welch_d, n_cat1, n_cat2

- Extended CSV (results/mannwhitney_and_welch_cat1_vs_cat2_extended.csv) includes medians and raw stats.

Notes:
- Percent difference uses median(Category2) as denominator when nonzero; otherwise a symmetric fallback.
- No multiple-comparison correction is applied here; adjust p-values if needed (Holm/FDR).
"""

import os
import math
import numpy as np
import pandas as pd
from scipy import stats

# ---------- CONFIG ----------
path_cat1 = "split_datasets/LIWC results/Category_1 - LIWC Analysis.csv"
path_cat2 = "split_datasets/LIWC results/Category_2 - LIWC Analysis.csv"

out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
main_out = os.path.join(out_dir, "mannwhitney_and_welch_cat1_vs_cat2.csv")
ext_out = os.path.join(out_dir, "mannwhitney_and_welch_cat1_vs_cat2_extended.csv")

# ---------- helper functions ----------
def safe_numeric_series(s):
    """Coerce series to numeric and drop NA."""
    return pd.to_numeric(s, errors="coerce").dropna()

def pct_diff_median(m1, m2):
    """
    Compute percent difference = (m1 - m2) / denom * 100.
    Primary denom = m2 (median of category 2).
    If denom==0, fallback denom = (|m1| + |m2|)/2.
    If fallback denom == 0, return 0.0.
    """
    if pd.isna(m1) or pd.isna(m2):
        return np.nan
    if m2 != 0:
        denom = m2
    else:
        denom = (abs(m1) + abs(m2)) / 2.0
    if denom == 0:
        return 0.0
    return (m1 - m2) / denom * 100.0

def p_to_stars(p):
    """Return significance stars string for p-value."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    try:
        p = float(p)
    except Exception:
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def cohens_d_welch(x, y):
    """Approximate Cohen's d for two independent samples with unequal variances."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = np.mean(x), np.mean(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    # Use the square root of the average of the two sample variances (common approximation)
    s_denom = math.sqrt((sx2 + sy2) / 2.0) if (sx2 + sy2) > 0 else 0.0
    if s_denom == 0:
        return np.nan
    return (mx - my) / s_denom

# ---------- load data ----------
df1 = pd.read_csv(path_cat1)
df2 = pd.read_csv(path_cat2)

# normalize headers (trim)
df1.columns = [str(c).strip() for c in df1.columns]
df2.columns = [str(c).strip() for c in df2.columns]

# columns C..end = skip first two
cols1 = list(df1.columns[2:])
cols2 = list(df2.columns[2:])

# Intersection of features (keep order of df1)
common_cols = [c for c in cols1 if c in cols2]
if len(common_cols) == 0:
    raise RuntimeError("No common feature columns found between the two files (after skipping first two columns).")

records = []
records_ext = []

for feat in common_cols:
    x_series = safe_numeric_series(df1[feat])
    y_series = safe_numeric_series(df2[feat])
    x = x_series.to_numpy()
    y = y_series.to_numpy()

    n1 = x.size
    n2 = y.size

    median_x = np.median(x) if n1 > 0 else np.nan
    median_y = np.median(y) if n2 > 0 else np.nan

    diff_pct = pct_diff_median(median_x, median_y)

    # Mann-Whitney U (two-sided)
    mw_p = np.nan
    mw_u = np.nan
    if n1 > 0 and n2 > 0:
        try:
            mw_u, mw_p = stats.mannwhitneyu(x, y, alternative="two-sided")
        except TypeError:
            # older scipy signature
            try:
                mw_u, mw_p = stats.mannwhitneyu(x, y)
            except Exception:
                mw_u, mw_p = np.nan, np.nan
        except Exception:
            mw_u, mw_p = np.nan, np.nan

    # Welch's t-test
    t_stat = np.nan
    t_p = np.nan
    if n1 > 1 and n2 > 1:
        try:
            t_res = stats.ttest_ind(x, y, equal_var=False, nan_policy='omit')
            t_stat = float(t_res.statistic) if t_res is not None else np.nan
            t_p = float(t_res.pvalue) if t_res is not None else np.nan
        except Exception:
            t_stat, t_p = np.nan, np.nan

    # Cohen's d (approx for Welch)
    cohens_d = cohens_d_welch(x, y)

    # stars
    mw_sig = p_to_stars(mw_p)
    welch_sig = p_to_stars(t_p)

    # formatted percent string
    diff_str = f"{diff_pct:+0.2f}%" if not (isinstance(diff_pct, float) and math.isnan(diff_pct)) else ""

    records.append({
        "Linguistic Feature": feat,
        "Difference_pct": diff_str,
        "MannWhitney_p": (float(mw_p) if not (mw_p is None or (isinstance(mw_p, float) and math.isnan(mw_p))) else np.nan),
        "MannWhitney_sig": mw_sig,
        "MannWhitney_U": (float(mw_u) if not (mw_u is None or (isinstance(mw_u, float) and math.isnan(mw_u))) else np.nan),
        "Welch_t": (float(t_stat) if not (t_stat is None or (isinstance(t_stat, float) and math.isnan(t_stat))) else np.nan),
        "Welch_p": (float(t_p) if not (t_p is None or (isinstance(t_p, float) and math.isnan(t_p))) else np.nan),
        "Welch_d": (float(cohens_d) if not (cohens_d is None or (isinstance(cohens_d, float) and math.isnan(cohens_d))) else np.nan),
        "Welch_sig": welch_sig,
        "n_cat1": int(n1),
        "n_cat2": int(n2)
    })

    records_ext.append({
        "Linguistic Feature": feat,
        "median_cat1": median_x,
        "median_cat2": median_y,
        "Difference_pct_raw": diff_pct,
        "MannWhitney_U": mw_u,
        "MannWhitney_p": mw_p,
        "MannWhitney_sig": mw_sig,
        "Welch_t": t_stat,
        "Welch_p": t_p,
        "Welch_d": cohens_d,
        "Welch_sig": welch_sig,
        "n_cat1": n1,
        "n_cat2": n2
    })

# ---------- build dataframes and save ----------
df_main = pd.DataFrame(records)
df_ext = pd.DataFrame(records_ext)

# order columns for main output
main_cols = [
    "Linguistic Feature", "Difference_pct",
    "MannWhitney_p", "MannWhitney_sig", "MannWhitney_U",
    "Welch_t", "Welch_p", "Welch_d", "Welch_sig",
    "n_cat1", "n_cat2"
]
df_main = df_main[main_cols]

df_main.to_csv(main_out, index=False)
df_ext.to_csv(ext_out, index=False)

# ---------- print quick summary ----------
with pd.option_context('display.max_rows', 200, 'display.max_columns', None):
    print("\nFirst 30 rows of main results:\n")
    print(df_main.head(30).to_string(index=False))

print(f"\nSaved main results to: {main_out}")
print(f"Saved extended results to: {ext_out}")
