#!/usr/bin/env python3
"""
ANOVA analysis for per-transcript dataset (all columns except first two)

- File: affect_sentence_outputs/per_transcript_from_sentences.csv
- Grouping variable: "Influencer/Mainstream" (if present) or 2nd column
- DVs: all columns except the first two
- Outputs saved to ./results/
    - anova_summary_per_transcript.csv
    - diagnostics_per_transcript.txt
    - tukey_<dv>.csv (if Tukey run)
    - gameshowell_<dv>.csv (if Welch & Games-Howell run)
"""

import os
import sys
import re
import glob
import subprocess
import warnings
from itertools import combinations

import pandas as pd
import numpy as np
from scipy import stats

# ---------------- utilities ----------------
def pip_install(pkg):
    """Install a Python package using the running interpreter."""
    print(f"Attempting to install {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def safe_filename(s):
    return re.sub(r'[^0-9A-Za-z_]+', '_', str(s)).strip('_')

def compute_ss(all_vals, groups_arrays):
    overall_mean = np.mean(all_vals)
    ss_between = sum([len(g) * (np.mean(g) - overall_mean) ** 2 for g in groups_arrays])
    ss_total = np.sum((all_vals - overall_mean) ** 2)
    ss_resid = ss_total - ss_between
    return ss_between, ss_resid, ss_total

# ---------------- imports (statsmodels + pingouin optional) ----------------
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception as e:
    print("statsmodels not available:", e)
    pip_install("statsmodels")
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

use_pingouin = True
try:
    import pingouin as pg
except Exception:
    try:
        pip_install("pingouin")
        import pingouin as pg
    except Exception as e:
        print("pingouin not available; Welch ANOVA / Games-Howell will be skipped.", e)
        use_pingouin = False

# ---------------- file paths ----------------
CSV_PATH = "affect_sentence_outputs/per_transcript_from_sentences.csv"
FALLBACK = "/mnt/data/per_transcript_from_sentences.csv"
if os.path.exists(CSV_PATH):
    csv_path = CSV_PATH
elif os.path.exists(FALLBACK):
    csv_path = FALLBACK
else:
    raise FileNotFoundError(f"CSV not found at '{CSV_PATH}' or fallback '{FALLBACK}'")

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
DIAG_PATH = os.path.join(OUT_DIR, "diagnostics_per_transcript.txt")
SUMMARY_PATH = os.path.join(OUT_DIR, "anova_summary_per_transcript.csv")

ALPHA = 0.05

# ---------------- read data ----------------
print("Reading CSV from:", csv_path)
df = pd.read_csv(csv_path)
# trim whitespace from headers
df.columns = [str(c).strip() for c in df.columns]

if df.shape[1] < 3:
    raise ValueError("Expected at least 3 columns (skip first two, test remaining).")

# choose grouping column
group_col = "Influencer/Mainstream" if "Influencer/Mainstream" in df.columns else df.columns[1]
print("Using grouping column:", group_col)

# DVs = all except first two
dv_cols = list(df.columns[2:])
print("Number of DVs to analyze:", len(dv_cols))

# ---------------- analysis containers ----------------
diag_lines = []
diag_lines.append(f"Source file: {csv_path}")
diag_lines.append(f"Grouping column: {group_col}")
diag_lines.append(f"DVs (excluded first two columns): {len(dv_cols)}")
diag_lines.append("")

anova_summaries = []

# ---------------- loop over DVs ----------------
for dv in dv_cols:
    diag_lines.append(f"\n==== DV: {dv} ====\n")
    print(f"\nProcessing DV: {dv}\n")
    # coerce to numeric
    df[dv] = pd.to_numeric(df[dv], errors='coerce')
    nobs = df[dv].dropna().shape[0]
    diag_lines.append(f"Non-missing numeric observations: {nobs}")

    if nobs < 3:
        msg = f"Not enough numeric observations for {dv} (n={nobs}). Skipping."
        print(msg)
        diag_lines.append(msg)
        continue

    # prepare analysis df
    analysis_df = df[[group_col, dv]].dropna().rename(columns={group_col: "category", dv: "dv_tmp"})
    analysis_df['category'] = analysis_df['category'].astype(str)

    groups = analysis_df['category'].unique().tolist()
    if len(groups) < 2:
        msg = f"Fewer than 2 groups present for {dv}. Skipping."
        print(msg)
        diag_lines.append(msg)
        continue

    # group arrays
    group_arrays = [analysis_df.loc[analysis_df['category']==g, 'dv_tmp'].values for g in groups]

    # Levene
    try:
        levene_stat, levene_p = stats.levene(*group_arrays)
    except Exception as e:
        levene_stat, levene_p = np.nan, np.nan
        diag_lines.append(f"Levene error: {e}")
    diag_lines.append(f"Levene W = {levene_stat:.4f}, p = {levene_p:.6f}")
    print(f"Levene W = {levene_stat:.4f}, p = {levene_p:.6f}\n")

    # Try statsmodels OLS ANOVA (Type II)
    anova_p = anova_F = np.nan
    ss_between = ss_resid = ss_total = np.nan
    df1 = df2 = None
    residuals = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ols("dv_tmp ~ C(category)", data=analysis_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
        # extract sums of squares and residuals
        ss_between = float(anova_table.loc['C(category)', 'sum_sq'])
        ss_resid = float(anova_table.loc['Residual', 'sum_sq'])
        ss_total = ss_between + ss_resid
        anova_F = float(anova_table.loc['C(category)', 'F'])
        anova_p = float(anova_table.loc['C(category)', 'PR(>F)'])
        df1 = anova_table.loc['C(category)', 'df']
        df2 = anova_table.loc['Residual', 'df']
        residuals = model.resid
        # print nicely
        diag_lines.append("ANOVA table (Type II):")
        diag_lines.append(anova_table.to_string())
        print("ANOVA table (Type II):")
        print(anova_table)
    except Exception as e:
        diag_lines.append(f"statsmodels ANOVA failed for {dv}: {e}")
        print("statsmodels ANOVA failed; falling back to scipy f_oneway. Error:", e)
        # fallback to scipy f_oneway
        try:
            f_stat, p_val = stats.f_oneway(*group_arrays)
            anova_F = float(f_stat)
            anova_p = float(p_val)
            df1 = len(group_arrays) - 1
            df2 = analysis_df.shape[0] - len(group_arrays)
            all_vals = np.concatenate(group_arrays)
            ss_between, ss_resid, ss_total = compute_ss(all_vals, group_arrays)
            # approximate residuals
            residuals = analysis_df['dv_tmp'] - analysis_df.groupby('category')['dv_tmp'].transform('mean')
            # create a synthetic ANOVA table printout
            at_lines = []
            at_lines.append("ANOVA table (Type II):")
            at_lines.append(f"{'':14s}{'sum_sq':>12s}{'df':>8s}{'F':>10s}{'PR(>F)':>12s}")
            at_lines.append(f"{'C(category)':14s}{ss_between:12.6f}{df1:8.1f}{anova_F:10.6f}{anova_p:12.6f}")
            at_lines.append(f"{'Residual':14s}{ss_resid:12.6f}{df2:8.1f}{'NaN':>10s}{'NaN':>12s}")
            diag_lines.extend(at_lines)
            print("\n".join(at_lines))
        except Exception as e2:
            diag_lines.append(f"Fallback ANOVA failed for {dv}: {e2}")
            print("Fallback ANOVA failed for", dv, ":", e2)
            continue

    # Shapiro on residuals
    try:
        sh_w, sh_p = stats.shapiro(residuals)
    except Exception:
        sh_w, sh_p = np.nan, np.nan
    diag_lines.append(f"Shapiro W (residuals) = {sh_w:.4f}, p = {sh_p:.6f}")
    print(f"\nShapiro W (residuals) = {sh_w:.4f}, p = {sh_p:.6f}\n")

    # compute effect sizes
    try:
        eta2 = ss_between / ss_total if (not np.isnan(ss_between) and not np.isnan(ss_total) and ss_total != 0) else np.nan
        partial_eta2 = ss_between / (ss_between + ss_resid) if (not np.isnan(ss_between) and not np.isnan(ss_resid) and (ss_between + ss_resid) != 0) else np.nan
    except Exception:
        eta2 = partial_eta2 = np.nan
    diag_lines.append(f"eta2 = {eta2:.4f}, partial_eta2 = {partial_eta2:.4f}")
    print(f"eta2 = {eta2:.4f}, partial_eta2 = {partial_eta2:.4f}\n")

    # If Levene p < .05 try Welch ANOVA & Games-Howell via pingouin
    if (not np.isnan(levene_p)) and (levene_p < 0.05):
        diag_lines.append("Warning: Levene p < .05 → variances unequal. Attempting Welch ANOVA (pingouin) if available.")
        print("Levene p < .05 → variances unequal. Attempting Welch ANOVA (pingouin) if available.")
        if use_pingouin:
            try:
                welch = pg.welch_anova(dv='dv_tmp', between='category', data=analysis_df)
                diag_lines.append("Welch ANOVA (pingouin):")
                diag_lines.append(welch.to_string())
                print("\nWelch ANOVA result (pingouin):")
                print(welch)
                if float(welch['p-unc'][0]) < ALPHA:
                    gh = pg.pairwise_gameshowell(dv='dv_tmp', between='category', data=analysis_df)
                    gh_path = os.path.join(OUT_DIR, f"gameshowell_{safe_filename(dv)}.csv")
                    gh.to_csv(gh_path, index=False)
                    diag_lines.append(f"Games-Howell pairwise saved to {gh_path}")
                    print("Games-Howell pairwise saved to:", gh_path)
            except Exception as e:
                diag_lines.append(f"pingouin Welch/GH error for {dv}: {e}")
                print("pingouin Welch/GH error:", e)
        else:
            diag_lines.append("pingouin not available; Welch ANOVA/Games-Howell skipped.")
            print("pingouin not available; Welch ANOVA/Games-Howell skipped.")

    # If standard ANOVA significant, run Tukey HSD
    if (anova_p is not None) and (not np.isnan(anova_p)) and (anova_p < ALPHA):
        try:
            tukey = pairwise_tukeyhsd(endog=analysis_df['dv_tmp'], groups=analysis_df['category'], alpha=ALPHA)
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_path = os.path.join(OUT_DIR, f"tukey_{safe_filename(dv)}.csv")
            tukey_df.to_csv(tukey_path, index=False)
            diag_lines.append(f"Tukey HSD performed; saved to {tukey_path}")
            print("ANOVA significant → Tukey HSD performed and saved to:", tukey_path)
        except Exception as e:
            diag_lines.append(f"Tukey HSD error for {dv}: {e}")
            print("Tukey HSD error for", dv, ":", e)
    else:
        diag_lines.append("ANOVA not significant → Tukey HSD not run.")
        print("ANOVA not significant → Tukey HSD not run.")

    # Append summary row
    anova_summaries.append({
        "DV": dv,
        "Levene_stat": round(float(levene_stat), 4) if not np.isnan(levene_stat) else None,
        "Levene_p": float(levene_p) if not np.isnan(levene_p) else None,
        "sum_sq_between": float(ss_between) if not np.isnan(ss_between) else None,
        "sum_sq_resid": float(ss_resid) if not np.isnan(ss_resid) else None,
        "ANOVA_df1": float(df1) if df1 is not None else None,
        "ANOVA_df2": float(df2) if df2 is not None else None,
        "ANOVA_F": float(anova_F) if not np.isnan(anova_F) else None,
        "ANOVA_p": float(anova_p) if not np.isnan(anova_p) else None,
        "eta2": round(float(eta2), 4) if not np.isnan(eta2) else None,
        "partial_eta2": round(float(partial_eta2), 4) if not np.isnan(partial_eta2) else None,
        "Shapiro_W": float(sh_w) if not np.isnan(sh_w) else None,
        "Shapiro_p": float(sh_p) if not np.isnan(sh_p) else None,
        "ANOVA_significant": bool((anova_p is not None) and (not np.isnan(anova_p)) and (anova_p < ALPHA))
    })

# ---------------- write outputs ----------------
summary_df = pd.DataFrame(anova_summaries)
summary_df.to_csv(SUMMARY_PATH, index=False)
with open(DIAG_PATH, "w", encoding="utf-8") as fh:
    fh.write("ANOVA diagnostics log\n\n")
    fh.write("\n".join(diag_lines))

print("\nSaved ANOVA summary to:", SUMMARY_PATH)
print("Saved diagnostics to:", DIAG_PATH)
print("If any post-hoc results were generated they are in the results/ folder as well.")
