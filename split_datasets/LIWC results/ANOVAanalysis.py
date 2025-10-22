#!/usr/bin/env python3
"""
ANOVA analysis for LIWC variables (Analytic, Clout, Authentic, Tone)

Reads CSVs from:
  split_datasets/LIWC results/Category_* - LIWC Analysis.csv

Saves results to ./results/:
  - anova_summary.csv
  - tukey_<dv>.csv (if Tukey run)
  - gameshowell_<dv>.csv (if Welch & Games-Howell run)
  - diagnostics.txt
"""

import os, sys, glob, re, subprocess
from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats

# ------------ helper to pip install if needed (optional) ------------
def pip_install(pkg):
    print(f"Attempting to install {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Try required imports; install if missing
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
    except Exception:
        print("pingouin unavailable. Welch ANOVA/Games-Howell will be skipped.")
        use_pingouin = False

# ------------- file discovery (your updated path) -------------
pattern = "split_datasets/LIWC results/Category_* - LIWC Analysis.csv"
files = sorted(glob.glob(pattern))
if not files:
    # fallback: any csv in folder
    files = sorted(glob.glob("split_datasets/LIWC results/*.csv"))

if not files:
    raise FileNotFoundError(f"No files found with pattern {pattern} or in folder 'split_datasets/LIWC results/'. "
                            "Check your working directory and file names.")

print("Files found:")
for f in files:
    print(" -", f)

# ------------- read CSVs and ensure category column -------------
dfs = []
for idx, path in enumerate(files, start=1):
    tmp = pd.read_csv(path)
    # normalize column names
    tmp.columns = [str(c).strip().lower().replace(" ", "_") for c in tmp.columns]
    # detect category-like column
    cat_col = None
    for cand in ("category","cat","group","label"):
        if cand in tmp.columns:
            cat_col = cand
            break
    if cat_col:
        tmp = tmp.rename(columns={cat_col: "category"})
    else:
        # extract "Category_1" style from filename or fallback to FILE_idx
        base = os.path.basename(path)
        m = re.search(r"(category[_\-]?\d+|c[_\-]?\d+|c\d+)", base, re.IGNORECASE)
        if m:
            label = m.group(1).upper().replace("-", "_")
        else:
            # as in your earlier files: "Category_1 - LIWC Analysis.csv" -> extract "Category_1"
            m2 = re.search(r"(Category[_\s]?\d+)", base, re.IGNORECASE)
            label = m2.group(1).replace(" ", "_") if m2 else f"FILE_{idx}"
        tmp["category"] = label
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True, sort=False)
print("\nCombined dataframe columns:\n", df.columns.tolist())

# ------------- detect the four DV columns by substring -------------
expected_substrings = {
    "analytic": ["analytic","analyt"],
    "clout": ["clout"],
    "authentic": ["authentic","authent"],
    "tone": ["tone"]
}
found = {}
for key, variants in expected_substrings.items():
    found[key] = None
    for col in df.columns:
        for v in variants:
            if v in col:
                found[key] = col
                break
        if found[key]:
            break

missing = [k for k,v in found.items() if v is None]
if missing:
    print("\nERROR: Could not find columns for:", missing)
    print("Available columns:", df.columns.tolist())
    raise KeyError("Make sure your LIWC columns include substrings for 'analytic', 'clout', 'authentic', and 'tone'.")

dv_cols = [found[k] for k in ("analytic","clout","authentic","tone")]
print("\nUsing DV columns:", dv_cols)

# ------------- prepare analysis dataframe -------------
for c in dv_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

analysis_df = df[["category"] + dv_cols].dropna(subset=dv_cols + ["category"]).copy()
analysis_df['category'] = analysis_df['category'].astype(str)
categories = analysis_df['category'].unique().tolist()

print("\nDetected categories and sizes:")
print(analysis_df.groupby('category').size())

# ------------- prepare output folder and diagnostics -------------
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
diag_lines = []
anova_summaries = []

# ------------- analysis loop per DV -------------
for dv in dv_cols:
    diag_lines.append(f"\n==== DV: {dv} ====\n")
    print(f"\nProcessing DV: {dv}")

    # Levene's test
    groups = [analysis_df.loc[analysis_df['category']==g, dv].dropna().values for g in categories]
    levene_stat, levene_p = stats.levene(*groups)
    diag_lines.append(f"Levene W = {levene_stat:.4f}, p = {levene_p:.6f}\n")
    print(f"Levene W={levene_stat:.4f}, p={levene_p:.6f}")

    # Fit OLS ANOVA (rename dv to safe column)
    tmp = analysis_df[[dv, "category"]].rename(columns={dv: "dv_tmp"})
    model = ols("dv_tmp ~ C(category)", data=tmp).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    diag_lines.append("ANOVA table (Type II):\n" + anova_table.to_string() + "\n")
    print("\nANOVA table:\n", anova_table)

    # Shapiro on residuals
    try:
        sh_w, sh_p = stats.shapiro(model.resid)
    except Exception:
        sh_w, sh_p = np.nan, np.nan
    diag_lines.append(f"Shapiro W (residuals) = {sh_w:.4f}, p = {sh_p:.6f}\n")
    print(f"Shapiro (resid) W={sh_w:.4f}, p={sh_p:.6f}")

    # effect sizes
    ss_between = anova_table.loc['C(category)','sum_sq']
    ss_resid = anova_table.loc['Residual','sum_sq']
    ss_total = ss_between + ss_resid
    eta2 = ss_between / ss_total if ss_total != 0 else np.nan
    partial_eta2 = ss_between / (ss_between + ss_resid) if (ss_between + ss_resid) != 0 else np.nan
    diag_lines.append(f"eta2 = {eta2:.4f}, partial_eta2 = {partial_eta2:.4f}\n")
    print(f"eta2={eta2:.4f}, partial_eta2={partial_eta2:.4f}")

    # append summary row
    row = {
        "DV": dv,
        "Levene_stat": round(float(levene_stat),4),
        "Levene_p": float(levene_p),
        "ANOVA_F": float(anova_table.loc['C(category)','F']),
        "ANOVA_df1": int(anova_table.loc['C(category)','df']),
        "ANOVA_df2": int(anova_table.loc['Residual','df']),
        "ANOVA_p": float(anova_table.loc['C(category)','PR(>F)']),
        "eta2": round(float(eta2),4),
        "partial_eta2": round(float(partial_eta2),4),
        "Shapiro_W_resid": float(sh_w) if not np.isnan(sh_w) else None,
        "Shapiro_p_resid": float(sh_p) if not np.isnan(sh_p) else None
    }
    anova_summaries.append(row)

    # If Levene significant -> try Welch & Games-Howell
    if levene_p < 0.05:
        diag_lines.append("Levene p < .05 (unequal variances). Attempting Welch ANOVA (pingouin) and Games-Howell pairwise if available.\n")
        print("Levene significant: trying Welch ANOVA (pingouin)")
        if use_pingouin:
            try:
                welch = pg.welch_anova(dv='dv_tmp', between='category', data=tmp)
                diag_lines.append("Welch ANOVA (pingouin):\n" + welch.to_string() + "\n")
                print("Welch ANOVA result:\n", welch)
                if float(welch['p-unc'][0]) < 0.05:
                    gh = pg.pairwise_gameshowell(dv='dv_tmp', between='category', data=tmp)
                    gh_path = os.path.join(out_dir, f"gameshowell_{dv}.csv")
                    gh.to_csv(gh_path, index=False)
                    diag_lines.append(f"Games-Howell pairwise saved to {gh_path}\n")
                    print("Games-Howell saved to:", gh_path)
            except Exception as e:
                diag_lines.append(f"Pingouin Welch/GH error: {e}\n")
                print("Pingouin Welch/GH error:", e)
        else:
            diag_lines.append("Pingouin not available; Welch/GH skipped.\n")

    # If standard ANOVA significant -> Tukey HSD
    if row["ANOVA_p"] < 0.05:
        diag_lines.append("ANOVA significant (p < .05). Running Tukey HSD and saving pairwise results.\n")
        try:
            tukey = pairwise_tukeyhsd(endog=tmp['dv_tmp'], groups=tmp['category'], alpha=0.05)
            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_path = os.path.join(out_dir, f"tukey_{dv}.csv")
            tukey_df.to_csv(tukey_path, index=False)
            diag_lines.append(f"Tukey HSD saved to {tukey_path}\n")
            print("Tukey HSD saved to:", tukey_path)
        except Exception as e:
            diag_lines.append(f"Tukey HSD error: {e}\n")
            print("Tukey HSD error:", e)
    else:
        diag_lines.append("ANOVA not significant -> Tukey not run.\n")

# ------------- save ANOVA summary and diagnostics -------------
anova_summary_df = pd.DataFrame(anova_summaries)
anova_summary_path = os.path.join(out_dir, "anova_summary_liwc.csv")
anova_summary_df.to_csv(anova_summary_path, index=False)
print("\nANOVA summary saved to:", anova_summary_path)

diag_path = os.path.join(out_dir, "diagnostics_liwc.txt")
with open(diag_path, "w", encoding="utf-8") as fh:
    fh.write("ANOVA diagnostics log\n\n")
    fh.write("\n".join(diag_lines))

print("Diagnostics saved to:", diag_path)
print("\nAll done. Check the 'results' folder for outputs.")
