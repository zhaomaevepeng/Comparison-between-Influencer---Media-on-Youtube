"""
One-way ANOVA analysis script (updated for your path)

Reads CSVs from:
  split_datasets/Narrativity arc/C_*Narrative Arc.csv

Outputs saved to ./results/
 - anova_summary.csv
 - tukey_<dv>.csv (if Tukey run)
 - gameshowell_<dv>.csv (if Welch & Games-Howell run)
 - diagnostics.txt (human-friendly log)
"""

import os, sys, glob, re, subprocess, json
from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats

# --------- Helper: try to import, else pip-install ----------
def pip_install(package):
    """Install a package using the same python interpreter running this script."""
    print(f"Attempting to install {package} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Attempt imports we need (statsmodels, pingouin for Welch/GH)
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception as e:
    print("statsmodels import failed:", e)
    pip_install("statsmodels")
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

# pingouin provides welch_anova and pairwise_gameshowell
use_pingouin = True
try:
    import pingouin as pg
except Exception:
    try:
        pip_install("pingouin")
        import pingouin as pg
    except Exception:
        print("Could not import pingouin. Welch ANOVA / Games-Howell won't be available.")
        use_pingouin = False

# ---------- File discovery (updated path) ----------
pattern = "split_datasets/Narrativity arc/C_*Narrative Arc.csv"
files = sorted(glob.glob(pattern))
if not files:
    # Try a slightly broader pattern
    files = sorted(glob.glob("split_datasets/Narrativity arc/*.csv"))
if not files:
    raise FileNotFoundError(f"No CSVs found with pattern {pattern} (or in that folder). "
                            "Check current working directory and file paths.")

print("Files discovered:")
for f in files:
    print(" -", f)

# ---------- Read CSVs; ensure a 'category' column exists ----------
dfs = []
for idx, path in enumerate(files, start=1):
    tmp = pd.read_csv(path)
    # normalize column names
    tmp.columns = [str(c).strip().lower().replace(" ", "_") for c in tmp.columns]
    # Look for an existing category-like column
    cat_col = None
    for cand in ("category","cat","group","label"):
        if cand in tmp.columns:
            cat_col = cand
            break
    if cat_col:
        tmp = tmp.rename(columns={cat_col: "category"})
    else:
        # extract 'C_1' or 'C1' etc from filename
        base = os.path.basename(path)
        m = re.search(r"(c[_\-]?\d+)", base, re.IGNORECASE)
        label = m.group(1).upper().replace("-", "_") if m else f"FILE_{idx}"
        tmp["category"] = label
    dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True, sort=False)
print("\nCombined dataframe loaded. Columns:")
print(df.columns.tolist())

# ---------- Detect DV columns by substring (adjust if needed) ----------
expected_substrings = {
    "narrativity_overall": ["narrativity_overall","narrativityoverall","overall_narrativity","overall_narrativity"],
    "narrativity_staging": ["narrativity_staging","narrativitystaging","staging"],
    "narrativity_plotprog": ["narrativity_plotprog","narrativityplotprog","plotprog","plot_prog","plot_progression"],
    "narrativity_cogtension": ["narrativity_cogtension","narrativitycogtension","cogtension","cognitive_tension"]
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
    print("ERROR: Couldn't find these variables automatically:", missing)
    print("Available columns:\n", df.columns.tolist())
    raise KeyError("Rename your columns or modify expected_substrings mapping in script.")

dv_cols = [found[k] for k in ("narrativity_overall","narrativity_staging","narrativity_plotprog","narrativity_cogtension")]
print("\nUsing DV columns:", dv_cols)

# ---------- Prepare analysis dataframe ----------
for c in dv_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')
analysis_df = df[["category"] + dv_cols].dropna(subset=dv_cols + ["category"]).copy()
analysis_df['category'] = analysis_df['category'].astype(str)

print("\nDetected categories and sizes:")
print(analysis_df.groupby('category').size())

# Create results folder
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
diag_path = os.path.join(out_dir, "diagnostics.txt")
diag_lines = []

# ---------- Analysis loop per DV ----------
anova_summaries = []
for dv in dv_cols:
    diag_lines.append(f"\n==== DV: {dv} ====\n")
    print(f"\nProcessing DV: {dv}")

    # Levene's test:
    groups = [analysis_df.loc[analysis_df['category']==g, dv].dropna().values
              for g in analysis_df['category'].unique()]
    levene_stat, levene_p = stats.levene(*groups)
    diag_lines.append(f"Levene W = {levene_stat:.4f}, p = {levene_p:.6f}\n")
    print(f"Levene W={levene_stat:.4f}, p={levene_p:.6f}")

    # Fit OLS and ANOVA table
    formula = f"`{dv}` ~ C(category)"
    # Because column name might have special chars, use DataFrame.assign to create safe var name
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

    # Compute effect sizes: eta2 and partial eta2
    # anova_table sum_sq rows: 'C(category)' and 'Residual'
    ss_between = anova_table.loc['C(category)','sum_sq']
    ss_resid = anova_table.loc['Residual','sum_sq']
    ss_total = ss_between + ss_resid
    eta2 = ss_between / ss_total if ss_total != 0 else np.nan
    partial_eta2 = ss_between / (ss_between + ss_resid) if (ss_between + ss_resid) != 0 else np.nan
    diag_lines.append(f"eta2 = {eta2:.4f}, partial_eta2 = {partial_eta2:.4f}\n")

    # Save ANOVA summary row
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

    # Decide post-hoc:
    anova_p = row["ANOVA_p"]
    # If Levene significant, we'll run Welch ANOVA as well (if pingouin available)
    if levene_p < 0.05:
        diag_lines.append("Warning: Levene p < .05 → variances unequal. Running Welch ANOVA (if available).\n")
        print("Levene suggests unequal variances (p < .05).")
        if use_pingouin:
            try:
                welch = pg.welch_anova(dv='dv_tmp', between='category', data=tmp)
                diag_lines.append("Welch ANOVA (pingouin):\n" + welch.to_string() + "\n")
                print("Welch ANOVA result:\n", welch)
                # if Welch significant, run Games-Howell pairwise
                if float(welch['p-unc'][0]) < 0.05:
                    gh = pg.pairwise_gameshowell(dv='dv_tmp', between='category', data=tmp)
                    gh.to_csv(os.path.join(out_dir, f"gameshowell_{dv}.csv"), index=False)
                    diag_lines.append("Games-Howell pairwise results saved.\n")
                    print("Games-Howell pairwise saved for", dv)
            except Exception as e:
                diag_lines.append(f"Could not run pingouin Welch/GH: {e}\n")
        else:
            diag_lines.append("pingouin not available; Welch ANOVA / Games-Howell skipped.\n")
    # If standard ANOVA is significant, run Tukey HSD
    if anova_p < 0.05:
        try:
            tukey = pairwise_tukeyhsd(endog=tmp['dv_tmp'], groups=tmp['category'], alpha=0.05)
            tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
            tukey_path = os.path.join(out_dir, f"tukey_{dv}.csv")
            tukey_df.to_csv(tukey_path, index=False)
            diag_lines.append(f"Tukey HSD performed; saved to {tukey_path}\n")
            print("Tukey HSD saved to:", tukey_path)
        except Exception as e:
            diag_lines.append(f"Tukey HSD error: {e}\n")
            print("Tukey HSD error:", e)
    else:
        diag_lines.append("ANOVA not significant → Tukey HSD not run.\n")

# Save ANOVA summary CSV
anova_summary_df = pd.DataFrame(anova_summaries)
anova_summary_path = os.path.join(out_dir, "anova_summary.csv")
anova_summary_df.to_csv(anova_summary_path, index=False)
print("\nANOVA summary saved to:", anova_summary_path)

# Save diagnostics file
with open(diag_path, "w", encoding="utf-8") as f:
    f.write("ANOVA diagnostics log\n\n")
    f.write("\n".join(diag_lines))
print("Diagnostics saved to:", diag_path)

print("\nDone. Check the results/ folder for CSV outputs and diagnostics.")
