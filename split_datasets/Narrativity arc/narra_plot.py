#!/usr/bin/env python3
"""
Compute narrativity summary table and stage plots for Category 1 and Category 2.

Outputs:
 - results/narrativity_overall_table.csv  (rows: metrics, cols: Category 1 & Category 2; values = mean (sd, n))
 - results/narrativity_stages_Category1.png
 - results/narrativity_stages_Category2.png
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

# ----------- User file paths (edit if needed) -------------
path_c1 = "split_datasets/Narrativity arc/C_1_Narrative Arc.csv"
path_c2 = "split_datasets/Narrativity arc/C_2_Narrative Arc.csv"

# fallback to /mnt/data (common when files were uploaded)
if not os.path.exists(path_c1) and os.path.exists("/mnt/data/C_1_Narrative Arc.csv"):
    path_c1 = "/mnt/data/C_1_Narrative Arc.csv"
if not os.path.exists(path_c2) and os.path.exists("/mnt/data/C_2_Narrative Arc.csv"):
    path_c2 = "/mnt/data/C_2_Narrative Arc.csv"

if not os.path.exists(path_c1) or not os.path.exists(path_c2):
    raise FileNotFoundError(f"Check both files exist. Paths tried:\n - {path_c1}\n - {path_c2}")

# ----------------- helpers -----------------
def soften_color(color, mix_with=(1,1,1), alpha=0.6):
    rgb = mcolors.to_rgb(color)
    return tuple(alpha*c + (1-alpha)*w for c,w in zip(rgb, mix_with))

def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def find_overall_col(df, candidates):
    """Find first column name containing any candidate substring (case-insensitive, spaces/underscores ignored)."""
    cols = df.columns.tolist()
    mod_cols = [c.lower().replace(" ", "").replace("_","") for c in cols]
    for cand in candidates:
        cand_mod = cand.lower().replace(" ", "").replace("_","")
        for i, mc in enumerate(mod_cols):
            if cand_mod in mc:
                return cols[i]
    return None

def find_stage_block_by_position(df):
    """Try to extract M..AA (15 cols) using positional slice: iloc[:, 12:27]"""
    if df.shape[1] >= 27:
        blk = df.iloc[:, 12:27].copy()
        return list(blk.columns), blk
    return None, None

def find_stage_block_by_name(df, keywords=("staging","plotprog","cogtension")):
    """
    Try to locate stage columns for each measure (1..5) by name heuristics.
    Returns ordered list of 15 column names (or None placeholders) and dataframe block.
    """
    cols = df.columns.tolist()
    lowered = [c.lower().replace(" ", "").replace("_","") for c in cols]
    found = []
    for key in keywords:
        k = key.lower().replace(" ", "").replace("_","")
        for i in range(1,6):
            target = f"{k}{i}"
            match = None
            # exact concat
            for idx, low in enumerate(lowered):
                if target in low:
                    match = cols[idx]; break
            if match is None:
                # key and digit anywhere
                for idx, low in enumerate(lowered):
                    if k in low and str(i) in low:
                        match = cols[idx]; break
            if match is None:
                # "key i" pattern
                for idx, col in enumerate(cols):
                    if k in col.lower() and f" {i}" in col.lower():
                        match = cols[idx]; break
            found.append(match)
    # Build block dataframe
    block = pd.DataFrame()
    for c in found:
        if c is None:
            block[c] = np.nan
        else:
            block[c] = safe_numeric(df[c])
    return found, block

def compute_overall_stats(df, mapping_candidates):
    """Return dict metric -> (colname, mean, sd, n, values_array)"""
    out = {}
    for metric, cands in mapping_candidates.items():
        col = find_overall_col(df, cands)
        if col is None:
            out[metric] = {"col": None, "mean": np.nan, "sd": np.nan, "n": 0, "values": np.array([])}
        else:
            vals = safe_numeric(df[col]).dropna().astype(float)
            mean = float(vals.mean()) if vals.size>0 else np.nan
            sd = float(vals.std(ddof=1)) if vals.size>1 else (np.nan if vals.size==1 else np.nan)
            out[metric] = {"col": col, "mean": mean, "sd": sd, "n": int(vals.size), "values": vals.values}
    return out

# ----------------- main -----------------
df1 = pd.read_csv(path_c1)
df2 = pd.read_csv(path_c2)
df1.columns = [str(c).strip() for c in df1.columns]
df2.columns = [str(c).strip() for c in df2.columns]

# mapping for overall metrics (substrings to search)
overall_candidates = {
    "Narrativity_overall": ["narrativity_overall","overallnarrativity","overall_narrativity","narrativity overall"],
    "Narrativity_staging": ["narrativity_staging","staging","narrativitystaging","narrativity_stag"],
    "Narrativity_PlotProg": ["narrativity_plotprog","plotprog","plot_progression","plot_progress"],
    "Narrativity_CogTension": ["narrativity_cogtension","cogtension","cognitive_tension","cog_tension"]
}

# compute overall stats for both categories
stats1 = compute_overall_stats(df1, overall_candidates)
stats2 = compute_overall_stats(df2, overall_candidates)

# Build comparison table: metrics as rows, columns Category 1 & Category 2 with "mean (sd, n)" strings
rows = []
for metric in overall_candidates.keys():
    a = stats1[metric]
    b = stats2[metric]
    def fmt(a):
        if a["n"] == 0 or (isinstance(a["mean"], float) and math.isnan(a["mean"])):
            return ""
        sd_str = f"{a['sd']:.2f}" if not (isinstance(a['sd'], float) and math.isnan(a['sd'])) else "NA"
        return f"{a['mean']:.4f} ({sd_str}, n={a['n']})"
    rows.append({
        "Metric": metric,
        "Category 1": fmt(a),
        "Category 2": fmt(b)
    })
comp_table = pd.DataFrame(rows).set_index("Metric")

# Save comparison table CSV
os.makedirs("results", exist_ok=True)
comp_csv = os.path.join("results", "narrativity_overall_table.csv")
comp_table.to_csv(comp_csv)
print("Saved overall comparison table to:", comp_csv)

# ----------------- Stage block extraction (15 columns) -----------------
# Prefer positional slice (M..AA -> iloc[:,12:27]); fallback to name heuristics.
cols_block1, block1 = find_stage_block_by_position(df1)
cols_block2, block2 = find_stage_block_by_position(df2)

if cols_block1 is None or cols_block2 is None:
    # fallback by name
    cols_block1, block1 = find_stage_block_by_name(df1, ("staging","plotprog","cogtension"))
    cols_block2, block2 = find_stage_block_by_name(df2, ("staging","plotprog","cogtension"))

# Validate we got 15 columns each
if block1.shape[1] < 15 or block2.shape[1] < 15:
    print("ERROR: Could not locate 15 stage columns for one or both files. Found columns:")
    print("Category1 columns (first 40):", df1.columns[:40].tolist())
    print("Category2 columns (first 40):", df2.columns[:40].tolist())
    raise SystemExit("Please ensure stage columns exist in positions M..AA or provide column names to the script.")

# split into three measures (first 5 staging, next 5 plotprog, last 5 cogtension)
staging_cols1 = block1.columns[0:5].tolist()
plotprog_cols1 = block1.columns[5:10].tolist()
cog_cols1 = block1.columns[10:15].tolist()

staging_cols2 = block2.columns[0:5].tolist()
plotprog_cols2 = block2.columns[5:10].tolist()
cog_cols2 = block2.columns[10:15].tolist()

def mean_of_cols(dfblock, cols):
    out = []
    for c in cols:
        vals = safe_numeric(dfblock[c]).dropna().astype(float)
        out.append(float(vals.mean()) if vals.size>0 else np.nan)
    return out

means1 = {
    "Staging": mean_of_cols(block1, staging_cols1),
    "PlotProg": mean_of_cols(block1, plotprog_cols1),
    "CogTension": mean_of_cols(block1, cog_cols1)
}
means2 = {
    "Staging": mean_of_cols(block2, staging_cols2),
    "PlotProg": mean_of_cols(block2, plotprog_cols2),
    "CogTension": mean_of_cols(block2, cog_cols2)
}

# ----------------- Plotting: two separate plots (Category 1 & 2) -----------------
def plot_single_category(means, category_label, outpath):
    x = np.arange(1,6)
    plt.figure(figsize=(7,4.2))
    base_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
    soft_colors = [soften_color(c, alpha=0.55) for c in base_colors]
    for i, key in enumerate(["Staging","PlotProg","CogTension"]):
        y = means[key]
        plt.plot(x, y, marker='o', linewidth=2.2, label=key, color=soft_colors[i], markersize=6)
    plt.xticks(x, [str(i) for i in x])
    plt.xlabel("Segments")
    plt.ylabel("Value")
    plt.title(f"Narrativity stage profiles — {category_label}")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print("Saved plot:", outpath)

out_dir = "results"
plot_single_category(means1, "Category 1 (influencers)", os.path.join(out_dir, "narrativity_stages_Category1.png"))
plot_single_category(means2, "Category 2 (media)", os.path.join(out_dir, "narrativity_stages_Category2.png"))

# Also save a compact numeric summary (CSV) including stage means
rows_summary = []
# overall metrics
for metric in overall_candidates.keys():
    rows_summary.append({
        "Metric": metric,
        "cat1_col": stats1[metric]["col"],
        "cat1_mean": stats1[metric]["mean"],
        "cat1_sd": stats1[metric]["sd"],
        "cat1_n": stats1[metric]["n"],
        "cat2_col": stats2[metric]["col"],
        "cat2_mean": stats2[metric]["mean"],
        "cat2_sd": stats2[metric]["sd"],
        "cat2_n": stats2[metric]["n"]
    })
# stage means
for measure in ["Staging","PlotProg","CogTension"]:
    for i in range(5):
        rows_summary.append({
            "Metric": f"{measure}_segment{i+1}",
            "cat1_col": (staging_cols1 + plotprog_cols1 + cog_cols1)[i if measure=="Staging" else (5+i) if measure=="PlotProg" else (10+i)],
            "cat1_mean": (means1[measure][i] if not math.isnan(means1[measure][i]) else None),
            "cat2_col": (staging_cols2 + plotprog_cols2 + cog_cols2)[i if measure=="Staging" else (5+i) if measure=="PlotProg" else (10+i)],
            "cat2_mean": (means2[measure][i] if not math.isnan(means2[measure][i]) else None)
        })

summary_df = pd.DataFrame(rows_summary)
summary_path = os.path.join(out_dir, "narrativity_stage_summary.csv")
summary_df.to_csv(summary_path, index=False)
print("Saved numeric summary CSV to:", summary_path)

# print the small comparison table to console
print("\nComparison table (Metric rows × Category columns):\n")
print(comp_table.to_string())
print("\nAll files saved in 'results' folder.")
