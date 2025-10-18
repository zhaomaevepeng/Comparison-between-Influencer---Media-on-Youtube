# split_by_category.py
# ------------------------------------------------------------
# Splits merged_transcripts.csv into three files based on
# the Influencer/Mainstream category column.
#   1 → category_1.csv
#   2 → category_2.csv
#   3 → category_3.csv
# ------------------------------------------------------------

import pandas as pd
import re
import os

# Input file and column names
INPUT_CSV = "merged_transcripts.csv"
CAT_COL = "Influencer/Mainstream"
OUT_DIR = "split_datasets"

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df):,} rows from {INPUT_CSV}")

# ------------------------------------------------------------
# 2. Clean category column — normalize values to 1, 2, or 3
# ------------------------------------------------------------
def to_cat(val):
    """
    Extract numeric category (1, 2, or 3) from messy column values.
    """
    m = re.search(r"\b([123])\b", str(val))
    return int(m.group(1)) if m else None

df[CAT_COL] = df[CAT_COL].apply(to_cat)
df = df[df[CAT_COL].isin([1, 2, 3])].reset_index(drop=True)

print("Category counts after cleaning:")
print(df[CAT_COL].value_counts().sort_index())

# ------------------------------------------------------------
# 3. Split and save
# ------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

for cat_value in [1, 2, 3]:
    subset = df[df[CAT_COL] == cat_value]
    out_path = os.path.join(OUT_DIR, f"category_{cat_value}.csv")
    subset.to_csv(out_path, index=False)
    print(f"Saved category {cat_value} → {out_path} ({len(subset)} rows)")

print("\n✅ Done! Files saved in:", OUT_DIR)
