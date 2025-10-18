# sentiment_emotions.py
# ------------------------------------------------------------
# Sentiment + Emotion analysis per category for merged_transcripts.csv
# - Sentiment: NLTK VADER (compound/pos/neu/neg)
# - Emotions:  NRCLex (8 Plutchik categories)
# - Emotional diversity: Shannon entropy (normalized) & Gini–Simpson
# - Outputs per-category CSVs + optional plots
# ------------------------------------------------------------
import nltk
nltk.download('vader_lexicon')

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "merged_transcripts.csv"
TEXT_COL  = "full_transcript"
CAT_COL   = "Influencer/Mainstream"    # categories {1,2,3} with one noisy row in your file
ID_COL    = "video_id"
OUT_DIR   = "affect_outputs"

MAKE_PLOTS = True   # set False if you only want CSVs
RANDOM_STATE = 42

# -----------------------------
# Imports (with graceful setup)
# -----------------------------
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    _ = SentimentIntensityAnalyzer()
except:
    nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer

try:
    from nrclex import NRCLex
except ImportError:
    raise SystemExit(
        "Missing dependency: nrclex\n"
        "Install with: pip install nrclex\n"
        "Then re-run:  python sentiment_emotions.py"
    )

# -----------------------------
# Helpers
# -----------------------------
def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize category to {1,2,3}
    def to_cat(val):
        m = re.search(r"\b([123])\b", str(val))
        return int(m.group(1)) if m else np.nan
    df[CAT_COL] = df[CAT_COL].apply(to_cat)
    df = df[df[CAT_COL].isin([1, 2, 3])]

    # Clean text and keep non-trivial docs
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(basic_clean)
    df = df[df[TEXT_COL].str.len() > 30].dropna(subset=[TEXT_COL, CAT_COL])
    df[ID_COL] = df[ID_COL].astype(str)
    return df.reset_index(drop=True)

def sentiment_scores(texts):
    """Return a DataFrame with VADER scores for a list of texts."""
    sia = SentimentIntensityAnalyzer()
    rows = []
    for t in texts:
        sc = sia.polarity_scores(t)  # dict: neg, neu, pos, compound
        rows.append(sc)
    return pd.DataFrame(rows)

EMOTIONS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]

def emotion_profile(text: str):
    """
    Return normalized emotion proportions for text using NRCLex.
    If no emotion words found, returns zeros.
    """
    emo = NRCLex(text)
    raw = emo.raw_emotion_scores  # dict emotion -> count
    vec = np.array([raw.get(e, 0) for e in EMOTIONS], dtype=float)
    total = vec.sum()
    if total <= 0:
        return np.zeros_like(vec)
    return vec / total

def shannon_entropy(p):
    """Normalized Shannon entropy in [0,1]. p must sum to 1 (or be zero-vector)."""
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    H = -np.sum(nz * np.log(nz))
    H_max = math.log(len(p)) if len(p) > 0 else 1.0
    return float(H / H_max) if H_max > 0 else 0.0

def gini_simpson(p):
    """1 - sum(p^2). p need not be normalized; we'll normalize."""
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    return float(1.0 - np.sum(p * p))

def analyze_per_category(df: pd.DataFrame):
    """
    For each transcript: compute sentiment + emotion profile + diversity metrics.
    Then aggregate per category.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Per-doc sentiment
    sent_df = sentiment_scores(df[TEXT_COL].tolist())
    sent_df.insert(0, ID_COL, df[ID_COL].values)
    sent_df.insert(1, CAT_COL, df[CAT_COL].values)

    # Per-doc emotions + diversity
    emo_rows = []
    for i, row in df.iterrows():
        vec = emotion_profile(row[TEXT_COL])
        sh = shannon_entropy(vec)
        gs = gini_simpson(vec)
        rec = {
            ID_COL: row[ID_COL],
            CAT_COL: row[CAT_COL],
            **{f"emo_{EMOTIONS[j]}": float(vec[j]) for j in range(len(EMOTIONS))},
            "emo_shannon_norm": sh,
            "emo_gini_simpson": gs
        }
        emo_rows.append(rec)
    emo_df = pd.DataFrame(emo_rows)

    # Save per-document outputs (joined)
    per_doc = sent_df.merge(emo_df, on=[ID_COL, CAT_COL], how="inner")
    per_doc_csv = os.path.join(OUT_DIR, "per_document_affect.csv")
    per_doc.to_csv(per_doc_csv, index=False)

    # ----- Aggregate per category
    agg_sent = per_doc.groupby(CAT_COL)[["compound","pos","neu","neg"]].mean().reset_index()
    emo_cols = [f"emo_{e}" for e in EMOTIONS]
    agg_emo = per_doc.groupby(CAT_COL)[emo_cols].mean().reset_index()
    agg_div = per_doc.groupby(CAT_COL)[["emo_shannon_norm","emo_gini_simpson"]].mean().reset_index()

    # Merge into one table for convenience
    agg_all = agg_sent.merge(agg_emo, on=CAT_COL, how="inner").merge(agg_div, on=CAT_COL, how="inner")

    # Save category-level outputs
    agg_sent.to_csv(os.path.join(OUT_DIR, "category_sentiment_means.csv"), index=False)
    agg_emo.to_csv(os.path.join(OUT_DIR, "category_emotion_means.csv"), index=False)
    agg_div.to_csv(os.path.join(OUT_DIR, "category_emotion_diversity.csv"), index=False)
    agg_all.to_csv(os.path.join(OUT_DIR, "category_affect_summary.csv"), index=False)

    print("Saved:")
    print(" ", per_doc_csv)
    print(" ", os.path.join(OUT_DIR, "category_sentiment_means.csv"))
    print(" ", os.path.join(OUT_DIR, "category_emotion_means.csv"))
    print(" ", os.path.join(OUT_DIR, "category_emotion_diversity.csv"))
    print(" ", os.path.join(OUT_DIR, "category_affect_summary.csv"))

    return per_doc, agg_all

def make_plots(agg_all: pd.DataFrame):
    """Minimal, clean matplotlib plots (one plot per figure, no styles/colors)."""
    # Sentiment bars per category
    cats = agg_all[CAT_COL].tolist()

    # 1) Compound sentiment
    plt.figure(figsize=(7,5))
    x = np.arange(len(cats))
    plt.bar(x, agg_all["compound"].values)
    plt.xticks(x, [str(c) for c in cats])
    plt.title("Average Sentiment (Compound) by Category")
    plt.xlabel("Category")
    plt.ylabel("VADER compound")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_sentiment_compound_by_category.png"), dpi=150)
    plt.close()

    # 2) Emotion proportions (stacked-like separate figures per emotion)
    for emo in EMOTIONS:
        col = f"emo_{emo}"
        plt.figure(figsize=(7,5))
        plt.bar(x, agg_all[col].values)
        plt.xticks(x, [str(c) for c in cats])
        plt.title(f"Average Emotion Proportion: {emo} (by Category)")
        plt.xlabel("Category")
        plt.ylabel("Mean proportion")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"plot_emotion_{emo}_by_category.png"), dpi=150)
        plt.close()

    # 3) Emotional diversity
    plt.figure(figsize=(7,5))
    plt.bar(x, agg_all["emo_shannon_norm"].values)
    plt.xticks(x, [str(c) for c in cats])
    plt.title("Emotional Diversity (Shannon, normalized) by Category")
    plt.xlabel("Category")
    plt.ylabel("Diversity [0–1]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_diversity_shannon_by_category.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.bar(x, agg_all["emo_gini_simpson"].values)
    plt.xticks(x, [str(c) for c in cats])
    plt.title("Emotional Diversity (Gini–Simpson) by Category")
    plt.xlabel("Category")
    plt.ylabel("1 - sum(p^2)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_diversity_ginisimpson_by_category.png"), dpi=150)
    plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_prepare(INPUT_CSV)
    print("Counts by category:", df[CAT_COL].value_counts().to_dict())

    per_doc, agg_all = analyze_per_category(df)

    if MAKE_PLOTS:
        make_plots(agg_all)

if __name__ == "__main__":
    main()
