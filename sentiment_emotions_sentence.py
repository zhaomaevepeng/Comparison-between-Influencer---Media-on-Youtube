# sentiment_emotions_sentence.py
# ------------------------------------------------------------
# Sentence-level Sentiment + Emotion + Diversity for merged_transcripts.csv
# - Sentence splitter: NLTK Punkt
# - Sentiment: VADER (compound/pos/neu/neg) per sentence
# - Emotions: NRCLex (anger, anticipation, disgust, fear, joy, sadness, surprise, trust)
# - Diversity per sentence: Shannon entropy (normalized 0–1) & Gini–Simpson (1 - sum p^2)
# - Saves per-sentence CSV, plus per-transcript (avg over sentences) and per-category summaries
# - Optional matplotlib plots (no seaborn; 1 plot per figure; no explicit colors)
# ------------------------------------------------------------

import os
import re
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "merged_transcripts.csv"
TEXT_COL  = "full_transcript"
CAT_COL   = "Influencer/Mainstream"    # {1,2,3} with a stray row — cleaned below
ID_COL    = "video_id"
OUT_DIR   = "affect_sentence_outputs"

MAKE_PLOTS = True   # set to False to skip charts
RANDOM_STATE = 42

# -----------------------------
# Robust NLTK setup (punkt + vader)
# -----------------------------
import nltk
def _ensure_nltk_resource(resource_path: str, download_name: str):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name, quiet=True)

# Punkt for sentence tokenization
_ensure_nltk_resource("tokenizers/punkt", "punkt")

# Try VADER via NLTK; if SSL/download issues occur, suggest manual install
from nltk.sentiment import SentimentIntensityAnalyzer
try:
    _ = SentimentIntensityAnalyzer()
except Exception:
    # Attempt auto download to user dir; if still fails, raise clear message
    try:
        nltk.download("vader_lexicon", quiet=True)
        _ = SentimentIntensityAnalyzer()
    except Exception as e:
        raise SystemExit(
            "Couldn't load VADER lexicon.\n"
            "Fix option 1 (recommended macOS): run once:\n"
            '  open "/Applications/Python 3.12/Install Certificates.command"\n'
            "Then re-run this script.\n\n"
            "Fix option 2: in terminal run:\n"
            "  python -m nltk.downloader vader_lexicon\n\n"
            f"Original error: {e}"
        )

# NRCLex for emotion lexicon
try:
    from nrclex import NRCLex
except ImportError:
    raise SystemExit("Missing dependency: nrclex. Install with: pip install nrclex")

from nltk import sent_tokenize

# -----------------------------
# Helpers
# -----------------------------
def clean_for_processing(s: str) -> str:
    """Light cleanup but KEEP sentence punctuation for tokenizer & VADER."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)  # strip URLs (they confuse tokenizers/sentiment)
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

    # Keep original text (just URL cleanup); also enforce basic length
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(clean_for_processing)
    df = df[df[TEXT_COL].str.len() > 10].dropna(subset=[TEXT_COL, CAT_COL])
    df[ID_COL] = df[ID_COL].astype(str)
    return df.reset_index(drop=True)

SENT_EMOTIONS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]

def emotion_profile(text: str) -> np.ndarray:
    """
    Return normalized emotion proportions for a sentence using NRCLex.
    If no emotion terms are found, returns zeros.
    """
    emo = NRCLex(text.lower())
    raw = emo.raw_emotion_scores  # dict
    vec = np.array([raw.get(e, 0) for e in SENT_EMOTIONS], dtype=float)
    tot = vec.sum()
    if tot <= 0:
        return np.zeros_like(vec)
    return vec / tot

def shannon_entropy(p: np.ndarray) -> float:
    """Normalized Shannon entropy ∈ [0,1]."""
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    H = -np.sum(nz * np.log(nz))
    H_max = math.log(len(p)) if len(p) > 0 else 1.0
    return float(H / H_max) if H_max > 0 else 0.0

def gini_simpson(p: np.ndarray) -> float:
    """Gini–Simpson index = 1 - ∑p^2."""
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    return float(1.0 - np.sum(p * p))

# -----------------------------
# Core analysis
# -----------------------------
def analyze_sentence_level(df: pd.DataFrame):
    """
    For each transcript:
      - split into sentences
      - compute VADER + NRCLex per sentence
      - compute diversity per sentence
    Returns:
      per_sentence_df, per_transcript_df, per_category_df
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    sia = SentimentIntensityAnalyzer()

    sent_records = []

    for _, row in df.iterrows():
        vid = row[ID_COL]
        cat = row[CAT_COL]
        text = row[TEXT_COL]
        # Sentence split
        sents = [s.strip() for s in sent_tokenize(text) if s and s.strip()]
        if not sents:
            continue

        for idx, s in enumerate(sents):
            # Sentiment per sentence
            sc = sia.polarity_scores(s)  # dict: neg, neu, pos, compound

            # Emotions per sentence
            emo_vec = emotion_profile(s)
            sh = shannon_entropy(emo_vec)
            gs = gini_simpson(emo_vec)

            rec = {
                ID_COL: vid,
                CAT_COL: cat,
                "sentence_index": idx,
                "sentence_text": s,
                "neg": sc["neg"],
                "neu": sc["neu"],
                "pos": sc["pos"],
                "compound": sc["compound"],
                **{f"emo_{SENT_EMOTIONS[j]}": float(emo_vec[j]) for j in range(len(SENT_EMOTIONS))},
                "emo_shannon_norm": sh,
                "emo_gini_simpson": gs
            }
            sent_records.append(rec)

    per_sentence = pd.DataFrame(sent_records)
    if per_sentence.empty:
        raise SystemExit("No sentences produced. Check the input CSV/text column.")

    # Save per-sentence
    per_sentence_csv = os.path.join(OUT_DIR, "per_sentence_affect.csv")
    per_sentence.to_csv(per_sentence_csv, index=False)

    # ---- Aggregate over sentences to per-transcript (mean of sentence metrics)
    metric_cols = ["neg","neu","pos","compound"] \
                  + [f"emo_{e}" for e in SENT_EMOTIONS] \
                  + ["emo_shannon_norm","emo_gini_simpson"]
    per_transcript = (per_sentence
                      .groupby([ID_COL, CAT_COL])[metric_cols]
                      .mean()
                      .reset_index())
    per_transcript_csv = os.path.join(OUT_DIR, "per_transcript_from_sentences.csv")
    per_transcript.to_csv(per_transcript_csv, index=False)

    # ---- Per-category summaries (averaging sentence-level metrics)
    per_category = (per_sentence
                    .groupby(CAT_COL)[metric_cols]
                    .mean()
                    .reset_index())
    per_category_csv = os.path.join(OUT_DIR, "category_sentence_level_summary.csv")
    per_category.to_csv(per_category_csv, index=False)

    print("Saved:")
    print(" ", per_sentence_csv)
    print(" ", per_transcript_csv)
    print(" ", per_category_csv)

    return per_sentence, per_transcript, per_category

# -----------------------------
# Plots (optional)
# -----------------------------
def make_plots(per_category: pd.DataFrame):
    cats = per_category[CAT_COL].tolist()
    x = np.arange(len(cats))

    # 1) Compound sentiment by category (sentence-level mean)
    plt.figure(figsize=(7,5))
    plt.bar(x, per_category["compound"].values)
    plt.xticks(x, [str(c) for c in cats])
    plt.title("Sentence-level Average Sentiment (Compound) by Category")
    plt.xlabel("Category")
    plt.ylabel("VADER compound")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_sentence_compound_by_category.png"), dpi=150)
    plt.close()

    # 2) Diversity (Shannon, normalized)
    plt.figure(figsize=(7,5))
    plt.bar(x, per_category["emo_shannon_norm"].values)
    plt.xticks(x, [str(c) for c in cats])
    plt.title("Sentence-level Emotional Diversity (Shannon, normalized)")
    plt.xlabel("Category")
    plt.ylabel("Diversity [0–1]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_sentence_diversity_shannon.png"), dpi=150)
    plt.close()

    # 3) Diversity (Gini–Simpson)
    plt.figure(figsize=(7,5))
    plt.bar(x, per_category["emo_gini_simpson"].values)
    plt.xticks(x, [str(c) for c in cats])
    plt.title("Sentence-level Emotional Diversity (Gini–Simpson)")
    plt.xlabel("Category")
    plt.ylabel("1 - sum(p^2)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "plot_sentence_diversity_ginisimpson.png"), dpi=150)
    plt.close()

    # 4) Each emotion’s average proportion by category
    for emo in SENT_EMOTIONS:
        col = f"emo_{emo}"
        plt.figure(figsize=(7,5))
        plt.bar(x, per_category[col].values)
        plt.xticks(x, [str(c) for c in cats])
        plt.title(f"Sentence-level Average Emotion Proportion: {emo}")
        plt.xlabel("Category")
        plt.ylabel("Mean proportion")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"plot_sentence_emotion_{emo}.png"), dpi=150)
        plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_prepare(INPUT_CSV)
    print("Docs by category:", df[CAT_COL].value_counts().to_dict())

    per_sentence, per_transcript, per_category = analyze_sentence_level(df)

    if MAKE_PLOTS:
        make_plots(per_category)

if __name__ == "__main__":
    main()
