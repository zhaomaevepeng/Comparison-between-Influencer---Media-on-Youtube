# bertopic_topics.py
# ------------------------------------------------------------
# BERTopic topic modeling per category (1/2/3) for merged_transcripts.csv
# - Uses all-MiniLM-L6-v2 embeddings (fast, solid)
# - Custom stopwords & vectorizer to reduce presenter filler
# - Saves: topic info, topic words, doc-topic assignments,
#          interactive HTML charts, and a static grid image per category.
# ------------------------------------------------------------

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple

# Core BERTopic stack
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.stem import SnowballStemmer



# -----------------------------
# Paths & column names
# -----------------------------
INPUT_CSV = "merged_transcripts.csv"          # <- your file
TEXT_COL  = "full_transcript"
CAT_COL   = "Influencer/Mainstream"           # contains {1,2,3} (one noisy row is handled)
ID_COL    = "video_id"
OUT_DIR   = "bertopic_outputs"

# -----------------------------
# Modeling configs
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small/fast (384-d)
MIN_TOPIC_SIZE = 5                # tweak per category size
N_GRAM_RANGE = (1, 2)             # unigrams+bigrams
TOP_N_WORDS = 12                   # words shown per topic internally
TOP_WORDS_FOR_GRID = 10            # words shown per tile in the static grid image
LOW_FREQ_MIN_DF = 1               # drop ultra-rare terms
HIGH_FREQ_MAX_DF = 1.0           # drop near-ubiquitous terms
RANDOM_STATE = 42

# Extra domain stopwords to kill boilerplate
EXTRA_STOPWORDS = {
    "subscribe","channel","video","like","comment","today","uh","um","yeah",
    "thing","things","gonna","kind","sort","lot","really","going","say","know",
    "want","see","think","just","dad","II","ll","right"
}

STEMMER = SnowballStemmer("english")
STOPSET = set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS)

# -----------------------------
# Helpers
# -----------------------------
def basic_clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)   # URLs
    s = re.sub(r"[^a-z\s']", " ", s)          # keep letters/apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s
def stem_analyzer(doc: str):
    """
    Analyzer for CountVectorizer:
    - lowercase/clean via basic_clean
    - tokenize by whitespace
    - remove stopwords
    - stem tokens so 'rates' and 'rate' -> 'rate'
    """
    doc = basic_clean(doc)
    tokens = doc.split()
    out = []
    for tok in tokens:
        if tok in STOPSET:
            continue
        stem = STEMMER.stem(tok)
        if stem and stem not in STOPSET:
            out.append(stem)
    return out

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize categories to {1,2,3} (there is a stray value)
    def to_cat(val):
        m = re.search(r"\b([123])\b", str(val))
        return int(m.group(1)) if m else np.nan

    df[CAT_COL] = df[CAT_COL].apply(to_cat)
    df = df[df[CAT_COL].isin([1, 2, 3])]

    df[TEXT_COL] = df[TEXT_COL].astype(str).map(basic_clean)
    df = df[df[TEXT_COL].str.len() > 30].dropna(subset=[TEXT_COL, CAT_COL])
    df[ID_COL] = df[ID_COL].astype(str)
    return df.reset_index(drop=True)

def make_vectorizer():
    return CountVectorizer(
        analyzer=stem_analyzer,          # <— use stemmed tokens
        max_df=HIGH_FREQ_MAX_DF,
        min_df=LOW_FREQ_MIN_DF,
        ngram_range=N_GRAM_RANGE
    )


def make_bertopic(min_cluster_size: int):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE
    )
    # More permissive HDBSCAN (smaller clusters allowed; tolerate noisier data)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,                 # <- be a bit more permissive
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=make_vectorizer(),
        top_n_words=TOP_N_WORDS,
        min_topic_size=min_cluster_size,  # keep consistent with HDBSCAN
        calculate_probabilities=True,
        verbose=True
    )
    return topic_model


def topics_to_long_df(topic_model: BERTopic, topic_info: pd.DataFrame) -> pd.DataFrame:
    """Expand each topic's word list into long form rows."""
    rows = []
    for topic_id in topic_info["Topic"].tolist():
        if topic_id == -1:
            # -1 is usually 'outliers' in BERTopic
            continue
        words = topic_model.get_topic(topic_id)  # list of (word, weight)
        for rank, (w, wt) in enumerate(words, start=1):
            rows.append({"topic": topic_id, "rank": rank, "word": w, "weight": float(wt)})
    return pd.DataFrame(rows)

def plot_topic_grid(long_topics_df: pd.DataFrame, cat_value: int, out_path: str, top_n: int = TOP_WORDS_FOR_GRID):
    """Static grid image like your screenshot: one small horizontal barchart per topic."""
    if long_topics_df.empty:
        return

    topics_sorted = sorted(long_topics_df["topic"].unique())
    k = len(topics_sorted)
    ncols = 4 if k >= 8 else min(3, k)
    nrows = int(math.ceil(k / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        squeeze=False
    )
    fig.suptitle("Topic Word Scores", fontsize=18, fontweight="bold")

    for idx, t in enumerate(topics_sorted):
        ax = axes[idx // ncols][idx % ncols]
        sub = long_topics_df[long_topics_df["topic"] == t].sort_values("rank").head(top_n)
        words = sub["word"].tolist()
        weights = sub["weight"].astype(float).tolist()

        # largest on top (reverse for barh)
        words_plot = list(reversed(words))
        weights_plot = list(reversed(weights))
        y = np.arange(len(words_plot))

        ax.barh(y, weights_plot)
        ax.set_yticks(y)
        ax.set_yticklabels(words_plot)
        ax.set_title(f"Topic {t}", fontsize=12, pad=6)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # hide empty tiles
    total_axes = nrows * ncols
    for j in range(k, total_axes):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def run_category(df_cat: pd.DataFrame, cat_value: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    docs = df_cat[TEXT_COL].tolist()
    ids  = df_cat[ID_COL].tolist()

    print(f"\n=== Category {cat_value} | docs={len(docs)} ===")
    if len(docs) < 5:
        print(f"[WARN] Very few docs; topics may be noisy.")

    # ---- choose a min_cluster_size that scales with category size
    n_docs = len(docs)
    # heuristic: smaller clusters for smaller categories
    min_cluster = max(2, min(8, n_docs // 5))   # value between 2 and 8

    def fit_and_collect(min_clust):
        tm = make_bertopic(min_clust)
        topics, probs = tm.fit_transform(docs)
        return tm, topics, probs

    # first run
    topic_model, topics, probs = fit_and_collect(min_cluster)

    # ---- if everything is outlier (-1), retry with smaller clusters
    unique_topics = set(np.unique(topics))
    if unique_topics == {-1}:
        print(f"[INFO] Category {cat_value}: all outliers at min_cluster_size={min_cluster}. Retrying with smaller clusters.")
        min_cluster_retry = max(2, min_cluster // 2)
        topic_model, topics, probs = fit_and_collect(min_cluster_retry)
        unique_topics = set(np.unique(topics))

    # ---- optionally reduce outliers if possible
    try:
        topics = topic_model.reduce_outliers(docs, topics)
        unique_topics = set(np.unique(topics))
    except Exception:
        pass

    # ---- Save doc-topic assignments
    doc_df = pd.DataFrame({
        ID_COL: ids,
        "topic": topics
    })
    if probs is not None and isinstance(probs, np.ndarray) and probs.ndim == 2:
        prob_df = pd.DataFrame(probs, columns=[f"topic_{i}" for i in range(probs.shape[1])])
        doc_df = pd.concat([doc_df, prob_df], axis=1)

    doc_csv = os.path.join(OUT_DIR, f"cat{cat_value}_bertopic_doc_topics.csv")
    doc_df.to_csv(doc_csv, index=False)

    # ---- Save topic info and topic words
    topic_info = topic_model.get_topic_info()
    info_csv = os.path.join(OUT_DIR, f"cat{cat_value}_bertopic_topic_info.csv")
    topic_info.to_csv(info_csv, index=False)

    long_topics = topics_to_long_df(topic_model, topic_info)
    long_csv = os.path.join(OUT_DIR, f"cat{cat_value}_bertopic_topics_words.csv")
    long_topics.to_csv(long_csv, index=False)

    # ---- Interactive HTML (Plotly)
    try:
        fig_bar = topic_model.visualize_barchart(top_n_topics=None, top_n_words=TOP_N_WORDS)
        fig_bar.write_html(os.path.join(OUT_DIR, f"cat{cat_value}_barchart.html"))
    except Exception as e:
        print("[viz] barchart:", e)

    try:
        fig_hier = topic_model.visualize_hierarchy()
        fig_hier.write_html(os.path.join(OUT_DIR, f"cat{cat_value}_hierarchy.html"))
    except Exception as e:
        print("[viz] hierarchy:", e)

    try:
        fig_heat = topic_model.visualize_heatmap()
        fig_heat.write_html(os.path.join(OUT_DIR, f"cat{cat_value}_heatmap.html"))
    except Exception as e:
        print("[viz] heatmap:", e)

    # ---- Static grid image like your screenshot
    grid_png = os.path.join(OUT_DIR, f"cat{cat_value}_topic_word_grid.png")
    if not long_topics.empty:
        plot_topic_grid(long_topics, cat_value, grid_png, top_n=TOP_WORDS_FOR_GRID)
    else:
        print(f"[INFO] Category {cat_value}: no non-outlier topics to plot.")

    print("Saved:")
    print(" ", doc_csv)
    print(" ", info_csv)
    print(" ", long_csv)
    if not long_topics.empty:
        print(" ", grid_png)
    print(" ", os.path.join(OUT_DIR, f"cat{cat_value}_barchart.html"))
    print(" ", os.path.join(OUT_DIR, f"cat{cat_value}_hierarchy.html"))
    print(" ", os.path.join(OUT_DIR, f"cat{cat_value}_heatmap.html"))


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_prepare(INPUT_CSV)
    print("Counts by category (after cleaning):")
    print(df[CAT_COL].value_counts().sort_index())

    for cat in [1, 2, 3]:
        sub = df[df[CAT_COL] == cat].reset_index(drop=True)
        if sub.empty:
            print(f"[SKIP] Category {cat} has 0 docs.")
            continue
        run_category(sub, cat)

if __name__ == "__main__":
    main()
