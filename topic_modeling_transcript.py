# lda_topics.py
# ------------------------------------------------------------
# LDA topic modeling for 3 categories (1/2/3) of transcripts.
# - CountVectorizer -> LDA (scikit-learn)
# - K selected by held-out perplexity
# - Saves topics, doc-topic matrices, and top docs per topic
# ------------------------------------------------------------

import os
import re
import numpy as np
import pandas as pd
from typing import List, Tuple
import math
import matplotlib.pyplot as plt

# How many words to show per topic tile in the grid image (screenshot uses 5)
TOP_WORDS_FOR_GRID = 10


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths & column names
# -----------------------------
INPUT_CSV = "merged_transcripts.csv"
TEXT_COL = "full_transcript"
CAT_COL = "Influencer/Mainstream"     # your 1/2/3 category
ID_COL = "video_id"
OUT_DIR = "lda_outputs"

# -----------------------------
# Modeling hyperparameters
# -----------------------------
K_CANDIDATES = list(range(4, 13))  # grid for topic count per category
TOP_N_WORDS = 15                # words printed/saved per topic
TOP_N_DOCS = 10                 # top docs saved per topic
MAX_FEATURES = 30000
MIN_DF = 3
MAX_DF = 0.95
RANDOM_STATE = 42

# LDA settings (good defaults)
LDA_MAX_ITER = 40
LDA_LEARNING_METHOD = "batch"    # "online" also works; "batch" is stable
LDA_EVALUATE_EVERY = 0           # no internal eval to save time

# Optional domain stopwords you might want to expand
EXTRA_STOPWORDS = {
    "subscribe", "channel", "video", "like", "comment", "today",
    "uh", "um", "yeah", "thing", "things", "gonna", "kind",
    "sort", "lot", "really","going", "say","know", "want", "see", "think",
    "dad","just","let","right","well","okay","ve"
}

# -----------------------------
# Helpers
# -----------------------------
def plot_topic_grid(topics, cat_value: int, out_path: str, top_n: int = TOP_WORDS_FOR_GRID):
    """
    Create a grid of small horizontal bar charts (one per topic), showing top words and weights.
    - topics: list like [(topic_idx, [(word, weight), ...]), ...] from topic_top_words()
    - cat_value: 1/2/3
    - out_path: where to save the figure PNG
    """
    k = len(topics)
    if k == 0:
        return

    # Choose a tidy grid (4 columns if many topics, else 3 or fewer)
    ncols = 4 if k >= 8 else min(3, k)
    nrows = int(math.ceil(k / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(4.2 * ncols, 3.2 * nrows),
        squeeze=False
    )
    fig.suptitle("Topic Word Scores", fontsize=18, fontweight="bold")

    # Draw each topic tile
    for idx, (t_idx, word_weights) in enumerate(topics):
        ax = axes[idx // ncols][idx % ncols]
        words = [w for w, _ in word_weights][:top_n]
        weights = [float(wt) for _, wt in word_weights][:top_n]

        # Horizontal bars, largest at top (reverse the order for nicer look)
        words_plot = list(reversed(words))
        weights_plot = list(reversed(weights))
        y = np.arange(len(words_plot))
        ax.barh(y, weights_plot)
        ax.set_yticks(y)
        ax.set_yticklabels(words_plot)
        ax.set_title(f"Topic {t_idx}", fontsize=12, pad=6)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
        # Tidy spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # Hide any unused subplots
    total_axes = nrows * ncols
    for j in range(k, total_axes):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def basic_clean(s: str) -> str:
    """Light normalization for transcripts."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)  # remove URLs
    s = re.sub(r"[^a-z\s']", " ", s)         # keep letters/apostrophes
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Clean category column: keep only 1/2/3 (there is a noisy row in your file)
    def to_cat(val):
        m = re.search(r"\b([123])\b", str(val))
        return int(m.group(1)) if m else np.nan

    df[CAT_COL] = df[CAT_COL].apply(to_cat)
    df = df[df[CAT_COL].isin([1, 2, 3])]

    # Text cleaning & filtering
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(basic_clean)
    df = df[df[TEXT_COL].str.len() > 30].dropna(subset=[TEXT_COL, CAT_COL])

    # Ensure IDs as string
    df[ID_COL] = df[ID_COL].astype(str)

    return df.reset_index(drop=True)

def build_vectorizer():
    # Merge english stop words with EXTRA_STOPWORDS
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stops = set(ENGLISH_STOP_WORDS).union(EXTRA_STOPWORDS)

    return CountVectorizer(
        stop_words=list(stops),
        max_df=MAX_DF,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        ngram_range=(1, 2)  # unigrams + bigrams help on transcripts
    )

def fit_lda(X, n_topics: int) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method=LDA_LEARNING_METHOD,
        max_iter=LDA_MAX_ITER,
        random_state=RANDOM_STATE,
        evaluate_every=LDA_EVALUATE_EVERY,
        doc_topic_prior=None,        # defaults
        topic_word_prior=None
    )
    lda.fit(X)
    return lda

def select_k_by_perplexity(docs: List[str]) -> Tuple[int, float]:
    """
    Choose K by held-out perplexity:
    - Split docs into train/test
    - Fit LDA on train for each K
    - Compute perplexity on test (lower is better)
    """
    if len(docs) < 40:
        # Small categories: skip split to avoid instability; just pick a mid K
        k_default = min(7, max(5, len(docs)//3 or 5))
        return k_default, float("nan")

    docs_train, docs_test = train_test_split(
        docs, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )

    cv = build_vectorizer()
    X_train = cv.fit_transform(docs_train)
    X_test = cv.transform(docs_test)

    results = []
    for k in K_CANDIDATES:
        lda = fit_lda(X_train, k)
        perp = lda.perplexity(X_test)
        results.append((k, perp))
        print(f"  K={k:<3}  held-out perplexity = {perp:.2f}")

    best_k, best_perp = min(results, key=lambda x: x[1])
    print(f"→ Selected K={best_k} (lowest held-out perplexity {best_perp:.2f})")
    return best_k, best_perp

def topic_top_words(components, feature_names, n_top: int) -> list:
    topics = []
    for t_idx, comp in enumerate(components):
        top_idx = np.argsort(comp)[::-1][:n_top]
        words = [feature_names[i] for i in top_idx]
        weights = [float(comp[i]) for i in top_idx]
        topics.append((t_idx, list(zip(words, weights))))
    return topics

def save_topics_csv(topics, out_csv):
    rows = []
    for t_idx, word_weights in topics:
        for rank, (w, wt) in enumerate(word_weights, start=1):
            rows.append({"topic": t_idx, "rank": rank, "word": w, "weight": wt})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

def save_doc_topics_csv(ids, theta, out_csv):
    doc_df = pd.DataFrame(theta, columns=[f"topic_{i}" for i in range(theta.shape[1])])
    doc_df.insert(0, ID_COL, ids)
    doc_df.to_csv(out_csv, index=False)

def save_top_docs_per_topic(ids, theta, n_docs, out_csv):
    rows = []
    for k in range(theta.shape[1]):
        scores = theta[:, k]
        top_idx = np.argsort(scores)[::-1][:n_docs]
        for rank, i in enumerate(top_idx, start=1):
            rows.append({"topic": k, "rank": rank, ID_COL: ids[i], "weight": float(scores[i])})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# -----------------------------
# Main per-category routine
# -----------------------------
def run_for_category(df_cat: pd.DataFrame, cat_value: int):
    os.makedirs(OUT_DIR, exist_ok=True)

    docs = df_cat[TEXT_COL].tolist()
    ids = df_cat[ID_COL].tolist()
    print(f"\n=== CATEGORY {cat_value} | docs: {len(docs)} ===")

    # 1) choose K via held-out perplexity (or a heuristic if too small)
    best_k, heldout_perp = select_k_by_perplexity(docs)

    # 2) fit final model on ALL docs with best_k
    cv_final = build_vectorizer()
    X_all = cv_final.fit_transform(docs)
    lda_final = fit_lda(X_all, best_k)
    feature_names = np.array(cv_final.get_feature_names_out())

    # 3) extract topics
    topics = topic_top_words(lda_final.components_, feature_names, TOP_N_WORDS)

    # 4) doc-topic distributions (theta)
    theta = lda_final.transform(X_all)  # rows sum to ~1

    # 5) save outputs
    topics_csv = os.path.join(OUT_DIR, f"cat{cat_value}_lda_topics.csv")
    save_topics_csv(topics, topics_csv)

    doc_topics_csv = os.path.join(OUT_DIR, f"cat{cat_value}_lda_doc_topics.csv")
    save_doc_topics_csv(ids, theta, doc_topics_csv)

    top_docs_csv = os.path.join(OUT_DIR, f"cat{cat_value}_top_docs_per_topic.csv")
    save_top_docs_per_topic(ids, theta, TOP_N_DOCS, top_docs_csv)

    # 6) console summary
    print(f"Selected K = {best_k}  (held-out perplexity: {heldout_perp})")
    print("Top words per topic:")
    for t_idx, word_weights in topics:
        print(f"  Topic {t_idx:02d}: " + ", ".join([w for w, _ in word_weights]))

    print("Saved files:")
    print(" ", topics_csv)
    print(" ", doc_topics_csv)
    print(" ", top_docs_csv)

        # 7) grid plot like the screenshot (no re-fit)
    grid_png = os.path.join(OUT_DIR, f"cat{cat_value}_topic_word_grid.png")
    plot_topic_grid(topics, cat_value, grid_png, top_n=TOP_WORDS_FOR_GRID)
    print(" ", grid_png)


# -----------------------------
# Entry point
# -----------------------------
def main():
    df = load_and_prepare(INPUT_CSV)
    print("Counts by category (after cleaning):")
    print(df[CAT_COL].value_counts().sort_index())

    for cat in [1, 2, 3]:
        df_cat = df[df[CAT_COL] == cat].reset_index(drop=True)
        if len(df_cat) == 0:
            print(f"\n[SKIP] Category {cat} has 0 docs.")
            continue
        if len(df_cat) < 5:
            print(f"\n[WARN] Category {cat} has few docs ({len(df_cat)}); topic quality may be unstable.")
        run_for_category(df_cat, cat)

if __name__ == "__main__":
    main()

