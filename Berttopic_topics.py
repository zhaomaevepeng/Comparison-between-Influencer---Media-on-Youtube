# bertopic_topics_enhanced.py
# ------------------------------------------------------------
# Enhanced BERTopic script:
# - Fixes & robustness
# - Saves topic word scores (top 12 words)
# - Colored static topic-word grid (distinct color per topic)
# - Hierarchical clustering (dendrogram) of topics
# - Documents & topics interactive HTML
# - Intertopic distance map (interactive HTML)
# - Static heatmap of topic similarity matrix + CSV
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
from umap import UMAP
from hdbscan import HDBSCAN
from nltk.stem import SnowballStemmer

# Extra utilities
from sklearn.metrics.pairwise import cosine_similarity
import scipy.cluster.hierarchy as sch

# -----------------------------
# Paths & column names
# -----------------------------
INPUT_CSV = "merged_transcripts.csv"          # <- your file
TEXT_COL  = "full_transcript"
CAT_COL   = "Influencer/Mainstream"           # contains {1,2,3}
ID_COL    = "video_id"
OUT_DIR   = "bertopic_outputs"

# -----------------------------
# Modeling configs
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small/fast (384-d)
MIN_TOPIC_SIZE = 5                # tweak per category size
N_GRAM_RANGE = (1, 2)             # unigrams+bigrams
TOP_N_WORDS = 12                  # words shown per topic (you requested 12)
TOP_WORDS_FOR_GRID = 12           # words shown per tile in the static grid image
LOW_FREQ_MIN_DF = 1               # drop ultra-rare terms
HIGH_FREQ_MAX_DF = 1.0            # drop near-ubiquitous terms
RANDOM_STATE = 42

# Extra domain stopwords to kill boilerplate
EXTRA_STOPWORDS = {
    "subscribe","channel","video","like","comment","today","uh","um","yeah",
    "thing","things","gonna","kind","sort","lot","really","going","say","know",
    "want","see","think","just","dad","ii","ll","right"
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
        analyzer=stem_analyzer,          # use stemmed tokens
        max_df=HIGH_FREQ_MAX_DF,
        min_df=LOW_FREQ_MIN_DF,
        ngram_range=N_GRAM_RANGE
    )

def make_bertopic(min_cluster_size: int):
    # instantiate embedding model (SentenceTransformer instance)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE
    )
    # HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
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
        min_topic_size=min_cluster_size,
        calculate_probabilities=True,
        verbose=True
    )
    return topic_model

def topics_to_long_df(topic_model: BERTopic, topic_info: pd.DataFrame, top_n: int = TOP_N_WORDS) -> pd.DataFrame:
    """Expand each topic's top-N word list into long form rows (topic, rank, word, weight)."""
    rows = []
    for topic_id in topic_info["Topic"].tolist():
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id)  # list of (word, weight)
        # keep exactly top_n words (if available)
        words = words[:top_n]
        for rank, (w, wt) in enumerate(words, start=1):
            rows.append({"topic": int(topic_id), "rank": int(rank), "word": w, "weight": float(wt)})
    return pd.DataFrame(rows)

def plot_topic_grid(long_topics_df: pd.DataFrame, cat_value: int, out_path: str, top_n: int = TOP_WORDS_FOR_GRID):
    """Static grid image: horizontal bar chart per topic with distinct colors."""
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
    fig.suptitle(f"Category {cat_value} — Topic Word Scores (top {top_n})", fontsize=16, fontweight="bold")

    # colormap with sufficiently many distinct colors
    cmap = plt.get_cmap("tab20")

    for idx, t in enumerate(topics_sorted):
        ax = axes[idx // ncols][idx % ncols]
        sub = long_topics_df[long_topics_df["topic"] == t].sort_values("rank").head(top_n)
        words = sub["word"].tolist()
        weights = sub["weight"].astype(float).tolist()

        # largest on top (reverse for barh)
        words_plot = list(reversed(words))
        weights_plot = list(reversed(weights))
        y = np.arange(len(words_plot))

        color = cmap(idx % cmap.N)
        ax.barh(y, weights_plot, color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(words_plot, fontsize=9)
        ax.set_title(f"Topic {t}", fontsize=11, pad=6)
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

def generate_topic_similarity_matrix(topic_model: BERTopic, docs: List[str], topics_list: List[int], out_csv: str, out_png: str):
    """
    Compute topic embeddings (robustly), similarity matrix (cosine),
    save CSV and static heatmap PNG.
    """
    topics_sorted = sorted([t for t in topics_list if t != -1])
    if not topics_sorted:
        print("[SIM] No non-outlier topics to compute similarity.")
        return None

    # Try to retrieve topic embeddings from model; otherwise compute by averaging doc embeddings
    topic_embs = None
    try:
        if hasattr(topic_model, "topic_embeddings_") and topic_model.topic_embeddings_ is not None:
            # topic_embeddings_ is typically an array aligned to topic ids order in get_topic_info()
            # We'll map from get_topic_info() order to topics_sorted.
            info = topic_model.get_topic_info()
            # info has 'Topic' and we can use that ordering to extract embeddings
            if hasattr(topic_model, "topic_embeddings_"):
                emb_array = np.array(topic_model.topic_embeddings_)
                # Ensure lengths match
                if emb_array.shape[0] == info.shape[0]:
                    # map embeddings by index; but get_topic_info returns Topic column that we can index
                    topic_order = info["Topic"].tolist()
                    emb_map = {}
                    for idx, t in enumerate(topic_order):
                        emb_map[int(t)] = emb_array[idx]
                    topic_embs = np.array([emb_map[t] for t in topics_sorted if t in emb_map])
    except Exception:
        topic_embs = None

    # Fallback: average document embeddings per topic
    if topic_embs is None:
        print("[SIM] Falling back to averaging document embeddings per topic.")
        try:
            doc_embs = topic_model.embedding_model.encode(docs, show_progress_bar=False)
            emb_list = []
            for t in topics_sorted:
                inds = [i for i, tt in enumerate(topics_list) if tt == t]
                if not inds:
                    # empty topic cluster (shouldn't happen) -> zero vector
                    emb_list.append(np.zeros(doc_embs.shape[1], dtype=float))
                else:
                    emb_list.append(doc_embs[inds].mean(axis=0))
            topic_embs = np.vstack(emb_list)
        except Exception as e:
            print("[SIM] Failed to compute doc embeddings fallback:", e)
            return None

    # cosine similarity
    sim = cosine_similarity(topic_embs)
    sim_df = pd.DataFrame(sim, index=topics_sorted, columns=topics_sorted)
    sim_df.to_csv(out_csv, index=True)

    # static heatmap (matplotlib)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(topics_sorted)))
    ax.set_yticks(np.arange(len(topics_sorted)))
    ax.set_xticklabels(topics_sorted, rotation=45, ha="right")
    ax.set_yticklabels(topics_sorted)
    ax.set_title("Topic Similarity (cosine)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("cosine similarity", rotation=270, labelpad=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return sim_df

def plot_topic_dendrogram(topic_model: BERTopic, docs: List[str], topics_list: List[int], out_png: str):
    """
    Compute topic embeddings (same fallback as similarity) and plot hierarchical dendrogram.
    """
    topics_sorted = sorted([t for t in topics_list if t != -1])
    if not topics_sorted:
        print("[DENDRO] No non-outlier topics for dendrogram.")
        return

    # Try to reuse similarity function code by computing embeddings similarly
    topic_embs = None
    try:
        if hasattr(topic_model, "topic_embeddings_") and topic_model.topic_embeddings_ is not None:
            info = topic_model.get_topic_info()
            emb_array = np.array(topic_model.topic_embeddings_)
            if emb_array.shape[0] == info.shape[0]:
                topic_order = info["Topic"].tolist()
                emb_map = {int(t): emb_array[idx] for idx, t in enumerate(topic_order)}
                topic_embs = np.array([emb_map[t] for t in topics_sorted if t in emb_map])
    except Exception:
        topic_embs = None

    if topic_embs is None:
        try:
            doc_embs = topic_model.embedding_model.encode(docs, show_progress_bar=False)
            emb_list = []
            for t in topics_sorted:
                inds = [i for i, tt in enumerate(topics_list) if tt == t]
                if not inds:
                    emb_list.append(np.zeros(doc_embs.shape[1], dtype=float))
                else:
                    emb_list.append(doc_embs[inds].mean(axis=0))
            topic_embs = np.vstack(emb_list)
        except Exception as e:
            print("[DENDRO] Failed to compute embeddings:", e)
            return

    # Create linkage and dendrogram
    try:
        linkage = sch.linkage(topic_embs, method='ward')
        fig, ax = plt.subplots(figsize=(10, 6))
        sch.dendrogram(linkage, labels=[str(t) for t in topics_sorted], leaf_rotation=45, ax=ax)
        ax.set_title("Hierarchical Clustering of Topics (Ward linkage)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        print("[DENDRO] Failed to draw dendrogram:", e)

# -----------------------------
# Run per category
# -----------------------------
def run_category(df_cat: pd.DataFrame, cat_value: int):
    os.makedirs(OUT_DIR, exist_ok=True)
    docs = df_cat[TEXT_COL].tolist()
    ids  = df_cat[ID_COL].tolist()

    print(f"\n=== Category {cat_value} | docs={len(docs)} ===")
    if len(docs) < 5:
        print(f"[WARN] Very few docs; topics may be noisy.")

    # ---- choose a min_cluster_size that scales with category size
    n_docs = len(docs)
    min_cluster = max(2, min(8, max(2, n_docs // 5)))   # value between 2 and 8

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
        # ensure column names correspond to discovered topics order; fallback to generic naming
        prob_df = pd.DataFrame(probs, columns=[f"prob_topic_{i}" for i in range(probs.shape[1])])
        doc_df = pd.concat([doc_df, prob_df], axis=1)

    doc_csv = os.path.join(OUT_DIR, f"cat{cat_value}_bertopic_doc_topics.csv")
    doc_df.to_csv(doc_csv, index=False)

    # ---- Save topic info and topic words
    topic_info = topic_model.get_topic_info()   # contains Topic, Count, Name
    info_csv = os.path.join(OUT_DIR, f"cat{cat_value}_bertopic_topic_info.csv")
    topic_info.to_csv(info_csv, index=False)

    long_topics = topics_to_long_df(topic_model, topic_info, top_n=TOP_N_WORDS)
    long_csv = os.path.join(OUT_DIR, f"cat{cat_value}_bertopic_topics_words_top{TOP_N_WORDS}.csv")
    long_topics.to_csv(long_csv, index=False)

    # ---- Interactive Plotly visualizations (HTML)
    try:
        fig_barch = topic_model.visualize_barchart(top_n_topics=None, top_n_words=TOP_N_WORDS)
        fig_barch.write_html(os.path.join(OUT_DIR, f"cat{cat_value}_barchart.html"))
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

    # visualize topics (intertopic distance map)
    try:
        fig_topics = topic_model.visualize_topics()
        fig_topics.write_html(os.path.join(OUT_DIR, f"cat{cat_value}_topics_intertopic_distance.html"))
    except Exception as e:
        print("[viz] intertopic distance map:", e)

    # visualize documents (interactive) - pass topics to color docs
    try:
        fig_docs = topic_model.visualize_documents(docs, topics=topics)
        fig_docs.write_html(os.path.join(OUT_DIR, f"cat{cat_value}_documents_topics.html"))
    except Exception as e:
        # some BERTopic versions accept (docs) only; fallback to docs only
        try:
            fig_docs = topic_model.visualize_documents(docs)
            fig_docs.write_html(os.path.join(OUT_DIR, f"cat{cat_value}_documents_topics.html"))
        except Exception as e2:
            print("[viz] documents:", e, e2)

    # ---- Static grid image: colored bars per topic
    grid_png = os.path.join(OUT_DIR, f"cat{cat_value}_topic_word_grid.png")
    if not long_topics.empty:
        plot_topic_grid(long_topics, cat_value, grid_png, top_n=TOP_WORDS_FOR_GRID)
    else:
        print(f"[INFO] Category {cat_value}: no non-outlier topics to plot.")

    # ---- Topic similarity heatmap + CSV (static)
    sim_csv = os.path.join(OUT_DIR, f"cat{cat_value}_topic_similarity.csv")
    sim_png = os.path.join(OUT_DIR, f"cat{cat_value}_topic_similarity_heatmap.png")
    sim_df = generate_topic_similarity_matrix(topic_model, docs, topics, sim_csv, sim_png)

    # ---- Dendrogram
    dendro_png = os.path.join(OUT_DIR, f"cat{cat_value}_topic_dendrogram.png")
    plot_topic_dendrogram(topic_model, docs, topics, dendro_png)

    # ---- Print saved outputs summary
    print("Saved:")
    print(" ", doc_csv)
    print(" ", info_csv)
    print(" ", long_csv)
    if not long_topics.empty:
        print(" ", grid_png)
    print(" ", os.path.join(OUT_DIR, f"cat{cat_value}_barchart.html"))
    print(" ", os.path.join(OUT_DIR, f"cat{cat_value}_hierarchy.html"))
    print(" ", os.path.join(OUT_DIR, f"cat{cat_value}_heatmap.html"))
    print(" ", os.path.join(OUT_DIR, f"cat{cat_value}_topics_intertopic_distance.html"))
    print(" ", os.path.join(OUT_DIR, f"cat{cat_value}_documents_topics.html"))
    if sim_df is not None:
        print(" ", sim_csv)
        print(" ", sim_png)
    print(" ", dendro_png)

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
