# complexity_plots.py
# ------------------------------------------------------------
# Make plots from outputs of complexity_features.py
# - Reads:
#     complexity_outputs/per_transcript_complexity.csv
#     complexity_outputs/category_complexity_means.csv
# - Produces:
#     complexity_outputs/plots/*.png + index.html
# - Uses matplotlib only, one plot per figure, no explicit colors
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "complexity_outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CAT_COL = "Influencer/Mainstream"

# Which per-category metrics to plot as bars
BAR_METRICS = [
    # Readability
    "flesch_reading_ease", "fk_grade", "smog_index",
    # Lexical richness
    "mtld_score", "ttr", "cttr", "pct_long_words", "pct_rare_words",
    # Syntax / structure
    "mean_tokens_per_sentence", "std_tokens_per_sentence",
    "mean_dep_tree_depth", "mean_subordination_index", "mean_verbs_per_sentence",
    # POS mix
    "content_function_ratio", "rate_noun", "rate_verb", "rate_adj", "rate_adv",
    # Extras
    "mean_chars_per_token",
]

# For boxplots (distributions per transcript by category)
BOX_METRICS = [
    "fk_grade", "mtld_score", "mean_tokens_per_sentence", "mean_dep_tree_depth"
]

def load_data():
    cat_csv = os.path.join(OUT_DIR, "category_complexity_means.csv")
    doc_csv = os.path.join(OUT_DIR, "per_transcript_complexity.csv")

    if not os.path.exists(cat_csv) or not os.path.exists(doc_csv):
        raise SystemExit(
            "Missing input files.\n"
            f"Expected:\n - {cat_csv}\n - {doc_csv}\n"
            "Run complexity_features.py first."
        )

    df_cat = pd.read_csv(cat_csv)
    df_doc = pd.read_csv(doc_csv)
    return df_cat, df_doc

def bar_plot(df_cat, metric, filename, title=None, ylabel=None):
    if metric not in df_cat.columns:
        return None
    cats = df_cat[CAT_COL].tolist()
    vals = df_cat[metric].values

    plt.figure(figsize=(7,5))
    x = np.arange(len(cats))
    plt.bar(x, vals)
    plt.xticks(x, [str(c) for c in cats])
    plt.title(title or metric)
    plt.xlabel("Category")
    plt.ylabel(ylabel or metric)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def box_plot(df_doc, metric, filename, title=None, ylabel=None):
    if metric not in df_doc.columns:
        return None
    # Build data per category
    cats = sorted(df_doc[CAT_COL].dropna().unique().tolist())
    data = [df_doc.loc[df_doc[CAT_COL] == c, metric].dropna().values for c in cats]
    if all(len(d) == 0 for d in data):
        return None

    plt.figure(figsize=(8,5))
    plt.boxplot(data, labels=[str(c) for c in cats], showmeans=True)
    plt.title(title or metric)
    plt.xlabel("Category")
    plt.ylabel(ylabel or metric)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path

def main():
    df_cat, df_doc = load_data()
    made = []

    # Friendly titles/labels if you want nicer names
    titles = {
        "flesch_reading_ease": "Flesch Reading Ease (↑ easier)",
        "fk_grade": "Flesch–Kincaid Grade (↑ harder)",
        "smog_index": "SMOG (↑ harder)",
        "mtld_score": "MTLD (Lexical Diversity)",
        "ttr": "Type–Token Ratio",
        "cttr": "Corrected TTR",
        "pct_long_words": "% Long Words (≥6 chars)",
        "pct_rare_words": "% Rare Words (ZIPF < 3.5)",
        "mean_tokens_per_sentence": "Mean Tokens per Sentence",
        "std_tokens_per_sentence": "Std. Tokens per Sentence",
        "mean_dep_tree_depth": "Mean Dependency Tree Depth",
        "mean_subordination_index": "Mean Subordination Index",
        "mean_verbs_per_sentence": "Mean Verbs per Sentence",
        "content_function_ratio": "Content/Function POS Ratio",
        "rate_noun": "Noun Rate",
        "rate_verb": "Verb Rate",
        "rate_adj": "Adjective Rate",
        "rate_adv": "Adverb Rate",
        "mean_chars_per_token": "Mean Characters per Token",
    }

    ylabels = {
        "pct_long_words": "Proportion",
        "pct_rare_words": "Proportion",
        "rate_noun": "Rate",
        "rate_verb": "Rate",
        "rate_adj": "Rate",
        "rate_adv": "Rate",
        "content_function_ratio": "Ratio",
    }

    # 1) Per-category bar charts
    for m in BAR_METRICS:
        if m in df_cat.columns:
            fn = f"bar_{m}.png"
            made_path = bar_plot(
                df_cat, m, fn,
                title=titles.get(m, m),
                ylabel=ylabels.get(m, m)
            )
            if made_path:
                made.append(made_path)

    # 2) Per-transcript distribution boxplots (by category)
    for m in BOX_METRICS:
        if m in df_doc.columns:
            fn = f"box_{m}.png"
            made_path = box_plot(
                df_doc, m, fn,
                title=f"{titles.get(m, m)} — per Transcript",
                ylabel=ylabels.get(m, m)
            )
            if made_path:
                made.append(made_path)

    # 3) Simple index.html to browse images
    if made:
        imgs = [os.path.basename(p) for p in made]
        imgs.sort()
        index_html = os.path.join(PLOTS_DIR, "index.html")
        lines = [
            "<html><head><meta charset='utf-8'><title>Complexity Plots</title></head><body>",
            "<h1>Language Complexity — Plots</h1>",
            "<p>Bar charts (category means) and boxplots (per transcript distributions).</p>",
        ]
        for img in imgs:
            lines.append(f"<div><p>{img}</p><img src='{img}' style='max-width:900px;display:block;'></div>")
        lines.append("</body></html>")
        with open(index_html, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print("Wrote", index_html)
    else:
        print("No plots generated (no matching columns). Check inputs.")

if __name__ == "__main__":
    main()
