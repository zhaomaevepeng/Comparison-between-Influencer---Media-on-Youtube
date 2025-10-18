# complexity_features.py
# ------------------------------------------------------------
# Language complexity features for 3 transcript categories (1/2/3)
# Input file schema:
#   - "Influencer/Mainstream" (category 1/2/3, with a noisy row handled)
#   - "full_transcript"
#   - "video_id"
# Outputs:
#   - complexity_outputs/per_transcript_complexity.csv
#   - complexity_outputs/category_complexity_means.csv
# ------------------------------------------------------------

import os
import re
import math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

# ---------- Config ----------
INPUT_CSV = "merged_transcripts.csv"
TEXT_COL  = "full_transcript"
CAT_COL   = "Influencer/Mainstream"
ID_COL    = "video_id"
OUT_DIR   = "complexity_outputs"
RANDOM_STATE = 42

# Optional rare-word detection using frequency lists (graceful fallback)
try:
    from wordfreq import zipf_frequency
    HAS_WORDFREQ = True
except Exception:
    HAS_WORDFREQ = False

# ---------- NLP backend (spaCy) ----------
import spacy
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])  # keep tagger+parser
except OSError as e:
    raise SystemExit(
        "spaCy model 'en_core_web_sm' is not installed.\n"
        "Install it with:\n"
        "  python -m spacy download en_core_web_sm\n"
        "Then rerun this script."
    )

# ---------- Basic cleaning ----------
def clean_text_keep_sentences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    def to_cat(val):
        m = re.search(r"\b([123])\b", str(val))
        return int(m.group(1)) if m else np.nan

    df[CAT_COL] = df[CAT_COL].apply(to_cat)
    df = df[df[CAT_COL].isin([1, 2, 3])]
    df[TEXT_COL] = df[TEXT_COL].astype(str).map(clean_text_keep_sentences)
    df = df[df[TEXT_COL].str.len() > 30].dropna(subset=[TEXT_COL, CAT_COL])
    df[ID_COL] = df[ID_COL].astype(str)
    return df.reset_index(drop=True)

# ---------- Utility: syllables (heuristic, good enough for readability) ----------
_vowel_re = re.compile(r"[aeiouy]+", re.I)
def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    # subtract a silent 'e'
    syls = len(_vowel_re.findall(w))
    if w.endswith("e") and syls > 1:
        syls -= 1
    return max(1, syls)

def polysyllabic_count(words) -> int:
    c = 0
    for w in words:
        if count_syllables(w) >= 3:
            c += 1
    return c

# ---------- Lexical richness: MTLD ----------
def mtld(tokens, ttr_threshold=0.72):
    """
    Measure of Textual Lexical Diversity (MTLD).
    Implementation of forward and backward pass; returns mean.
    """
    def _mtld_seq(seq):
        types = set()
        token_count = 0
        factor_count = 0.0
        cur_types = set()
        cur_count = 0
        for tok in seq:
            token_count += 1
            cur_count += 1
            cur_types.add(tok)
            if (len(cur_types) / max(1, cur_count)) <= ttr_threshold:
                factor_count += 1.0
                cur_types = set()
                cur_count = 0
        # partial factor
        if cur_count > 0:
            factor_count += (1 - ((len(cur_types) / cur_count) / ttr_threshold))
        return token_count / max(1e-9, factor_count)

    if len(tokens) < 50:
        return float("nan")
    fwd = _mtld_seq(tokens)
    bwd = _mtld_seq(list(reversed(tokens)))
    return (fwd + bwd) / 2.0

# ---------- Feature extraction ----------
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "DET", "INTJ", "PART", "PRON", "SCONJ"}

def dep_tree_depth(sent) -> int:
    """Approximate dependency tree depth: max distance from a token to the sentence root."""
    depths = []
    for t in sent:
        d = 0
        cur = t
        while cur.head is not cur:
            d += 1
            cur = cur.head
            if d > 100:  # safety
                break
        depths.append(d)
    return int(max(depths)) if depths else 0

def subordination_index(sent) -> int:
    """Counts subordinating cues: SCONJ POS, 'mark' dependencies, and 'relcl' (relative clauses)."""
    count = 0
    for t in sent:
        if t.pos_ == "SCONJ":
            count += 1
        if t.dep_ in {"mark", "relcl"}:
            count += 1
    return count

def tokenize_words(doc):
    return [t.text.lower() for t in doc if t.is_alpha]

def lexical_features(tokens):
    if not tokens:
        return dict(
            ttr=float("nan"),
            cttr=float("nan"),
            mtld_score=float("nan"),
            pct_long_words=float("nan"),
            pct_rare_words=float("nan") if HAS_WORDFREQ else None
        )
    types = set(tokens)
    n = len(tokens)
    ttr = len(types) / n
    cttr = len(types) / math.sqrt(2 * n)  # corrected TTR

    # MTLD requires lemma-like tokens; lowercase fine for transcripts
    mtld_score = mtld(tokens)

    long_words = sum(1 for w in tokens if len(w) >= 6)
    pct_long = long_words / n

    pct_rare = None
    if HAS_WORDFREQ:
        rare = 0
        for w in tokens:
            # wordfreq ZIPF < 3.5 ≈ rarer lexicon
            try:
                if zipf_frequency(w, "en") < 3.5:
                    rare += 1
            except Exception:
                pass
        pct_rare = rare / n

    return dict(
        ttr=ttr,
        cttr=cttr,
        mtld_score=mtld_score,
        pct_long_words=pct_long,
        pct_rare_words=pct_rare
    )

def readability(sentences, tokens):
    # Flesch & FK use words, sentences, syllables; SMOG uses polysyllables & sentences
    n_sent = max(1, len(sentences))
    words = [w for w in tokens]  # already alpha lowercase strings
    n_words = max(1, len(words))
    sylls = sum(count_syllables(w) for w in words)
    polys = polysyllabic_count(words)

    flesch = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (sylls / n_words)
    fk_grade = 0.39 * (n_words / n_sent) + 11.8 * (sylls / n_words) - 15.59
    smog = (1.0430 * math.sqrt(30.0 * (polys / n_sent))) + 3.1291 if n_sent > 0 else float("nan")

    return dict(flesch_reading_ease=flesch, fk_grade=fk_grade, smog_index=smog)

def pos_mix(doc):
    total = sum(1 for t in doc if t.is_alpha)
    counts = Counter(t.pos_ for t in doc if t.is_alpha)
    def rate(tag):
        return (counts.get(tag, 0) / total) if total else float("nan")
    content = sum(counts.get(p, 0) for p in CONTENT_POS)
    function = sum(counts.get(p, 0) for p in FUNCTION_POS)
    return dict(
        rate_noun=rate("NOUN"),
        rate_verb=rate("VERB"),
        rate_adj=rate("ADJ"),
        rate_adv=rate("ADV"),
        content_function_ratio=(content / function) if function else float("nan")
    )

def syntactic_complexity(doc):
    sent_lengths = []
    depths = []
    subords = []
    verbs_per_sent = []
    for sent in doc.sents:
        toks = [t for t in sent if t.is_alpha]
        sent_lengths.append(len(toks))
        depths.append(dep_tree_depth(sent))
        subords.append(subordination_index(sent))
        verbs_per_sent.append(sum(1 for t in sent if t.pos_ == "VERB" or t.tag_.startswith("VB")))
    return dict(
        mean_tokens_per_sentence=np.mean(sent_lengths) if sent_lengths else float("nan"),
        std_tokens_per_sentence=np.std(sent_lengths) if sent_lengths else float("nan"),
        mean_dep_tree_depth=np.mean(depths) if depths else float("nan"),
        mean_subordination_index=np.mean(subords) if subords else float("nan"),
        mean_verbs_per_sentence=np.mean(verbs_per_sent) if verbs_per_sent else float("nan")
    )

# ---------- Main per-transcript computation ----------
def analyze_transcript(text: str):
    doc = nlp(text)
    tokens = tokenize_words(doc)
    sents = list(doc.sents)

    feats = {}
    feats.update(lexical_features(tokens))
    feats.update(readability(sents, tokens))
    feats.update(pos_mix(doc))
    feats.update(syntactic_complexity(doc))

    # average chars per token
    feats["mean_chars_per_token"] = (np.mean([len(w) for w in tokens]) if tokens else float("nan"))
    return feats

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_and_prepare(INPUT_CSV)
    print("Counts by category:", df[CAT_COL].value_counts().to_dict())

    rows = []
    for _, r in df.iterrows():
        feats = analyze_transcript(r[TEXT_COL])
        feats[ID_COL] = r[ID_COL]
        feats[CAT_COL] = r[CAT_COL]
        rows.append(feats)

    per_doc = pd.DataFrame(rows)
    per_doc = per_doc[[ID_COL, CAT_COL] + [c for c in per_doc.columns if c not in {ID_COL, CAT_COL}]]

    per_doc_csv = os.path.join(OUT_DIR, "per_transcript_complexity.csv")
    per_doc.to_csv(per_doc_csv, index=False)

    # Per-category means
    per_cat = per_doc.groupby(CAT_COL).mean(numeric_only=True).reset_index()
    per_cat_csv = os.path.join(OUT_DIR, "category_complexity_means.csv")
    per_cat.to_csv(per_cat_csv, index=False)

    print("Saved:")
    print(" ", per_doc_csv)
    print(" ", per_cat_csv)

if __name__ == "__main__":
    run()
