# -*- coding: utf-8 -*-
"""
Customer-Rating Prediction (Trustpilot/Temu) – Streamlit App

✓ Loads artifacts from src/models/ (or models/ as fallback)
✓ Accepts best_model.pkl OR best_classification_model.pkl
✓ Wordclouds (neg/neu/pos) on the Preprocessing page
✓ 100-sample evaluation and Live prediction WITHOUT retraining
✓ Optional light baseline training (off by default)

Author: you :)
"""

import os, io, sys, pickle, json, time, random
from typing import Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix

# Optional viz libs
import matplotlib.pyplot as plt

# WordCloud is optional; app runs without it
try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


# --------------------------------------------------------------------------------------
# PATHS & HELPERS
# --------------------------------------------------------------------------------------
ROOT = os.getcwd()
PATH_MODELS_CANDIDATES = ["src/models", "models"]
PATH_MODELS = next((p for p in PATH_MODELS_CANDIDATES if os.path.isdir(p)), "src/models")
PATH_DATA_PROCESSED = "src/data/processed/temu_reviews_preprocessed.csv"

NUM_PREVIEW_ROWS = 10
DEFAULT_RANDOM_SEED = 42


def _exists(p): return os.path.exists(p)


@st.cache_resource(show_spinner=False)
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_best_model(path_models: str) -> Tuple[object, dict]:
    """
    Loads the best model from either `best_model.pkl` or `best_classification_model.pkl`.
    Accepts both:
      • a dict containing {"estimator": <model>, ...}
      • a plain pickled estimator object
    """
    candidates = ["best_model.pkl", "best_classification_model.pkl"]
    chosen = None
    for name in candidates:
        p = os.path.join(path_models, name)
        if _exists(p):
            chosen = p
            break
    if not chosen:
        raise FileNotFoundError(
            f"No best model pickle found in {path_models}. "
            f"Tried: {', '.join(candidates)}"
        )

    obj = load_pickle(chosen)
    if isinstance(obj, dict) and "estimator" in obj:
        return obj["estimator"], obj
    # else a bare estimator
    return obj, {"estimator": obj, "model_name": type(obj).__name__}


def find_artifacts(path_models: str) -> Dict[str, Optional[str]]:
    """Return artifact paths or None if missing."""
    paths = {
        "tfidf": os.path.join(path_models, "tfidf_vectorizer.pkl"),
        "scaler": os.path.join(path_models, "scaler.pkl"),
        "feature_info": os.path.join(path_models, "feature_info.pkl"),
        "processed_data": os.path.join(path_models, "processed_data.pkl"),
        "train_test_splits": os.path.join(path_models, "train_test_splits.pkl"),
    }
    return {k: (v if _exists(v) else None) for k, v in paths.items()}


def artifact_status_msg(art: Dict[str, Optional[str]], need_model=True) -> Tuple[bool, str]:
    missing = []
    if need_model:
        try:
            _ = load_best_model(PATH_MODELS)
        except Exception:
            missing.append("best_model (pickled)")
    for k, v in art.items():
        if v is None:
            missing.append(k)
    ok = (len(missing) == 0)
    msg = "All artifacts present." if ok else f"Missing artifacts: {missing}. Looked in: {PATH_MODELS}"
    return ok, msg


import pickle
try:
    import dill  # used for processed_data.pkl
except ImportError:
    dill = None

@st.cache_resource(show_spinner=False)
def load_pickle_any(path: str):
    """Load a pickled object. Try pickle first, then dill."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except Exception as e_pickle:
            if dill is None:
                raise RuntimeError(
                    f"Failed to load '{path}' with pickle ({e_pickle}). "
                    f"'dill' is not installed."
                )
            f.seek(0)
            try:
                return dill.load(f)
            except Exception as e_dill:
                raise RuntimeError(
                    f"Failed to load '{path}'. pickle error: {e_pickle}; dill error: {e_dill}"
                )

# --------------------------------------------------------------------------------------
# TEXT PREPROCESS (lightweight – matches your notebooks’ intent)
# --------------------------------------------------------------------------------------
import re

URL_RE = re.compile(r"http\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
NONALPHA_RE = re.compile(r"[^a-zA-Z\s]+")

def clean_text_basic(text: str) -> str:
    if pd.isna(text): return ""
    x = str(text).lower()
    x = HTML_RE.sub(" ", x)
    x = URL_RE.sub(" ", x)
    x = NONALPHA_RE.sub(" ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def build_neg_neu_pos_text(df: pd.DataFrame, text_col="processed_text", rating_col="ReviewRating"):
    neg = " ".join(df.loc[df[rating_col] <= 2, text_col].dropna().astype(str))
    neu = " ".join(df.loc[df[rating_col] == 3, text_col].dropna().astype(str))
    pos = " ".join(df.loc[df[rating_col] >= 4, text_col].dropna().astype(str))
    return neg, neu, pos


def plot_wordclouds(neg: str, neu: str, pos: str):
    if not WORDCLOUD_AVAILABLE:
        st.info("wordcloud is not installed; skipping clouds.")
        return

    stop = STOPWORDS.union({"temu", "item", "order"})
    cols = st.columns(3)
    groups = [
        ("Negative (1–2★)", neg, "Reds"),
        ("Neutral (3★)", neu, "Blues"),
        ("Positive (4–5★)", pos, "Greens"),
    ]
    for col, (title, txt, cmap) in zip(cols, groups):
        with col:
            if txt.strip():
                wc = WordCloud(width=600, height=400, background_color="white",
                               stopwords=stop, colormap=cmap).generate(txt)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                ax.set_title(title)
                st.pyplot(fig, use_container_width=True)
            else:
                st.write(f"*No {title.lower()} reviews available.*")


# --------------------------------------------------------------------------------------
# UI HELPERS
# --------------------------------------------------------------------------------------
def section_header(title: str, emoji: str = "🧭"):
    st.markdown(
        f"<div style='padding:10px 12px;background:linear-gradient(90deg,#667eea,#764ba2);"
        f"border-radius:8px;color:white;font-weight:600'>{emoji} {title}</div>",
        unsafe_allow_html=True
    )


def dataset_preview(df: pd.DataFrame, caption=""):
    st.dataframe(df.head(NUM_PREVIEW_ROWS), use_container_width=True)
    if caption:
        st.caption(caption)


def combine_features(tfidf_vec, scaler, tfidf_texts, numeric_df: Optional[pd.DataFrame]):
    X_tfidf = tfidf_vec.transform(tfidf_texts)
    if numeric_df is None or numeric_df.empty:
        # Use scaler mean_ to create a neutral numeric vector for each sample
        base = np.tile(scaler.mean_, (X_tfidf.shape[0], 1))
        X_num_scaled = scaler.transform(base)
    else:
        X_num_scaled = scaler.transform(numeric_df.values)
    return hstack([X_tfidf, X_num_scaled])


# --------------------------------------------------------------------------------------
# SIDEBAR – NAV
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Customer-Rating Prediction", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "1) Introduction",
        "2) Load & Preprocess",
        "3) Feature Engineering (view)",
        "4) Train & Compare (optional)",
        "5) 100-Sample Evaluation",
        "6) Live Prediction",
        "7) Conclusion"
    ],
    index=0
)

# Cache artifacts once per session
if "art" not in st.session_state:
    st.session_state.art = find_artifacts(PATH_MODELS)


# --------------------------------------------------------------------------------------
# 1) INTRO
# --------------------------------------------------------------------------------------
if page.startswith("1)"):
    section_header("Introduction", "📘")
    st.markdown("""
**Goal.** Predict Trustpilot star ratings (1–5★) from review text.

**Pipeline overview**
1. _Load & Preprocess_ – clean text → `processed_text`, quick EDA, **neg/neu/pos wordclouds**.  
2. _Feature Engineering_ – TF-IDF (1–2-grams) + engineered numeric features (length, VADER, etc.).  
3. _Train & Compare_ – baseline models, plus ensembles; **we deploy the winner**.  
4. _100-Sample Evaluation_ – quick health-check without retraining.  
5. _Live Prediction_ – try arbitrary text and see probabilities + sentiment group.  

**Artifacts directory**: `{}`  
*(the app accepts `best_model.pkl` **or** `best_classification_model.pkl`)*  
""".format(PATH_MODELS))

    ok, msg = artifact_status_msg(st.session_state.art, need_model=True)
    st.info(msg)


# --------------------------------------------------------------------------------------
# 2) LOAD & PREPROCESS
# --------------------------------------------------------------------------------------
elif page.startswith("2)"):
    section_header("Load & Preprocess", "📦")
    st.write("Upload a **raw** CSV (with at least `ReviewText` and `ReviewRating`) "
             "or skip to use the preprocessed file at "
             f"`{PATH_DATA_PROCESSED}` if present.")

    up = st.file_uploader("Upload raw CSV", type=["csv"])
    df = None

    if up is not None:
        try:
            df = pd.read_csv(up)
            st.success(f"Loaded uploaded file with shape {df.shape}.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if df is None and _exists(PATH_DATA_PROCESSED):
        try:
            df = pd.read_csv(PATH_DATA_PROCESSED)
            st.success(f"Loaded existing processed dataset: {PATH_DATA_PROCESSED} (shape {df.shape}).")
        except Exception as e:
            st.error(f"Could not load default processed CSV: {e}")

    if df is None:
        st.warning("No data loaded yet.")
        st.stop()

    # Ensure key columns
    if "ReviewText" not in df.columns:
        st.error("Column `ReviewText` missing.")
        st.stop()
    if "ReviewRating" not in df.columns:
        st.error("Column `ReviewRating` missing.")
        st.stop()

    # Create processed_text if missing
    if "processed_text" not in df.columns:
        st.info("Creating `processed_text` (basic clean)…")
        df["processed_text"] = df["ReviewText"].astype(str).apply(clean_text_basic)

    # Quick summary + preview
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Rows", len(df))
    with c2: st.metric("Avg. text length", int(df["processed_text"].str.len().mean()))
    with c3: st.metric("Distinct ratings", df["ReviewRating"].nunique())

    dataset_preview(df, "Top rows")

    st.markdown("#### Wordclouds by sentiment group")
    neg, neu, pos = build_neg_neu_pos_text(df)
    plot_wordclouds(neg, neu, pos)

    # Option to save processed CSV
    if st.button("💾 Save as processed CSV", use_container_width=True):
        os.makedirs(os.path.dirname(PATH_DATA_PROCESSED), exist_ok=True)
        df.to_csv(PATH_DATA_PROCESSED, index=False)
        st.success(f"Saved → {PATH_DATA_PROCESSED}")


# --------------------------------------------------------------------------------------
# 3) FEATURE ENGINEERING (VIEW ONLY)
# --------------------------------------------------------------------------------------
elif page.startswith("3)"):
    section_header("Feature Engineering – Artifacts view", "🧱")

    art = st.session_state.art
    ok, msg = artifact_status_msg(art, need_model=False)
    st.info(msg)

    if art["feature_info"]:
        finfo = load_pickle(art["feature_info"])
        st.write("**Numerical feature columns used in training:**")
        st.code("\n".join(map(str, finfo.get("numerical_features", []))) or "(not found)")
        st.write(f"TF-IDF feature count: {finfo.get('feature_count', 'n/a')}")
    else:
        st.warning("feature_info.pkl not found. (The app will still run if model & vectorizer exist.)")

    if art["tfidf"]:
        st.success("TF-IDF vectorizer found.")
    if art["scaler"]:
        st.success("Scaler found.")
    if art["processed_data"]:
        st.success("processed_data.pkl found (contains preprocessed DataFrame).")


# --------------------------------------------------------------------------------------
# 4) TRAIN & COMPARE (OPTIONAL, LIGHT)
# --------------------------------------------------------------------------------------
elif page.startswith("4)"):
    section_header("Train & Compare (optional)", "🏋️")
    st.write("**Skip this in presentations.** Your saved model will be used. "
             "Below is a *tiny* baseline training if you explicitly run it.")

    art = st.session_state.art
    data_source = None
    df_proc = None

    if art["processed_data"]:
        try:
            # processed_data.pkl contains a dict with 'df'
            obj = load_pickle_any(art["processed_data"])
            df_proc = obj["df"] if isinstance(obj, dict) and "df" in obj else None
            data_source = "processed_data.pkl"
        except Exception:
            df_proc = None

    if df_proc is None and _exists(PATH_DATA_PROCESSED):
        try:
            df_proc = pd.read_csv(PATH_DATA_PROCESSED)
            data_source = PATH_DATA_PROCESSED
        except Exception:
            pass

    if df_proc is None:
        st.warning("No processed dataset available for training demo.")
        st.stop()

    st.caption(f"Using: {data_source} | shape={df_proc.shape}")
    dataset_preview(df_proc)

    run_small = st.checkbox("Run a very small Logistic Regression baseline (1–2 minutes)", value=False)
    if not run_small:
        st.info("Baseline training **not** executed.")
        st.stop()

    # Minimal example: TF-IDF only, to keep it fast
    if not art["tfidf"]:
        st.error("Need tfidf_vectorizer.pkl to run this minimal demo.")
        st.stop()

    tfidf = load_pickle(art["tfidf"])
    X = tfidf.transform(df_proc["processed_text"].astype(str).values)
    y = df_proc["ReviewRating"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=DEFAULT_RANDOM_SEED, stratify=y)

    # NOTE: fix the iloc bug – if y_tr is a Series, index via iloc
    if isinstance(y_tr, pd.Series):
        y_tr = y_tr.values
        y_te = y_te.values

    lr = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced", multi_class="ovr")
    with st.spinner("Training small baseline…"):
        lr.fit(X_tr, y_tr)
    preds = lr.predict(X_te)
    acc = accuracy_score(y_te, preds)
    f1w = f1_score(y_te, preds, average="weighted")
    st.success(f"Baseline LogisticRegression → Acc={acc:.3f}, Weighted F1={f1w:.3f}")

# --------------------------------------------------------------------------------------
# 5) 100-SAMPLE EVAL (uses saved artifacts)
# --------------------------------------------------------------------------------------
elif page.startswith("5)"):
    section_header("100-Sample Evaluation (no retraining)", "🧪")

    art = st.session_state.art
    ok, msg = artifact_status_msg(art, need_model=True)
    if not ok:
        st.error(msg)
        st.stop()

    # ---------- helpers (local to this page) ----------
    def plot_cm(cm, labels, title, cmap="Blues"):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap=cmap)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        plt.tight_layout()
        return fig

    STAR = {i: f"{i}★" for i in range(1, 6)}
    GROUP_OF = {1: "neg", 2: "neg", 3: "neu", 4: "pos", 5: "pos"}

    def make_colored_html_table(df, max_rows=100):
        css = """
        <style>
          .legend{margin:10px 0;padding:10px;border:1px solid #ddd;border-radius:5px;background:#f9f9f9}
          .legend .item{display:inline-block;margin:5px 10px;padding:5px 10px;border-radius:4px;font-weight:600}
          .dark{background:#2d5a27;color:#fff}.light{background:#90ee90;color:#000}
          .yellow{background:#fff8dc;color:#000}.red{background:#ffcccb;color:#000}
          table{border-collapse:collapse;margin:10px 0;width:100%}
          th,td{padding:8px 10px;text-align:center;border:1px solid #eee}
          th{background:#f5f5f5}
        </style>
        """
        html = css + """
        <div class="legend">
          <span class="item dark">Perfect Initial Prediction</span>
          <span class="item light">Corrected by Adjustment</span>
          <span class="item yellow">1 Star Difference</span>
          <span class="item red">2+ Stars Difference</span>
        </div>
        <table>
        <tr>
          <th>Index</th>
          <th>True★</th><th>Pred★</th><th>Adj★</th>
          <th>Group True</th><th>Group Pred</th><th>Group Adj</th>
        </tr>
        """
        for i, row in df.head(max_rows).iterrows():
            cls = row["Color"]
            if   cls == "dark_green":  cls = "dark"
            elif cls == "light_green": cls = "light"
            elif cls == "yellow":      cls = "yellow"
            else:                      cls = "red"
            cells = [row["True★"], row["Pred★"], row["Adj★"],
                     row["Group True"], row["Group Pred"], row["Group Adj"]]
            html += f'<tr class="{cls}">'
            html += f"<td><b>{i+1}</b></td>"
            for c in cells:
                html += f"<td>{c}</td>"
            html += "</tr>"
        html += "</table>"
        return html

    def ensure_numeric_features(df_in, needed_cols):
        df = df_in.copy()
        missing = [c for c in needed_cols if c not in df.columns]
        if not missing:
            return df

        txt = df.get("ReviewText", df.get("processed_text", "")).fillna("").astype(str)
        df["word_count"]       = txt.str.split().apply(len)
        df["char_count"]       = txt.str.len()
        df["sentence_count"]   = txt.str.count(r"[.!?]") + 1
        df["avg_word_length"]  = (df["char_count"] / df["word_count"]).replace([np.inf, np.nan], 0)
        df["exclamation_count"]= txt.str.count("!")
        df["question_count"]   = txt.str.count(r"\?")
        df["capital_ratio"]    = txt.apply(lambda s: sum(c.isupper() for c in s) / len(s) if len(s) else 0)

        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            try: nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError: nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
            pol = txt.apply(sia.polarity_scores)
            df["sentiment_compound"] = pol.apply(lambda d: d["compound"])
            df["sentiment_pos"]      = pol.apply(lambda d: d["pos"])
            df["sentiment_neu"]      = pol.apply(lambda d: d["neu"])
            df["sentiment_neg"]      = pol.apply(lambda d: d["neg"])
        except Exception:
            for c in ["sentiment_compound","sentiment_pos","sentiment_neu","sentiment_neg"]:
                df[c] = 0.0

        for c in needed_cols:
            if c not in df.columns:
                df[c] = 0.0
        return df

    # ---------- load artifacts ----------
    model, meta = load_best_model(PATH_MODELS)
    tfidf       = load_pickle(art["tfidf"])
    scaler      = load_pickle(art["scaler"])
    finfo       = load_pickle(art["feature_info"])
    num_cols    = finfo.get("numerical_features", [])

    df_proc = None
    if art["processed_data"]:
        obj = load_pickle_any(art["processed_data"])
        if isinstance(obj, dict) and "df" in obj:
            df_proc = obj["df"]
    if df_proc is None and _exists(PATH_DATA_PROCESSED):
        df_proc = pd.read_csv(PATH_DATA_PROCESSED)

    if df_proc is None:
        st.error("Need processed dataset and model artifacts. Please run earlier steps or copy files.")
        st.stop()

    df_eval = df_proc[df_proc["processed_text"].astype(str).str.len() > 0].copy()
    if df_eval.empty:
        st.error("No rows with processed_text found.")
        st.stop()

    # ---------- controls ----------
    left, mid, right = st.columns([1,1,1])
    with left:
        seed = st.number_input("Random seed", value=DEFAULT_RANDOM_SEED, step=1)
    with mid:
        apply_penalty = st.checkbox("Apply 20% penalty to 3★", value=True)
    with right:
        show_rows = st.slider("Rows to display", 20, 100, 100, step=10)

    n = min(100, len(df_eval))
    df_s = df_eval.sample(n=n, random_state=int(seed)).reset_index(drop=True)

    # ---------- build features ----------
    X = tfidf.transform(df_s["processed_text"].astype(str).values)
    df_s = ensure_numeric_features(df_s, num_cols)
    Xn = scaler.transform(df_s[num_cols].values)
    Xc = hstack([X, Xn])
    y_true = df_s["ReviewRating"].astype(int).values

    # ---------- scoring (KEEP y_pred as a VECTOR!) ----------
    with st.spinner("Scoring…"):
        # vector prediction
        y_pred = np.asarray(model.predict(Xc)).astype(int).ravel()

        # probabilities if available
        proba = model.predict_proba(Xc) if hasattr(model, "predict_proba") else None

    # ----- OPTIONAL: apply 20% penalty to class 3, vectorised -----
    if apply_penalty and proba is not None:
        classes = [int(c) for c in getattr(model, "classes_", [1, 2, 3, 4, 5])]  # keep model's order!
        st.caption(f"Classes: {classes}")
        if 3 in classes:
            i3 = classes.index(3)
            proba[:, i3] *= 0.80
            proba = proba / proba.sum(axis=1, keepdims=True)
        # choose labels from (possibly penalised) probabilities
        y_pred = np.array([classes[i] for i in proba.argmax(axis=1)], dtype=int)

    # ---------- metrics ----------
    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")
    f1m = f1_score(y_true, y_pred, average="macro")
    st.success(f"Accuracy: **{acc:.1%}**   |   Weighted F1: **{f1w:.3f}**   |   Macro F1: **{f1m:.3f}**")

    # ---------- 5-class confusion matrix ----------
    labels_5 = [1, 2, 3, 4, 5]
    cm5 = confusion_matrix(y_true, y_pred, labels=labels_5)
    fig_cm5 = plot_cm(cm5, [STAR[i] for i in labels_5], "Confusion Matrix (5-class)", cmap="Blues")
    st.pyplot(fig_cm5, use_container_width=True)

    # ---------- smart grouping & adjustment (Δ>1 only) ----------
    has_probs = proba is not None
    group_true = np.array([GROUP_OF[t] for t in y_true])
    group_pred = np.array([GROUP_OF[p] for p in y_pred])
    adj_star = y_pred.copy()

    if has_probs:
        cls_order = [int(c) for c in getattr(model, "classes_", labels_5)]
        class_to_idx = {c: i for i, c in enumerate(cls_order)}
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if abs(t - p) <= 1:
                continue
            row = proba[i]
            p_neg = row[class_to_idx.get(1,0)] + row[class_to_idx.get(2,0)]
            p_neu = row[class_to_idx.get(3,0)]
            p_pos = row[class_to_idx.get(4,0)] + row[class_to_idx.get(5,0)]
            if p_neu >= max(p_neg, p_pos):
                adj_star[i] = 3
            elif p_neg >= p_pos:
                adj_star[i] = 1 if row[class_to_idx.get(1,0)] >= row[class_to_idx.get(2,0)] else 2
            else:
                adj_star[i] = 4 if row[class_to_idx.get(4,0)] >= row[class_to_idx.get(5,0)] else 5

    group_adj = np.array([GROUP_OF[a] for a in adj_star])

    # summary like notebook
    perfect_initial = int(np.sum(y_true == y_pred))
    corrected_by_adj = int(np.sum((y_true != y_pred) & (y_true == adj_star)))
    close_1 = int(np.sum((y_true != adj_star) & (np.abs(y_true - adj_star) == 1)))
    poor_2p = int(np.sum(np.abs(y_true - adj_star) >= 2))
    total = len(y_true)

    st.markdown(
        f"""
**Prediction Quality Summary**
- 🟢 Perfect Initial Predictions: **{perfect_initial} ({perfect_initial/total:.1%})**
- 🟡 Corrected by Adjustment: **{corrected_by_adj} ({corrected_by_adj/total:.1%})**
- 🟨 Close Predictions (±1): **{close_1} ({close_1/total:.1%})**
- 🔴 Poor Predictions (≥2): **{poor_2p} ({poor_2p/total:.1%})**
        """
    )

    # ---------- grouped confusion matrix ----------
    order = ["neg", "neu", "pos"]
    cm_grp = confusion_matrix(group_true, group_adj, labels=order)
    fig_grp = plot_cm(cm_grp, order, "Grouped Confusion Matrix (neg / neu / pos)", cmap="YlGnBu")
    st.pyplot(fig_grp, use_container_width=True)

    # ---------- color-coded table ----------
    def color_class(t, p, a):
        if t == p: return "dark_green"
        if t == a: return "light_green"
        return "yellow" if abs(t - a) == 1 else "red"

    table_df = pd.DataFrame({
        "True★": [STAR[x] for x in y_true],
        "Pred★": [STAR[x] for x in y_pred],
        "Adj★":  [STAR[x] for x in adj_star],
        "Group True": group_true,
        "Group Pred": group_pred,
        "Group Adj":  group_adj,
        "Color": [color_class(t, p, a) for t, p, a in zip(y_true, y_pred, adj_star)]
    })

    st.markdown("#### Color-coded results (first 100 rows)")
    st.markdown(make_colored_html_table(table_df, max_rows=show_rows), unsafe_allow_html=True)

    # ---------- detailed sample predictions ----------
    st.markdown("#### Sample predictions with texts")
    show_n = st.slider("How many rows to show below", 10, 100, 100, step=10)
    out = df_s[["ReviewText", "processed_text", "ReviewRating"]].copy()
    out.rename(columns={"ReviewRating":"True"}, inplace=True)
    out["Predicted"] = y_pred
    out["Adjusted"]  = adj_star
    st.dataframe(out.head(show_n), use_container_width=True)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download table as CSV", csv, "sample_predictions.csv", "text/csv")
# --------------------------------------------------------------------------------------
# 6) LIVE PREDICTION (Notebook-style UI, polished model card)
# --------------------------------------------------------------------------------------
elif page.startswith("6)"):
    from datetime import datetime
    import inspect

    section_header("Live Prediction", "🎭")

    # --- check artifacts ---
    art = st.session_state.art
    ok, msg = artifact_status_msg(art, need_model=True)
    if not ok:
        st.error(msg); st.stop()

    # --- load artifacts ---
    model, meta = load_best_model(PATH_MODELS)
    tfidf   = load_pickle(art["tfidf"])
    scaler  = load_pickle(art["scaler"])
    finfo   = load_pickle(art["feature_info"])
    num_cols = finfo.get("numerical_features", [])

    # --- helpers ---
    STAR = {i: f"{i}⭐" for i in range(1, 6)}
    GROUP_OF = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    EMOJI = {"negative": "😞", "neutral": "😐", "positive": "😊"}

    def neutral_num_vector():
        # neutral numeric input with correct dimensionality
        return scaler.transform(scaler.mean_.reshape(1, -1))

    def refine_from_probs(probs, classes):
        # probs: shape (n_classes,), classes: e.g. [1,2,3,4,5]
        c2i = {c: i for i, c in enumerate(classes)}
        p1 = probs[c2i.get(1, 0)]; p2 = probs[c2i.get(2, 0)]
        p3 = probs[c2i.get(3, 0)]; p4 = probs[c2i.get(4, 0)]; p5 = probs[c2i.get(5, 0)]
        p_neg, p_neu, p_pos = p1 + p2, p3, p4 + p5
        if p_neu >= max(p_neg, p_pos):    return 3, "neutral"
        if p_neg >= p_pos:                return (1 if p1 >= p2 else 2), "negative"
        return (4 if p4 >= p5 else 5), "positive"

    def model_card(model, meta, tfidf, num_cols):
        # friendly summary
        name = meta.get("model_name", type(model).__name__)
        acc  = meta.get("test_accuracy", None)
        f1w  = meta.get("weighted_f1", None)
        f1m  = meta.get("macro_f1", None)

        # vocab size
        try:
            n_vocab = len(tfidf.get_feature_names_out())
        except Exception:
            n_vocab = len(getattr(tfidf, "vocabulary_", {}))

        n_num = len(num_cols)

        # stacking details (if applicable)
        stack_lines = []
        base_learners = None
        final_est = None
        if hasattr(model, "estimators_") or hasattr(model, "named_estimators_"):
            try:
                # sklearn StackingClassifier
                if hasattr(model, "named_estimators_"):
                    base_learners = [(k, type(v).__name__) for k, v in model.named_estimators_.items()]
                elif hasattr(model, "estimators_"):
                    base_learners = [(f"est_{i}", type(est[1]).__name__) for i, est in enumerate(model.estimators_)]
                final_est = type(getattr(model, "final_estimator_", model)).__name__
            except Exception:
                pass

        # classes as stars (clean ints)
        try:
            classes = [int(c) for c in getattr(model, "classes_", [1,2,3,4,5])]
        except Exception:
            classes = [1,2,3,4,5]

        # render
        st.markdown("**Model**")
        st.markdown(f"**{name}**")
        st.caption(
            " • " +
            " • ".join(
                [f"Test Acc: {acc:.1%}" if acc is not None else "",
                 f"Weighted F1: {f1w:.3f}" if f1w is not None else "",
                 f"Macro F1: {f1m:.3f}" if f1m is not None else ""]
            ).strip(" • ")
        )
        st.caption(f"Features: TF-IDF vocab **{n_vocab}** + numeric **{n_num}**")
        st.caption(f"Classes: {', '.join([f'{c}★' for c in classes])}")

        with st.expander("Advanced details", expanded=False):
            if base_learners:
                st.write("Base learners:")
                st.write(pd.DataFrame(base_learners, columns=["Name","Estimator"]))
            if final_est:
                st.write(f"Final estimator: **{final_est}**")
            st.write("Raw meta:", meta)

        return classes

    # --- header like your notebook ---
    st.markdown(
        """
        <div style="background:linear-gradient(90deg,#667eea 0%,#764ba2 100%);
                    padding:14px;border-radius:10px;margin:4px 0 14px 0;">
          <h3 style="color:white;margin:0;text-align:center;">🌟 Live Sentiment Analysis Interface</h3>
          <p style="color:white;margin:6px 0 0 0;text-align:center;">
            Enter your review text below and get instant star rating prediction!
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    colL, colR = st.columns([2, 1])
    with colL:
        text = st.text_area(
            "Input text",
            height=150,
            placeholder="e.g., 'This product is amazing, I love it!'"
        )
    with colR:
        apply_penalty = st.checkbox("Apply 20% penalty to 3★ (reduce mid bias)", value=True)
        # show a compact, friendly model card (no np.int64 noise)
        classes = model_card(model, meta, tfidf, num_cols)

    # init history
    if "live_history" not in st.session_state:
        st.session_state.live_history = []

    if st.button("🎯 Predict Rating", type="primary"):
        if not text.strip():
            st.warning("Please enter some text."); st.stop()

        # --- features ---
        processed = clean_text_basic(text)
        X_tfidf   = tfidf.transform([processed])
        X_num     = neutral_num_vector()
        Xc        = hstack([X_tfidf, X_num])

        # --- predictions ---
        y_init  = int(model.predict(Xc)[0])

        proba_used = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xc)  # (1, n_classes)
            if apply_penalty and 3 in classes:
                i3 = classes.index(3)
                proba[:, i3] *= 0.80
                proba /= proba.sum(axis=1, keepdims=True)
            proba_used = proba[0]

        y_pred = y_init if proba_used is None else classes[int(np.argmax(proba_used))]

        # refined (group-aware) prediction
        if proba_used is not None:
            y_refined, group = refine_from_probs(proba_used, classes)
        else:
            y_refined = y_pred
            group = GROUP_OF.get(y_pred, "neutral")

        # --- 3 result cards (like your notebook) ---
        grp_color = {"negative":"#ff6b6b","neutral":"#feca57","positive":"#48dbfb"}[group]
        st.markdown(
            f"""
            <div style="background:white;padding:16px;border-radius:10px;
                        box-shadow:0 4px 8px rgba(0,0,0,.08);margin:8px 0 14px 0;">
              <h4 style="margin:0 0 12px 0;color:#2c3e50;">🎯 Prediction Results</h4>
              <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <div style="flex:1;min-width:180px;text-align:center;background:{grp_color};
                            color:white;border-radius:10px;padding:12px;">
                  <div style="font-weight:700;">Sentiment Group</div>
                  <div style="font-size:28px;margin:6px 0;">{EMOJI[group]}</div>
                  <div style="font-size:18px;font-weight:700;">{group.upper()}</div>
                </div>
                <div style="flex:1;min-width:180px;text-align:center;background:#3498db;
                            color:white;border-radius:10px;padding:12px;">
                  <div style="font-weight:700;">Initial Prediction</div>
                  <div style="font-size:28px;margin:6px 0;">⭐</div>
                  <div style="font-size:18px;font-weight:700;">{STAR[y_init]}</div>
                </div>
                <div style="flex:1;min-width:180px;text-align:center;background:#27ae60;
                            color:white;border-radius:10px;padding:12px;">
                  <div style="font-weight:700;">Refined Prediction</div>
                  <div style="font-size:28px;margin:6px 0;">🎯</div>
                  <div style="font-size:18px;font-weight:700;">{STAR[y_refined]}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(f"Processed: _{processed[:160]}{'…' if len(processed)>160 else ''}_")

        # --- charts (bar + pie) ---
        if proba_used is not None:
            stars = classes
            probs = proba_used

            c1, c2 = st.columns(2)

            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                barlist = ax.bar(stars, probs, color=["#e74c3c","#f39c12","#f1c40f","#2ecc71","#27ae60"][:len(stars)])
                ax.set_ylim(0, 1); ax.set_xlabel("Star Rating"); ax.set_ylabel("Probability")
                ax.set_title("★ Star Rating Probabilities")
                if y_refined in stars:
                    k = stars.index(y_refined)
                    barlist[k].set_edgecolor("black"); barlist[k].set_linewidth(2.5)
                for b, p in zip(barlist, probs):
                    ax.text(b.get_x()+b.get_width()/2., b.get_height()+0.02, f"{p:.3f}",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")
                st.pyplot(fig, use_container_width=True)

            with c2:
                idx = {c:i for i,c in enumerate(stars)}
                p1 = probs[idx.get(1,0)]; p2 = probs[idx.get(2,0)]
                p3 = probs[idx.get(3,0)]; p4 = probs[idx.get(4,0)]; p5 = probs[idx.get(5,0)]
                group_vals = [p1+p2, p3, p4+p5]
                group_lbls = ["negative","neutral","positive"]
                group_cols = ["#ff6b6b","#feca57","#48dbfb"]

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                wedges, txts, autotxts = ax2.pie(
                    group_vals, labels=group_lbls, colors=group_cols,
                    autopct="%1.1f%%", startangle=90, textprops={"fontweight":"bold"}
                )
                ax2.set_title("😊 Sentiment Group Distribution")
                for i, g in enumerate(group_lbls):
                    if g == group:
                        wedges[i].set_edgecolor("black"); wedges[i].set_linewidth(2.5)
                st.pyplot(fig2, use_container_width=True)

        # --- add to history ---
        st.session_state.live_history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Text (preview)": (text[:120] + "…") if len(text) > 120 else text,
            "Initial": y_init,
            "Refined": y_refined,
            "Sentiment": group
        })

    # --- history block ---
    with st.expander("📚 Prediction History", expanded=False):
        if st.session_state.live_history:
            hist_df = pd.DataFrame(st.session_state.live_history)
            hist_df["Sentiment"] = hist_df["Sentiment"].map(lambda g: f"{EMOJI.get(g,'🤔')} {g}")
            hist_df["Initial"] = hist_df["Initial"].map(STAR)
            hist_df["Refined"] = hist_df["Refined"].map(STAR)
            st.dataframe(hist_df, use_container_width=True, height=min(420, 60 + 28*len(hist_df)))
            c1, c2 = st.columns(2)
            if c1.button("🧹 Clear history"):
                st.session_state.live_history = []
            csv = pd.DataFrame(st.session_state.live_history).to_csv(index=False).encode("utf-8")
            c2.download_button("⬇️ Download history CSV", data=csv, file_name="live_history.csv", mime="text/csv")
        else:
            st.info("No predictions yet.")
            
# --------------------------------------------------------------------------------------
# 7) CONCLUSION
# --------------------------------------------------------------------------------------
elif page.startswith("7)"):
    section_header("Conclusion", "✅")
    st.markdown("""
**What worked well**
- **Stacking** delivered the best trade-off on imbalanced data (strong Weighted-F1).
- Simple **TF-IDF (1–2-grams)** + a few pragmatic numeric features (length, VADER).
- Clear split between **negative** (1–2★) and **positive** (4–5★); **neutral (3★)** remains hardest.

**Practical takeaways**
- Keep the deployed **stacking** model for production; use **Logistic Regression / LinearSVC** as robust fallbacks.
- Apply a small **3★ penalty** at inference to curb mid-class over-prediction if your business needs clearer polarities.
- Use grouped evaluation (neg/neu/pos) alongside 5-class metrics for business-friendly reporting.

**Artifacts**
- Place your pickles in **`src/models/`** (or `models/`) with these names:
  - `tfidf_vectorizer.pkl`, `scaler.pkl`, `feature_info.pkl`, `processed_data.pkl`, and
  - either `best_model.pkl` **or** `best_classification_model.pkl`.

You can run everything above **without retraining**.
""" )
