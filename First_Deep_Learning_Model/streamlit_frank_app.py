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
from pathlib import Path
import os, pickle
import streamlit as st
from typing import Tuple, Dict, Optional

# Ankerpunkte relativ zur Datei
APP_DIR = Path(__file__).resolve().parent  # .../src/streamlit
SRC_DIR = APP_DIR.parent  # .../src
PROJ_DIR = SRC_DIR.parent  # repo root


def _pick_existing(*cands) -> Path:
    """Gib den ersten existierenden Pfad zurück (ansonsten den ersten Kandidaten für klare Fehlermeldungen)."""
    for p in cands:
        p = Path(p)
        if p.exists():
            return p
    return Path(cands[0])


# Haupt-Verzeichnisse/Dateien (mit Fallbacks)
PATH_MODELS = _pick_existing(
    SRC_DIR / "models",
    PROJ_DIR / "src" / "models",
    Path.cwd() / "src" / "models",
    Path.cwd() / "models",
)

PATH_DATA_PROCESSED = _pick_existing(
    SRC_DIR / "data" / "processed" / "temu_reviews_preprocessed.csv",
    PROJ_DIR / "src" / "data" / "processed" / "temu_reviews_preprocessed.csv",
)

PATH_RESULTS = _pick_existing(
    PROJ_DIR / "results",
    SRC_DIR / "results",
    Path.cwd() / "results",
)

# Optional: overrides (env/secrets) — never crash if secrets are missing
_env_model_dir = os.getenv("MODEL_DIR", None)
if _env_model_dir:
    PATH_MODELS = Path(_env_model_dir)

# Access st.secrets only inside a try/except to avoid StreamlitSecretNotFoundError
_sec_model_dir = None
try:
    _sec = st.secrets  # may raise if no secrets.toml exists
    _sec_model_dir = _sec.get("model_dir", None)
except Exception:
    _sec_model_dir = None

if _sec_model_dir:
    PATH_MODELS = Path(_sec_model_dir)

NUM_PREVIEW_ROWS = 10
DEFAULT_RANDOM_SEED = 42


def _exists(p) -> bool:
    return Path(p).exists()


@st.cache_resource(show_spinner=False)
def load_pickle(path: str | Path):
    with open(Path(path), "rb") as f:
        return pickle.load(f)


def load_best_model(models_dir: str | Path) -> Tuple[object, dict]:
    """
    Lädt das beste Modell aus best_model.pkl oder best_classification_model.pkl.
    Unterstützt:
      • dict mit {'estimator': <model>, ...}
      • direkt gepickeltes Estimator-Objekt
    """
    models_dir = Path(models_dir)
    candidates = ["best_model.pkl", "best_classification_model.pkl"]
    chosen = None
    for name in candidates:
        p = models_dir / name
        if p.exists():
            chosen = p
            break
    if not chosen:
        raise FileNotFoundError(
            f"Missing artifacts: ['best_model (pickled)']. Looked in: {models_dir}"
        )

    obj = load_pickle(chosen)
    if isinstance(obj, dict) and "estimator" in obj:
        return obj["estimator"], obj
    return obj, {"estimator": obj, "model_name": type(obj).__name__}


def find_artifacts(path_models: str | Path) -> Dict[str, Optional[Path]]:
    """Gibt Pfade zu Artefakten (oder None) zurück."""
    path_models = Path(path_models)
    files = {
        "tfidf": path_models / "tfidf_vectorizer.pkl",
        "scaler": path_models / "scaler.pkl",
        "feature_info": path_models / "feature_info.pkl",
        "processed_data": path_models / "processed_data.pkl",
        "train_test_splits": path_models / "train_test_splits.pkl",
    }
    return {k: (p if p.exists() else None) for k, p in files.items()}


def artifact_status_msg(art: Dict[str, Optional[Path]], need_model=True) -> Tuple[bool, str]:
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


# dill-Loader beibehalten
try:
    import dill  # used for processed_data.pkl
except ImportError:
    dill = None


# tolerant unpickler: try pickle → dill (if present) → return None when allowed
@st.cache_resource(show_spinner=False)
def load_pickle_any(path: str, allow_fail: bool = False):
    try:
        with open(path, "rb") as f:
            try:
                return pickle.load(f)  # standard pickle
            except Exception as e_pickle:
                if dill is None:
                    if allow_fail:
                        return None
                    raise RuntimeError(
                        f"Failed to load '{path}' with pickle ({e_pickle}). 'dill' is not installed."
                    )
                f.seek(0)
                try:
                    return dill.load(f)  # try dill
                except Exception as e_dill:
                    if allow_fail:
                        return None
                    raise RuntimeError(
                        f"Failed to load '{path}'. pickle error: {e_pickle}; dill error: {e_dill}"
                    )
    except FileNotFoundError:
        if allow_fail:
            return None
        raise


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
# --------------------------------------------------------------------------------------
# PAGE CONFIG + SIDEBAR NAV (icons + credits)
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Star Rating Prediction", page_icon="⭐", layout="wide")

# —— tiny CSS to tighten the radio and polish fonts
st.markdown("""
<style>
/* tighten radio spacing + make labels cleaner */
section[data-testid="stSidebar"] .stRadio > div { gap: 0.35rem !important; }
section[data-testid="stSidebar"] label { font-size: 14px !important; }
section[data-testid="stSidebar"] h3, 
section[data-testid="stSidebar"] .st-emotion-cache-1y4p8pa { margin-bottom: .25rem !important; }
/* light divider */
.sidebar-divider { border-top:1px solid #eaeaea; margin:.5rem 0 1rem 0; }
/* small header card in the sidebar */
.sidebar-card {
  background:#f6f7fb; border:1px solid #e9ecf3; border-radius:10px;
  padding:10px 12px; margin-bottom:.6rem;
}
.sidebar-card-title {
  display:flex; align-items:center; gap:8px; font-weight:700; color:#2b2f38;
}
.sidebar-card-sub { font-size:12px; color:#6b7280; margin-top:2px; }
.sidebar-credits { font-size:12.5px; line-height:1.45; color:#666; }
</style>
""", unsafe_allow_html=True)

# —— small title card INSIDE the sidebar (instead of a big banner in main area)
st.sidebar.markdown(
    """
    <div class="sidebar-card">
      <div class="sidebar-card-title"><span style="font-size:18px;">⭐</span>Star Rating Prediction</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "1) Introduction",
        "2) Load & Preprocess",
        "3) Feature Engineering",
        "4) Compare Models",
        "5) 100-Sample Evaluation",
        "6) Live Prediction",
        "7) Conclusion",
    ],
    index=0,
    label_visibility="collapsed",  # keeps things tidy
)

# — credits
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div class="sidebar-credits">
      <b>Created by</b><br/>
      Frank · Sebastian · Mohamed<br/>
      DataScientest – Data Science Program
    </div>
    """,
    unsafe_allow_html=True,
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
    with c1:
        st.metric("Rows", len(df))
    with c2:
        st.metric("Avg. text length", int(df["processed_text"].str.len().mean()))
    with c3:
        st.metric("Distinct ratings", df["ReviewRating"].nunique())

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

    # --- load artifacts we can visualize ---
    finfo = load_pickle(art["feature_info"]) if art["feature_info"] else None
    tfidf = load_pickle(art["tfidf"]) if art["tfidf"] else None
    scaler = load_pickle(art["scaler"]) if art["scaler"] else None

    # try to load processed dataframe for charts
    df_proc = None
    if art["processed_data"]:
        obj = load_pickle_any(art["processed_data"], allow_fail=True)
        if isinstance(obj, dict) and "df" in obj:
            df_proc = obj["df"]
    if df_proc is None and _exists(PATH_DATA_PROCESSED):
        try:
            df_proc = pd.read_csv(PATH_DATA_PROCESSED)
        except Exception:
            df_proc = None

    # ----------------- NUMERIC FEATURES -----------------
    if finfo:
        num_cols = finfo.get("numerical_features", [])
        st.subheader("Numerical feature columns used in training:")
        colA, colB = st.columns([1, 2])
        with colA:
            st.code("\n".join(map(str, num_cols)) or "(not found)")
        with colB:
            # one-liners to explain each numeric feature
            nice = {
                "word_count": "How long the review is (words).",
                "char_count": "Length in characters.",
                "sentence_count": "How many sentences.",
                "avg_word_length": "Average characters per word.",
                "exclamation_count": "Number of exclamation marks (!).",
                "question_count": "Number of question marks (?).",
                "capital_ratio": "Share of uppercase letters (INTENSITY).",
                "sentiment_compound": "VADER overall polarity (−1…+1).",
                "sentiment_pos": "VADER positive share.",
                "sentiment_neu": "VADER neutral share.",
                "sentiment_neg": "VADER negative share.",
            }
            bullets = [f"- **{c}** – {nice.get(c, 'auxiliary signal.')}" for c in num_cols]
            st.markdown("\n".join(bullets))

    # ----------------- TF-IDF – SETTINGS & WHY -----------------
    if tfidf:
        with st.expander("TF-IDF settings (what & why)", expanded=True):
            st.code(
                "TfidfVectorizer(\n"
                f"  max_features={getattr(tfidf, 'max_features', 'n/a')},\n"
                f"  min_df={getattr(tfidf, 'min_df', 'n/a')}, max_df={getattr(tfidf, 'max_df', 'n/a')},\n"
                "  stop_words='english',\n"
                f"  ngram_range={getattr(tfidf, 'ngram_range', '(1,1)')}\n"
                ")",
                language="python",
            )
            st.markdown(
                """
- **TF-IDF** turns words/phrases into numbers ∝ “how characteristic” a term is for a document.
- **max_features** keeps only the top terms to prevent overfitting/speed issues.
- **min_df** drops super-rare typos; **max_df** drops boiler-plate words.
- **ngram_range (1,2)** = unigrams + bigrams like *“not good”*, which capture context.
                """
            )

    # ----------------- TOP WORDS CHART -----------------
    if tfidf and df_proc is not None and "processed_text" in df_proc:
        dfv = df_proc[df_proc["processed_text"].astype(str).str.len() > 0].copy()
        if not dfv.empty:
            X = tfidf.transform(dfv["processed_text"].astype(str).values)
            feat = tfidf.get_feature_names_out()
            mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
            k = min(30, len(mean_tfidf))
            top_idx = mean_tfidf.argsort()[-k:][::-1]
            words = [feat[i] for i in top_idx]
            scores = [float(mean_tfidf[i]) for i in top_idx]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(range(len(words)), scores)
            ax.set_yticks(range(len(words)));
            ax.set_yticklabels(words)
            ax.set_xlabel("Average TF-IDF Score");
            ax.set_title("Top 30 Most Important Words")
            ax.invert_yaxis();
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        # ---- Top terms per star (quick table) ----
        if "ReviewRating" in dfv.columns:
            st.markdown("##### Top terms by rating")


            def top_for(r, n=8):
                mask = (dfv["ReviewRating"].astype(int) == r).values
                if mask.sum() == 0: return []
                m = np.asarray(X[mask].mean(axis=0)).ravel()
                idx = m.argsort()[-n:][::-1]
                return [feat[i] for i in idx]


            data = {f"{r}★": ", ".join(top_for(r)) for r in [1, 2, 3, 4, 5]}
            st.table(pd.DataFrame(data, index=["Top terms"]).T)

    # ----------------- SCALER EXPLANATION -----------------
    if scaler and finfo:
        with st.expander("Numeric scaling", expanded=False):
            st.markdown(
                f"""
We scale numeric features with **StandardScaler** so they are comparable and play nicely with
linear models. Formula:  \n
\\( z = (x - \\mu) / \\sigma \\).  \n
**Shape** expected by the model: {len(finfo.get('numerical_features', []))} numeric columns.
                """
            )
            st.success("Scaler found.")
    else:
        if art["scaler"]:
            st.success("Scaler found.")

    # always show presence flags (kept from your version)
    if art["tfidf"]:
        st.success("TF-IDF vectorizer found.")
    if art["processed_data"]:
        st.success("processed_data.pkl found (contains preprocessed DataFrame).")

    # ----------------- Optional: tiny 2D projection -----------------
    if tfidf and df_proc is not None and st.checkbox("Show tiny 2D PCA projection (sample)", value=False):
        dfs = df_proc.sample(n=min(600, len(df_proc)), random_state=42)
        Xs = tfidf.transform(dfs["processed_text"].astype(str).values)
        try:
            # dense just for the small sample
            Xdense = Xs.toarray()
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2, random_state=42)
            Z = pca.fit_transform(Xdense)
            fig, ax = plt.subplots(figsize=(6, 4))
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=dfs["ReviewRating"].astype(int), cmap="viridis", alpha=.6)
            ax.set_title("2D projection (PCA) – color = rating");
            plt.colorbar(sc, ax=ax)
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.info(f"PCA skipped: {e}")


# --------------------------------------------------------------------------------------
# 4) TRAIN & COMPARE (Notebook results, no heavy retraining)
# --------------------------------------------------------------------------------------
elif page.startswith("4)"):
    section_header("Train & Compare (optional)", "🏋️")
    st.caption("This page reads the comparison results exported by your notebook "
               "(`classification_comparison_results.pkl`). No heavy training here.")

    # ---------- locate artefacts ----------
    # ---------- locate artefacts (robust across working dirs) ----------
    from pathlib import Path

    APP_DIR = Path(__file__).resolve().parent
    ROOT_CANDIDATES = [
        Path.cwd(),  # wo streamlit gestartet wurde
        APP_DIR,  # Ordner der Datei
        APP_DIR.parent,  # …/src
        APP_DIR.parent.parent  # …/ (Repo-Root)
    ]


    def first_existing(paths):
        for p in paths:
            if p and Path(p).exists():
                return str(p)
        return None


    def candidates_for(relpath_list):
        cands = []
        for root in ROOT_CANDIDATES:
            for rel in relpath_list:
                cands.append(root / rel)
        return cands


    # zusätzlich direkt im Modell-Ordner nachsehen (funktioniert auch bei Secrets/ENV Overrides)
    extra_model_candidates = []
    try:
        extra_model_candidates.append(Path(PATH_MODELS) / "classification_summary.json")
        extra_model_candidates.append(Path(PATH_MODELS) / "classification_comparison_results.pkl")
    except Exception:
        pass

    # JSON bevorzugt (klein & git-freundlich)
    sum_path = first_existing(
        [*candidates_for([
            "src/models/classification_summary.json",
            "results/classification_summary.json",
        ])] + extra_model_candidates
    )

    # Großer PKL nur als Notlösung (kann fehlen – ist okay)
    res_path = first_existing(
        [*candidates_for([
            "src/models/classification_comparison_results.pkl",
            "results/classification_comparison_results.pkl",
        ])] + extra_model_candidates
    )

    cmp_df = None
    source_used = ""

    # ---- 1) preferred: small JSON summary ----
    if sum_path:
        try:
            with open(sum_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload.get("summary", payload)
            cmp_df = (
                pd.DataFrame(rows)
                .rename(columns={
                    "model_name": "Model",
                    "accuracy": "Accuracy",
                    "weighted_f1": "Weighted F1",
                    "macro_f1": "Macro F1",
                    "weighted_precision": "W. Precision",
                    "weighted_recall": "W. Recall",
                })
                .sort_values("Weighted F1", ascending=False)
                .reset_index(drop=True)
            )
            # auf Nummer sicher: alle Metriken als float casten
            for c in ["Accuracy", "Weighted F1", "Macro F1", "W. Precision", "W. Recall"]:
                if c in cmp_df.columns:
                    cmp_df[c] = cmp_df[c].astype(float)
            source_used = f"summary JSON ({Path(sum_path).as_posix()})"
            st.info("Loaded lightweight summary (no heavy objects).")
        except Exception as e:
            st.warning(f"Could not read summary JSON: {e}")

    # ---- 2) fallback: big pickle (optional) ----
    if cmp_df is None and res_path:
        if st.checkbox("Load large comparison pickle (~GB) instead?", value=False):
            import pickle

            with open(res_path, "rb") as f:
                all_results = pickle.load(f)
            rows = []
            for r in all_results:
                rows.append({
                    "Model": r.get("model_name", type(r.get("estimator")).__name__),
                    "Estimator": r.get("estimator"),
                    "Accuracy": float(r.get("test_accuracy", 0.0)),
                    "Weighted F1": float(r.get("weighted_f1", 0.0)),
                    "Macro F1": float(r.get("macro_f1", 0.0)),
                    "W. Precision": float(r.get("weighted_precision", 0.0)),
                    "W. Recall": float(r.get("weighted_recall", 0.0)),
                })
            cmp_df = (
                pd.DataFrame(rows)
                .sort_values("Weighted F1", ascending=False)
                .reset_index(drop=True)
            )
            source_used = f"pickle ({Path(res_path).as_posix()})"

    # ---- 3) nothing found
    if cmp_df is None:
        looked = [
            *(str(p) for p in
              candidates_for(["src/models/classification_summary.json", "results/classification_summary.json"])),
            *(str(p) for p in candidates_for(
                ["src/models/classification_comparison_results.pkl", "results/classification_comparison_results.pkl"])),
            *(str(p) for p in extra_model_candidates),
        ]
        st.warning(
            "No comparison artifacts found.\n\n"
            "Please export `classification_summary.json` from the notebook (recommended)."
        )
        st.caption("Looked in:\n" + "\n".join(looked))
        st.stop()

    st.caption(f"Loaded: {source_used}")

    best_row = cmp_df.iloc[0]

    # ---------- header cards ----------
    c1, c2, c3 = st.columns(3)
    c1.metric("Champion model", best_row["Model"], "")
    c2.metric("Weighted F1", f"{best_row['Weighted F1']:.3f}")
    c3.metric("Accuracy", f"{best_row['Accuracy']:.3f}")

    # ---------- nice ranking table ----------
    st.markdown("#### Model ranking (sorted by **Weighted F1**)")

    df_show = cmp_df.copy()
    df_show.index = np.arange(1, len(df_show) + 1)  # 1-basierter Index
    df_show.index.name = "#"


    def _highlight_best_column(col):
        if col.name in ["Accuracy", "Weighted F1", "Macro F1", "W. Precision", "W. Recall"]:
            m = col.max()
            return ["font-weight:700; color:#c0392b" if v == m else "" for v in col]
        return [""] * len(col)


    styled = (
        df_show[["Model", "Accuracy", "Weighted F1", "Macro F1", "W. Precision", "W. Recall"]]
        .round(3)
        .style.apply(_highlight_best_column, axis=0)
    )

    row_h, header_h, max_h = 32, 38, 700
    height = min(max_h, header_h + row_h * len(df_show))
    st.dataframe(styled, use_container_width=True, height=height)

    st.markdown("#### Performance overview")

    # ---------- Visuals (confusion grid etc.) ----------
    IMG_CANDIDATES = []
    for root in ROOT_CANDIDATES:
        IMG_CANDIDATES += [
            root / "results" / "confusion_grid_all_models.png",
            root / "results" / "model_bars.png",
        ]

    shown = False
    seen = set()
    for p in IMG_CANDIDATES:
        p = Path(p).resolve()
        if p.exists():
            key = p.as_posix()
            if key in seen:
                continue
            st.image(str(p), use_container_width=True)
            seen.add(key)
            shown = True
    if not shown:
        st.info("Place PNGs like `results/confusion_grid_all_models.png` to show them here.")

    # ---------- Key hyper-parameters (from params_preview or estimator) ----------
    st.markdown("#### Key hyper-parameters (expand)")


    def _fmt_val(v):
        if isinstance(v, dict) and v.get("_type"):
            t = v["_type"]
            rest = {k: vv for k, vv in v.items() if k != "_type"}
            inner = ", ".join(f"{k}={_fmt_val(vv)}" for k, vv in rest.items())
            return f"{t}({inner})"
        if isinstance(v, (list, tuple)):
            return ", ".join(map(str, v))
        if v is None:
            return "None"
        return str(v)


    def _pretty_params(d):
        if not d:
            return "(no public params)"
        lines = []
        for k, v in d.items():
            lines.append(f"{k}: {_fmt_val(v)}")
        return "\n".join(lines)


    for _, row in cmp_df.iterrows():
        name = str(row["Model"])
        with st.expander(f"{name} – parameters"):
            shown = False
            # (1) bevorzugt: params_preview aus summary JSON
            if "params_preview" in row and isinstance(row["params_preview"], dict) and row["params_preview"]:
                st.code(_pretty_params(row["params_preview"]))
                shown = True
            # (2) fallback: falls großes PKL geladen und Estimator verfügbar
            if not shown and "Estimator" in cmp_df.columns:
                try:
                    est = row["Estimator"]
                    params = getattr(est, "get_params", lambda: {})()
                    # Voting/Stacking: Basismodelle dazu
                    if hasattr(est, "estimators"):
                        base_names = [n for n, _ in est.estimators]
                        params = dict(params)
                        params["base_estimators"] = base_names
                    st.code(_pretty_params(params))
                    shown = True
                except Exception:
                    pass
            if not shown:
                st.write("No parameter info available for this model.")

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
        ax.set_xticks(range(len(labels)));
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels);
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted");
        ax.set_ylabel("Actual")
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
            if cls == "dark_green":
                cls = "dark"
            elif cls == "light_green":
                cls = "light"
            elif cls == "yellow":
                cls = "yellow"
            else:
                cls = "red"
            cells = [row["True★"], row["Pred★"], row["Adj★"],
                     row["Group True"], row["Group Pred"], row["Group Adj"]]
            html += f'<tr class="{cls}">'
            html += f"<td><b>{i + 1}</b></td>"
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
        df["word_count"] = txt.str.split().apply(len)
        df["char_count"] = txt.str.len()
        df["sentence_count"] = txt.str.count(r"[.!?]") + 1
        df["avg_word_length"] = (df["char_count"] / df["word_count"]).replace([np.inf, np.nan], 0)
        df["exclamation_count"] = txt.str.count("!")
        df["question_count"] = txt.str.count(r"\?")
        df["capital_ratio"] = txt.apply(lambda s: sum(c.isupper() for c in s) / len(s) if len(s) else 0)

        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk
            try:
                nltk.data.find("sentiment/vader_lexicon.zip")
            except LookupError:
                nltk.download("vader_lexicon", quiet=True)
            sia = SentimentIntensityAnalyzer()
            pol = txt.apply(sia.polarity_scores)
            df["sentiment_compound"] = pol.apply(lambda d: d["compound"])
            df["sentiment_pos"] = pol.apply(lambda d: d["pos"])
            df["sentiment_neu"] = pol.apply(lambda d: d["neu"])
            df["sentiment_neg"] = pol.apply(lambda d: d["neg"])
        except Exception:
            for c in ["sentiment_compound", "sentiment_pos", "sentiment_neu", "sentiment_neg"]:
                df[c] = 0.0

        for c in needed_cols:
            if c not in df.columns:
                df[c] = 0.0
        return df


    # ---------- load artifacts ----------
    model, meta = load_best_model(PATH_MODELS)
    tfidf = load_pickle(art["tfidf"])
    scaler = load_pickle(art["scaler"])
    finfo = load_pickle(art["feature_info"])
    num_cols = finfo.get("numerical_features", [])

    df_proc = None
    if art["processed_data"]:
        obj = load_pickle_any(art["processed_data"], allow_fail=True)
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
    left, mid, right = st.columns([1, 1, 1])
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
            p_neg = row[class_to_idx.get(1, 0)] + row[class_to_idx.get(2, 0)]
            p_neu = row[class_to_idx.get(3, 0)]
            p_pos = row[class_to_idx.get(4, 0)] + row[class_to_idx.get(5, 0)]
            if p_neu >= max(p_neg, p_pos):
                adj_star[i] = 3
            elif p_neg >= p_pos:
                adj_star[i] = 1 if row[class_to_idx.get(1, 0)] >= row[class_to_idx.get(2, 0)] else 2
            else:
                adj_star[i] = 4 if row[class_to_idx.get(4, 0)] >= row[class_to_idx.get(5, 0)] else 5

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
- 🟢 Perfect Initial Predictions: **{perfect_initial} ({perfect_initial / total:.1%})**
- 🟡 Corrected by Adjustment: **{corrected_by_adj} ({corrected_by_adj / total:.1%})**
- 🟨 Close Predictions (±1): **{close_1} ({close_1 / total:.1%})**
- 🔴 Poor Predictions (≥2): **{poor_2p} ({poor_2p / total:.1%})**
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
        "Adj★": [STAR[x] for x in adj_star],
        "Group True": group_true,
        "Group Pred": group_pred,
        "Group Adj": group_adj,
        "Color": [color_class(t, p, a) for t, p, a in zip(y_true, y_pred, adj_star)]
    })

    st.markdown("#### Color-coded results (first 100 rows)")
    st.markdown(make_colored_html_table(table_df, max_rows=show_rows), unsafe_allow_html=True)

    # ---------- detailed sample predictions ----------
    st.markdown("#### Sample predictions with texts")
    show_n = st.slider("How many rows to show below", 10, 100, 100, step=10)

    out = df_s[["ReviewText", "processed_text", "ReviewRating"]].copy()
    out.rename(columns={"ReviewRating": "True"}, inplace=True)
    out["Predicted"] = y_pred
    out["Adjusted"] = adj_star

    # 👉 1-basierter Index nur für die Anzeige
    out_display = out.head(show_n).copy()
    out_display.index = np.arange(1, len(out_display) + 1)
    out_display.index.name = "#"

    st.dataframe(out_display, use_container_width=True)

    # Download weiterhin ohne Index
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
        st.error(msg);
        st.stop()

    # --- load artifacts ---
    model, meta = load_best_model(PATH_MODELS)
    tfidf = load_pickle(art["tfidf"])
    scaler = load_pickle(art["scaler"])
    finfo = load_pickle(art["feature_info"])
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
        p1 = probs[c2i.get(1, 0)];
        p2 = probs[c2i.get(2, 0)]
        p3 = probs[c2i.get(3, 0)];
        p4 = probs[c2i.get(4, 0)];
        p5 = probs[c2i.get(5, 0)]
        p_neg, p_neu, p_pos = p1 + p2, p3, p4 + p5
        if p_neu >= max(p_neg, p_pos):    return 3, "neutral"
        if p_neg >= p_pos:                return (1 if p1 >= p2 else 2), "negative"
        return (4 if p4 >= p5 else 5), "positive"


    def model_card(model, meta, tfidf, num_cols):
        # friendly summary
        name = meta.get("model_name", type(model).__name__)
        acc = meta.get("test_accuracy", None)
        f1w = meta.get("weighted_f1", None)
        f1m = meta.get("macro_f1", None)

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
            classes = [int(c) for c in getattr(model, "classes_", [1, 2, 3, 4, 5])]
        except Exception:
            classes = [1, 2, 3, 4, 5]

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
                st.write(pd.DataFrame(base_learners, columns=["Name", "Estimator"]))
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
            st.warning("Please enter some text.");
            st.stop()

        # --- features ---
        processed = clean_text_basic(text)
        X_tfidf = tfidf.transform([processed])
        X_num = neutral_num_vector()
        Xc = hstack([X_tfidf, X_num])

        # --- predictions ---
        y_init = int(model.predict(Xc)[0])

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
        grp_color = {"negative": "#ff6b6b", "neutral": "#feca57", "positive": "#48dbfb"}[group]
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

        st.caption(f"Processed: _{processed[:160]}{'…' if len(processed) > 160 else ''}_")

        # --- charts (bar + pie) ---
        if proba_used is not None:
            stars = classes
            probs = proba_used

            c1, c2 = st.columns(2)

            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                barlist = ax.bar(stars, probs,
                                 color=["#e74c3c", "#f39c12", "#f1c40f", "#2ecc71", "#27ae60"][:len(stars)])
                ax.set_ylim(0, 1);
                ax.set_xlabel("Star Rating");
                ax.set_ylabel("Probability")
                ax.set_title("★ Star Rating Probabilities")
                if y_refined in stars:
                    k = stars.index(y_refined)
                    barlist[k].set_edgecolor("black");
                    barlist[k].set_linewidth(2.5)
                for b, p in zip(barlist, probs):
                    ax.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.02, f"{p:.3f}",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")
                st.pyplot(fig, use_container_width=True)

            with c2:
                idx = {c: i for i, c in enumerate(stars)}
                p1 = probs[idx.get(1, 0)];
                p2 = probs[idx.get(2, 0)]
                p3 = probs[idx.get(3, 0)];
                p4 = probs[idx.get(4, 0)];
                p5 = probs[idx.get(5, 0)]
                group_vals = [p1 + p2, p3, p4 + p5]
                group_lbls = ["negative", "neutral", "positive"]
                group_cols = ["#ff6b6b", "#feca57", "#48dbfb"]

                fig2, ax2 = plt.subplots(figsize=(6, 4))
                wedges, txts, autotxts = ax2.pie(
                    group_vals, labels=group_lbls, colors=group_cols,
                    autopct="%1.1f%%", startangle=90, textprops={"fontweight": "bold"}
                )
                ax2.set_title("😊 Sentiment Group Distribution")
                for i, g in enumerate(group_lbls):
                    if g == group:
                        wedges[i].set_edgecolor("black");
                        wedges[i].set_linewidth(2.5)
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
            hist_df["Sentiment"] = hist_df["Sentiment"].map(lambda g: f"{EMOJI.get(g, '🤔')} {g}")
            hist_df["Initial"] = hist_df["Initial"].map(STAR)
            hist_df["Refined"] = hist_df["Refined"].map(STAR)
            st.dataframe(hist_df, use_container_width=True, height=min(420, 60 + 28 * len(hist_df)))
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
""")
