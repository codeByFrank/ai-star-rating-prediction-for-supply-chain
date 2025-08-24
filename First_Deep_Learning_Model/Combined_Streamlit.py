import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, io, sys, pickle, json, time, random
from typing import Tuple, Dict, Optional

from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
)
from scipy.sparse import hstack, csr_matrix
from pathlib import Path
from glob import glob

# ---- robuste Root-Suche (CWD, Dateipfad, darüberliegende Ordner) ----
APP_DIR = Path(__file__).resolve().parent
ROOT_CANDIDATES = [Path.cwd(), APP_DIR, APP_DIR.parent, APP_DIR.parent.parent]

def _first_existing(relpaths):
    """Gib den ersten existierenden Pfad (als str) aus einer Liste relativer Pfade zurück."""
    if isinstance(relpaths, (str, Path)):
        relpaths = [relpaths]
    for root in ROOT_CANDIDATES:
        for rel in relpaths:
            p = (root / rel).resolve()
            if p.exists():
                return p.as_posix()
    return None

import os, json
from glob import glob

def _bases_for(name: str):
    """Mögliche Basispfade, unter denen api_models/<name>/ liegen könnte."""
    for root in ROOT_CANDIDATES:
        yield (root / "api_models" / name).resolve()
        yield (root / "src" / "api_models" / name).resolve()


def _ensure_nltk():
    import nltk
    def _has(path: str) -> bool:
        try:
            nltk.data.find(path); return True
        except LookupError:
            return False

    # punkt_tab (neuere NLTK) oder punkt (ältere)
    if not (_has("tokenizers/punkt_tab") or _has("tokenizers/punkt")):
        for pkg, path in [("punkt_tab", "tokenizers/punkt_tab"), ("punkt", "tokenizers/punkt")]:
            try:
                nltk.download(pkg, quiet=True)
                if _has(path): break
            except Exception:
                pass

    if not _has("corpora/stopwords"): nltk.download("stopwords", quiet=True)
    if not _has("corpora/wordnet"):   nltk.download("wordnet", quiet=True)
    if not _has("sentiment/vader_lexicon"): nltk.download("vader_lexicon", quiet=True)


def find_dl_model_path(name: str):
    """Suche .keras/.h5/.hdf5 oder SavedModel-Ordner unter allen Roots."""
    patterns = ["model.keras", "*.keras", "*.h5", "*.hdf5", "saved_model", "*saved_model*"]
    for base in _bases_for(name):
        if not base.exists():
            continue
        for pat in patterns:
            for p in base.glob(pat):
                if p.is_dir():
                    # SavedModel-Struktur?
                    if (p / "saved_model.pb").exists() or (p / "variables").exists():
                        return p.as_posix()
                else:
                    return p.as_posix()
    return None

def find_dl_metadata_path(name: str):
    for base in _bases_for(name):
        p = (base / "metadata.json")
        if p.exists():
            return p.as_posix()
    return None
    
# WordCloud is optional; app runs without it
try:
    from wordcloud import WordCloud, STOPWORDS
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

PATH_MODELS = _first_existing(["src/models", "models"]) or "models"
PATH_DATA_PROCESSED = _first_existing([
    "src/data/processed/temu_reviews_preprocessed.csv",
    "data/processed/temu_reviews_preprocessed.csv"
])

NUM_PREVIEW_ROWS = 10
DEFAULT_RANDOM_SEED = 42


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

def _safe_stopwords():
    # Try NLTK; else lightweight fallback
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return set("""
        a an the and or is are was were be been being to of in on for with at by from up about into over after
        before between under again further then once here there all any both each few more most other some such no nor
        not only own same so than too very s t can will just don don should now
        """.split())

STOP_EN = _safe_stopwords()

def _safe_lemmatize(tokens):
    try:
        from nltk.stem import WordNetLemmatizer
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        wnl = WordNetLemmatizer()
        return [wnl.lemmatize(t) for t in tokens]
    except Exception:
        return tokens 
    
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


# Set page config
st.set_page_config(page_title="Star Rating Prediction", page_icon="⭐", layout="wide", initial_sidebar_state="expanded")

# ——— CSS (kompakt, wie früher)
st.markdown("""
<style>
section[data-testid="stSidebar"] .stRadio > div { gap: 0.35rem !important; }
section[data-testid="stSidebar"] label { font-size: 14px !important; }
section[data-testid="stSidebar"] h3, 
section[data-testid="stSidebar"] .st-emotion-cache-1y4p8pa { margin-bottom: .25rem !important; }
.sidebar-divider { border-top:1px solid #eaeaea; margin:.5rem 0 1rem 0; }
.sidebar-card { background:#f6f7fb; border:1px solid #e9ecf3; border-radius:10px; padding:10px 12px; margin-bottom:.6rem; }
.sidebar-card-title { display:flex; align-items:center; gap:8px; font-weight:700; color:#2b2f38; }
.sidebar-card-sub { font-size:12px; color:#6b7280; margin-top:2px; }
.sidebar-credits { font-size:12.5px; line-height:1.45; color:#666; }
</style>
""", unsafe_allow_html=True)

# ——— kleines Titel-Kärtchen in der Sidebar
st.sidebar.markdown(
    """
    <div class="sidebar-card">
      <div class="sidebar-card-title"><span style="font-size:18px;">⭐</span>Star Rating Prediction</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============
# NAVIGATION
# ============
# Labels (sichtbar) + Keys (intern)
PAGES = [
    ("1) Introduction",          "intro"),
    ("2) Load Data",             "load"),
    ("3) Data Exploration",      "dataexp"),
    ("4) Preprocess",            "preprocess"),
    ("5) Feature Engineering",   "features"),
    ("6) Compare Models (ML)",   "compare"),
    ("7) 100-Sample Evaluation", "eval100"),
    ("8) Live Prediction (ML)",  "live_ml"),
    ("9) Live Prediction (DL)",  "live_dl"),
    ("10) Compare Models (DL)",  "results"),
    ("11) Conclusion",           "conclusion"),
    ("12) About",                "about"),
    
]
LABEL_TO_KEY = {label: key for label, key in PAGES}

st.sidebar.markdown("### Navigation")
selected_label = st.sidebar.radio(
    "Go to",
    [label for label, _ in PAGES],
    index=0,
    label_visibility="collapsed",
)

selected_key = LABEL_TO_KEY[selected_label]

# — Credits
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div class="sidebar-credits">
      <b>Created by</b><br/>
      Frank · Sebastian · Mohamed<br/>
      DataScientest – Data Science Project
    </div>
    """,
    unsafe_allow_html=True,
)

# Available models
MODEL_NAMES = [
    "deep_mlp_with_tf-idf",
    "lstm_model",
    "bilstm_with_attention",
    "cnn_model",
    "transformer_model",
    "hybrid_cnn-lstm"
]

# Initialize text preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Load preprocessing artifacts
@st.cache_resource(show_spinner=False)
def load_artifacts():
    files = {
        "metadata":        ["data/metadata.pkl",        "src/data/metadata.pkl",        "metadata.pkl"],
        "tokenizer":       ["data/tokenizer.pkl",       "src/data/tokenizer.pkl",       "tokenizer.pkl"],
        "scaler":          ["data/scaler.pkl",          "src/data/scaler.pkl",          "scaler.pkl"],
        "label_encoders":  ["data/label_encoders.pkl",  "src/data/label_encoders.pkl",  "label_encoders.pkl"],
    }
    out = {}
    for key, rels in files.items():
        p = _first_existing(rels)
        if p:
            with open(p, "rb") as f:
                out[key] = pickle.load(f)
        else:
            out[key] = None  # nicht stoppen – Seiten, die es brauchen, prüfen selbst
    return out


artifacts = load_artifacts()


def load_mlp_specific_assets():
    vec_path  = _first_existing(["api_models/deep_mlp_with_tf-idf/tfidf.pkl",
                                 "src/api_models/deep_mlp_with_tf-idf/tfidf.pkl"])
    meta_path = _first_existing(["api_models/deep_mlp_with_tf-idf/metadata.json",
                                 "src/api_models/deep_mlp_with_tf-idf/metadata.json"])
    if not vec_path or not meta_path:
        raise FileNotFoundError("TF-IDF/metadata for MLP not found under api_models/…")
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        mlp_metadata = json.load(f)
    return vectorizer, mlp_metadata


def clean_text(text: str) -> str:
    """Lightweight clean for inference (no NLTK requirement)."""
    if pd.isna(text) or text == "":
        return ""
    x = str(text).lower()
    x = re.sub(r"http\S+|www\.\S+|https?\S+", " ", x)
    x = re.sub(r"\S+@\S+", " ", x)
    x = re.sub(r"<.*?>", " ", x)
    # keep letters/spaces only
    x = re.sub(r"[^a-z\s]", " ", x)
    # regex tokenization
    toks = re.findall(r"[a-z]+", x)
    toks = [t for t in toks if len(t) > 2 and t not in STOP_EN]
    toks = _safe_lemmatize(toks)  # no-op if wordnet not present
    return " ".join(toks)


def preprocess_text(text, max_len=100):
    """Tokenize and preprocess text for models that don't use TF-IDF"""
    try:
        # Use the cached tokenizer if available
        tokenizer = artifacts.get('tokenizer')
        if tokenizer:
            sequence = tokenizer.texts_to_sequences([text])
            return sequence
    except:
        pass

    # Fallback simple tokenization (should match your training preprocessing)
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens
              if token not in stop_words and len(token) > 2]
    return [[len(token) for token in tokens[:max_len]]]  # Simplified example

def preprocess_for_mlp(text, vectorizer, num_features):
    """Process inputs specifically for MLP model"""
    # Transform text to TF-IDF
    text_features = vectorizer.transform([text]).toarray()

    # Combine with numerical features
    full_features = np.concatenate([text_features, num_features], axis=1)
    return full_features


# Load models and metadata
@st.cache_resource(show_spinner=False)
def load_model_and_metadata(model_name):
    model_path = find_dl_model_path(model_name)
    metadata_path = find_dl_metadata_path(model_name)
    if not model_path or not metadata_path:
        raise FileNotFoundError(f"Model or metadata for '{model_name}' not found.")
    model = load_model(model_path)  # kann Datei (.keras/.h5) oder Ordner (SavedModel) sein
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def preprocess_input(review_text, review_title, review_count, user_country):
    """Preprocess user input for prediction"""
    # Clean text
    clean_review = clean_text_basic(review_text)
    clean_title = clean_text_basic(review_title)
    combined_text = clean_review + ' ' + clean_title

    # Tokenize and pad text
    tokenizer = artifacts['tokenizer']
    sequence = tokenizer.texts_to_sequences([combined_text])
    padded_sequence = pad_sequences(sequence, maxlen=artifacts['metadata']['max_sequence_length'],
                                    padding='post', truncating='post')

    # Process numerical features
    country_encoded = artifacts['label_encoders']['UserCountry'].transform([user_country])[0]

    # Extract text features
    text_features = {
        'text_length': len(review_text),
        'word_count': len(review_text.split()),
        'avg_word_length': np.mean([len(word) for word in review_text.split()]) if review_text else 0,
        'exclamation_count': review_text.count('!'),
        'question_count': review_text.count('?'),
        'upper_case_ratio': sum(1 for c in review_text if c.isupper()) / len(review_text) if review_text else 0,
        'title_text_length': len(review_title),
        'title_word_count': len(review_title.split()),
        'title_avg_word_length': np.mean([len(word) for word in review_title.split()]) if review_title else 0,
        'title_exclamation_count': review_title.count('!'),
        'title_question_count': review_title.count('?'),
        'title_upper_case_ratio': sum(1 for c in review_title if c.isupper()) / len(review_title) if review_title else 0
    }

    # Create numerical feature array
    numerical_features = [
        review_count, country_encoded,
        text_features['text_length'], text_features['word_count'], text_features['avg_word_length'],
        text_features['exclamation_count'], text_features['question_count'], text_features['upper_case_ratio'],
        text_features['title_text_length'], text_features['title_word_count'], text_features['title_avg_word_length'],
        text_features['title_exclamation_count'], text_features['title_question_count'],
        text_features['title_upper_case_ratio']
    ]

    # Scale numerical features
    scaled_numerical = artifacts['scaler'].transform([numerical_features])

    return padded_sequence, scaled_numerical


def preprocess_inputs(model_name, review_text, numerical_features):
    if model_name == "deep_mlp_with_tf-idf":
        # Load TF-IDF vectorizer
        with open('api_models/deep_mlp_with_tf-idf/tfidf.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

        # Process text and combine with numerical features
        return preprocess_for_mlp(review_text, vectorizer, numerical_features)
    else:
        # Text sequence processing for other models
        text_seq = preprocess_text(review_text, max_len=100)
        text_seq = tf.keras.preprocessing.sequence.pad_sequences(
            text_seq,
            maxlen=100,
            padding='post'
        )
        return [text_seq, numerical_features]


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


# ---------- PAGE: Load Data (leicht) ----------
def show_load_page():
    section_header("Load Data", "📥")
    st.write("Upload a **raw** CSV file with at least the columns `ReviewText` and `ReviewRating`.")

    up = st.file_uploader("Upload raw CSV", type=["csv"])
    df = None

    # 1) Upload bevorzugt
    if up is not None:
        try:
            df = pd.read_csv(up)
            st.success(f"Loaded uploaded file with shape {df.shape}.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    # 2) Falls nichts hochgeladen: versuche typische Roh-Pfade
    if df is None:
        for cand in ["data/temu_reviews.csv", "src/data/temu_reviews.csv",
                     "data/raw/temu_reviews.csv", "src/data/raw/temu_reviews.csv"]:
            if os.path.exists(cand):
                try:
                    df = pd.read_csv(cand)
                    st.info(f"Loaded fallback raw dataset: `{cand}` (shape {df.shape}).")
                    break
                except Exception:
                    pass

    if df is None:
        st.warning("No raw data found. Please upload a CSV.")
        st.stop()

    # Pflichtspalten prüfen
    need = {"ReviewText", "ReviewRating"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        st.error(f"Missing required columns: {miss}")
        st.stop()

    # Für Vorschau: processed_text erzeugen (leicht)
    if "processed_text" not in df.columns:
        df["processed_text"] = df["ReviewText"].astype(str).apply(clean_text_basic)

    # KPIs + Vorschau
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Avg. text length", int(df["processed_text"].str.len().mean()))
    c3.metric("Distinct ratings", int(df["ReviewRating"].nunique()))
    dataset_preview(df, "Top rows")

    # Wordclouds (leicht, aus processed_text)
    st.markdown("#### Wordclouds by sentiment group")

    def _find_file(relpaths):
        roots = []
        if "ROOT_CANDIDATES" in globals():
            roots += list(ROOT_CANDIDATES)
        here = Path(__file__).resolve().parent
        roots += [Path.cwd(), here, here.parent]
        for root in roots:
            for rel in relpaths:
                p = (Path(root) / rel).resolve()
                if p.exists():
                    return p.as_posix()
        return None

    img_path = _find_file([
        "charts/wordcloud.png",                 # dein genannter Pfad
        "api_models/charts/wordcloud.png",      # falls dort liegt
        "src/charts/wordcloud.png"              # fallback
    ])

    if img_path:
        st.image(img_path, use_container_width=True,
                 caption="Wordclouds by sentiment group (PNG)")
    else:
        st.info("Wordcloud PNG not found (expected at `charts/wordcloud.png`).")

    # In Session parken für die nächste Seite
    st.session_state["raw_df"] = df.copy()
    st.success("Raw dataset cached for preprocessing.")

# ---------- PAGE: Preprocess (bewusst „schwer“) ----------
def show_preprocess_page():
    section_header("Preprocess", "🧪")

    # Quelle festlegen
    df_raw = st.session_state.get("raw_df")
    if df_raw is None:
        # Fallback: bereits vorhandene Preprocessed-Datei
        if os.path.exists(PATH_DATA_PROCESSED):
            st.info(f"No raw_df in session. Using `{PATH_DATA_PROCESSED}` as input.")
            df_raw = pd.read_csv(PATH_DATA_PROCESSED)
        else:
            st.warning("No input data found. Please go to **Load Data** first.")
            st.stop()

    # Optionen
    st.subheader("Options")
    c1, c2, c3 = st.columns(3)
    with c1:
        do_emoji = st.checkbox("Map emojis → sentiments", value=True)
    with c2:
        do_vader = st.checkbox("Add VADER sentiment features", value=True)
    with c3:
        sample_n = st.number_input("Sample (0 = all)", min_value=0, value=0, step=1000)

    with st.spinner("Setting up NLTK resources…"):
            _ensure_nltk()

    # Pipeline (gecached)
    @st.cache_data(show_spinner=False)
    def _preprocess_df(df_in, do_emoji_, do_vader_):
        import re
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        _ensure_nltk()
        
        lemm = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        def remove_special_chars(x):
            x = re.sub(r"[^a-zA-Z\s]", " ", str(x))
            return re.sub(r"\s+", " ", x).strip()

        def emoji_to_sentiment(x):
            if not do_emoji_:
                return x
            try:
                from emoji import demojize
                s = demojize(str(x).lower())
                s = re.sub(r":\w*?(smil|grin|laugh|heart|thumbs_up|star|party|kiss)\w*?:", " positive_emoji ", s)
                s = re.sub(r":\w*?(angry|cry|sad|thumbs_down|sick|vomit|rage|poop|devil|skull)\w*?:", " negative_emoji ", s)
                s = re.sub(r":[a-z_]+:", " neutral_emoji ", s)
                return s
            except Exception:
                return x
        
        from nltk.tokenize import wordpunct_tokenize, word_tokenize
        def tok_stop_lemma(x):
            s = str(x)
            try:
                toks = word_tokenize(s)  # „schöner“, aber braucht punkt(_tab)
            except LookupError:
                # Fallback ohne punkt/punkt_tab
                toks = wordpunct_tokenize(s)  # trennt an Wort-/Satzzeichen, keine Downloads
            except Exception:
                # Ultimativer Fallback
                toks = re.findall(r"[A-Za-z]+", s)

            toks = [t for t in toks if t not in stop_words and len(t) > 2]
            toks = [lemm.lemmatize(t) for t in toks]
            return " ".join(toks)

        def text_feats(txt):
            s = str(txt)
            words = s.split()
            wc = len(words)
            return {
                "word_count": wc,
                "char_count": len(s),
                "sentence_count": max(1, len(re.split(r"[.!?]+", s))),
                "avg_word_length": (sum(len(w) for w in words) / wc) if wc else 0.0,
                "exclamation_count": s.count("!"),
                "question_count": s.count("?"),
                "capital_ratio": (sum(c.isupper() for c in s) / len(s)) if len(s) else 0.0,
            }

        df = df_in.copy()
        if "ReviewText" not in df or "ReviewRating" not in df:
            raise ValueError("`ReviewText` and `ReviewRating` required.")

        # Textverarbeitung
        t = df["ReviewText"].astype(str).apply(clean_text_basic)
        t = t.apply(remove_special_chars)
        t = t.apply(emoji_to_sentiment)
        df["processed_text"] = t.apply(tok_stop_lemma)

        # Numerische Textfeatures
        feats = df["ReviewText"].apply(text_feats).apply(pd.Series)
        for c in feats.columns:
            df[c] = feats[c] 

        # VADER (optional)
        if do_vader_:
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                pol = df["ReviewText"].astype(str).apply(sia.polarity_scores)
                df["sentiment_compound"] = pol.apply(lambda d: d["compound"])
                df["sentiment_pos"] = pol.apply(lambda d: d["pos"])
                df["sentiment_neu"] = pol.apply(lambda d: d["neu"])
                df["sentiment_neg"] = pol.apply(lambda d: d["neg"])
            except Exception:
                for c in ["sentiment_compound", "sentiment_pos", "sentiment_neu", "sentiment_neg"]:
                    df[c] = 0.0

        df = df.loc[:, ~df.columns.duplicated()]
        return df

    # ggf. Sampling für schnellere Iteration
    df_in = df_raw.sample(n=min(sample_n, len(df_raw)), random_state=42).copy() if sample_n else df_raw

    with st.spinner("Preprocessing…"):
        df_proc = _preprocess_df(df_in, do_emoji, do_vader)
        df_proc = df_proc.loc[:, ~df_proc.columns.duplicated()]

    st.success(f"Preprocessed rows: {len(df_proc):,}")
    dataset_preview(df_proc, "Preview of preprocessed data")

    # kleine QC-KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg processed length", int(df_proc["processed_text"].str.len().mean()))
    if "sentiment_compound" in df_proc:
        c2.metric("Mean compound", f"{df_proc['sentiment_compound'].mean():.3f}")
    c3.metric("Empty processed", int((df_proc["processed_text"].str.len() == 0).sum()))

    # Speichern
    if st.button("💾 Save as processed CSV", use_container_width=True):
        os.makedirs(os.path.dirname(PATH_DATA_PROCESSED), exist_ok=True)
        df_proc.to_csv(PATH_DATA_PROCESSED, index=False)
        st.session_state["processed_df"] = df_proc
        st.success(f"Saved → {PATH_DATA_PROCESSED}")

def show_feature_engineering_page():
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

def show_compare_models():
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
    from pathlib import Path

    # Nur je Bildname die erste gefundene Datei anzeigen (keine Duplikate)
    WANTED = ["confusion_grid_all_models.png", "model_bars.png"]

    found_by_name = {}  # name -> Path
    for root in ROOT_CANDIDATES:
        for name in WANTED:
            if name in found_by_name:
                continue
            p = (Path(root) / "results" / name).resolve()
            if p.exists():
                found_by_name[name] = p

    if found_by_name:
        for name in WANTED:
            p = found_by_name.get(name)
            if p:
                st.image(p.as_posix(), use_container_width=True)
    else:
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
def show_100_sample_evaluation_page():
    section_header("100-Sample Evaluation (no retraining)", "🧪")

    art = st.session_state.art
    ok, msg = artifact_status_msg(art, need_model=True)
    if not ok:
        st.error(msg)
        st.stop()

    # ---------- helpers (local to this page) ----------
    def plot_cm(cm, labels, title, cmap="Blues"):
        fig, ax = plt.subplots(figsize=(4, 3))
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

def show_live_prediction_ml_page():
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
        processed = clean_text(text)
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


def show_live_prediction_DL_page():
    import os, json
    from pathlib import Path

    section_header("Make a Prediction")

    # --- Preprocessing-Checks wie gehabt ---
    needed = ["metadata", "tokenizer", "scaler", "label_encoders"]
    missing = [k for k in needed if artifacts.get(k) is None]
    if missing:
        st.error(
            "Preprocessing artifacts not found: "
            + ", ".join(missing)
            + ". Place them under one of: 'data/', 'src/data/' or the repo root."
        )
        st.stop()

    # ---------- Verfügbarkeit der DL-Modelle prüfen ----------
    def _exists_any(paths):
        for p in paths:
            if p and os.path.exists(p):
                return p
        return None

    def _dl_paths(name):
        model_path = _exists_any([
            f"api_models/{name}/model.keras",
            f"src/api_models/{name}/model.keras",
        ])
        meta_path = _exists_any([
            f"api_models/{name}/metadata.json",
            f"src/api_models/{name}/metadata.json",
        ])
        return model_path, meta_path

    # Verfügbarkeit je Modell prüfen (Model + Metadata)
    availability = {name: bool(find_dl_model_path(name) and find_dl_metadata_path(name))
                    for name in MODEL_NAMES}

    # Default = erstes verfügbares Modell (damit kein „unavailable“ vorausgewählt ist)
    default_idx = next((i for i, n in enumerate(MODEL_NAMES) if availability.get(n)), 0)

    model_name = st.selectbox(
        "Select Model",
        MODEL_NAMES,
        index=default_idx,
        format_func=lambda n: f"{n} (unavailable)" if not availability.get(n) else n,
    )

    # Wenn das ausgewählte Modell in diesem Deployment fehlt → nur Hinweis, kein Trace
    if not availability.get(model_name):
        st.warning(
            "This model isn't included in this deployment (likely excluded due to size). "
            f"Add the files under `api_models/{model_name}/` "
            "(a `.keras`/`.h5` file or a `saved_model/` folder plus `metadata.json`)."
        )
        with st.expander("Diagnostics", expanded=False):
            st.write("Roots searched:", [r.as_posix() for r in ROOT_CANDIDATES])
            st.write("Tried base folders:", [b.as_posix() for b in _bases_for(model_name)])
            st.write("Found model path:", find_dl_model_path(model_name))
            st.write("Found metadata path:", find_dl_metadata_path(model_name))
        return
    # ---------- Ab hier: nur wenn das Modell wirklich vorhanden ist ----------
    model, metadata = load_model_and_metadata(model_name)

    # Nur für das MLP die Zusatzassets versuchen
    tfidf_vectorizer = None
    if model_name == "deep_mlp_with_tf-idf":
        try:
            tfidf_vectorizer, mlp_metadata = load_mlp_specific_assets()
        except Exception:
            tfidf_vectorizer = None  # nicht kritisch für die Seite

    # Sample data
    sample_data = {
        "ReviewText": "This product is very good",
        "ReviewTitle": "Best purchase ever",
        "ReviewCount": 5,
        "UserCountry": "US",
    }

    # User input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            review_text = st.text_area("Review Text", value=sample_data["ReviewText"], height=150)
            review_count = st.number_input("Review Count", min_value=1, value=sample_data["ReviewCount"])
        with col2:
            review_title = st.text_input("Review Title", value=sample_data["ReviewTitle"])
            user_country = st.selectbox(
                "User Country",
                options=artifacts['label_encoders']['UserCountry'].classes_,
                index=0
            )
        submitted = st.form_submit_button("Predict Rating")

    if not submitted:
        return

    with st.spinner("Processing your review..."):
        # Preprocess input
        text_seq, num_features = preprocess_input(
            review_text, review_title, review_count, user_country
        )
        # Prepare inputs for the selected model
        inputs = preprocess_inputs(model_name, review_text, num_features)

        # Display cleaned text
        st.subheader("Preprocessed Text")
        cleaned_review = clean_text_basic(review_text)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Review**")
            st.write(review_text)
        with col2:
            st.markdown("**Cleaned Review**")
            st.write(cleaned_review)

        # Make prediction (robust)
        try:
            prediction = model.predict(inputs)
            predicted_class = np.argmax(prediction, axis=1)[0] + 1
            probabilities = prediction[0]

            st.subheader("Prediction Results")
            st.write(f"Predicted Rating: {predicted_class} stars")

            fig, ax = plt.subplots()
            sns.barplot(x=list(range(1, 6)), y=probabilities, ax=ax)
            ax.set_title("Rating Probability Distribution")
            ax.set_xlabel("Star Rating"); ax.set_ylabel("Probability")
            st.pyplot(fig)

            st.write(f"Confidence: {probabilities[predicted_class - 1] * 100:.1f}%")

            st.subheader("Key Influencing Factors")
            for i, factor in enumerate(
                ["Positive sentiment in review", "Review length",
                 "Use of exclamation marks", "User's review history"], 1
            ):
                st.markdown(f"{i}. {factor}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Model input requirements:")
            if hasattr(model, 'input'):
                st.json([inp.shape for inp in model.input])
            else:
                st.json(getattr(model, "input_shape", "n/a"))
            st.write("What you provided:")
            if isinstance(inputs, list):
                st.json([getattr(x, "shape", None) for x in inputs])
            else:
                st.json(getattr(inputs, "shape", None))

def show_data_exploration_page():
    import pandas as pd, numpy as np
    import matplotlib.pyplot as plt, seaborn as sns
    from pathlib import Path

    section_header("Data Exploration")

    # ---------- Daten laden (Upload + Fallback + Cache) ----------
    @st.cache_data(show_spinner=False)
    def _load_csv(path: str):
        return pd.read_csv(path)

    def _first_existing(relpaths):
        roots = [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
        for root in roots:
            for rel in relpaths:
                p = (root / rel).resolve()
                if p.exists():
                    return p.as_posix()
        return None

    up = st.file_uploader("Upload CSV (optional)", type=["csv"])
    if up:
        df = pd.read_csv(up)
        source = "uploaded file"
    else:
        fallback = _first_existing([
            "data/temu_reviews.csv",
            "src/data/temu_reviews.csv",
            "src/data/processed/temu_reviews_preprocessed.csv",
            "data/processed/temu_reviews_preprocessed.csv",
        ])
        if not fallback:
            st.error("No dataset found. Upload a CSV or place it under data/ or src/data/.")
            return
        df = _load_csv(fallback)
        source = fallback

    st.caption(f"Loaded: {source}")
    if df.empty:
        st.warning("Dataset is empty."); return

    # ---------- Grundinfos ----------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Users", f"{df['UserId'].nunique():,}" if 'UserId' in df else "–")
    c3.metric("Countries", f"{df['UserCountry'].nunique():,}" if 'UserCountry' in df else "–")
    c4.metric("Features", f"{len(df.columns)}")

    st.dataframe(df.head(10), use_container_width=True)

    # ---------- Filter ----------
    with st.expander("Filters", expanded=True):
        stars = sorted(df['ReviewRating'].dropna().unique()) if 'ReviewRating' in df else []
        star_sel = st.multiselect("Star ratings", stars, default=stars)
        top_countries = (df['UserCountry'].value_counts().head(25).index.tolist()
                         if 'UserCountry' in df else [])
        country_sel = st.multiselect("Countries (top 25 shown)", top_countries, default=top_countries)

        # Datumsfilter
        from datetime import timedelta

        if 'ReviewDate' in df:
            # Timestamps als UTC; NaT raus
            d = pd.to_datetime(df['ReviewDate'], errors="coerce", utc=True)
            d_valid = d.dropna()
            if not d_valid.empty:
                # -> Python datetime (tz-aware) für den Slider
                dmin_py = d_valid.min().to_pydatetime()
                dmax_py = d_valid.max().to_pydatetime()
                r = st.slider(
                    "Date range",
                    min_value=dmin_py,
                    max_value=dmax_py,
                    value=(dmin_py, dmax_py),
                    step=timedelta(days=1),
                    format="YYYY-MM-DD",
                )
            else:
                r = None
        else:
            r = None

    # Filter anwenden
    dfv = df.copy()
    if 'ReviewRating' in dfv and star_sel:
        dfv = dfv[dfv['ReviewRating'].isin(star_sel)]
    if 'UserCountry' in dfv and country_sel:
        dfv = dfv[dfv['UserCountry'].isin(country_sel)]
    if r and 'ReviewDate' in dfv:
        d = pd.to_datetime(dfv['ReviewDate'], errors="coerce", utc=True)
        start, end = r  # Python datetime (mit tzinfo=UTC)
        # in pandas Timestamps (tz-aware) umwandeln
        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)
        dfv = dfv[(d >= start_ts) & (d <= end_ts)]

    st.caption(f"Filtered rows: {len(dfv):,}")

    # ---------- Tabs ----------
    tab1, tab2, tab3, tab4 = st.tabs(["Target distribution", "Lengths", "Geography & Time", "Quality"])

    # === Tab 1: Ratings ===
    with tab1:
        if 'ReviewRating' not in dfv:
            st.info("Column `ReviewRating` missing."); 
        else:
            counts = dfv['ReviewRating'].value_counts().sort_index()
            # Bar
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(counts.index, counts.values, edgecolor="black")
            ax.set_xlabel("Star"); ax.set_ylabel("Count"); ax.set_title("Distribution of Review Ratings")
            ax.set_xticks(range(int(counts.index.min()), int(counts.index.max())+1))
            st.pyplot(fig); plt.close(fig)

            # Pie
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
            ax2.set_title("Rating Distribution (%)")
            st.pyplot(fig2); plt.close(fig2)

    # === Tab 2: Lengths ===
    with tab2:
        if 'ReviewText' in dfv:
            dfv['review_length'] = dfv['ReviewText'].astype(str).str.len()
            c1, c2 = st.columns(2)
            with c1:
                st.write(dfv['review_length'].describe())
            with c2:
                if 'ReviewRating' in dfv:
                    fig, ax = plt.subplots(figsize=(6,4))
                    sns.boxplot(data=dfv, x='ReviewRating', y='review_length', ax=ax)
                    ax.set_title("Review Length vs Rating"); ax.set_xlabel("Star"); ax.set_ylabel("Length (chars)")
                    st.pyplot(fig); plt.close(fig)
            # Histogram
            fig3, ax3 = plt.subplots(figsize=(6,4))
            ax3.hist(dfv['review_length'], bins=40)
            ax3.set_title("Histogram of review lengths"); ax3.set_xlabel("Length"); ax3.set_ylabel("Count")
            st.pyplot(fig3); plt.close(fig3)
        else:
            st.info("Column `ReviewText` missing.")

    # === Tab 3: Geo & Time ===
    with tab3:
        if 'UserCountry' in dfv:
            top = dfv['UserCountry'].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(7,4))
            ax.bar(top.index, top.values)
            ax.set_title("Top countries by number of reviews"); ax.set_xlabel("Country"); ax.set_ylabel("Count")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig); plt.close(fig)

        if 'ReviewDate' in dfv and 'ReviewRating' in dfv:
            d = pd.to_datetime(dfv['ReviewDate'], errors="coerce", utc=True)
            # Zeitzone Berlin korrekt: tz_aware → convert, dann naive für Gruppierung
            d_ber = d.dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
            dfv2 = dfv.copy(); dfv2['ReviewDate'] = d_ber
            dfv2['month'] = dfv2['ReviewDate'].dt.to_period('M').astype(str)
            monthly = dfv2.groupby('month')['ReviewRating'].mean()
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(monthly.index, monthly.values, marker='o'); ax.set_title("Average rating by month")
            ax.set_xlabel("Month"); ax.set_ylabel("Avg rating"); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            ax.grid(True, alpha=.3); st.pyplot(fig); plt.close(fig)

            # Heatmap Day x Hour
            dfv2['day'] = dfv2['ReviewDate'].dt.day_name()
            dfv2['hour'] = dfv2['ReviewDate'].dt.hour
            order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            heat = dfv2.pivot_table(index='day', columns='hour', values='ReviewRating', aggfunc='mean').reindex(order)
            fig, ax = plt.subplots(figsize=(7,4))
            sns.heatmap(heat, cmap='RdYlBu_r', center=3, ax=ax)
            ax.set_title("Rating heatmap by day & hour"); st.pyplot(fig); plt.close(fig)

    # === Tab 4: Quality ===
    with tab4:
        issues = []
        if 'UserId' in dfv and 'ReviewText' in dfv:
            duplicates = dfv.duplicated(subset=['UserId','ReviewText']).sum()
            issues.append(("Duplicate reviews", duplicates))
        missing_text = dfv['ReviewText'].isna().sum() if 'ReviewText' in dfv else 0
        issues.append(("Missing review text", missing_text))
        invalid = dfv[~dfv.get('ReviewRating', pd.Series(dtype=int)).isin([1,2,3,4,5])].shape[0] if 'ReviewRating' in dfv else 0
        issues.append(("Invalid ratings (not 1-5)", invalid))

        st.table(pd.DataFrame(issues, columns=["Check","Count"]))


def show_results():
    import os, json
    from pathlib import Path

    section_header("Model Evaluation Results")

    # --- robustes Root-Finder
    def _first_across_roots(relpaths):
        roots = []
        if "ROOT_CANDIDATES" in globals():
            roots += list(ROOT_CANDIDATES)
        here = Path(__file__).resolve().parent
        roots += [Path.cwd(), here, here.parent]
        for root in roots:
            for rel in relpaths:
                p = (Path(root) / rel).resolve()
                if p.exists():
                    return p.as_posix()
        return None

    # =======================
    # 1) "Classical DL" RESULTS (dein gespeichertes model_results.pkl)
    # =======================
    try:
        pkl_path = _first_across_roots([
            "data/model_results.pkl",
            "results/model_results.pkl",
            "models/model_results.pkl",
            "src/models/model_results.pkl",
        ])

        if pkl_path:
            with open(pkl_path, "rb") as f:
                all_results = pickle.load(f)

            # --- y_test robust laden (NPZ oder Fallback aus all_results)
            def _load_y_test(all_results):
                npz_path = _first_across_roots([
                    "data/preprocessed_data.npz",
                    "results/preprocessed_data.npz",
                    "models/preprocessed_data.npz",
                    "src/data/preprocessed_data.npz",
                    "src/models/preprocessed_data.npz",
                ])
                if npz_path:
                    try:
                        with np.load(npz_path, allow_pickle=True) as z:
                            for k in ("y_test", "y_true", "labels", "y"):
                                if k in z:
                                    y = np.asarray(z[k]).astype(int).ravel()
                                    if y.size:
                                        return y
                    except Exception as e:
                        st.info(f"Could not read NPZ at {npz_path}: {e}")

                if isinstance(all_results, (list, tuple)):
                    for r in all_results:
                        if not isinstance(r, dict):
                            continue
                        for k in ("y_test", "y_true", "labels", "y"):
                            if k in r:
                                try:
                                    y = np.asarray(r[k]).astype(int).ravel()
                                    if y.size:
                                        return y
                                except Exception:
                                    pass
                return None

            y_test = _load_y_test(all_results)
            has_roc_data = y_test is not None

            class_names = ["1","2","3","4","5"]

            st.subheader("Performance Summary")

            # --- Tabelle robust bauen
            summary_data = [{
                "Model": r.get("model_name"),
                "Accuracy": r.get("accuracy"),
                "F1-Weighted": r.get("f1_weighted"),
                "F1-Macro": r.get("f1_macro"),
                "AUC Score": r.get("auc_score"),
            } for r in all_results]
            summary_df = pd.DataFrame(summary_data)

            metric_cols = ["Accuracy","F1-Weighted","F1-Macro","AUC Score"]
            for c in metric_cols:
                if c in summary_df.columns:
                    summary_df[c] = pd.to_numeric(summary_df[c], errors="coerce")

            sort_col = next((c for c in ["F1-Weighted","Accuracy","F1-Macro","AUC Score"]
                             if c in summary_df.columns and summary_df[c].notna().any()), None)
            if sort_col:
                summary_df = summary_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

            fmt = {c: (lambda v: "" if pd.isna(v) else f"{v:.3f}")
                   for c in metric_cols if c in summary_df.columns}
            st.dataframe(summary_df.style.format(fmt), use_container_width=True)

            # --- Balkendiagramm
            st.subheader("Performance Metrics Comparison (DL)")
            metrics = [c for c in metric_cols if c in summary_df.columns and summary_df[c].notna().any()]
            if metrics:
                fig1, ax1 = plt.subplots(figsize=(12,6))
                x = np.arange(len(summary_df)); width = 0.8/max(1,len(metrics))
                plot_df = summary_df.copy()
                plot_df[metrics] = plot_df[metrics].fillna(0.0)
                for i, metric in enumerate(metrics):
                    ax1.bar(x + i*width, plot_df[metric], width, label=metric, alpha=0.85)
                ax1.set_xlabel('Models'); ax1.set_ylabel('Score'); ax1.set_title('Model Performance Comparison (DL)')
                ax1.set_xticks(x + width*(len(metrics)-1)/2); ax1.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
                ax1.legend(); ax1.grid(True, alpha=0.3)
                st.pyplot(fig1); plt.close(fig1)

            # --- Confusion Matrices
            st.subheader("Confusion Matrices (DL)")
            n_models = len(all_results)
            fig2, axes = plt.subplots(1, n_models, figsize=(6*n_models,5))
            if n_models == 1: axes = [axes]
            for ax, result in zip(axes, all_results):
                cm = result['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names, ax=ax)
                ax.set_title(result['model_name']); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
            st.pyplot(fig2); plt.close(fig2)

            # --- ROC
            st.subheader("ROC Curves Comparison (DL)")

            # 1) Versuch: PNG finden/anzeigen
            roc_img = _first_across_roots([
                "charts/micro-average_roc_curve_comparison.png",
                "api_models/charts/micro-average_roc_curve_comparison.png",
                "results/micro-average_roc_curve_comparison.png",
                "charts/micro-average-roc-curve-comparison.png",
                "api_models/charts/micro-average-roc-curve-comparison.png",
                "results/micro-average-roc-curve-comparison.png",
            ])
            if roc_img:
                st.image(roc_img, caption="Micro-average ROC curve comparison", use_container_width=True)
            else:
                # 2) Fallback: wie bisher berechnen/plotten (nur wenn Labels vorhanden)
                if has_roc_data:
                    try:
                        classes = sorted(np.unique(y_test).tolist())
                        y_true_bin = label_binarize(y_test, classes=classes)
                        models_with_proba = [r for r in all_results if isinstance(r, dict) and 'y_pred_proba' in r]

                        if models_with_proba:
                            fig3 = plt.figure(figsize=(10, 8))
                            drew_any = False
                            for r in models_with_proba:
                                proba = np.asarray(r['y_pred_proba'])
                                if proba.ndim == 1:
                                    continue
                                n = min(len(y_test), proba.shape[0])
                                if n == 0:
                                    continue
                                proba = proba[:n, :len(classes)]
                                y_bin_n = y_true_bin[:n, :len(classes)]
                                fpr, tpr, _ = roc_curve(y_bin_n.ravel(), proba.ravel())
                                roc_auc = auc(fpr, tpr)
                                plt.plot(fpr, tpr, label=f"{r.get('model_name','model')} (AUC = {roc_auc:.2f})")
                                drew_any = True

                            if drew_any:
                                plt.plot([0, 1], [0, 1], 'k--')
                                plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
                                plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
                                plt.title('Micro-average ROC Curve Comparison (DL)')
                                plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
                                st.pyplot(fig3); plt.close(fig3)
                            else:
                                st.info("No usable probability arrays found to draw ROC. "
                                        "Add the PNG under api_models/charts/ to show it here.")
                        else:
                            st.info("No models with prediction probabilities; place the ROC PNG under /charts/ to display it.")
                    except Exception as e:
                        st.warning(f"Could not generate ROC curves (DL): {e}")
                else:
                    st.info("Test labels not found; place the ROC PNG under /charts/ to display it.")
        else:
            st.info("No DL pickle found (model_results.pkl). Skipping DL details.")
    except Exception as e:
        st.error(f"Error loading DL results: {e}")

def get_intro_image():
    # Ordner der aktuellen Datei (Combined_Streamlit.py)
    here = Path(__file__).resolve().parent
    candidates = [
        here / "satisfied-customer.png",
        here / "assets" / "satisfied-customer.png",
        here.parent / "satisfied-customer.png",
        here.parent / "assets" / "satisfied-customer.png",
        Path.cwd() / "satisfied-customer.png",  # fallback
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def show_intro_page():
    section_header("Customer Satisfaction from Reviews (Temu)", "📦")

    # small centered image (optional)
    import os



    st.markdown("""
### Introduction
In today’s digital marketplace, customer reviews are more than opinions—they are a critical source of insight for businesses.  
Reading thousands of free-text reviews manually is slow and inconsistent, which makes it hard to spot recurring issues such as delivery delays, poor packaging, or product quality problems.

Our project addresses this by building a machine-learning app that predicts **1–5 star ratings** from review text. Using ~**14k** English-language Temu reviews (from Trustpilot), we compared **classical ML** and **deep-learning** models. The system not only predicts a star rating, it also groups reviews into **positive / neutral / negative** for faster triage.

The result is a tool that turns raw feedback into **actionable monitoring** for supply-chain and logistics teams—helping them respond faster and improve customer satisfaction.
""")

    st.markdown("""
### Web scraping – data collection
We built a robust, polite scraper to collect high-quality Trustpilot reviews for **Temu**, **AliExpress**, and **Wish**.

**What we collect (per review)**
- User (anonymized) and **country**
- **Star rating**, review **title** and **text**
- **Timestamps** and **company replies**

**Tech stack**
- `requests` – reliable HTTP requests  
- `beautifulsoup4` – fast HTML parsing  
- `pandas` – structured CSV output

**Reliability & quality controls**
- **Dynamic targeting**: switch between `temu`, `aliexpress`, `wish`
- **Text cleaning & date normalization** (e.g. “2 days ago” → ISO date)
- **Polite rate limiting** (configurable delays / max pages)
- **Comprehensive logging** for progress & errors

Output is a clean, analysis-ready CSV (e.g. `temu_reviews.csv`) used throughout this app.
""")

    st.markdown("""
### Dataset & scope
We first scraped AliExpress (~57k) and Wish (~99k), but training at that scale was impractical on local hardware.  
On our mentor’s advice that ~10k observations are sufficient, we pivoted to **Temu** (~14k). Temu’s ratings are strongly **polarized** (many **1★** and **5★**), which is ideal for testing **class-imbalance** strategies.
""")

    st.markdown("""
### Objectives
1. Scrape, clean, and explore the Temu review corpus.  
2. Predict **1–5 star ratings from free text** (multi-class classification).  
3. Compare baseline ML vs. deep models under class imbalance.  
4. Ship a **live demo** that returns a rating and sentiment group for any new text.
""")

    st.markdown("""
### Design decisions
- Moved from **regression to classification** to align with star labels and business usage.  
- Explored automated **reply templates** driven by predicted sentiment (interesting, but out of scope for this app).  
- Favored **transparent features** (TF-IDF + light text signals) with a champion model and robust fallbacks.
""")

    st.markdown("""
### Why it matters for supply chain
The pipeline highlights issue clusters (e.g., *late delivery*, *poor packaging*, *not as described*) and turns raw feedback into **real-time monitoring** for quality and logistics—helping teams prioritize fixes that raise satisfaction.
""")

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews (Temu)", "≈ 14,000")
    c2.metric("Target", "1–5 Stars")
    c3.metric("Demo", "Real-time prediction")

    img_path = get_intro_image()
    if img_path:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(str(img_path), width=360)
            # st.caption(f"Loaded image: {img_path.name}")  # optional zum Debuggen
    else:
        st.caption("Image not found. Place 'satisfied-customer.png' next to Combined_Streamlit.py or in an 'assets/' folder.")

def show_conclusion_page():
    section_header("Conclusion")
    st.header("Supply Chain Sentiment AI Application")

    st.markdown("""
    **Advanced Model Benchmarking:** The selection of the final classification model was determined through a structured 
        empirical evaluation of an ensemble of 18 candidate algorithms, comprising both traditional machine learning and
          modern deep learning architectures.

    - **Machine Learning (11 Models):** Implemented and tuned a wide range of classical algorithms:
        - Stacking
        - LinearSVC
        - Logistic Regression
        - Hard Voting
        - Soft Voting
        - XGBoost
        - Gradient Boosting
        - Random Forest
        - K-Nearest Neighbours
        - Gaussian Naive Bayes
        - Decision Tree) 
        using TF-IDF feature extraction.

    - **Deep Learning (6 Models):** Experimented with state-of-the-art neural networks, almost certainly including pre-trained transformer models :
        - Deep MLP with TF-IDF 
        - LSTM Model 
        - BiLSTM with Attention 
        - CNN Model 
        - Transformer Model 
        - Hybrid CNN-LSTM 
     for their superior context understanding, alongside other architectures like LSTMs.

    **Data-Driven Deployment:** The final application is powered by the champion model from this rigorous testing process,
                                ensuring high-accuracy sentiment classification for supply chain terminology.

    **Accessible Interface:** The complex AI backend is delivered through a user-friendly, web-based interface built with 
                                    Streamlit, making advanced analytics accessible to non-technical domain experts.

    **Outcome:** Provides a reliable, automated system for monitoring market sentiment, turning news analysis from a manual
                                    chore into a scalable, real-time strategic function.
    """)

    st.subheader("Model Comparison: Key Considerations")

    st.markdown("""
    *   **Cost & Complexity:** DL models demand significant computational resources (GPUs) and time, while ML models are faster and cheaper to train and run.
    *   **Data Hunger:** DL requires vast amounts of data to excel, whereas ML can deliver strong results with smaller, well-structured datasets.
    *   **Business Reality:** The "best" model is a trade-off. Performance gains must be weighed against explainability, operational costs, and maintenance complexity.
    """)

    st.markdown("---")
    with st.expander("Acknowledgements"):
        st.markdown("""
        We would like to express our gratitude to:

        *   **DataScientest** for providing the platform and knowledge for this project.
        *   **Mr. Kylian**, our tutor, for his expert advice, insightful feedback, and constant availability which were crucial to our success.
        """)
    #import plotly.express as px
    #comparison_data = {
    #    'Model': ['LinearSVC (ML)', 'Random Forest (ML)', 'DistilBERT (DL)', 'LSTM (DL)'],
    #    'Accuracy': [0.89, 0.87, 0.92, 0.85],
    #    'F1-Score': [0.89, 0.87, 0.92, 0.85],
    #    'Training Time (min)': [2, 10, 240, 120],
    #    'Interpretability': ['High', 'Medium', 'Low', 'Low']
    #}
    #df = pd.DataFrame(comparison_data)

    #st.header("Model Comparison: Machine Learning vs. Deep Learning")

    # Metrics Table
    #st.subheader("Performance Metrics")
    #st.dataframe(df.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score'], color='lightgreen'),
    #             use_container_width=True)

    # Bar Chart for Accuracy
    #st.subheader("Accuracy Comparison")
    #fig = px.bar(df, x='Model', y='Accuracy', title='Model Accuracy', color='Model')
    #st.plotly_chart(fig)


def show_about_page():
    section_header("About this App", "ℹ️")

    st.markdown("""
**What it does**  
This app predicts customer satisfaction (**1–5 stars**) from review text. It also groups predictions into **negative / neutral / positive** for fast triage.

**How it works**
- **Preprocessing:** light text cleaning, optional emoji mapping, basic text features (length, punctuation, capitalization), optional VADER sentiment.
- **Models:** classical ML with TF-IDF + numeric features (champion with robust fallbacks) and deep-learning variants (e.g., LSTM, BiLSTM-Attention, CNN, Transformer, Deep MLP with TF-IDF).
- **UI:** load data → explore → preprocess → inspect artifacts → compare models → evaluate → live prediction.

**Why it’s useful**
- Surfaces supply-chain issues in real time (delivery, packaging, quality, pricing).
- Reduces manual reading; standardizes decisions under class imbalance.

**Available model families**
- Deep MLP (TF-IDF), LSTM, BiLSTM-Attention, CNN, Transformer, Hybrid CNN-LSTM  
- Classical baselines (e.g., Logistic Regression / Linear SVC / Stacking)

**Data source**
- English-language Temu reviews from Trustpilot (~14k)

**Team**
- Frank · Sebastian · Mohamed – DataScientest Project
""")


def main():
    # Artefakt-Suche einmal pro Session cachen
    if "art" not in st.session_state:
        st.session_state.art = find_artifacts(PATH_MODELS)

    # Handler je Seite
    HANDLERS = {
        "about":      show_about_page,
        "intro":      show_intro_page,
        "load":       show_load_page,
        "dataexp":    show_data_exploration_page,
        "preprocess": show_preprocess_page,
        "features":   show_feature_engineering_page,
        "compare":    show_compare_models,
        "eval100":    show_100_sample_evaluation_page,
        "live_ml":    show_live_prediction_ml_page,
        "live_dl":    show_live_prediction_DL_page,
        "results":    show_results,
        "conclusion": show_conclusion_page,
    }

    # ausgewählte Seite ausführen
    HANDLERS.get(selected_key, show_intro_page)()


if __name__ == "__main__":
    main()