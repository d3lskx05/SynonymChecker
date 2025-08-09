import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
import json
import tempfile
import os
from typing import List, Tuple, Any

from sentence_transformers import SentenceTransformer, util
import altair as alt

# --------------------
# Утилиты
# --------------------

def preprocess_text(t: Any) -> str:
    if pd.isna(t):
        return ""
    return " ".join(str(t).lower().strip().split())

def file_md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

def read_uploaded_file_bytes(uploaded) -> Tuple[pd.DataFrame, str]:
    raw = uploaded.read()
    h = file_md5(raw)
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        try:
            df = pd.read_excel(io.BytesIO(raw))
        except Exception as e:
            raise ValueError("Файл должен быть CSV или Excel. Ошибка: " + str(e))
    return df, h

def parse_topics_field(val) -> List[str]:
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    for sep in [";", "|", ","]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            return parts
    return [s] if s else []

def jaccard_tokens(a: str, b: str) -> float:
    sa = set([t for t in a.split() if t])
    sb = set([t for t in b.split() if t])
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0

def style_low_score_rows(df, threshold=0.75):
    def highlight(row):
        return ['background-color: #ffcccc' if (pd.notna(row['score']) and row['score'] < threshold) else '' for _ in row]
    return df.style.apply(highlight, axis=1)

def download_and_extract_gdrive_model(file_id: str) -> str:
    import gdown
    import zipfile
    import tarfile

    tmp_dir = tempfile.gettempdir()
    archive_path = os.path.join(tmp_dir, f"model_gdrive_{file_id}")

    # Скачиваем файл, если его нет
    if not os.path.exists(archive_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, archive_path, quiet=True)

    # Папка для распаковки
    extract_path = os.path.join(tmp_dir, f"model_gdrive_extracted_{file_id}")
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        # Проверяем, архив ли это
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_path)
        else:
            # Не архив — возвращаем путь к файлу, предположим, что это папка модели
            return archive_path
    return extract_path

@st.cache_resource(show_spinner=False)
def load_model_from_source(source: str, identifier: str) -> SentenceTransformer:
    if source == "huggingface":
        model_path = identifier
    elif source == "google_drive":
        model_path = download_and_extract_gdrive_model(identifier)
    else:
        raise ValueError("Unknown model source")
    model = SentenceTransformer(model_path)
    return model

def encode_texts_in_batches(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.array([])
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embs)

# --------------------
# Streamlit UI
# --------------------

st.set_page_config(page_title="Synonym Checker (with A/B, History)", layout="wide")
st.title("🔎 Synonym Checker — с выбором модели, историей, A/B тестом и визуализацией")

st.sidebar.header("Настройки модели")

model_source = st.sidebar.selectbox("Источник модели", ["huggingface", "google_drive"], index=0)
if model_source == "huggingface":
    model_id = st.sidebar.text_input("Hugging Face Model ID", value="fine_tuned_model")
else:
    model_id = st.sidebar.text_input("Google Drive File ID", value="1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf")

enable_ab_test = st.sidebar.checkbox("Включить A/B тест двух моделей", value=False)
if enable_ab_test:
    ab_model_source = st.sidebar.selectbox("Источник второй модели", ["huggingface", "google_drive"], index=0, key="ab_source")
    if ab_model_source == "huggingface":
        ab_model_id = st.sidebar.text_input("Hugging Face Model ID (B)", value="all-mpnet-base-v2", key="ab_id")
    else:
        ab_model_id = st.sidebar.text_input("Google Drive File ID (B)", value="", key="ab_id")
else:
    ab_model_id = None

batch_size = st.sidebar.number_input("Batch size для энкодинга", min_value=8, max_value=1024, value=64, step=8)

try:
    with st.spinner("Загружаю основную модель..."):
        model_a = load_model_from_source(model_source, model_id)
    st.sidebar.success("Основная модель загружена")
except Exception as e:
    st.sidebar.error(f"Не удалось загрузить основную модель: {e}")
    st.stop()

model_b = None
if enable_ab_test and ab_model_id and ab_model_id.strip() != "":
    try:
        with st.spinner("Загружаю модель B..."):
            model_b = load_model_from_source(ab_model_source, ab_model_id)
        st.sidebar.success("Модель B загружена")
    except Exception as e:
        st.sidebar.error(f"Не удалось загрузить модель B: {e}")
        st.stop()

if "history" not in st.session_state:
    st.session_state["history"] = []

def add_to_history(record: dict):
    st.session_state["history"].append(record)

def clear_history():
    st.session_state["history"] = []

st.sidebar.header("История проверок")
if st.sidebar.button("Очистить историю"):
    clear_history()

if st.session_state["history"]:
    history_bytes = json.dumps(st.session_state["history"], indent=2, ensure_ascii=False).encode('utf-8')
    st.sidebar.download_button(
        label="Скачать историю в JSON",
        data=history_bytes,
        file_name="history.json",
        mime="application/json"
    )
else:
    st.sidebar.info("История пустая")

st.header("1. Загрузите CSV или Excel с колонками: phrase_1, phrase_2, topics (опционально)")

uploaded_file = st.file_uploader("Выберите файл с парами фраз", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df, file_hash = read_uploaded_file_bytes(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка чтения файла: {e}")
        st.stop()

    required_cols = {"phrase_1", "phrase_2"}
    if not required_cols.issubset(set(df.columns)):
        st.error(f"Файл должен содержать колонки: {required_cols}")
        st.stop()

    df["phrase_1"] = df["phrase_1"].map(preprocess_text)
    df["phrase_2"] = df["phrase_2"].map(preprocess_text)
    if "topics" in df.columns:
        df["topics_list"] = df["topics"].map(parse_topics_field)
    else:
        df["topics_list"] = [[] for _ in range(len(df))]

    phrases_all = list(set(df["phrase_1"].tolist() + df["phrase_2"].tolist()))
    phrase2idx = {p: i for i, p in enumerate(phrases_all)}

    with st.spinner("Кодирую фразы моделью A..."):
        embeddings_a = encode_texts_in_batches(model_a, phrases_all, batch_size)

    embeddings_b = None
    if enable_ab_test and model_b is not None:
        with st.spinner("Кодирую фразы моделью B..."):
            embeddings_b = encode_texts_in_batches(model_b, phrases_all, batch_size)

    scores = []
    scores_b = []
    lexical_scores = []
    for idx, row in df.iterrows():
        p1 = row["phrase_1"]
        p2 = row["phrase_2"]
        emb1_a = embeddings_a[phrase2idx[p1]]
        emb2_a = embeddings_a[phrase2idx[p2]]
        score_a = float(util.cos_sim(emb1_a, emb2_a).item())
        scores.append(score_a)

        if embeddings_b is not None:
            emb1_b = embeddings_b[phrase2idx[p1]]
            emb2_b = embeddings_b[phrase2idx[p2]]
            score_b = float(util.cos_sim(emb1_b, emb2_b).item())
            scores_b.append(score_b)

        lex_score = jaccard_tokens(p1, p2)
        lexical_scores.append(lex_score)

    df["score"] = scores
    if embeddings_b is not None:
        df["score_b"] = scores_b
    df["lexical_score"] = lexical_scores

    highlight_threshold = st.slider("Порог подсветки низкой схожести (score)", min_value=0.0, max_value=1.0, value=0.75)

    st.subheader("Результаты проверки пар")

    result_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Скачать результаты CSV", data=result_csv, file_name="results.csv", mime="text/csv")

    styled_df = style_low_score_rows(df, threshold=highlight_threshold)
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("Гистограмма распределения similarity score (модель A)")
    chart = alt.Chart(pd.DataFrame({"score": df["score"]})).mark_bar().encode(
        alt.X("score:Q", bin=alt.Bin(maxbins=30), title="Cosine similarity score"),
        y='count()', tooltip=['count()']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    if embeddings_b is not None:
        st.subheader("Гистограмма распределения similarity score (модель B)")
        chart_b = alt.Chart(pd.DataFrame({"score_b": df["score_b"]})).mark_bar().encode(
            alt.X("score_b:Q", bin=alt.Bin(maxbins=30), title="Cosine similarity score (B)"),
            y='count()', tooltip=['count()']
        ).interactive()
        st.altair_chart(chart_b, use_container_width=True)

    if st.button("Сохранить результаты проверки в историю"):
        record = {
            "file_hash": file_hash,
            "file_name": uploaded_file.name,
            "results": df.to_dict(orient="records"),
            "model_a": model_id,
            "model_b": ab_model_id if enable_ab_test else None,
            "timestamp": pd.Timestamp.now().isoformat(),
            "highlight_threshold": highlight_threshold,
        }
        add_to_history(record)
        st.success("Результаты сохранены в историю.")

else:
    st.info("Загрузите файл для начала проверки.")

if st.session_state["history"]:
    st.header("История проверок")
    for idx, rec in enumerate(reversed(st.session_state["history"])):
        st.markdown(f"### Проверка #{len(st.session_state['history']) - idx}")
        st.markdown(f"**Файл:** {rec.get('file_name','-')}  |  **Дата:** {rec.get
