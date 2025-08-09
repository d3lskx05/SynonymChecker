import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
import json
from typing import List, Tuple, Dict, Any

from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
import torch

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
    """Прочитать CSV или Excel из streamlit uploader и вернуть DataFrame + md5 хэш содержимого."""
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
    """Преобразует поле topics в список строк.
    Поддерживает: NaN -> [], JSON-список -> parsed, строка с разделителем , ; | -> split
    """
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    # Попробуем как JSON
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    # Иначе split по разделителям
    for sep in [";", "|", ","]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep) if p.strip()]
            return parts
    # одиночная тема
    return [s] if s else []


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    return SentenceTransformer(model_path)


def encode_texts_in_batches(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Энкодим список текстов и возвращаем numpy array (CPU).
    Используем SentenceTransformer.encode с convert_to_numpy=True.
    """
    if not texts:
        return np.array([])
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embs)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def jaccard_tokens(a: str, b: str) -> float:
    sa = set([t for t in a.split() if t])
    sb = set([t for t in b.split() if t])
    if not sa and not sb:
        return 0.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / len(union) if union else 0.0


def dedupe_pairs(pairs: List[Tuple[int, int, float, float]]) -> List[Tuple[int, int, float, float]]:
    """Убираем зеркальные дубликаты (i,j) и (j,i) — оставляем первую встречную пару."""
    best = {}
    for i, j, mscore, lex in pairs:
        key = tuple(sorted((i, j)))
        if key not in best:
            best[key] = (i, j, mscore, lex)
    return list(best.values())


# --------------------
# UI и параметры
# --------------------
st.set_page_config(page_title="Synonym Checker (with topics)", layout="wide")
st.title("🔎 Synonym Checker — Ручной и Автоматический режимы (с поддержкой topics)")

# Sidebar settings
st.sidebar.header("Настройки модели и обучения")
model_path = st.sidebar.text_input("Путь или имя модели (SentenceTransformer)", value="fine_tuned_model")
output_model_path = st.sidebar.text_input("Куда сохранять дообученную модель (папка)", value="fine_tuned_model_updated")
batch_size = st.sidebar.number_input("Batch size для энкодинга", min_value=8, max_value=1024, value=64, step=8)
train_batch_size = st.sidebar.number_input("Batch size для дообучения", min_value=4, max_value=64, value=8, step=1)
train_epochs = st.sidebar.number_input("Эпохи для дообучения", min_value=1, max_value=10, value=1, step=1)
warmup_steps = st.sidebar.number_input("Warmup steps (train)", min_value=0, max_value=1000, value=10, step=1)
DEFAULT_MAX_ROWS = st.sidebar.number_input("Максимум строк (auto mode)", min_value=100, max_value=20000, value=5000, step=100)

# Загрузка модели
try:
    with st.spinner("Загружаю модель..."):
        model = load_model_cached(model_path)
    st.sidebar.success("Модель загружена")
except Exception as e:
    st.sidebar.error(f"Не удалось загрузить модель: {e}")
    st.stop()

# session_state cache для эмбеддингов (по хэшу файла)
if "emb_cache" not in st.session_state:
    st.session_state["emb_cache"] = {}

# Основные вкладки
tab_manual, tab_auto = st.tabs(["🔧 Ручной режим (CSV/Excel пар)", "🤖 Автоматический режим (корпус фраз)"])

# ----------------------
# Ручной режим
# ----------------------
with tab_manual:
    st.header("🔧 Ручной режим — загрузите CSV/XLSX с колонками: phrase, synonym (опционально: topics)")
    uploaded = st.file_uploader("Файл с парами (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="manual_upl")

    if uploaded is not None:
        try:
            df_pairs, file_hash = read_uploaded_file_bytes(uploaded)
        except Exception as e:
            st.error(str(e))
            st.stop()

        # нормализуем имена колонок (храним mapping оригинального имени)
        cols_lower = {c.lower(): c for c in df_pairs.columns}
        if "phrase" not in cols_lower or "synonym" not in cols_lower:
            st.error("Файл должен содержать колонки: phrase и synonym (регистр букв не важен)")
            st.stop()

        phrase_col = cols_lower["phrase"]
        synonym_col = cols_lower["synonym"]
        topics_col = cols_lower.get("topics")  # может быть None

        # preprocess
        df_pairs = df_pairs.copy()
        df_pairs["phrase_proc"] = df_pairs[phrase_col].apply(preprocess_text)
        df_pairs["syn_proc"] = df_pairs[synonym_col].apply(preprocess_text)
        if topics_col:
            df_pairs["topics"] = df_pairs[topics_col].apply(parse_topics_field)
            # уникальные темы
            all_topics = sorted({t for ts in df_pairs["topics"] for t in ts})
            selected_topics = st.multiselect("Фильтр по тематикам (manual)", all_topics)
            if selected_topics:
                df_pairs = df_pairs[df_pairs["topics"].apply(lambda ts: bool(set(ts) & set(selected_topics)))].reset_index(drop=True)
        else:
            df_pairs["topics"] = [[] for _ in range(len(df_pairs))]
            selected_topics = []

        if df_pairs.empty:
            st.warning("Нет пар после фильтрации по темам (или файл пуст).")
            st.stop()

        # энкодим уникальные тексты
        unique_texts = list({t for t in pd.concat([df_pairs["phrase_proc"], df_pairs["syn_proc"]]) if pd.notna(t) and t})
        st.info(f"Уникальных строк для энкодинга: {len(unique_texts)}")

        cache_key = f"manual_{file_hash}_{model_path}_{batch_size}"
        if cache_key in st.session_state["emb_cache"]:
            emb_map = st.session_state["emb_cache"][cache_key]["map"]
            st.success("Эмбеддинги загружены из кэша")
        else:
            with st.spinner("Энкодим уникальные фразы..."):
                emb_arr = encode_texts_in_batches(model, unique_texts, batch_size=int(batch_size))
                emb_map = {text: emb for text, emb in zip(unique_texts, emb_arr)}
            st.session_state["emb_cache"][cache_key] = {"map": emb_map}
            st.success("Эмбеддинги вычислены")

        # вычисление сходства
        results = []
        for _, row in df_pairs.iterrows():
            a_proc = row["phrase_proc"]
            b_proc = row["syn_proc"]
            emb_a = emb_map.get(a_proc)
            emb_b = emb_map.get(b_proc)
            score = cos_sim(emb_a, emb_b) if emb_a is not None and emb_b is not None else None
            results.append({
                "phrase": row[phrase_col],
                "synonym": row[synonym_col],
                "topics": row["topics"],
                "score": score
            })

        res_df = pd.DataFrame(results).sort_values(by="score", ascending=True, na_position="last")
        st.subheader("Результаты (manual)")
        st.dataframe(res_df, use_container_width=True)

        cut = st.slider("Порог: считать пару слабой, если сходство < ", 0.0, 1.0, 0.75, 0.01, key="manual_thresh")
        bad_df = res_df[res_df["score"].notna() & (res_df["score"] < float(cut))].reset_index(drop=True)
        st.markdown(f"Найдено слабых пар: **{len(bad_df)}** (score &lt; {cut})")

        # Скачивание результатов
        if not res_df.empty:
            csv_buf = io.StringIO()
            res_df.to_csv(csv_buf, index=False, encoding="utf-8-sig")
            st.download_button("💾 Скачать результаты (CSV)", csv_buf.getvalue(), file_name="synonym_check_results_manual.csv", mime="text/csv")

        # train_data.json для дообучения (на слабых парах)
        train_data = []
        for _, r in bad_df.iterrows():
            train_data.append({"texts": [r["phrase"], r["synonym"]], "label": 1.0, "topics": r.get("topics", [])})

        json_buf = io.StringIO()
        json.dump(train_data, json_buf, ensure_ascii=False, indent=2)
        if train_data:
            st.download_button("💾 Скачать датасет для дообучения (JSON)", json_buf.getvalue(), file_name="train_data_manual.json", mime="application/json")
        else:
            st.info("Нет слабых пар — ничего не будет сгенерировано для дообучения.")

        # Дообучение
        if st.button("🚀 Дообучить модель на слабых парах (manual)"):
            if not train_data:
                st.warning("Нет слабых пар для дообучения.")
            else:
                with st.spinner("Дообучаем модель..."):
                    train_examples = [InputExample(texts=[t[0], t[1]], label=1.0) for t in [(x["texts"][0], x["texts"][1]) for x in train_data]]
                    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=int(train_batch_size))
                    train_loss = losses.CosineSimilarityLoss(model)
                    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=int(train_epochs), warmup_steps=int(warmup_steps))
                    try:
                        model.save(output_model_path)
                        st.success(f"Модель дообучена и сохранена в {output_model_path}")
                    except Exception as e:
                        st.error(f"Ошибка при сохранении модели: {e}")

# ----------------------
# Автоматический режим
# ----------------------
with tab_auto:
    st.header("🤖 Автоматический режим — загрузите файл с фразами (CSV/XLSX)")
    st.markdown("Мы ищем лексически похожие пары (Jaccard токенов), которые модель считает НЕ похожими — лучшие кандидаты для обучения.")
    uploaded_auto = st.file_uploader("CSV/Excel с колонкой 'phrase' (опционально: topics, comment)", type=["csv", "xlsx", "xls"], key="auto_upl")

    if uploaded_auto is not None:
        try:
            df_all, file_hash = read_uploaded_file_bytes(uploaded_auto)
        except Exception as e:
            st.error(str(e))
            st.stop()

        cols_lower = {c.lower(): c for c in df_all.columns}
        if "phrase" not in cols_lower:
            st.error("Файл должен содержать колонку 'phrase' (регистр не важен).")
            st.stop()

        phrase_col = cols_lower["phrase"]
        topics_col = cols_lower.get("topics")

        df_all = df_all.copy()
        df_all["phrase_proc"] = df_all[phrase_col].apply(preprocess_text)
        if topics_col:
            df_all["topics"] = df_all[topics_col].apply(parse_topics_field)
            all_topics = sorted({t for ts in df_all["topics"] for t in ts})
            selected_topics = st.multiselect("Фильтр по тематикам (auto)", all_topics)
            if selected_topics:
                df_all = df_all[df_all["topics"].apply(lambda ts: bool(set(ts) & set(selected_topics)))].reset_index(drop=True)
        else:
            df_all["topics"] = [[] for _ in range(len(df_all))]
            selected_topics = []

        texts = df_all["phrase_proc"].fillna("").tolist()
        n = len(texts)
        st.info(f"Загружено фраз: {n}")

        if n == 0:
            st.warning("Нет фраз для анализа (после фильтрации по темам пусто).")
            st.stop()

        if n > int(DEFAULT_MAX_ROWS):
            st.warning(f"Файл слишком большой ({n} строк). Обработаем первые {int(DEFAULT_MAX_ROWS)} строк.")
            df_all = df_all.head(int(DEFAULT_MAX_ROWS))
            texts = df_all["phrase_proc"].tolist()
            n = len(texts)

        st.subheader("Параметры автоматического поиска кандидатов")
        col1, col2, col3 = st.columns(3)
        top_k = int(col1.number_input("Top-K соседей (на фразу)", min_value=1, max_value=50, value=5, step=1))
        lexical_thresh = float(col2.slider("Лексич. схожесть (Jaccard) >= ", 0.0, 1.0, 0.5, 0.05))
        model_thresh = float(col3.slider("Модель считает схожими если score >= ", 0.0, 1.0, 0.75, 0.01))

        cache_key = f"auto_{file_hash}_{model_path}_{batch_size}"
        if cache_key in st.session_state["emb_cache"]:
            embs = st.session_state["emb_cache"][cache_key]["emb"]
            st.success("Эмбеддинги для файла загружены из кэша")
        else:
            with st.spinner("Энкодим все фразы (может занять некоторое время)..."):
                embs = encode_texts_in_batches(model, texts, batch_size=int(batch_size))
                st.session_state["emb_cache"][cache_key] = {"emb": embs, "texts": texts, "df": df_all}
            st.success("Эмбеддинги вычислены")

        # semantic_search (all-against-all)
        with st.spinner("Ищу ближайших соседей и формирую кандидатов..."):
            corpus_emb = torch.from_numpy(embs)
            query_emb = corpus_emb
            hits = util.semantic_search(query_embedding=query_emb, corpus_embeddings=corpus_emb, top_k=top_k)

            candidate_pairs: List[Tuple[int, int, float, float]] = []
            for i, neighbors in enumerate(hits):
                for hit in neighbors:
                    j = int(hit["corpus_id"])
                    if i == j:
                        continue
                    mscore = float(hit["score"])
                    lex = jaccard_tokens(texts[i], texts[j])
                    if lex >= float(lexical_thresh) and mscore < float(model_thresh):
                        candidate_pairs.append((i, j, mscore, lex))

            candidate_pairs = dedupe_pairs(candidate_pairs)
            candidate_pairs = sorted(candidate_pairs, key=lambda x: (x[2], -x[3]))

        st.markdown(f"Найдено кандидатных пар: **{len(candidate_pairs)}** (лексич. схожесть ≥ {lexical_thresh} и модель score &lt; {model_thresh})")

        if not candidate_pairs:
            st.info("Не найдено кандидатов. Попробуйте изменить пороги.")
        else:
            # Подготовим таблицу кандидатов для показа
            display_rows = []
            for idx, (i, j, mscore, lex) in enumerate(candidate_pairs):
                pa_orig = df_all.iloc[i][phrase_col]
                pb_orig = df_all.iloc[j][phrase_col]
                ta = df_all.iloc[i]["topics"] if "topics" in df_all.columns else []
                tb = df_all.iloc[j]["topics"] if "topics" in df_all.columns else []
                display_rows.append({
                    "idx": idx,
                    "phrase_a": pa_orig,
                    "phrase_b": pb_orig,
                    "model_score": mscore,
                    "lex_jaccard": lex,
                    "topics_a": ta,
                    "topics_b": tb,
                })

            cand_df = pd.DataFrame(display_rows)
            st.subheader("Кандидаты (пример) — таблица")
            st.dataframe(cand_df[["idx", "phrase_a", "phrase_b", "model_score", "lex_jaccard", "topics_a", "topics_b"]], use_container_width=True)

            st.subheader("Отметьте пары для включения в датасет (train)")
            select_all = st.checkbox("Выбрать все кандидаты", value=True, key=f"auto_sel_all_{file_hash}")
            selected_indices = []
            # Вывод каждой строки с чекбоксом
            for row in display_rows:
                cols = st.columns([4, 4, 1, 1, 1])
                cols[0].write(row["phrase_a"])
                cols[1].write(row["phrase_b"])
                cols[2].write(f"{row['model_score']:.3f}")
                cols[3].write(f"{row['lex_jaccard']:.3f}")
                cols[4].write(", ".join(row['topics_a']) if row['topics_a'] else "")
                key = f"cand_{file_hash}_{row['idx']}"
                default = True if select_all else False
                sel = st.checkbox("", value=default, key=key)
                if sel:
                    selected_indices.append(row['idx'])

            st.markdown(f"Отмечено кандидатов: **{len(selected_indices)}**")

            # Сформировать train_data.json
            if st.button("📂 Сформировать и скачать train_data_auto.json"):
                chosen = []
                for idx in selected_indices:
                    i, j, mscore, lex = candidate_pairs[idx]
                    a = df_all.iloc[i][phrase_col]
                    b = df_all.iloc[j][phrase_col]
                    chosen.append({"texts": [a, b], "label": 1.0, "topics": df_all.iloc[i]["topics"]})
                if not chosen:
                    st.warning("Не выбрано ни одной пары.")
                else:
                    json_buf = io.StringIO()
                    json.dump(chosen, json_buf, ensure_ascii=False, indent=2)
                    st.download_button("💾 Скачать train_data_auto.json", json_buf.getvalue(), file_name="train_data_auto.json", mime="application/json")
                    st.success(f"Сформировано {len(chosen)} пар для дообучения.")

            # Дообучение по выбранным
            if st.button("🚀 Дообучить модель на выбранных парах (auto)"):
                chosen = []
                for idx in selected_indices:
                    i, j, mscore, lex = candidate_pairs[idx]
                    a = df_all.iloc[i][phrase_col]
                    b = df_all.iloc[j][phrase_col]
                    chosen.append({"texts": [a, b], "label": 1.0})
                if not chosen:
                    st.warning("Нечего дообучать — нет выбранных пар.")
                else:
                    with st.spinner("Дообучаем..."):
                        train_examples = [InputExample(texts=item["texts"], label=item["label"]) for item in chosen]
                        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=int(train_batch_size))
                        train_loss = losses.CosineSimilarityLoss(model)
                        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=int(train_epochs), warmup_steps=int(warmup_steps))
                        try:
                            model.save(output_model_path)
                            st.success(f"Модель дообучена и сохранена в {output_model_path}")
                        except Exception as e:
                            st.error(f"Ошибка при сохранении модели: {e}")

# Конец
st.markdown("---")
st.caption("Инструмент ищет и формирует кандидатов для дообучения понимания синонимов. \nManual — вы загружаете пары. Auto — вы даёте корпус фраз и (опционально) темы; система предлагает кандидатов на основе лексической похожести и слабой модельной связи.")
