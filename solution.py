"""Минимальный пример сабмита ПСБ: embeddings + rerank + LLM.

Это учебный каркас, а не полноценное RAG-решение: используется только
простой поиск по словам среди документов и короткие текстовые фрагменты.
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import bm25s
import faiss
import httpx
import numpy as np
import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(env(name))))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return max(0.0, min(2.0, float(env(name))))
    except ValueError:
        return default


# Настройки приходят с платформы. В ZIP НЕ НУЖНО класть секреты и .env.
DATA_DIR = Path(env("DATA_DIR", "data"))
API_BASE = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY

LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "qwen/qwen3-reranker-8b")

#расширил лимиты для качественного гибридного поиска
RETRIEVAL_TOP_K = max(50, env_int("TOP_K", 50, 20, 120))
RERANKER_TOP_N = env_int("RERANKER_TOP_N", 5, 3, 10)
MAX_TOKENS = env_int("MAX_TOKENS", 550, 128, 8192)
MAX_WORKERS = env_int("MAX_WORKERS", 6, 1, 16)
TEMPERATURE = env_float("TEMPERATURE", 0.0)
CHUNKER = env("CHUNKER", "common")


FALLBACK = (
    "В базе знаний нет документов по данному вопросу, поэтому ответить по документам я не могу. "
    "По памяти отвечать не буду, так как нормы могли измениться. Для ответа обратитесь к юристу."
)

#def count_tokens(text: str) -> int:
#   return len(TOKENIZER.encode(text))


def find_file(*names: str) -> Path:
    for root in (DATA_DIR, Path(__file__).resolve().parent / DATA_DIR, Path.cwd()):
        for name in names:
            path = root / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"Не найден один из файлов: {', '.join(names)}")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
        rows = [(str(r["id"]).strip(), str(r["question"]).strip()) for r in reader]
    if not rows or any(not qid for qid, _ in rows):
        raise ValueError("В questions.csv отсутствуют вопросы или id")
    return rows


#добваил возможность чтения таблиц 
def read_docx(raw: bytes) -> str:
    """Читает DOCX с сохранением порядка следования абзацев и Markdown-разметкой таблиц."""
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    body = root.find(ns + "body")
    if body is None:
        return ""

    blocks: list[str] = []
    for child in body:
        if child.tag == ns + "p":
            line = "".join(t.text or "" for t in child.iter(ns + "t")).strip()
            if line:
                blocks.append(line)
        elif child.tag == ns + "tbl":
            table_rows: list[str] = []
            for r_idx, tr in enumerate(child.iter(ns + "tr")):
                cells: list[str] = []
                for tc in tr.iter(ns + "tc"):
                    cell_parts = [
                        "".join(t.text or "" for t in p.iter(ns + "t")).strip()
                        for p in tc.iter(ns + "p")
                    ]
                    cell_text = " ".join(part for part in cell_parts if part)
                    cells.append(cell_text)
                if any(cells):
                    table_rows.append("| " + " | ".join(cells) + " |")
                    if r_idx == 0:
                        table_rows.append("|" + "|".join(["---"] * len(cells)) + "|")
            if table_rows:
                blocks.append("\n".join(table_rows))

    return "\n\n".join(blocks)


#читаем pdf
def read_pdf(raw: bytes) -> str:
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append(txt.strip())
    return "\n".join(pages)

#читаем html 
def read_html(raw: bytes) -> str:
    soup = BeautifulSoup(raw.decode("utf-8", errors="ignore"), "lxml")
    return soup.get_text(separator="\n", strip=True)



def fix_zip_filename(name: str, flag_bits: int) -> str:
    """Исправляет все виды битых кодировок русских имен файлов в ZIP (CP437/CP866 <-> UTF-8)."""
    if "╨" in name or "╤" in name:
        try:
            return name.encode("cp866").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    if flag_bits & 0x800:
        return name
    try:
        raw_bytes = name.encode("cp437")
        for enc in ("utf-8", "cp866", "cp1251"):
            try:
                return raw_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
    except UnicodeEncodeError:
        pass
    return name


def load_documents(path: Path) -> list[tuple[str, str]]:
    documents = []
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda x: x.filename):
            raw_name = info.filename
            if raw_name.endswith("/") or raw_name.startswith("__MACOSX/"):
                continue
            clean_name = fix_zip_filename(raw_name, info.flag_bits)
            suffix = Path(clean_name).suffix.lower()
            raw = archive.read(raw_name)
            try:
                if suffix == ".docx":
                    text = read_docx(raw)
                elif suffix == ".pdf":
                    text = read_pdf(raw)
                elif suffix in {".html", ".htm"}:
                    text = read_html(raw)
                elif suffix in {".txt", ".md", ".csv", ".json"}:
                    text = raw.decode("utf-8-sig", errors="replace")
                else:
                    continue
            except Exception as exc:
                print(f"Пропущен нечитаемый документ {clean_name}: {type(exc).__name__}", file=sys.stderr)
                continue

            cleaned = re.sub(r"\n{3,}", "\n\n", text).strip()
            if cleaned:
                documents.append((clean_name, cleaned))

    if not documents:
        raise ValueError("В архиве базы знаний нет читаемых документов")
    return documents





#безопасная инициализация токенизатора на случай изолированного docker
try:
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")

    def count_tockens(text: str) -> int:
        return len(_TOKENIZER.encode(text))
except Exception:
    def count_tockens(text:str) -> int:
        #для русского текста 1 токен в среднем равен ~3 символам
        return max(1, len(text) // 3)


@dataclass
class ParentChunk:
    parent_id: int 
    doc_name: str
    header: str
    text: str

@dataclass
class ChildChunk:
    child_id: int 
    parent_id: int 
    doc_name: str
    header: str
    text: str

#Регулярное выражение для поиска юредических заголовков 
_KB_HEADINGS = [
    (1, re.compile(r"^#\s+(.+)$")),
    (2, re.compile(r"^#{2}\s+(.+)$")),
    (3, re.compile(r"^#{3,6}\s+(.+)$")),
    (1, re.compile(r"^(?:ЧАСТЬ|Часть)\s+[IVXLCDM\d]+.*$")),
    (2, re.compile(r"^(?:ГЛАВА|Глава)\s+[IVXLCDM\d]+.*$")),
    (3, re.compile(r"^(?:РАЗДЕЛ|Раздел)\s+[IVXLCDM\d]+.*$")),
    (4, re.compile(r"^(?:Статья|Ст\.)\s*\d+(?:\.\d+)*\.?.*$")),
    
]


def _kb_heading(line: str) -> tuple[int, str] | None:
    """Если строка — юридический или структурный заголовок, возвращает (уровень, текст)."""
    if not line or len(line) > 160:
        return None
    for level, pattern in _KB_HEADINGS:
        m = pattern.match(line)
        if m:
            title = m.group(1) if m.groups() else line
            return level, title.strip().rstrip("#").strip()[:120]
    if 8 <= len(line) <= 90 and line == line.upper() and any(c.isalpha() for c in line) and not line.endswith("."):
        return 2, line.strip()
    return None


def _kb_sections(text: str) -> list[tuple[str, str]]:
    """Разбивает документ на логические секции со стеком заголовков (Глава > Раздел > Статья)."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))

    for line in text.splitlines():
        clean_line = line.strip()
        h = _kb_heading(clean_line)
        if h:
            flush()
            buf = [clean_line]
            level, title = h
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def build_chunks(
    documents: list[tuple[str, str]],
    parent_size: int = 600,
    parent_overlap: int = 100,
    child_size: int = 200,
    child_overlap: int = 35,
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Строит Parent-Document чанки с иерархическими хлебными крошками."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_overlap,
        length_function=count_tockens,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        length_function=count_tockens,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []

    for doc_name, full_text in documents:
        clean_doc_name = str(Path(doc_name).with_suffix("")).replace("_", " ").strip()
        sections = _kb_sections(full_text) or [("", full_text)]

        for crumb, section_body in sections:
            full_header = f"{clean_doc_name} > {crumb}" if crumb else clean_doc_name
            parent_texts = (
                [section_body]
                if CHUNKER.strip().lower() in ("off", "none", "disabled")
                else parent_splitter.split_text(section_body)
            )

            for p_text in parent_texts:
                p_text = p_text.strip()
                if not p_text:
                    continue

                parent_id = len(parents)
                enriched_parent = f"[Документ: {clean_doc_name} | Раздел: {full_header}]\n{p_text}"
                parents.append(
                    ParentChunk(
                        parent_id=parent_id,
                        doc_name=clean_doc_name,
                        header=full_header,
                        text=enriched_parent,
                    )
                )

                child_texts = child_splitter.split_text(p_text)
                for c_text in child_texts:
                    c_text = c_text.strip()
                    if not c_text:
                        continue
                    enriched_child = f"{full_header}\n{c_text}"
                    children.append(
                        ChildChunk(
                            child_id=len(children),
                            parent_id=parent_id,
                            doc_name=clean_doc_name,
                            header=full_header,
                            text=enriched_child,
                        )
                    )

    return parents, children


TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[\.\-][0-9A-Za-zА-Яа-яЁё]+)*")

def tokenize_ru(text: str) -> list[str]:
    """Токенизация для BM25 с сохранением номеров статей/пунктов и базовой нормализацией."""
    raw_tokens = TOKEN_RE.findall(text.lower().replace("ё", "е"))
    tokens = []
    for tok in raw_tokens:
        if len(tok) <= 1:
            continue
        tokens.append(tok)
        # Для длинных русских слов отсекаем 2 последних символа 
        if len(tok) >= 6 and tok.isalpha():
            tokens.append(tok[:-2])
    return tokens        

@dataclass
class HybridIndex:
    parents: list[ParentChunk]
    children: list[ChildChunk]
    bm25: bm25s.BM25
    faiss_index: faiss.IndexFlatIP | None


def post_with_retry(
    client: httpx.Client,
    path: str,
    key: str,
    payload: dict,
    retries: int = 4,
) -> dict:
    """Универсальный POST-запрос с экспоненциальной паузой при 429 / 5xx."""
    if not key:
        raise ValueError("На платформе отсутствует API-ключ")
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.post(
                API_BASE + path,
                headers={"Authorization": "Bearer " + key},
                json=payload,
            )
            if response.status_code in {429, 500, 502, 503, 504}:
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (2 ** attempt))
    raise last_exc or RuntimeError("Неизвестная ошибка сети")


def fetch_embeddings_batch(client: httpx.Client, texts: list[str]) -> np.ndarray:
    """Получает нормализованные L2-векторы для батча текстов."""
    result = post_with_retry(
        client,
        "/embeddings",
        EMBED_KEY,
        {
            "model": EMB_MODEL,
            "input": texts,
            "encoding_format": "float",
        },
    )
    vectors: list[list[float] | None] = [None] * len(texts)
    for pos, item in enumerate(result["data"]):
        idx = int(item.get("index", pos))
        if 0 <= idx < len(texts):
            vectors[idx] = item["embedding"]
    if any(v is None for v in vectors):
        raise ValueError("Получены не все embeddings от прокси")

    arr = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr


def build_hybrid_index(
    parents: list[ParentChunk],
    children: list[ChildChunk],
    batch_size: int = 32,
) -> HybridIndex:
    """Строит лексический (BM25s) и векторный (FAISS) индексы по дочерним чанкам."""
    # 1. Лексический индекс BM25s (работает локально за миллисекунды)
    corpus_tokens = [tokenize_ru(c.text) for c in children]
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=False)

    # 2. Векторный индекс FAISS через локальный прокси BAAI/bge-m3
    faiss_index: faiss.IndexFlatIP | None = None
    try:
        all_vectors = []
        with httpx.Client(timeout=120.0) as client:
            for start in range(0, len(children), batch_size):
                batch_texts = [c.text for c in children[start : start + batch_size]]
                batch_vecs = fetch_embeddings_batch(client, batch_texts)
                all_vectors.append(batch_vecs)

        if all_vectors:
            matrix = np.vstack(all_vectors)
            faiss_index = faiss.IndexFlatIP(matrix.shape[1])
            faiss_index.add(matrix)
            print(f"FAISS индекс построен: {faiss_index.ntotal} векторов (dim={matrix.shape[1]})", flush=True)
    except Exception as exc:
        print(
            f"Предупреждение: не удалось построить FAISS ({type(exc).__name__}), работаем только на BM25s",
            file=sys.stderr,
            flush=True,
        )
        faiss_index = None

    return HybridIndex(
        parents=parents,
        children=children,
        bm25=retriever,
        faiss_index=faiss_index,
    )


DIALOGUE_MARKERS_RE = re.compile(
    r"(?:^|\n)\s*(?:user|assistant|пользователь|ассистент|вопрос|ответ)\s*:",
    re.IGNORECASE,
)


def generate_search_queries(client: httpx.Client, question: str) -> list[str]:
    """Формирует 1–3 точечных поисковых запроса (раскрывает диалоги и декомпозирует сложные вопросы)."""
    is_dialogue = "\n" in question or bool(DIALOGUE_MARKERS_RE.search(question))
    words_count = len(question.split())

    # Короткий одиночный вопрос не требует вызова LLM
    if not is_dialogue and words_count <= 10:
        return [question]

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты — модуль подготовки поисковых запросов для юридической базы знаний. "
                    "Проанализируй вопрос пользователя (или историю диалога) и сформируй от 1 до 3 коротких, "
                    "самодостаточных поисковых запросов для поиска по договорам и законам.\n"
                    "Правила:\n"
                    "1. Раскрой все местоимения и контекст из предыдущих реплик диалога.\n"
                    "2. Если вопрос составной (спрашивается о нескольких условиях, сроках, штрафах или документах), "
                    "разбей его на 2–3 отдельных поисковых запроса по каждой теме.\n"
                    "3. Сохраняй точные номера договоров, статей, дат и названия компаний.\n"
                    "4. Верни ТОЛЬКО сами запросы — каждый с новой строки, без нумерации и пояснений."
                ),
            },
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": 250,
    }
    try:
        result = post_with_retry(client, "/chat/completions", LLM_KEY, payload, retries=2)
        raw_text = result["choices"][0]["message"].get("content", "").strip()
        queries = [
            re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]
        # Оставляем до 3 уникальных непустых запросов
        unique_queries: list[str] = []
        for q in queries:
            if 3 <= len(q) <= 500 and q not in unique_queries:
                unique_queries.append(q)
        if unique_queries:
            return unique_queries[:3]
    except Exception as exc:
        print(f"Multi-query пропущен ({type(exc).__name__}), используем исходный вопрос", file=sys.stderr)

    return [question]


def rerank_parents(
    client: httpx.Client,
    search_query: str,
    candidates: list[ParentChunk],
) -> list[ParentChunk]:
    """Переранжирует родительские чанки с помощью Qwen3 Reranker 8B."""
    if len(candidates) <= 1:
        return candidates

    top_n = min(RERANKER_TOP_N, len(candidates))
    result = post_with_retry(
        client,
        "/rerank",
        RERANK_KEY,
        {
            "model": RERANKER_MODEL,
            "query": search_query,
            "documents": [p.text for p in candidates],
            "top_n": top_n,
        },
        retries=3,
    )
    items = result.get("results", result.get("data", []))
    ranked: list[ParentChunk] = []
    seen_indices: set[int] = set()
    doc_counts: dict[str, int] = {}
    max_per_doc = 2

    # Первый проход: берем лучшие чанки с ограничением не более 2 на один документ
    for item in items:
        idx = int(item["index"])
        if 0 <= idx < len(candidates) and idx not in seen_indices:
            cand = candidates[idx]
            if doc_counts.get(cand.doc_name, 0) < max_per_doc:
                seen_indices.add(idx)
                doc_counts[cand.doc_name] = doc_counts.get(cand.doc_name, 0) + 1
                ranked.append(cand)
                if len(ranked) >= top_n:
                    break

    # Второй проход: если уникальных документов не хватило до top_n, добираем оставшиеся лучшие
    if len(ranked) < top_n:
        for item in items:
            idx = int(item["index"])
            if 0 <= idx < len(candidates) and idx not in seen_indices:
                seen_indices.add(idx)
                ranked.append(candidates[idx])
                if len(ranked) >= top_n:
                    break

    return ranked[:top_n] if ranked else candidates[:top_n]


def retrieve_context(
    client: httpx.Client,
    question: str,
    index: HybridIndex,
    rrf_k: int = 60,
) -> list[ParentChunk]:
    """Выполняет Multi-Query гибридный поиск (BM25s + FAISS), RRF-слияние и реранкинг."""
    if not index.children:
        return []

    search_queries = generate_search_queries(client, question)
    primary_query = search_queries[0]
    k = min(RETRIEVAL_TOP_K, len(index.children))
    rrf_scores: dict[int, float] = {}

    # 1. Лексический поиск BM25s по всем подзапросам
    for q in search_queries:
        q_tokens = tokenize_ru(q)
        if not q_tokens:
            continue
        bm25_indices, _ = index.bm25.retrieve([q_tokens], k=k, show_progress=False)
        for rank, child_idx in enumerate(bm25_indices[0]):
            cid = int(child_idx)
            if 0 <= cid < len(index.children):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

    # 2. Векторный поиск FAISS батчем сразу по всем подзапросам (1 сетевой запрос)
    if index.faiss_index is not None:
        try:
            q_vecs = fetch_embeddings_batch(client, search_queries)
            _, faiss_indices = index.faiss_index.search(q_vecs, k)
            for row in faiss_indices:
                for rank, child_idx in enumerate(row):
                    cid = int(child_idx)
                    if 0 <= cid < len(index.children):
                        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
        except Exception as exc:
            print(f"Ошибка векторного поиска ({type(exc).__name__}), используем BM25", file=sys.stderr)

    if not rrf_scores:
        sorted_child_ids = list(range(min(k, len(index.children))))
    else:
        sorted_child_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    # 3. Агрегация оценок дочерних чанков на уровень родительских документов
    parent_scores: dict[int, float] = {}
    for cid in sorted_child_ids:
        pid = index.children[cid].parent_id
        parent_scores[pid] = parent_scores.get(pid, 0.0) + rrf_scores.get(cid, 0.0)

    # Передаем топ-24 уникальных родительских блока в Qwen3 Reranker 8B
    rerank_pool_size = max(RERANKER_TOP_N * 3, 24)
    top_parent_ids = sorted(parent_scores.keys(), key=lambda pid: parent_scores[pid], reverse=True)[:rerank_pool_size]
    candidate_parents = [index.parents[pid] for pid in top_parent_ids]

    # 4. Финальный реранкинг по объединенному поисковому запросу
    rerank_query = " | ".join(search_queries) if len(search_queries) > 1 else primary_query
    try:
        return rerank_parents(client, rerank_query, candidate_parents)
    except Exception as exc:
        print(f"Reranker недоступен ({type(exc).__name__}), возвращаем RRF-топ", file=sys.stderr)
        return candidate_parents[:RERANKER_TOP_N]


SYSTEM_PROMPT = (
    "Ты — юридический ассистент компании. Дай точный, фактологически безупречный и максимально лаконичный ответ "
    "на вопрос пользователя ИСКЛЮЧИТЕЛЬНО на основе предоставленных ниже фрагментов документов.\n\n"
    "Строгие правила:\n"
    "1. Запрещено использовать внешние знания, законы или статьи, которых нет в переданных фрагментах. "
    "Любой выдуманный факт или домысел недопустимы.\n"
    "2. Если информации для ответа в фрагментах нет — напиши: «В базе знаний нет документов по данному вопросу, "
    "поэтому ответить по документам я не могу. По памяти отвечать не буду, так как нормы могли измениться. "
    "Для ответа обратитесь к юристу.» Если в тексте упомянуты другие статьи, которых нет в базе — прямо укажи, "
    "что этих статей в базе нет.\n"
    "3. Приоритет редакций: если в фрагментах есть несколько редакций одного регламента или закона, всегда применяй "
    "самую свежую (действующую) редакцию, если только в вопросе прямо не указана дата события из прошлого.\n"
    "4. Если пользователь в вопросе называет неверную цифру, срок или условие — прямо опровергни это со ссылкой на документ.\n"
    "5. Формат ответа: ОДИН плотный абзац строго на 40–75 слов (без списков, заголовков и вводных фраз). "
    "Если вопрос предполагает «да/нет», начни со слова «Да.» или «Нет.». В конце или в скобках укажи точный документ и пункт/статью.\n\n"
    "Эталоны стиля:\n"
    "— Арендная плата по договору аренды № А-08/2024 с ООО «Северный Склад» — 450 000 рублей в месяц, включая НДС (п. 3.1). "
    "Вносится ежемесячно, не позднее 5-го числа текущего месяца (п. 3.2).\n"
    "— Да. Во время испытания на работника распространяются положения трудового законодательства, коллективного договора, "
    "соглашений и локальных нормативных актов работодателя. Источник: ТК РФ, ст. 70."
)


def generate_answer(client: httpx.Client, question: str, chunks: list[ParentChunk]) -> str:
    """Генерирует эталонный краткий юридический ответ строго по отобранным фрагментам."""
    if not chunks:
        return FALLBACK

    context_blocks = [f"--- Фрагмент {i} ---\n{chunk.text}" for i, chunk in enumerate(chunks, 1)]
    context_str = "\n\n".join(context_blocks)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Контекст из базы знаний:\n{context_str}\n\nВопрос пользователя:\n{question}",
            },
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    try:
        result = post_with_retry(client, "/chat/completions", LLM_KEY, payload, retries=4)
        content = result["choices"][0]["message"].get("content", "")
        if isinstance(content, str) and content.strip():
            cleaned = " ".join(content.strip().split())
            if len(cleaned) > 9500:
                cleaned = cleaned[:9500].rsplit(" ", 1)[0] + "..."
            return cleaned
    except Exception as exc:
        print(f"Ошибка генерации LLM: {type(exc).__name__}", file=sys.stderr)

    return FALLBACK


def main() -> None:
    start_time = time.time()
    if not API_BASE or not LLM_KEY:
        raise ValueError("Нужны переменные окружения LLM_BASE_URL и LLM_API_KEY (или их аналоги)")

    questions_path = find_file("questions.csv", "demo_questions.csv")
    kb_path = find_file("kb.zip", "knowledge-base.zip", "knowledge_base.zip")


    questions = load_questions(questions_path)
    documents = load_documents(kb_path)
    print(f"Загружено вопросов: {len(questions)}, документов: {len(documents)}", flush=True)

    # Этап 1: Нарезка на родительские и дочерние чанки
    parents, children = build_chunks(documents)
    print(f"Сформировано чанков: {len(parents)} родительских, {len(children)} дочерних", flush=True)

    # Этап 2: Построение гибридного индекса (BM25s + FAISS) один раз на весь запуск
    index = build_hybrid_index(parents, children)
    print(f"Индексация завершена за {time.time() - start_time:.1f} сек.", flush=True)

    # Этап 3: Многопоточная обработка вопросов
    def work(position: int, question: str) -> tuple[int, str]:
        try:
            with httpx.Client(timeout=90.0) as client:
                selected_chunks = retrieve_context(client, question, index)
                answer = generate_answer(client, question, selected_chunks)
                return position, answer
        except Exception as exc:
            print(f"Критическая ошибка на вопросе {position}: {type(exc).__name__}", file=sys.stderr)
            return position, FALLBACK

    answers = [FALLBACK] * len(questions)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(work, i, q) for i, (_, q) in enumerate(questions)]
        for finished, future in enumerate(as_completed(futures), 1):
            idx, ans = future.result()
            answers[idx] = ans
            if finished % 10 == 0 or finished == len(questions):
                elapsed = time.time() - start_time
                print(f"Обработано {finished}/{len(questions)} ({elapsed:.1f} сек.)", flush=True)

    # Этап 4: Сохранение результата в строгом соответствии с форматом регламента
    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for i, (qid, _) in enumerate(questions):
            safe_answer = answers[i] if answers[i].strip() else FALLBACK
            writer.writerow([qid, safe_answer])

    print(f"Готово! Создан submission.csv ({len(questions)} строк) за {time.time() - start_time:.1f} сек.", flush=True)


if __name__ == "__main__":
    main()
