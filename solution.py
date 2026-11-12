"""Сабмит ПСБ: embeddings + rerank + LLM + структурный чанкинг базы знаний.

Доработка базового каркаса:
- документы режутся на чанки ~250 токенов по заголовкам (главы/статьи/markdown);
- каждый чанк несёт "хлебную крошку" документ > глава > статья (источник для юриста);
- слова чанков кэшируются (лимит 20 минут на прогон);
- формат данных list[(name, text)] не меняется — retrieval работает поверх чанков.
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
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx


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
RERANKER_MODEL = env("RERANKER_MODEL", "cohere/rerank-5-fast")

TOP_K = env_int("TOP_K", 4, 1, 8)
RERANKER_TOP_N = env_int("RERANKER_TOP_N", 2, 1, 5)
MAX_TOKENS = env_int("MAX_TOKENS", 1024, 128, 8192)
MAX_WORKERS = env_int("MAX_WORKERS", 2, 1, 8)
TEMPERATURE = env_float("TEMPERATURE", 0.0)
CHUNKER = env("CHUNKER", "common")

SNIPPET_CHARS = 1200
FALLBACK = "Не удалось получить ответ."
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")


@lru_cache(maxsize=200_000)
def words(text: str) -> set[str]:
    # Кэш: чанки не меняются между вопросами, слова считаем один раз (лимит 20 мин).
    # Возвращаемое множество НЕ мутировать (используется только через &).
    return set(WORD_RE.findall(text.lower().replace("ё", "е")))


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


def read_docx(raw: bytes) -> str:
    # DOCX — это ZIP с XML. Пример обходится без дополнительной библиотеки.
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join(
        "".join(x.text or "" for x in paragraph.iter(ns + "t"))
        for paragraph in root.iter(ns + "p")
    )


def load_documents(path: Path) -> list[tuple[str, str]]:
    documents = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or name.startswith("__MACOSX/"):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in {".txt", ".md", ".docx"}:
                continue
            raw = archive.read(name)
            try:
                text = read_docx(raw) if suffix == ".docx" else raw.decode("utf-8-sig")
            except (UnicodeDecodeError, KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
                print(f"Пропущен нечитаемый документ: {name}", file=sys.stderr)
                continue
            if text.strip():
                documents.append((name, text.strip()))
    if not documents:
        raise ValueError("В архиве базы знаний нет читаемых документов")
    return documents


def snippet(text: str) -> str:
    # Минимальный пример применения CHUNKER, не полноценное разбиение на чанки.
    piece = text[:SNIPPET_CHARS]
    if CHUNKER == "common" and len(text) > SNIPPET_CHARS:
        boundary = piece.rfind("\n")
        if boundary > SNIPPET_CHARS // 2:
            piece = piece[:boundary]
    return piece.strip()


# ============ СТРУКТУРНЫЙ ЧАНКИНГ БАЗЫ ЗНАНИЙ (feature/structural-chunking) ============
# Было: поиску и LLM видны только первые 1200 символов документа (snippet).
# Стало: документ режется на чанки ~250 токенов по заголовкам (главы/статьи),
# каждый чанк несёт "хлебную крошку" документ > глава > статья.
# Формат данных не меняется: list[(name, text)] — весь остальной код работает как раньше.

KB_CHUNK_TOKENS = env_int("KB_CHUNK_TOKENS", 250, 64, 1024)   # платформа не передаёт; тюнинг здесь
KB_CHUNK_OVERLAP = env_int("KB_CHUNK_OVERLAP", 50, 0, 200)    # 250/50 — стартовые значения из статей

# Шаблоны заголовков: (уровень вложенности, регулярка). Уровень нужен, чтобы
# "Статья" не затирала "Главу" в хлебной крошке.
_KB_HEADINGS = [
    (1, re.compile(r"^#\s+(.+)$")),
    (2, re.compile(r"^#{2}\s+(.+)$")),
    (3, re.compile(r"^#{3,6}\s+(.+)$")),
    (1, re.compile(r"^(?:ЧАСТЬ|Часть)\s+[IVXLCDM\d]+.*$")),
    (2, re.compile(r"^(?:ГЛАВА|Глава)\s+[IVXLCDM\d]+.*$")),
    (3, re.compile(r"^(?:РАЗДЕЛ|Раздел)\s+[IVXLCDM\d]+.*$")),
    (4, re.compile(r"^(?:Статья|Ст\.)\s*\d+(?:\.\d+)*\.?.*$")),
    (5, re.compile(r"^\d+(?:\.\d+)*\.?\s+\S.{0,120}$")),
]


def _kb_heading(line: str):
    """Если строка — заголовок, вернуть (уровень, текст), иначе None."""
    if not line or len(line) > 160:
        return None
    for level, pattern in _KB_HEADINGS:
        m = pattern.match(line)
        if m:
            title = m.group(1) if m.groups() else line
            return level, title.strip().rstrip("#").strip()[:120]
    # ЗАГОЛОВКИ КАПСОМ (типично для законов), без точки на конце
    if 8 <= len(line) <= 90 and line == line.upper() and any(c.isalpha() for c in line) and not line.endswith("."):
        return 2, line.strip()
    return None


def _kb_sections(text: str) -> list[tuple[str, str]]:
    """Разбить документ на секции по заголовкам. Возвращает [(путь_заголовков, тело)]."""
    sections = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))

    for line in text.splitlines():
        h = _kb_heading(line.strip())
        if h:
            flush()                      # закрываем ПРЕДЫДУЩУЮ секцию со старым путём
            buf = [line.strip()]         # сам заголовок оставляем в тексте секции
            level, title = h
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _kb_split_fallback(body: str) -> list[str]:
    """Запасной резак без внешних библиотек: ~3 символа на токен для русского текста."""
    limit = KB_CHUNK_TOKENS * 3
    overlap = min(KB_CHUNK_OVERLAP * 3, limit // 2)
    pieces, start = [], 0
    while start < len(body):
        end = min(len(body), start + limit)
        if end < len(body):              # стараемся резать по границе абзаца/предложения
            for sep in ("\n\n", "\n", ". ", " "):
                pos = body.rfind(sep, start + limit // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        piece = body[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(body):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _kb_make_splitter():
    """Основной путь: разрешённые langchain-text-splitters + tiktoken (точный счёт токенов)."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=KB_CHUNK_TOKENS,
            chunk_overlap=min(KB_CHUNK_OVERLAP, KB_CHUNK_TOKENS // 2),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except Exception as exc:             # нет библиотеки / tiktoken оффлайн не скачал словарь
        print(f"Чанкинг: токенный сплиттер недоступен ({type(exc).__name__}), символьный fallback", file=sys.stderr)
        return None


def build_chunks(documents: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Документы -> чанки с хлебными крошками. Интерфейс совместим со всем остальным кодом."""
    if CHUNKER.strip().lower() in ("off", "none", "disabled"):
        return documents                 # платформенный CHUNKER может выключить чанкинг
    splitter = _kb_make_splitter()
    chunks: list[tuple[str, str]] = []
    for name, text in documents:
        for crumb, body in _kb_sections(text):
            pieces = None
            if splitter is not None:
                try:
                    pieces = splitter.split_text(body)
                except Exception:
                    splitter = None      # один сбой — дальше работаем запасным резаком
            if pieces is None:
                pieces = _kb_split_fallback(body)
            for i, piece in enumerate(pieces, 1):
                piece = piece.strip()
                if not piece:
                    continue
                label = f"{name} > {crumb}" if crumb else name
                if len(pieces) > 1:
                    label += f" [фрагмент {i}/{len(pieces)}]"
                # крошку кладём и в текст: её видят эмбеддер, реранкер и LLM
                chunk_text = f"{crumb}\n{piece}" if crumb else piece
                chunks.append((label, chunk_text))
    if not chunks:                       # совсем пустая база — не ломаем пайплайн
        return documents
    print(f"Чанкинг: {len(documents)} документов -> {len(chunks)} чанков "
          f"(цель ~{KB_CHUNK_TOKENS} токенов, overlap {KB_CHUNK_OVERLAP})", flush=True)
    return chunks
# ======================= КОНЕЦ БЛОКА СТРУКТУРНОГО ЧАНКИНГА =======================


def post(client: httpx.Client, path: str, key: str, payload: dict) -> dict:
    if not key:
        raise ValueError("На платформе отсутствует API-ключ")
    response = client.post(
        API_BASE + path,
        headers={"Authorization": "Bearer " + key},
        json=payload,
    )
    response.raise_for_status()
    return response.json()


def embeddings(client: httpx.Client, texts: list[str]) -> list[list[float]]:
    """Пример вызова EMB_MODEL через POST /embeddings."""
    result = post(client, "/embeddings", EMBED_KEY, {
        "model": EMB_MODEL,
        "input": texts,
        "encoding_format": "float",
    })
    vectors: list[list[float] | None] = [None] * len(texts)
    for position, item in enumerate(result["data"]):
        index = int(item.get("index", position))
        if index < 0 or index >= len(texts):
            raise ValueError("Некорректный индекс embedding")
        vector = item["embedding"]
        norm = math.sqrt(sum(float(x) ** 2 for x in vector)) or 1.0
        vectors[index] = [float(x) / norm for x in vector]
    if any(v is None for v in vectors):
        raise ValueError("Получены не все embeddings")
    return vectors  # type: ignore[return-value]


def rerank(client: httpx.Client, question: str, docs: list[tuple[str, str]]) -> list[int]:
    """Пример вызова RERANKER_MODEL через POST /rerank."""
    result = post(client, "/rerank", RERANK_KEY, {
        "model": RERANKER_MODEL,
        "query": question,
        "documents": [name + "\n" + snippet(text) for name, text in docs],
        "top_n": min(RERANKER_TOP_N, len(docs)),
    })
    items = result.get("results", result.get("data", []))
    indices = [int(item["index"]) for item in items]
    return [i for i in indices if 0 <= i < len(docs)][:RERANKER_TOP_N]


def select_documents(client: httpx.Client, question: str,
                     documents: list[tuple[str, str]]) -> list[tuple[str, str]]:
    # Простая предварительная выборка по совпадению слов; не полноценный RAG.
    query_words = words(question)
    candidates = sorted(
        documents,
        key=lambda d: len(query_words & words(d[0] + " " + d[1])),
        reverse=True,
    )[:TOP_K]

    # Шаг 1: демонстрация использования эмбеддера.
    try:
        vectors = embeddings(client, [question] + [snippet(text) for _, text in candidates])
        query_vec = vectors[0]
        scores = [sum(a * b for a, b in zip(query_vec, vec)) for vec in vectors[1:]]
        candidates = [candidates[i] for i in sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"Embeddings недоступны: {type(exc).__name__}; используем поиск по словам", file=sys.stderr)

    # Шаг 2: демонстрация реранкера. При ошибке продолжаем без него.
    try:
        indices = rerank(client, question, candidates)
        if indices:
            return [candidates[i] for i in indices]
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"Reranker недоступен: {type(exc).__name__}; используем текущий порядок", file=sys.stderr)
    return candidates[:RERANKER_TOP_N]


def generate(client: httpx.Client, question: str, docs: list[tuple[str, str]]) -> str:
    """Пример вызова LLM_MODEL через POST /chat/completions."""
    context = "\n\n".join(name + "\n" + snippet(text) for name, text in docs)
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "Ответь на вопрос по приложенным фрагментам. Не выдумывай юридические нормы. Если ответа нет, сообщи об этом."},
            {"role": "user", "content": "Фрагменты:\n" + context + "\n\nВопрос:\n" + question},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    for attempt in range(2):
        try:
            result = post(client, "/chat/completions", LLM_KEY, payload)
            content = result["choices"][0]["message"].get("content", "")
            if isinstance(content, str) and content.strip():
                return " ".join(content.split())  # одна физическая строка CSV
            raise ValueError("Модель вернула пустой ответ")
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            print(f"LLM attempt {attempt + 1}: {type(exc).__name__}", file=sys.stderr)
            if attempt == 0:
                time.sleep(1)
    return FALLBACK


def main() -> None:
    if not API_BASE or not LLM_KEY:
        raise ValueError("Нужны LLM_BASE_URL и LLM_API_KEY (или их аналоги)")
    questions = load_questions(find_file("questions.csv"))
    documents = load_documents(find_file("kb.zip", "knowledge-base.zip", "knowledge_base.zip"))
    documents = build_chunks(documents)   # структурный чанкинг вместо обрезки первых 1200 символов
    print(f"Вопросы: {len(questions)}; чанков: {len(documents)}; TOP_K={TOP_K}", flush=True)

    def work(position: int, question: str) -> tuple[int, str]:
        try:
            with httpx.Client(timeout=90.0) as client:
                selected = select_documents(client, question, documents)
                return position, generate(client, question, selected)
        except Exception as exc:
            print(f"Ошибка вопроса {position}: {type(exc).__name__}", file=sys.stderr)
            return position, FALLBACK

    answers = [FALLBACK] * len(questions)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(work, i, question) for i, (_, question) in enumerate(questions)]
        for finished, future in enumerate(as_completed(futures), 1):
            index, answer = future.result()
            answers[index] = answer
            if finished % 10 == 0 or finished == len(questions):
                print(f"Обработано {finished}/{len(questions)}", flush=True)

    with open("submission.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        for i, (qid, _) in enumerate(questions):
            writer.writerow([qid, answers[i]])
    print(f"Создан submission.csv: {len(questions)} ответов", flush=True)


if __name__ == "__main__":
    main()