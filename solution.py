"""ПСБ / Hack&Change 2026 — юридический RAG-ассистент.
только бы все заработало
Пайплайн:
  1. Чтение knowledge-base.zip (docx c таблицами, pdf, html, txt) + questions.csv.
  2. Структурная нарезка по юридическим единицам (статья / раздел / пункт),
     метаданные: номер договора, контрагент, редакция, оглавление.
  3. Гибридный поиск: BM25 (свой, numpy) + BAAI/bge-m3 (FAISS/numpy), RRF.
  4. Бустинг по контрагенту/номеру договора и по применимой редакции.
  5. Qwen3 Reranker -> сборка контекста целыми пунктами + каталог базы.
  6. Генерация строго по контексту + проход-контролёр против выдуманных фактов.
  7. Инкрементальная запись submission.csv (файл валиден с первой секунды).
"""
from __future__ import annotations

import csv
import io
import os
import random
import re
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

try:
    import faiss
except Exception:
    faiss = None


# --------------------------------------------------------------------------- #
# Конфигурация (только из окружения, никаких секретов в коде)
# --------------------------------------------------------------------------- #
def env(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is None:
            continue
        val = raw.strip()
        if val and val != name and val.lower() not in {"none", "null", "-"}:
            return val
    return default


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(float(env(name, default=str(default))))))
    except Exception:
        return default


def env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(env(name, default=str(default)))))
    except Exception:
        return default


T0 = time.time()
DATA_DIR = Path(env("DATA_DIR", default="data"))
API_BASE = env("LLM_BASE_URL", "OPENAI_BASE_URL", "BASE_URL").rstrip("/")
GEN_MODEL = env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL")
EMB_MODEL = env("EMB_MODEL", "EMBEDDER_MODEL", default="BAAI/bge-m3")
RERANK_MODEL = env("RERANKER_MODEL", default="Qwen/Qwen3-Reranker-8B")

LLM_KEY = env("LLM_API_KEY", "OPENAI_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY
API_KEY = LLM_KEY  

TEMPERATURE = env_float("TEMPERATURE", 0.0, 0.0, 1.0)
ANSWER_TOKENS = max(512, env_int("MAX_TOKENS", 700, 128, 1600))
MAX_WORKERS = env_int("MAX_WORKERS", 6, 1, 12)
TOP_K = env_int("TOP_K", 40, 10, 120)
RERANK_TOP_N = env_int("RERANKER_TOP_N", 8, 3, 20)

RUN_BUDGET = env_int("RUN_BUDGET_SEC", 1020, 60, 1140)   # 17 мин из 20 на всякий случай
DEADLINE = T0 + RUN_BUDGET
INDEX_BUDGET = min(420.0, RUN_BUDGET * 0.40)
CTX_CHARS = env_int("CTX_CHARS", 14000, 3000, 30000)

USE_RERANK = env("USE_RERANK", default="1").lower() not in {"0", "false", "off", "no"}
USE_VERIFY = env("USE_VERIFY", default="1").lower() not in {"0", "false", "off", "no"}
USE_EMBED = env("USE_EMBED", default="1").lower() not in {"0", "false", "off", "no"}

CHILD_MAX = 900
CHILD_OVL = 120
BLOCK_MAX = 2000
MAX_EMBED_CHILDREN = env_int("MAX_EMBED_CHILDREN", 16000, 500, 60000)

FALLBACK = (
    "В базе знаний нет документов по данному вопросу, поэтому ответить по документам я не могу. "
    "По памяти отвечать не буду, так как нормы могли измениться. Для ответа обратитесь в "
    "юридический отдел или отдел кадров."
)

_LOG_LOCK = threading.Lock()


def log(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def warn(msg: str) -> None:
    with _LOG_LOCK:
        print(f"[{time.time() - T0:7.1f}s] WARN {msg}", file=sys.stderr, flush=True)


def left() -> float:
    return DEADLINE - time.time()


# --------------------------------------------------------------------------- #
# HTTP-слой: один пул соединений, ограниченные ретраи, учёт дедлайна
# --------------------------------------------------------------------------- #
class ApiError(RuntimeError):
    def __init__(self, msg: str, permanent: bool = False):
        super().__init__(msg)
        self.permanent = permanent


_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def client() -> httpx.Client:
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = httpx.Client(
                timeout=httpx.Timeout(180.0, connect=15.0),
                limits=httpx.Limits(
                    max_connections=MAX_WORKERS * 3 + 6,
                    max_keepalive_connections=MAX_WORKERS * 2 + 4,
                ),
                headers={"Authorization": "Bearer " + API_KEY},
            )
        return _CLIENT


def _bases() -> list[str]:
    out = [API_BASE]
    if API_BASE.endswith("/v1"):
        out.append(API_BASE[:-3])
    else:
        out.append(API_BASE + "/v1")
    return out


def post(path: str, payload: dict, *, key: str = "", retries: int = 3,
         timeout: float = 180.0, deadline: float | None = None) -> dict:
    auth_key = key or LLM_KEY
    if not API_BASE or not auth_key:
        raise ApiError("нет LLM_BASE_URL / LLM_API_KEY", permanent=True)
    dl = DEADLINE if deadline is None else deadline
    last: Exception | None = None
    for base in _bases():
        for attempt in range(retries):
            if time.time() > dl - 2:
                raise ApiError("дедлайн запроса")
            try:
                resp = client().post(base + path, json=payload, timeout=timeout,
                                     headers={"Authorization": "Bearer " + auth_key})
            except Exception as exc:
                last = exc
                time.sleep(min(6.0, 1.0 + attempt) + random.uniform(0, 0.4))
                continue
            code = resp.status_code
            if code == 404:
                last = ApiError(f"{path} HTTP 404", permanent=True)
                break
            if code in (400, 401, 403, 422):
                raise ApiError(f"{path} HTTP {code}", permanent=True)
            if code == 429 or code >= 500:
                try:
                    wait = float(resp.headers.get("Retry-After", "0"))
                except Exception:
                    wait = 0.0
                wait = max(wait, min(8.0, 1.2 * (2 ** attempt)))
                last = ApiError(f"{path} HTTP {code}")
                if attempt < retries - 1 and time.time() + wait < dl - 2:
                    time.sleep(wait + random.uniform(0, 0.4))
                    continue
                break
            try:
                return resp.json()
            except Exception as exc:
                last = exc
                break
    raise last or ApiError(f"{path}: нет ответа")


def discover_models() -> None:
    """Прокси может отдавать ID в другом регистре — берём фактические ID из /models."""
    global GEN_MODEL, EMB_MODEL, RERANK_MODEL
    ids: list[str] = []
    for base in _bases():
        try:
            resp = client().get(base + "/models", timeout=20.0,
                                headers={"Authorization": "Bearer " + LLM_KEY})
            if resp.status_code == 200:
                items = resp.json().get("data") or []
                ids = [str(i.get("id", "")) for i in items if isinstance(i, dict)]
                if ids:
                    break
        except Exception:
            continue
    if not ids:
        return

    def pick(pred, current: str) -> str:
        if current and current in ids:
            return current
        cands = [i for i in ids if pred(i.lower())]
        return cands[0] if cands else current

    GEN_MODEL = pick(lambda s: not any(k in s for k in ("bge", "embed", "rerank", "whisper")), GEN_MODEL)
    EMB_MODEL = pick(lambda s: "bge" in s or "embed" in s, EMB_MODEL)
    RERANK_MODEL = pick(lambda s: "rerank" in s, RERANK_MODEL)
    log(f"модели: gen={GEN_MODEL} | emb={EMB_MODEL} | rerank={RERANK_MODEL}")


def chat(messages: list[dict], *, max_tokens: int, temperature: float = 0.0,
         retries: int = 3, timeout: float = 180.0, deadline: float | None = None) -> str:
    data = post(
        "/chat/completions",
        {"model": GEN_MODEL, "messages": messages,
         "temperature": temperature, "max_tokens": max_tokens},
        retries=retries, timeout=timeout, deadline=deadline,
    )
    text = ""
    try:
        content = data["choices"][0]["message"].get("content")
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
            )
        text = (content or "").strip()
    except Exception:
        text = ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()


def _embed_call(texts: list[str], deadline: float, retries: int = 2) -> np.ndarray:
    data = post("/embeddings", {"model": EMB_MODEL, "input": texts},
        key=EMBED_KEY, retries=retries, timeout=180.0, deadline=deadline)
    items = data.get("data") or []
    vecs: list = [None] * len(texts)
    for pos, item in enumerate(items):
        try:
            idx = int(item.get("index", pos))
        except Exception:
            idx = pos
        if 0 <= idx < len(texts):
            vecs[idx] = item.get("embedding")
    if any(v is None for v in vecs):
        raise ApiError("неполный ответ /embeddings")
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def _embed_recursive(texts: list[str], base: int, deadline: float, depth: int = 0) -> list:
    try:
        arr = _embed_call(texts, deadline)
        return [(base + i, arr[i]) for i in range(len(texts))]
    except ApiError as exc:
        if exc.permanent:
            raise
    except Exception:
        pass
    if depth >= 2 or len(texts) <= 1 or time.time() > deadline - 5:
        return []
    mid = len(texts) // 2
    return (_embed_recursive(texts[:mid], base, deadline, depth + 1)
            + _embed_recursive(texts[mid:], base + mid, deadline, depth + 1))


def embed_many(texts: list[str], deadline: float, batch: int = 48, workers: int = 4):
    pairs: list = []
    lock = threading.Lock()
    starts = list(range(0, len(texts), batch))

    def job(start: int) -> None:
        try:
            got = _embed_recursive(texts[start:start + batch], start, deadline)
        except Exception as exc:
            warn(f"embeddings недоступны: {type(exc).__name__}")
            got = []
        if got:
            with lock:
                pairs.extend(got)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(job, starts))
    if not pairs:
        return [], None
    pairs.sort(key=lambda p: p[0])
    ids = [p[0] for p in pairs]
    matrix = np.vstack([p[1] for p in pairs]).astype(np.float32)
    return ids, matrix


class VecIndex:
    def __init__(self, ids: list[int], matrix: np.ndarray | None):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.matrix = matrix
        self.index = None
        if matrix is not None and len(ids) and faiss is not None:
            try:
                idx = faiss.IndexFlatIP(matrix.shape[1])
                idx.add(matrix)
                self.index = idx
            except Exception:
                self.index = None

    def __len__(self) -> int:
        return len(self.ids)

    def search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self.matrix is None or not len(self.ids):
            return []
        k = max(1, min(k, len(self.ids)))
        if self.index is not None:
            sims, pos = self.index.search(q.reshape(1, -1), k)
            return [(int(self.ids[p]), float(s)) for p, s in zip(pos[0], sims[0]) if p >= 0]
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        return [(int(self.ids[p]), float(sims[p])) for p in top]


def rerank(query: str, docs_text: list[str], top_n: int, deadline: float) -> list[int]:
    payload = {"model": RERANK_MODEL, "query": query, "documents": docs_text, "top_n": top_n}
    try:
        data = post("/rerank", payload, key=RERANK_KEY, retries=2, timeout=120.0, deadline=deadline)
    except ApiError as exc:
        if not exc.permanent:
            raise
        data = post("/rerank", {"model": RERANK_MODEL, "query": query, "documents": docs_text},
        key=RERANK_KEY, retries=1, timeout=120.0, deadline=deadline)
    items = data.get("results") or data.get("data") or []
    out: list[int] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            idx = int(it.get("index", -1))
        except Exception:
            continue
        if 0 <= idx < len(docs_text) and idx not in out:
            out.append(idx)
    return out[:top_n]


# --------------------------------------------------------------------------- #
# Чтение документов
# --------------------------------------------------------------------------- #
def fix_zip_name(name: str, flag_bits: int) -> str:
    if flag_bits & 0x800:
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    for enc in ("utf-8", "cp866", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


def read_docx(raw: bytes) -> str:
    """Абзацы И таблицы в порядке следования по документу."""
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = docx.Document(io.BytesIO(raw))
        out: list[str] = []
        for child in document.element.body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                text = Paragraph(child, document).text.strip()
                if text:
                    out.append(text)
            elif tag == "tbl":
                table = Table(child, document)
                for row in table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    line = " | ".join(c for c in cells if c)
                    if line:
                        out.append(line)
        if out:
            return "\n".join(out)
    except Exception:
        pass
    from xml.etree import ElementTree as ET

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    lines = []
    for para in root.iter(ns + "p"):
        text = "".join(node.text or "" for node in para.iter(ns + "t")).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def read_pdf(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n".join(p for p in pages if p)


def read_html(raw: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw.decode("utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_file(name: str, raw: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".docx":
        text = read_docx(raw)
    elif suffix == ".pdf":
        text = read_pdf(raw)
    elif suffix in {".html", ".htm"}:
        text = read_html(raw)
    elif suffix in {".txt", ".md", ".csv", ".json"}:
        text = decode_bytes(raw)
    else:
        return ""
    text = text.replace("\u00a0", " ").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def iter_kb_files(kb_path: Path):
    if kb_path.is_dir():
        for path in sorted(kb_path.rglob("*")):
            if path.is_file():
                yield str(path.relative_to(kb_path)), path.read_bytes()
        return
    with zipfile.ZipFile(kb_path) as archive:
        for info in sorted(archive.infolist(), key=lambda i: i.filename):
            if info.filename.endswith("/") or "__MACOSX" in info.filename:
                continue
            name = fix_zip_name(info.filename, info.flag_bits)
            if Path(name).name.startswith("."):
                continue
            yield name, archive.read(info.filename)


# --------------------------------------------------------------------------- #
# Метаданные документа
# --------------------------------------------------------------------------- #
NUM_RE = re.compile(r"№\s*([A-Za-zА-Яа-я]{0,4}[-–]?\d{1,6}(?:/\d{1,4})?)")
ORG_RE = re.compile(r"(?:ООО|АО|ПАО|ЗАО|ОАО|ИП)\s*[«\"“]([^»\"”]{2,60})[»\"”]")
EDITION_RE = re.compile(r"(?:редакци\w*|ред\.)\s*(?:от\s*)?(\d{2}\.\d{2}\.\d{4})", re.I)
DATE_RE = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")


def date_ord(day: str, month: str, year: str) -> int:
    return int(year) * 10000 + int(month) * 100 + int(day)


@dataclass
class Doc:
    did: int
    name: str
    text: str
    kind: str = "other"
    number: str = ""
    parties: list[str] = field(default_factory=list)
    edition: str = ""
    edition_ord: int = 0
    family: str = ""
    outline: str = ""


def norm_family(name: str) -> str:
    text = name.lower().replace("ё", "е")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = DATE_RE.sub(" ", text)
    text = re.sub(r"редакци\w*|ред\.|от\s+\d", " ", text)
    text = re.sub(r"[^a-zа-я ]+", " ", text)
    return " ".join(sorted(set(w for w in text.split() if len(w) > 3)))

#тут косяк был, исправил
def make_doc(did: int, path_name: str, text: str) -> Doc:
    name = re.sub(r"\s+", " ", Path(path_name).stem.replace("_", " ")).strip()
    head = name + "\n" + text[:2500]
    number = ""
    m = NUM_RE.search(head)
    if m:
        number = m.group(1).replace("–", "-").strip()
    parties: list[str] = []
    for org in ORG_RE.findall(head):
        org = org.strip()
        if org and org not in parties:
            parties.append(org)
    edition, e_ord = "", 0
    m = EDITION_RE.search(head) or DATE_RE.search(name) or DATE_RE.search(head)
    if m:
        edition = m.group(0) if m.re is EDITION_RE else m.group(0)
        dm = DATE_RE.search(edition)
        if dm:
            e_ord = date_ord(dm.group(1), dm.group(2), dm.group(3))
            edition = dm.group(0)
    low = name.lower()
    articles = len(re.findall(r"(?im)^\s*статья\s+\d+", text))
    if articles >= 2 or "кодекс" in low or re.search(r"\bфз\b", low):
        kind = "law"
    elif "регламент" in low or "порядок" in low or "политик" in low:
        kind = "regulation"
    elif "договор" in low or "контракт" in low:
        kind = "contract"
    elif "записк" in low or "письм" in low:
        kind = "memo"
    else:
        kind = "other"
    return Doc(did=did, name=name, text=text, kind=kind, number=number,
               parties=parties, edition=edition, edition_ord=e_ord, family=norm_family(name))


def doc_label(doc: Doc) -> str:
    parts = [doc.name]
    if doc.number and doc.number not in doc.name:
        parts.append(f"№ {doc.number}")
    if doc.parties:
        parts.append("(" + ", ".join(doc.parties[:3]) + ")")
    if doc.edition:
        parts.append(f"редакция от {doc.edition}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Структурная нарезка: заголовки -> статьи/разделы -> пункты
# --------------------------------------------------------------------------- #
HEAD_PATTERNS = [
    (1, re.compile(r"^#{1,2}\s+(.{3,150})$")),
    (1, re.compile(r"^(?:ЧАСТЬ|Часть)\s+[IVXLCDM\d]+")),
    (2, re.compile(r"^(?:ГЛАВА|Глава|РАЗДЕЛ|Раздел)\s+[IVXLCDM\d]+")),
    (2, re.compile(r"^\d{1,2}\.\s+[А-ЯA-ZЁ][^.]{3,120}$")),
    (3, re.compile(r"^(?:Статья|СТАТЬЯ|ст\.)\s*\d+(?:\.\d+)*\.?.{0,120}$")),
    (3, re.compile(r"^#{3,6}\s+(.{3,150})$")),
]
CLAUSE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,3}){1,3})\.?\s+\S")


def match_heading(line: str):
    if not line or len(line) > 170:
        return None
    for level, pattern in HEAD_PATTERNS:
        if pattern.match(line):
            return level, line.strip().rstrip("#").strip()[:130]
    if (8 <= len(line) <= 90 and line == line.upper()
            and any(c.isalpha() for c in line) and not line.endswith(".")):
        return 1, line.strip()
    return None


def split_units(doc: Doc) -> list[tuple[str, str, str]]:
    units: list[tuple[str, str, str]] = []
    stack: list[tuple[int, str]] = []
    clause = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            units.append((" > ".join(t for _, t in stack), clause, body))

    for raw_line in doc.text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        head = match_heading(line)
        if head:
            flush()
            level, title = head
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            clause = ""
            buf = [line]
            continue
        m = CLAUSE_RE.match(line)
        if m:
            flush()
            clause = m.group(1)
            buf = [line]
            continue
        buf.append(line)
    flush()
    return units or [("", "", doc.text)]


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[.;!?])\s+", text)
    chunks: list[str] = []
    cur = ""
    for sent in sentences:
        if len(sent) > size:
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(1, size - overlap)
            for i in range(0, len(sent), step):
                chunks.append(sent[i:i + size])
            continue
        if len(cur) + len(sent) + 1 <= size:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                chunks.append(cur)
            tail = chunks[-1][-overlap:] if (chunks and overlap) else ""
            cur = (tail + " " + sent).strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()]


@dataclass
class Block:
    bid: int
    did: int
    crumb: str
    text: str
    clauses: list[str] = field(default_factory=list)


@dataclass
class Child:
    cid: int
    bid: int
    did: int
    text: str


def build_blocks(docs: list[Doc]):
    blocks: list[Block] = []
    children: list[Child] = []
    for doc in docs:
        label = doc_label(doc)
        units = split_units(doc)
        groups: list[list] = []
        for crumb, clause, body in units:
            if groups and groups[-1][0] == crumb and groups[-1][2] + len(body) <= BLOCK_MAX:
                groups[-1][1].append((clause, body))
                groups[-1][2] += len(body)
            else:
                groups.append([crumb, [(clause, body)], len(body)])
        outline_parts: list[str] = []
        for crumb, group_units, _ in groups:
            header = label + (f" | {crumb}" if crumb else "")
            body = "\n".join(text for _, text in group_units)
            block = Block(bid=len(blocks), did=doc.did, crumb=crumb,
                          text=f"[{header}]\n{body}",
                          clauses=[c for c, _ in group_units if c])
            blocks.append(block)
            if crumb:
                outline_parts.append(crumb.split(" > ")[-1])
            for clause, unit_text in group_units:
                for piece in split_text(unit_text, CHILD_MAX, CHILD_OVL):
                    children.append(Child(cid=len(children), bid=block.bid, did=doc.did,
                                          text=f"{header}\n{piece}"))
        doc.outline = build_outline(doc, units, outline_parts)
    return blocks, children


def build_outline(doc: Doc, units, sections: list[str]) -> str:
    articles: list[str] = []
    for crumb, _, body in units:
        for m in re.finditer(r"(?:Статья|ст\.)\s*(\d+(?:\.\d+)?)", crumb + "\n" + body[:120]):
            if m.group(1) not in articles:
                articles.append(m.group(1))
    clauses: list[str] = []
    for _, clause, _ in units:
        if clause and clause not in clauses:
            clauses.append(clause)
    bits: list[str] = []
    if articles:
        bits.append("статьи: " + ", ".join(sorted(articles, key=lambda s: float(s))[:40]))
    uniq_sections: list[str] = []
    for sec in sections:
        short = sec[:70]
        if short not in uniq_sections:
            uniq_sections.append(short)
    if uniq_sections and not articles:
        bits.append("разделы: " + "; ".join(uniq_sections[:10]))
    if clauses:
        bits.append("пункты: " + ", ".join(clauses[:45]))
    return "; ".join(bits)[:400]


def build_catalog(docs: list[Doc]) -> str:
    lines = []
    for doc in docs:
        line = f"- {doc_label(doc)}"
        if doc.outline:
            line += f" [{doc.outline}]"
        lines.append(line[:420])
    text = "\n".join(lines)
    return text[:3000]


# --------------------------------------------------------------------------- #
# Лексический индекс: BM25 на numpy + аккуратная русская нормализация
# --------------------------------------------------------------------------- #
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+(?:[.\-/][0-9A-Za-zА-Яа-яЁё]+)*")
SUFFIXES = (
    "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими", "ее", "ие",
    "ые", "ой", "ей", "ем", "ам", "ах", "ях", "ов", "ев", "ий", "ый", "ая", "яя",
    "ое", "ых", "их", "ют", "ят", "ат", "ет", "ит", "ла", "ло", "ли", "ть", "у",
    "ю", "а", "я", "ы", "и", "е", "о", "ь", "й",
)


def stem(word: str) -> str:
    if len(word) <= 4 or not word.isalpha():
        return word
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[:-len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for raw in WORD_RE.findall(text.lower().replace("ё", "е")):
        if len(raw) < 2:
            continue
        out.append(stem(raw))
        if re.search(r"[.\-/]", raw):
            for part in re.split(r"[.\-/]", raw):
                if len(part) >= 2:
                    out.append(stem(part))
    return out


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.dl = np.asarray([len(t) for t in corpus], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.n else 1.0
        counts: dict[str, dict[int, int]] = {}
        for did, tokens in enumerate(corpus):
            for tok in tokens:
                bucket = counts.get(tok)
                if bucket is None:
                    bucket = counts[tok] = {}
                bucket[did] = bucket.get(did, 0) + 1
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.idf: dict[str, float] = {}
        for tok, bucket in counts.items():
            df = len(bucket)
            self.idf[tok] = float(np.log(1.0 + (self.n - df + 0.5) / (df + 0.5)))
            self.postings[tok] = (
                np.asarray(list(bucket.keys()), dtype=np.int32),
                np.asarray(list(bucket.values()), dtype=np.float32),
            )

    def search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        if not self.n or not query_tokens:
            return []
        scores = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * self.dl / max(self.avgdl, 1e-6))
        for tok in set(query_tokens):
            posting = self.postings.get(tok)
            if posting is None:
                continue
            ids, tf = posting
            scores[ids] += self.idf[tok] * (tf * (self.k1 + 1.0)) / (tf + norm[ids])
        k = max(1, min(k, self.n))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0.0]


# --------------------------------------------------------------------------- #
# Разбор вопроса: диалог, дата события, контрагент, номер договора
# --------------------------------------------------------------------------- #
ROLE_RE = re.compile(r"(?:^|\n)\s*(пользователь|ассистент|user|assistant)\s*:", re.I)
MONTHS = {"январ": 1, "феврал": 2, "март": 3, "апрел": 4, "май": 5, "мая": 5, "июн": 6,
          "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12}


@dataclass
class QueryInfo:
    raw: str
    last_turn: str
    is_dialogue: bool
    search: str = ""
    event_ord: int = 0
    event_text: str = ""
    doc_hits: set = field(default_factory=set)
    number_hits: set = field(default_factory=set)
    ambiguous: bool = False


def last_user_turn(question: str) -> tuple[str, bool]:
    matches = list(ROLE_RE.finditer(question))
    if not matches:
        return question.strip(), False
    last = question.strip()
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        if m.group(1).lower() in {"пользователь", "user"}:
            last = question[m.end():end].strip()
    return last, True


def find_event_date(text: str) -> tuple[int, str]:
    m = DATE_RE.search(text)
    if m:
        return date_ord(m.group(1), m.group(2), m.group(3)), m.group(0)
    for m in re.finditer(r"([А-Яа-яЁё]{3,10})\s+(20\d{2})", text):
        word = m.group(1).lower().replace("ё", "е")
        for key, month in MONTHS.items():
            if word.startswith(key):
                return int(m.group(2)) * 10000 + month * 100 + 1, m.group(0)
    m = re.search(r"\b(20\d{2})\s*год", text)
    if m:
        return int(m.group(1)) * 10000 + 701, m.group(0)
    return 0, ""


def detect_entities(query: str, docs: list[Doc], own_orgs: set) -> tuple[set, set, bool]:
    q = query.lower().replace("ё", "е")
    q_norm = re.sub(r"[–—]", "-", query)
    q_tokens = set(re.findall(r"[a-zа-я0-9]+", q))
    number_hits: set = set()
    name_hits: set = set()
    for doc in docs:
        if doc.number and re.search(re.escape(doc.number), q_norm, re.I):
            number_hits.add(doc.did)
        for party in doc.parties:
            if party in own_orgs:
                continue
            pname = party.lower().replace("ё", "е")
            if re.search(r"(?<![а-яa-z0-9])" + re.escape(pname) + r"(?![а-яa-z0-9])", q):
                name_hits.add(doc.did)
    if number_hits or name_hits:
        return number_hits, name_hits, len(name_hits) > 1
    exact: set = set()
    prefix: set = set()
    for doc in docs:
        for party in doc.parties:
            if party in own_orgs:
                continue
            ptoks = [t for t in re.findall(r"[a-zа-я0-9]+", party.lower().replace("ё", "е"))
                     if len(t) >= 4]
            if not ptoks:
                continue
            if any(t in q_tokens for t in ptoks):
                exact.add(doc.did)
            if all(any(qt.startswith(t[:max(4, len(t) - 2)]) for qt in q_tokens) for t in ptoks):
                prefix.add(doc.did)
    hits = exact or prefix
    return number_hits, hits, len(hits) > 1


def analyze(question: str, docs: list[Doc], own_orgs: set) -> QueryInfo:
    last, is_dialogue = last_user_turn(question)
    event_ord, event_text = find_event_date(question)
    numbers, names, ambiguous = detect_entities(question, docs, own_orgs)
    return QueryInfo(raw=question, last_turn=last, is_dialogue=is_dialogue, search=last,
                     event_ord=event_ord, event_text=event_text,
                     doc_hits=set(numbers) | set(names), number_hits=set(numbers),
                     ambiguous=ambiguous)


REWRITE_SYSTEM = (
    "Ты — модуль подготовки поисковых запросов к юридической базе знаний. "
    "Тебе дан вопрос сотрудника, возможно с историей диалога. Собери ОДИН самодостаточный "
    "поисковый запрос по последней реплике пользователя: раскрой местоимения, подставь "
    "название контрагента, номер договора и предмет из истории. Добавь 2-4 юридических "
    "термина-синонима, которые встречаются в договорах и законах по этой теме. "
    "Ничего не выдумывай, не отвечай на вопрос. Верни только текст запроса одной строкой."
)


def rewrite(info: QueryInfo, deadline: float) -> str:
    need = info.is_dialogue or len(info.last_turn.split()) < 7 or re.search(
        r"\b(это|так|там|они|него|неё|них|такой|этот|эта)\b", info.last_turn.lower())
    if not need:
        return info.last_turn
    try:
        text = chat(
            [{"role": "system", "content": REWRITE_SYSTEM},
             {"role": "user", "content": info.raw}],
            max_tokens=180, retries=2, timeout=60.0, deadline=deadline,
        )
    except Exception:
        return info.last_turn
    text = text.strip().strip('"').split("\n")[0].strip()
    if 8 <= len(text) <= 400:
        return text
    return info.last_turn


def edition_plan(docs: list[Doc], event_ord: int) -> dict:
    families: dict = {}
    for doc in docs:
        # редакции бывают у регламентов/законов, но не у разных договоров
        if doc.edition_ord and doc.kind != "contract":
            families.setdefault((doc.family, doc.number), []).append(doc)
    plan: dict = {}
    for group in families.values():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda d: d.edition_ord)
        applicable = group[-1]
        if event_ord:
            applicable = group[0]
            for doc in group:
                if doc.edition_ord <= event_ord:
                    applicable = doc
        for doc in group:
            plan[doc.did] = {
                "applicable": doc.did == applicable.did,
                "applicable_edition": applicable.edition,
                "current_edition": group[-1].edition,
                "editions": [d.edition for d in group],
            }
    return plan


# --------------------------------------------------------------------------- #
# Поиск
# --------------------------------------------------------------------------- #
@dataclass
class SearchEngine:
    docs: list[Doc]
    blocks: list[Block]
    children: list[Child]
    bm25: BM25
    vec: VecIndex
    catalog: str
    own_orgs: set


def build_engine(docs: list[Doc]) -> SearchEngine:
    blocks, children = build_blocks(docs)
    log(f"чанки: {len(blocks)} блоков / {len(children)} поисковых фрагментов")
    bm25 = BM25([tokenize(c.text) for c in children])
    vec = VecIndex([], None)
    if USE_EMBED and children and API_BASE and API_KEY:
        subset = children[:MAX_EMBED_CHILDREN]
        deadline = min(T0 + INDEX_BUDGET, DEADLINE - 120)
        ids, matrix = embed_many([c.text for c in subset], deadline)
        if matrix is not None:
            vec = VecIndex(ids, matrix)
            log(f"векторный индекс: {len(vec)} фрагментов (dim={matrix.shape[1]})")
        else:
            warn("эмбеддинги недоступны — работаем на BM25")
    org_counts: dict[str, int] = {}
    for doc in docs:
        for party in doc.parties:
            org_counts[party] = org_counts.get(party, 0) + 1
    own = {p for p, n in org_counts.items() if n > max(1, len(docs) // 2)}
    return SearchEngine(docs=docs, blocks=blocks, children=children, bm25=bm25, vec=vec,
                        catalog=build_catalog(docs), own_orgs=own)


def doc_multiplier(doc: Doc, info: QueryInfo, plan: dict, docs: list[Doc]) -> float:
    mult = 1.0
    if info.doc_hits:
        if doc.did in info.number_hits:
            mult *= 3.0
        elif doc.did in info.doc_hits:
            mult *= 2.2
        else:
            mult *= 0.9
            hit_parties = [p.lower() for d in info.doc_hits for p in docs[d].parties]
            for party in doc.parties:
                if any(party.lower() != o and party.lower()[:5] == o[:5] for o in hit_parties):
                    mult *= 0.3
                    break
    entry = plan.get(doc.did)
    if entry:
        mult *= 1.8 if entry["applicable"] else 0.55
    return mult


def retrieve(engine: SearchEngine, info: QueryInfo, deadline: float) -> list[Block]:
    if not engine.children:
        return []
    plan = edition_plan(engine.docs, info.event_ord)
    
    variants = [info.search]
    if info.last_turn and info.last_turn != info.search:
        variants.append(info.last_turn)
    cues = " ".join(sorted({engine.docs[d].number for d in info.number_hits if engine.docs[d].number}))
    if cues:
        variants.append(f"{info.search} {cues}")
    variants = variants[:3]

    q_tokens = [tokenize(v) for v in variants]
    q_vecs: list = []
    if len(engine.vec):
        try:
            arr = _embed_call(variants, min(deadline, time.time() + 45))
            q_vecs = [arr[i] for i in range(len(variants))]
        except Exception as exc:
            warn(f"вектор запроса недоступен ({type(exc).__name__})")

    rrf: dict[int, float] = {}
    k = min(TOP_K, len(engine.children))
    for vi, tokens in enumerate(q_tokens):
        weight = 1.0 if vi == 0 else 0.6
        for rank, (cid, _) in enumerate(engine.bm25.search(tokens, k)):
            rrf[cid] = rrf.get(cid, 0.0) + weight / (60.0 + rank + 1)
        if vi < len(q_vecs):
            for rank, (cid, _) in enumerate(engine.vec.search(q_vecs[vi], k)):
                rrf[cid] = rrf.get(cid, 0.0) + weight / (60.0 + rank + 1)
    if not rrf:
        return []

    agg: dict[int, float] = {}
    for cid, score in rrf.items():
        bid = engine.children[cid].bid
        agg[bid] = agg.get(bid, 0.0) + score
    best: dict[int, float] = {}
    for cid, score in rrf.items():
        bid = engine.children[cid].bid
        best[bid] = max(best.get(bid, 0.0), score)
    scored = {bid: best[bid] + 0.25 * (total - best[bid]) for bid, total in agg.items()}
    for bid in scored:
        scored[bid] *= doc_multiplier(engine.docs[engine.blocks[bid].did], info, plan, engine.docs)
    ranked = sorted(scored, key=lambda b: scored[b], reverse=True)
    if len(engine.blocks) <= 12:
        pool = [engine.blocks[b] for b in sorted(ranked)]
    else:
        pool = [engine.blocks[b] for b in ranked[:max(24, RERANK_TOP_N * 3)]]

    selected: list[Block] = pool[:RERANK_TOP_N]
    if USE_RERANK and len(pool) > 1 and left() > 60:
        query = info.search + (" " + cues if cues else "")
        try:
            order = rerank(query, [b.text for b in pool], min(RERANK_TOP_N, len(pool)), deadline)
            if order:
                selected = [pool[i] for i in order]
        except Exception as exc:
            warn(f"реранк недоступен ({type(exc).__name__})")

    chosen = {b.bid for b in selected}
    # гарантированно добавляем нужный договор / применимую и действующую редакции
    for bid in ranked[:40]:
        block = engine.blocks[bid]
        if block.bid in chosen:
            continue
        need = block.did in info.number_hits
        entry = plan.get(block.did)
        if entry and any(engine.blocks[s.bid].did in plan for s in selected):
            fam = engine.docs[block.did].family
            if any(engine.docs[s.did].family == fam for s in selected):
                need = True
        if need:
            selected.append(block)
            chosen.add(block.bid)
        if len(selected) >= RERANK_TOP_N + 4:
            break
    return selected


def assemble_context(engine: SearchEngine, selected: list[Block]) -> str:
    if not selected:
        return ""
    ordered = sorted(selected, key=lambda b: (b.did, b.bid))
    parts: list[str] = []
    used = 0
    for block in ordered:
        text = block.text
        if used + len(text) > CTX_CHARS:
            room = CTX_CHARS - used
            if room < 400:
                break
            text = text[:room]
        parts.append(text)
        used += len(text)
    # соседние блоки того же раздела — чтобы не потерять «п. 6.2» рядом с «п. 6.1»
    if used < CTX_CHARS * 0.75:
        have = {b.bid for b in ordered}
        extra: list[Block] = []
        for block in ordered:
            for delta in (-1, 1):
                nb = block.bid + delta
                if 0 <= nb < len(engine.blocks) and nb not in have:
                    cand = engine.blocks[nb]
                    if cand.did == block.did and cand.crumb.split(" > ")[0] == block.crumb.split(" > ")[0]:
                        extra.append(cand)
                        have.add(nb)
        for block in sorted(extra, key=lambda b: (b.did, b.bid)):
            if used + len(block.text) > CTX_CHARS:
                break
            parts.append(block.text)
            used += len(block.text)
        merged = sorted(ordered + extra, key=lambda b: (b.did, b.bid))
        parts, used = [], 0
        for block in merged:
            if used + len(block.text) > CTX_CHARS:
                break
            parts.append(block.text)
            used += len(block.text)
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Генерация и контроль достоверности
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """Ты — юридический ассистент компании. Отвечаешь сотруднику строго по документам из базы знаний, приведённым в сообщении пользователя.

ПРАВИЛА ДОСТОВЕРНОСТИ:
1. Используй только текст приведённых документов. Никаких внешних знаний, «общих норм права» и предположений. Каждая сумма, срок, номер пункта, статья, дата и название должны буквально следовать из текста документов.
2. Никогда не выдумывай номера пунктов/статей и не приписывай документу условия, которых в нём нет. Выдуманный факт недопустим.
3. Если в вопросе неверная предпосылка (другая сумма, срок, условие) — прямо исправь её и приведи верное значение со ссылкой на пункт.
4. Учитывай редакции и даты: для события в прошлом применяй редакцию, действовавшую на дату события; при необходимости кратко отметь, что действующая редакция иная и к тому событию не применяется.
5. Не путай похожие названия контрагентов (например, «Альфа-Трейд» и «Альфа-Трейдинг» — разные компании) и разные договоры. Отвечай по тому документу, о котором спрашивает пользователь.
6. Если нужных документов в базе нет — прямо скажи, что в базе знаний нет документов по этому вопросу, по памяти отвечать нельзя, и направь к юристу или в отдел кадров. Не подменяй ответ общими рассуждениями.
7. Если данные есть частично — ответь по имеющемуся тексту и прямо укажи, каких статей или документов в базе не хватает (сверяйся с каталогом базы знаний). Не делай вид, что отсутствующие нормы тебе известны.
8. Если в диалоге уточнение уже дано — отвечай по уточнению, повторно ничего не спрашивай.

ФОРМАТ ОТВЕТА:
- Один связный абзац, 40–90 слов, без списков, заголовков и markdown.
- На вопрос «да/нет» отвечай первым словом «Да.» или «Нет.», затем обоснование.
- Всегда указывай источник: название/номер документа и пункт (п. 3.1) либо статью (ТК РФ, ст. 70).
- Деловой тон, без вступлений «согласно предоставленным фрагментам», без пересказа вопроса."""

VERIFY_SYSTEM = """Ты — контролёр достоверности юридических ответов. Тебе даны выдержки документов, вопрос сотрудника и черновик ответа.
Проверь по документам: (1) каждую сумму, срок, номер пункта/статьи, название контрагента, номер договора и дату редакции; (2) не перепутаны ли похожие контрагенты или договоры; (3) применена ли редакция, действовавшая на дату события; (4) нет ли выдуманных фактов; (5) нет ли ложного отказа, когда ответ в документах есть.
Верни итоговый ответ: один абзац 40–90 слов, с точными ссылками на пункт/статью, в том же деловом стиле. Если черновик полностью верен — верни его без изменений. Если в черновике есть факт, которого нет в документах, убери его. Никаких пояснений и комментариев — только текст ответа."""


def hints_block(engine: SearchEngine, info: QueryInfo) -> str:
    lines: list[str] = []
    if info.event_text:
        plan = edition_plan(engine.docs, info.event_ord)
        applicable = sorted({v["applicable_edition"] for v in plan.values() if v["applicable"]})
        current = sorted({v["current_edition"] for v in plan.values()})
        lines.append(f"Дата события в вопросе: {info.event_text}.")
        if applicable:
            lines.append("Применимая на эту дату редакция: " + ", ".join(applicable) + ".")
        if current and current != applicable:
            lines.append("Действующая сейчас редакция: " + ", ".join(current)
                         + " (к событию в прошлом не применяется).")
    if info.doc_hits and not info.ambiguous:
        labels = [doc_label(engine.docs[d]) for d in sorted(info.doc_hits)]
        lines.append("Документы, о которых спрашивает пользователь: " + "; ".join(labels[:4])
                     + ". Условия других договоров и контрагентов не используй.")
    elif info.ambiguous:
        lines.append("Название контрагента в вопросе неоднозначно: опирайся на уточнение из диалога "
                     "и предмет закупки, не смешивай условия разных договоров.")
    return "\n".join(lines)


def suspicious_numbers(answer: str, context: str) -> list[str]:
    ctx = context.replace("\u00a0", " ")
    plain = set(re.findall(r"\d+", ctx))
    glued = set(re.findall(r"\d+", re.sub(r"(?<=\d) (?=\d)", "", ctx)))
    out: list[str] = []
    for m in re.finditer(r"\d[\d \u00a0]*", answer):
        raw = m.group(0).strip()
        compact = re.sub(r"[ \u00a0]", "", raw)
        if len(compact) < 2 or raw in ctx:
            continue
        if compact in plain or compact in glued:
            continue
        out.append(raw)
    return out[:8]


def clean_answer(text: str) -> str:
    out = text.strip()
    out = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", out).strip()
    out = re.sub(r"^(?:ответ|answer)\s*:\s*", "", out, flags=re.I)
    out = out.replace("**", "").replace("__", "")
    out = re.sub(r"^\s*[-•*]\s+", "", out, flags=re.M)
    out = re.sub(r"^#+\s*", "", out)
    out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    if len(out) > 9500:
        out = out[:9500].rsplit(" ", 1)[0] + "…"
    return out


def answer_question(engine: SearchEngine, question: str) -> str:
    if left() < 25:
        return FALLBACK
    info = analyze(question, engine.docs, engine.own_orgs)
    stage_dl = min(time.time() + 70, DEADLINE - 40)
    info.search = rewrite(info, stage_dl)
    selected = retrieve(engine, info, stage_dl)
    context = assemble_context(engine, selected)
    if not context.strip():
        return FALLBACK
    hints = hints_block(engine, info)
    user = (
        f"ВЫДЕРЖКИ ИЗ ДОКУМЕНТОВ:\n{context}\n\n"
        + (f"ПОДСКАЗКИ ПО ВОПРОСУ:\n{hints}\n\n" if hints else "")
        + f"ВОПРОС СОТРУДНИКА (если есть история диалога — отвечай на последнюю реплику "
          f"пользователя с учётом истории):\n{question}\n\n"
        + f"СПРАВОЧНО — КАТАЛОГ БАЗЫ ЗНАНИЙ (используй только чтобы понять, "
          f"каких документов в базе нет):\n{engine.catalog}"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    gen_timeout = max(30.0, min(180.0, left() - 20))
    try:
        draft = clean_answer(chat(messages, max_tokens=ANSWER_TOKENS, temperature=TEMPERATURE,
                                  retries=3, timeout=gen_timeout))
    except Exception as exc:
        warn(f"генерация не удалась ({type(exc).__name__})")
        return FALLBACK
    if not draft:
        return FALLBACK
    if not USE_VERIFY or left() < 100 or draft == FALLBACK:
        return draft
    bad = suspicious_numbers(draft, context)
    check_user = user + f"\n\nЧЕРНОВИК ОТВЕТА:\n{draft}"
    if bad:
        check_user += ("\n\nОСОБО ПРОВЕРЬ ЧИСЛА, которых нет в тексте документов: "
                       + ", ".join(bad) + ". Если подтверждения нет — убери их.")
    try:
        fixed = clean_answer(chat(
            [{"role": "system", "content": VERIFY_SYSTEM}, {"role": "user", "content": check_user}],
            max_tokens=ANSWER_TOKENS, temperature=0.0, retries=2,
            timeout=max(30.0, min(150.0, left() - 15)),
        ))
    except Exception:
        return draft
    if fixed and 20 <= len(fixed) <= 9500:
        return fixed
    return draft


# --------------------------------------------------------------------------- #
# Ввод/вывод и orchestration
# --------------------------------------------------------------------------- #
QUESTIONS: list[tuple[str, str]] = []
ANSWERS: list[str] = []
CSV_LOCK = threading.Lock()


def find_input(*names: str) -> Path | None:
    roots = [DATA_DIR, Path.cwd(), Path(__file__).resolve().parent,
             Path(__file__).resolve().parent / DATA_DIR, Path("data")]
    for root in roots:
        for name in names:
            path = root / name
            if path.exists():
                return path
    return None


def find_kb() -> Path | None:
    direct = find_input("knowledge-base.zip", "kb.zip", "knowledge_base.zip", "knowledge-base")
    if direct is not None:
        return direct
    for root in (DATA_DIR, Path.cwd()):
        if root.is_dir():
            zips = sorted(p for p in root.glob("*.zip") if p.is_file())
            if zips:
                return zips[0]
    return None


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {(f or "").strip().lower(): f for f in (reader.fieldnames or [])}
        id_col = fields.get("id")
        q_col = fields.get("question") or fields.get("query") or fields.get("text")
        if not id_col or not q_col:
            raise ValueError("в questions.csv нужны колонки id и question")
        rows = []
        for row in reader:
            qid = str(row.get(id_col, "")).strip()
            text = str(row.get(q_col, "")).strip()
            if qid:
                rows.append((qid, text))
    return rows


def write_submission() -> None:
    with CSV_LOCK:
        tmp = Path("submission.csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writerow(["id", "answer"])
            for (qid, _), answer in zip(QUESTIONS, ANSWERS):
                text = answer.strip() if answer and answer.strip() else FALLBACK
                writer.writerow([qid, text[:9500]])
        os.replace(tmp, "submission.csv")


def main() -> None:
    global QUESTIONS, ANSWERS
    qpath = find_input("questions.csv", "demo_questions.csv")
    if qpath is None:
        raise FileNotFoundError("не найден questions.csv")
    QUESTIONS = load_questions(qpath)
    ANSWERS = [FALLBACK] * len(QUESTIONS)
    write_submission()                       # файл валиден уже сейчас
    log(f"вопросов: {len(QUESTIONS)} | бюджет: {RUN_BUDGET}c | workers={MAX_WORKERS}")

    kb = find_kb()
    if kb is None:
        warn("база знаний не найдена — все ответы будут отказом")
        return
    docs: list[Doc] = []
    for name, raw in iter_kb_files(kb):
        try:
            text = parse_file(name, raw)
        except Exception as exc:
            warn(f"пропущен {name}: {type(exc).__name__}")
            continue
        if text:
            docs.append(make_doc(len(docs), name, text))
    if not docs:
        warn("в базе знаний нет читаемых документов")
        return
    log(f"документов: {len(docs)}")
    if API_BASE and API_KEY:
        discover_models()
    engine = build_engine(docs)
    log(f"индекс готов, осталось {left():.0f}c")

    done = [0]
    counter_lock = threading.Lock()

    def work(index: int) -> None:
        qid, question = QUESTIONS[index]
        if not question:
            return
        try:
            answer = answer_question(engine, question)
        except Exception as exc:
            warn(f"вопрос {qid}: {type(exc).__name__}")
            answer = FALLBACK
        ANSWERS[index] = answer or FALLBACK
        with counter_lock:
            done[0] += 1
            n = done[0]
        if n % 5 == 0 or n == len(QUESTIONS):
            write_submission()
            log(f"готово {n}/{len(QUESTIONS)}, осталось {left():.0f}c")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(pool.map(work, range(len(QUESTIONS))))
    write_submission()
    refusals = sum(1 for a in ANSWERS if a == FALLBACK)
    log(f"submission.csv записан: {len(ANSWERS)} строк, отказов {refusals}, "
        f"время {time.time() - T0:.0f}c")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        warn(f"фатальная ошибка: {type(exc).__name__}: {exc}")
        try:
            if QUESTIONS:
                write_submission()
        except Exception:
            pass
    sys.exit(0)