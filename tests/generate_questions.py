from __future__ import annotations
import argparse, csv, io, json, os, re, sys, time, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import httpx
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

def env(n, d=""):
    v = os.getenv(n, "").strip(); return d if not v or v == n else v

BASE = (env("GEN_BASE_URL") or env("JUDGE_BASE_URL") or env("LLM_BASE_URL")).rstrip("/")
KEY = env("GEN_API_KEY") or env("JUDGE_API_KEY") or env("LLM_API_KEY")
MODEL = env("GEN_MODEL") or env("JUDGE_MODEL", "openai/gpt-oss-120b")
MAX_CHARS = 7000

def post(client, payload, retries=6):
    last = None
    for a in range(retries):
        try:
            r = client.post(BASE + "/chat/completions", headers={"Authorization": "Bearer " + KEY}, json=payload)
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("retry-after", 0) or 0) or min(40, 6 * (a + 1))
                time.sleep(wait); last = RuntimeError("HTTP " + str(r.status_code)); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            last = e
            if a < retries - 1: time.sleep(min(30, 4 * (a + 1)))
    raise last

def extract(name, raw):
    s = Path(name).suffix.lower()
    if s in (".txt",".md",".csv",".json",".html",".htm"): return raw.decode("utf-8-sig", errors="replace")
    if s == ".docx":
        with zipfile.ZipFile(io.BytesIO(raw)) as d: root = ET.fromstring(d.read("word/document.xml"))
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        return "\n".join("".join(t.text or "" for t in p.iter(ns+"t")) for p in root.iter(ns+"p"))
    if s == ".pdf":
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(io.BytesIO(raw)).pages)
    return ""

def load_kb(args):
    if args.kb_zip:
        parts = []
        with zipfile.ZipFile(args.kb_zip) as z:
            for n in sorted(z.namelist()):
                if n.endswith("/") or n.startswith("__MACOSX/"): continue
                t = extract(n, z.read(n)).strip()
                if t: parts.append("### " + n + "\n" + t)
        return "\n\n".join(parts)
    return Path(args.kb).read_text(encoding="utf-8")

BOUND = re.compile(r"(?m)^(?:Источник:\s|ДОГОВОР\s|ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ|РЕГЛАМЕНТ\s|ООО «МЕРИДИАН»|Исх\.\s)")
def split_docs(text):
    raw_parts = [p.strip() for p in BOUND.split(text) if p.strip()]
    docs, buf = [], ""
    for p in raw_parts:
        while len(p) > MAX_CHARS:
            cut = p.rfind("\n", 0, MAX_CHARS)
            if cut < MAX_CHARS // 2: cut = MAX_CHARS
            docs.append(p[:cut]); p = p[cut:]
        if len(buf) + len(p) > MAX_CHARS and buf:
            docs.append(buf); buf = p
        else:
            buf = (buf + "\n\n" + p).strip()
    if buf: docs.append(buf)
    return docs

SYS = ("Ты составляешь строгий экзамен для юридического AI-ассистента по базе знаний. "
       "Вопросы answerable ТОЛЬКО из переданного текста. Для каждого — точный эталонный ответ "
       "со ссылкой на пункт/статью и документ. Не выдумывай. Верни ТОЛЬКО JSON-массив без пояснений.")

def gen_doc(client, doc, n):
    user = ("БАЗА ЗНАНИЙ:\n" + doc + "\n\nСоставь " + str(n) + " вопросов разных типов: факты (цифры/сроки/суммы/ставки), "
            "редакции/даты, различение похожих контрагентов, multi-hop, adversarial (ложная предпосылка — эталон опровергает), "
            "unanswerable (нет в базе — эталон-отказ), dialogue (2-3 реплики, ответ на последнюю). "
            'Верни JSON-массив: [{"question","answer","source","category","type"}]')
    res = post(client, {"model": MODEL, "temperature": 0.3, "max_tokens": 2000, "reasoning_effort": "low",
                        "messages":[{"role":"system","content":SYS},{"role":"user","content":user}]})
    txt = res["choices"][0]["message"].get("content","") or ""
    m = re.search(r"\[.*\]", txt, re.DOTALL)
    if not m: return []
    try: return [x for x in json.loads(m.group(0)) if isinstance(x,dict) and x.get("question") and x.get("answer")]
    except Exception: return []

def load_demo(data_dir):
    pq = Path(data_dir)/"demo_questions.csv"; pa = Path(data_dir)/"demo_output.csv"
    if not pq.exists() or not pa.exists(): return []
    qs = {r["id"]: r["question"] for r in csv.DictReader(open(pq, encoding="utf-8-sig"))}
    ans = {r["id"]: r["answer"] for r in csv.DictReader(open(pa, encoding="utf-8-sig"))}
    return [{"question": q, "answer": ans.get(i,""), "source":"demo", "category":"demo", "type":"text"} for i,q in qs.items()]

SEED = [
 {"question":"Какой срок оплаты по договору аренды № А-08/2024?","answer":"Арендная плата вносится ежемесячно не позднее 5-го числа текущего месяца (п. 3.2 договора аренды № А-08/2024).","source":"Договор аренды № А-08/2024, п. 3.2","category":"adversarial","type":"date"},
 {"question":"Какой адрес доставки по договору № П-17/2024 для партии со спецификацией от 15 октября 2024?","answer":"Московская область, г. Химки, ул. Заводская, д. 15, склад № 3 (доп. соглашение № 1; новый адрес для спецификаций после 01.10.2024).","source":"Доп. соглашение № 1 к договору № П-17/2024","category":"version","type":"text"},
 {"question":"Какой гарантийный срок на компьютеры по договору с ООО Альфа-Трейдинг?","answer":"Конкретный срок в договоре не установлен: определяется производителем и указывается в гарантийном талоне (п. 6.1 договора № ПТ-05/2023).","source":"Договор № ПТ-05/2023, п. 6.1","category":"cross_doc","type":"text"},
 {"question":"Какой штраф за нарушение авторских прав?","answer":"В предоставленной базе знаний нет документов об ответственности за нарушение авторских прав, поэтому ответить по документам я не могу. Обратитесь к юристу.","source":"нет в базе","category":"unanswerable","type":"text"},
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb-zip", default="data/kb.zip")
    ap.add_argument("--kb", default="")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--per-doc", type=int, default=5)
    ap.add_argument("--out-questions", default="questions.csv")
    ap.add_argument("--out-golden", default="golden.json")
    args = ap.parse_args()
    if not KEY: raise SystemExit("Нет ключа в .env")
    docs = split_docs(load_kb(args))
    base = load_demo(args.data_dir) + SEED
    need = max(0, args.target - len(base))
    print("фрагментов:", len(docs), "база:", len(base), "нужно ещё:", need, flush=True)
    gen, seen = [], set()
    with httpx.Client(timeout=180) as client:
        for i, d in enumerate(docs, 1):
            if len(gen) >= need: break
            try:
                items = gen_doc(client, d, args.per_doc)
            except Exception as e:
                print("фрагмент", i, "пропущен:", type(e).__name__, file=sys.stderr); items = []
            for it in items:
                k = re.sub(r"\W+","",it["question"].lower())[:120]
                if k and k not in seen: seen.add(k); gen.append(it)
            print("фрагмент", i, "/", len(docs), "собрано", len(gen), flush=True)
            time.sleep(1)
    bank = base + gen[:need]
    golden = {}
    with open(args.out_questions,"w",encoding="utf-8",newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, lineterminator="\n"); w.writerow(["id","question"])
        for i, it in enumerate(bank, 1):
            w.writerow([str(i), it["question"]])
            golden[str(i)] = {"answer":it["answer"],"source":it.get("source",""),"category":it.get("category","fact"),"type":it.get("type","text")}
    Path(args.out_golden).write_text(json.dumps(golden, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Готово:", len(bank), "вопросов ->", args.out_questions, "+", args.out_golden)

if __name__ == "__main__":
    main()
