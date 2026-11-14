import os, httpx
from dotenv import load_dotenv
load_dotenv()
base = (os.getenv("GEN_BASE_URL") or os.getenv("JUDGE_BASE_URL") or "").rstrip("/")
key = os.getenv("GEN_API_KEY") or os.getenv("JUDGE_API_KEY") or ""
model = os.getenv("GEN_MODEL") or os.getenv("JUDGE_MODEL") or ""
print("BASE:", base)
print("MODEL:", model)
print("KEY:", (key[:4] + "... (" + str(len(key)) + " символов)") if key else "НЕТ КЛЮЧА")
try:
    r = httpx.post(base + "/chat/completions",
                   headers={"Authorization": "Bearer " + key},
                   json={"model": model, "messages":[{"role":"user","content":"Скажи OK"}], "max_tokens": 10},
                   timeout=60)
    print("STATUS:", r.status_code)
    print("BODY:", r.text[:1000])
except Exception as e:
    print("ОШИБКА:", type(e).__name__, str(e)[:500])
