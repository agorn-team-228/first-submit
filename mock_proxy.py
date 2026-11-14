import hashlib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class EmbedRequest(BaseModel):
    input: list[str] | str


class RerankRequest(BaseModel):
    documents: list[str]
    top_n: int = 5


class ChatRequest(BaseModel):
    messages: list[dict]


@app.post("/embeddings")
def mock_embeddings(req: EmbedRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    data = []
    for idx, text in enumerate(texts):
        # Генерируем детерминированный вектор размерности 1024 (как у bge-m3)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [((digest[i % 32] / 255.0) - 0.5) for i in range(1024)]
        data.append({"index": idx, "embedding": vec})
    return {"data": data}


@app.post("/rerank")
def mock_rerank(req: RerankRequest):
    count = min(req.top_n, len(req.documents))
    results = [{"index": i, "relevance_score": 1.0 - i * 0.05} for i in range(count)]
    return {"results": results}


@app.post("/chat/completions")
def mock_chat(req: ChatRequest):
    user_msg = req.messages[-1].get("content", "")
    preview = user_msg[:180].replace("\n", " ")
    answer = f"[ТЕСТОВЫЙ ОТВЕТ] Контекст успешно передан в модель. Начало запроса: {preview}..."
    return {"choices": [{"message": {"content": answer}}]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
