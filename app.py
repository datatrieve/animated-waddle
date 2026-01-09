# app.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="LFM2-350M Chat API", version="1.0")

# --- Load model ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "LFM2-350M-Q4_0.gguf")

if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

print(f"🔍 Loading model from: {MODEL_PATH}")
try:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        verbose=False
    )
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"💥 Failed to load model: {e}")
    raise

# --- API ---
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant."

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ]
        response = llm.create_chat_completion(messages=messages)
        reply = response["choices"][0]["message"]["content"]
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
