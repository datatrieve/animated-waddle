# app.py
import os
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="LFM2-350M Chat API", version="1.0")

# --- Paths ---
BUILD_MODEL_DIR = "/app/model"
MODEL_FILENAME = "LFM2-350M-Q4_0.gguf"
RUNTIME_MODEL_PATH = f"/tmp/{MODEL_FILENAME}"

# --- Copy model to /tmp on startup (if not already there) ---
if not os.path.exists(RUNTIME_MODEL_PATH):
    source_path = os.path.join(BUILD_MODEL_DIR, MODEL_FILENAME)
    if not os.path.exists(source_path):
        raise RuntimeError(f"Model not found at build path: {source_path}")
    print(f"Copying model from {source_path} to {RUNTIME_MODEL_PATH}...")
    shutil.copyfile(source_path, RUNTIME_MODEL_PATH)
    print("✅ Model copied to /tmp")

# --- Load model from /tmp ---
print(f"🔍 Loading model from: {RUNTIME_MODEL_PATH}")
llm = Llama(
    model_path=RUNTIME_MODEL_PATH,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("✅ Model loaded successfully!")

# --- API ---
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant. Provide clear, concise, accurate responses."

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
