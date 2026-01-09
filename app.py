# app.py
import os
import shutil
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="LFM2-350M Chat API")

# --- Config ---
BUILD_MODEL_PATH = "/app/model/LFM2-350M-Q4_0.gguf"
RUNTIME_MODEL_PATH = "/tmp/LFM2-350M-Q4_0.gguf"
llm = None
_model_loading_error = None
_model_loaded = threading.Event()

def load_model_background():
    global llm, _model_loading_error
    try:
        # Copy model to /tmp if not already there
        if not os.path.exists(RUNTIME_MODEL_PATH):
            print("🔄 Copying model to /tmp...")
            shutil.copyfile(BUILD_MODEL_PATH, RUNTIME_MODEL_PATH)
            print("✅ Copy complete.")

        # Validate file size (~2.2 GB expected)
        size_gb = os.path.getsize(RUNTIME_MODEL_PATH) / (1024**3)
        print(f"📏 Model size: {size_gb:.2f} GB")
        if size_gb < 1.5:
            raise RuntimeError("Model file too small — likely corrupted")

        print("🧠 Loading model...")
        llm = Llama(
            model_path=RUNTIME_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        _model_loading_error = str(e)
        print(f"💥 Model load failed: {e}")
    finally:
        _model_loaded.set()

# Start loading in background thread
threading.Thread(target=load_model_background, daemon=True).start()

# --- Health check (required by Leapcell) ---
@app.get("/kaithheathcheck")
@app.get("/kaithhealth")
async def health_check():
    # Always return OK during startup
    return {"status": "starting"}

@app.get("/health")
async def full_health():
    if _model_loading_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {_model_loading_error}")
    if llm is None:
        return {"status": "loading"}
    return {"status": "ready"}

# --- Chat endpoint ---
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant."

@app.post("/chat")
async def chat(request: ChatRequest):
    # Wait up to 60 seconds for model to load
    if not _model_loaded.wait(timeout=60):
        raise HTTPException(status_code=500, detail="Model loading timeout")
    
    if _model_loading_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {_model_loading_error}")
    
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
