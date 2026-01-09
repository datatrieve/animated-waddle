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
_loading_complete = False  # New flag

def load_model_background():
    global llm, _model_loading_error, _loading_complete
    try:
        # Ensure source model exists
        if not os.path.exists(BUILD_MODEL_PATH):
            raise FileNotFoundError(f"Build model not found at {BUILD_MODEL_PATH}")

        # Copy to /tmp if needed
        if not os.path.exists(RUNTIME_MODEL_PATH):
            print("🔄 Copying model to /tmp...")
            shutil.copyfile(BUILD_MODEL_PATH, RUNTIME_MODEL_PATH)
            print("✅ Copy complete.")

        # Validate size
        size_bytes = os.path.getsize(RUNTIME_MODEL_PATH)
        size_gb = size_bytes / (1024**3)
        print(f"📏 Model size: {size_gb:.2f} GB ({size_bytes} bytes)")
        if size_bytes < 1_500_000_000:  # 1.5 GB
            raise RuntimeError(f"Model too small: {size_bytes} bytes")

        # Try to open as binary to check basic integrity
        with open(RUNTIME_MODEL_PATH, "rb") as f:
            header = f.read(4)
            if header != b"GGUF":
                raise ValueError(f"Invalid GGUF header: {header}. Expected b'GGUF'")

        print("🧠 Loading model with llama.cpp...")
        llm = Llama(
            model_path=RUNTIME_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=True  # Temporarily enable for debugging
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        _model_loading_error = str(e)
        print(f"💥 FATAL: Model failed to load: {e}")
    finally:
        _loading_complete = True
        _model_loaded.set()

# Start background loading
threading.Thread(target=load_model_background, daemon=True).start()

# --- Leapcell health checks (must return 200 quickly) ---
@app.get("/kaithheathcheck")
@app.get("/kaithhealth")
async def leapcell_health():
    return {"status": "ok"}  # Must be 200 OK immediately

# --- Real health endpoint ---
@app.get("/health")
async def full_health():
    if not _loading_complete:
        return {"status": "starting"}
    if _model_loading_error:
        return {"status": "error", "error": _model_loading_error}
    if llm is not None:
        return {"status": "ready"}
    return {"status": "unknown"}

# --- Chat endpoint ---
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant."

@app.post("/chat")
async def chat(request: ChatRequest):
    if not _model_loaded.wait(timeout=120):  # Wait up to 2 mins
        raise HTTPException(status_code=500, detail="Model loading timeout")
    
    if _model_loading_error:
        raise HTTPException(status_code=500, detail=f"Model load failed: {_model_loading_error}")
    
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ]
        response = llm.create_chat_completion(messages=messages)
        reply = response["choices"][0]["message"]["content"]
        return {"response": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
