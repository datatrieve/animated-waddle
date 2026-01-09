# app.py
import os
import shutil
import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI(title="LFM2-350M Chat API")

BUILD_MODEL_PATH = "/app/model/LFM2-350M-Q4_0.gguf"
RUNTIME_MODEL_PATH = "/tmp/LFM2-350M-Q4_0.gguf"

llm = None
_model_loading_error = None
_loading_complete = False

def load_model_background():
    global llm, _model_loading_error, _loading_complete
    start_time = time.time()
    try:
        if not os.path.exists(RUNTIME_MODEL_PATH):
            print("🔄 Copying model to /tmp...")
            shutil.copyfile(BUILD_MODEL_PATH, RUNTIME_MODEL_PATH)
            print("✅ Copy done.")

        size_mb = os.path.getsize(RUNTIME_MODEL_PATH) / (1024 * 1024)
        print(f"📏 Model size: {size_mb:.1f} MB")

        # --- Critical: reduce memory usage ---
        print("🧠 Loading model with low-memory settings...")
        llm = Llama(
            model_path=RUNTIME_MODEL_PATH,
            n_ctx=512,          # ⬅️ Reduced from 2048
            n_threads=2,        # ⬅️ Use fewer threads
            verbose=False,
            use_mmap=False,     # ⬅️ Disable memory mapping (helps on constrained envs)
            use_mlock=False,    # ⬅️ Don't lock pages in RAM
        )
        print(f"✅ Model loaded in {time.time() - start_time:.1f}s")
    except Exception as e:
        _model_loading_error = str(e)
        print(f"💥 Load failed: {e}")
    finally:
        _loading_complete = True

# Start loading
threading.Thread(target=load_model_background, daemon=True).start()

# --- Required health endpoints ---
@app.get("/kaithheathcheck")
@app.get("/kaithhealth")
async def leapcell_health():
    return {"status": "ok"}  # Must return 200 immediately

@app.get("/health")
async def full_health():
    if not _loading_complete:
        return {"status": "starting"}
    if _model_loading_error:
        return {"status": "error", "error": _model_loading_error}
    return {"status": "ready"}

# --- Chat endpoint ---
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful AI assistant."

@app.post("/chat")
async def chat(request: ChatRequest):
    # Wait up to 90 seconds for model
    for _ in range(90):
        if _loading_complete:
            break
        time.sleep(1)
    else:
        raise HTTPException(status_code=500, detail="Model loading timeout")

    if _model_loading_error:
        raise HTTPException(status_code=500, detail=_model_loading_error)
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.message}
        ]
        response = llm.create_chat_completion(messages=messages, timeout=30)
        return {"response": response["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
