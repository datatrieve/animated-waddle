# download_model.py
import os
import sys
import urllib.request

MODEL_URL = "https://huggingface.co/LiquidAI/LFM2-350M-GGUF/resolve/main/LFM2-350M-Q4_0.gguf?download=true"
OUTPUT_DIR = "./model"
MODEL_FILENAME = "LFM2-350M-Q4_0.gguf"
MODEL_PATH = os.path.join(OUTPUT_DIR, MODEL_FILENAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if os.path.exists(MODEL_PATH):
    print(f"✅ Model already exists at {MODEL_PATH}")
    sys.exit(0)

print(f"📥 Downloading model from: {MODEL_URL}")
print(f"   Saving to: {MODEL_PATH}")

try:
    def progress_hook(count, block_size, total_size):
        percent = min(100, int(count * block_size * 100 / total_size))
        if percent % 10 == 0:
            print(f"   Progress: {percent}%")

    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=progress_hook)
    
    size_gb = os.path.getsize(MODEL_PATH) / (1024**3)
    print(f"✅ Download complete! Size: {size_gb:.2f} GB")
    
except Exception as e:
    print(f"❌ Download failed: {e}")
    sys.exit(1)
