# download_model.py
import os
from leap_bundle import download

MODEL_NAME = "LFM2-350M"
QUANTIZATION = "Q4_0"
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}-{QUANTIZATION}.gguf")

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Downloading {MODEL_NAME} ({QUANTIZATION}) to {MODEL_PATH}...")
    download(model=MODEL_NAME, quantization=QUANTIZATION, output_dir=MODEL_DIR)
    print("Download complete.")
else:
    print("Model already exists. Skipping download.")
