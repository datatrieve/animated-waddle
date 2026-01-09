# download_model.py
import os
import subprocess
import sys

MODEL_NAME = "LFM2-350M"
QUANTIZATION = "Q4_0"
OUTPUT_DIR = "./model"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Check if model already exists (optional but saves time on rebuilds)
expected_file = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}-{QUANTIZATION}.gguf")
if os.path.exists(expected_file):
    print(f"Model already exists at {expected_file}. Skipping download.")
    sys.exit(0)

print(f"Downloading {MODEL_NAME} ({QUANTIZATION}) to {OUTPUT_DIR}...")
try:
    result = subprocess.run([
        "leap-bundle", "download",
        MODEL_NAME,
        "--quantization", QUANTIZATION,
        "--output-dir", OUTPUT_DIR
    ], check=True, text=True, capture_output=True)
    print("Download successful!")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("Download failed!")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
    sys.exit(1)
except FileNotFoundError:
    print("Error: 'leap-bundle' command not found. Make sure it's installed.")
    sys.exit(1)
