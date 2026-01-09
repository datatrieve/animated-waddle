# download_model.py
import os
import subprocess
import sys

MODEL_NAME = "LFM2-350M"
QUANTIZATION = "Q4_0"
OUTPUT_DIR = "./model"
EXPECTED_FILE = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}-{QUANTIZATION}.gguf")

# Create output dir if needed
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Skip if already downloaded
if os.path.exists(EXPECTED_FILE):
    print(f"Model already exists at {EXPECTED_FILE}. Skipping download.")
    sys.exit(0)

print(f"Downloading {MODEL_NAME} ({QUANTIZATION}) to {OUTPUT_DIR}...")

try:
    # Note: --output-path (not --output-dir)
    result = subprocess.run([
        "leap-bundle", "download",
        MODEL_NAME,
        "--quantization", QUANTIZATION,
        "--output-path", OUTPUT_DIR  # ✅ Correct flag
    ], check=True, text=True, capture_output=True)
    print("✅ Download successful!")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("❌ Download failed!")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
    sys.exit(1)
except FileNotFoundError:
    print("❌ 'leap-bundle' command not found.")
    sys.exit(1)
