# load_model.py - Script to download the LFM2 model during build
import os
import sys
import urllib.request

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "./model"
MODEL_FILE = "LFM2-1.2B-RAG-Q4_0.gguf"
# Direct download URL from Hugging Face
MODEL_URL = "https://huggingface.co/LiquidAI/LFM2-1.2B-RAG-GGUF/resolve/main/LFM2-1.2B-RAG-Q4_0.gguf?download=true"

def download_with_progress(url, destination):
    """Download file with progress bar"""
    def progress_hook(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        mb_downloaded = (count * block_size) / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f'\r   [{bar}] {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
        sys.stdout.flush()
    
    print(f"   Downloading from: {url}")
    urllib.request.urlretrieve(url, destination, progress_hook)
    print()  # New line after progress bar

def main():
    """Download the LFM2 model"""
    
    print("=" * 60)
    print("🚀 LFM2 Model Download Script")
    print("=" * 60)
    
    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        print(f"📁 Creating model directory: {MODEL_PATH}")
        os.makedirs(MODEL_PATH, exist_ok=True)
    
    model_file_path = os.path.join(MODEL_PATH, MODEL_FILE)
    
    print(f"📦 Model: {MODEL_FILE}")
    print(f"💾 Save path: {model_file_path}")
    print()
    
    # Check if model already exists
    if os.path.exists(model_file_path):
        size_mb = os.path.getsize(model_file_path) / (1024 * 1024)
        print(f"✅ Model already exists ({size_mb:.2f} MB)")
        print(f"   Skipping download")
        print()
        print("=" * 60)
        print("✅ Model setup complete!")
        print("=" * 60)
        return
    
    try:
        print(f"⬇️  Downloading {MODEL_FILE}...")
        print("   This will take 3-5 minutes...")
        print()
        
        # Download the model
        download_with_progress(MODEL_URL, model_file_path)
        
        print()
        print("✅ Model downloaded successfully!")
        
        # Verify file size
        size_mb = os.path.getsize(model_file_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        
        print()
        print("=" * 60)
        print("✅ Model setup complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during model download: {e}")
        # Clean up partial download
        if os.path.exists(model_file_path):
            os.remove(model_file_path)
        sys.exit(1)

if __name__ == "__main__":
    main()
