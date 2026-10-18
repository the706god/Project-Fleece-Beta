#!/usr/bin/env python3
"""
Project: Fleece Beta - Multimodal Model Setup Utility
Downloads verified VLM GGUF weights directly from Hugging Face into models/
"""

import os
import sys
import time
import shutil
import urllib.request
import subprocess

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = [
    {
        "name": "Qwen 3.5 4B Vision Instruct (Q4_K_M GGUF)",
        "filename": "model.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
        "expected_bytes": 2740937888,
        "min_bytes": 2_000_000_000,
    },
    {
        "name": "Qwen 3.5 4B Multimodal Projector (mmproj-F16 GGUF)",
        "filename": "mmproj.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf",
        "expected_bytes": 672423616,
        "min_bytes": 500_000_000,
    }
]

def download_file(item: dict, force: bool = False) -> str:
    target_path = os.path.join(MODEL_DIR, item["filename"])

    # If it is a symlink, verify if target exists
    if os.path.islink(target_path):
        if not os.path.exists(target_path):
            print(f"[Fleece Beta] Removing broken symlink at {target_path}...")
            os.remove(target_path)
        else:
            size = os.path.getsize(target_path)
            if size >= item["min_bytes"]:
                print(f"[✓] {item['name']} linked and verified ({size / (1024*1024):.1f} MB).")
                return target_path

    if os.path.exists(target_path) and not force:
        size = os.path.getsize(target_path)
        if size >= item["min_bytes"]:
            print(f"[✓] {item['name']} already installed ({size / (1024*1024):.1f} MB).")
            return target_path

    print("=" * 65)
    print(f"📦 [Fleece Beta] Installing {item['name']}")
    print(f"Target: {target_path}")
    print(f"Source: {item['url']}")
    print("=" * 65)

    temp_path = target_path + ".download"

    # Try curl first for maximum speed and resumability, fallback to urllib
    curl_bin = shutil.which("curl")
    if curl_bin:
        cmd = [
            curl_bin,
            "-L",
            "-C", "-",
            "--retry", "5",
            "--retry-delay", "2",
            "--connect-timeout", "30",
            "-#",
            "-o", temp_path,
            item["url"]
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) >= item["min_bytes"]:
            os.replace(temp_path, target_path)
            print(f"\n[✓] Successfully installed {item['name']} ({os.path.getsize(target_path) / (1024*1024):.1f} MB).")
            return target_path

    # Fallback: urllib.request with progress reporter
    t0 = time.time()
    def _progress(count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = count * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_tot = total_size / (1024 * 1024)
        elapsed = time.time() - t0
        speed = (downloaded / (1024 * 1024)) / max(0.1, elapsed)
        sys.stdout.write(f"\rDownloading: [{mb_down:.1f} / {mb_tot:.1f} MB] ({pct:.1f}%) - {speed:.1f} MB/s   ")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(item["url"], temp_path, reporthook=_progress)
        if os.path.getsize(temp_path) >= item["min_bytes"]:
            os.replace(temp_path, target_path)
            print(f"\n[✓] Successfully installed {item['name']} ({os.path.getsize(target_path) / (1024*1024):.1f} MB).")
            return target_path
        else:
            raise RuntimeError("Downloaded file size smaller than expected.")
    except Exception as exc:
        print(f"\n[✗] Download failed: {exc}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

def download_models(force: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for item in MODELS:
        download_file(item, force=force)

if __name__ == "__main__":
    download_models()
