#!/usr/bin/env python3
"""
Project: Fleece Beta - Multimodal & Mimir OCR Model Setup Utility
Downloads verified VLM GGUF weights directly into models/
"""

import os
import sys
import time
import re
import urllib.request
import urllib.parse
import http.cookiejar

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = [
    {
        "filename": "model.gguf",
        "gdrive_id": "1uFyr6_AGL8Du3mjqRiB-2y015T01muFd",
        "fallback_url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
        "min_bytes": 2_000_000_000,
    },
    {
        "filename": "mmproj.gguf",
        "gdrive_id": "1FO6gsFbACn4IIEQ_3DkNhbxOeX0Y2ta2",
        "fallback_url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/mmproj-F16.gguf",
        "min_bytes": 500_000_000,
    },
]

MIMIR_OCR_MODELS = [
    {
        "filename": "MiniCPM-V-2.6-Q4_0.gguf",
        "gdrive_id": "1iYN45Gfyeez0cY7vHAVUxtT6c2hr8j4A",
        "fallback_url": "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/ggml-model-Q4_0.gguf",
        "min_bytes": 3_500_000_000,
    },
    {
        "filename": "MiniCPM-V-2.6-mmproj-f16.gguf",
        "gdrive_id": "1xV1iwTb7jtLR6n4pKZwdNAUKLiYge9sc",
        "fallback_url": "https://huggingface.co/openbmb/MiniCPM-V-2_6-gguf/resolve/main/mmproj-model-f16.gguf",
        "min_bytes": 800_000_000,
    },
]


def _download_stream(resp, target_path, total_size, filename, min_bytes):
    temp_path = target_path + ".download"
    t0 = time.time()
    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(temp_path, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = min(100.0, downloaded * 100.0 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                elapsed = time.time() - t0
                speed = mb_down / max(0.1, elapsed)
                sys.stdout.write(
                    f"\rDownloading {filename}: [{mb_down:.1f} / {mb_tot:.1f} MB] ({pct:.1f}%) - {speed:.1f} MB/s   "
                )
                sys.stdout.flush()

    if downloaded >= min_bytes:
        os.replace(temp_path, target_path)
        sys.stdout.write(
            f"\n[✓] {filename} verified and ready ({os.path.getsize(target_path) / (1024*1024):.1f} MB).\n"
        )
        sys.stdout.flush()
        return target_path
    else:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Download of {filename} incomplete or corrupt.")


def download_file(item: dict, force: bool = False) -> str:
    target_path = os.path.join(MODEL_DIR, item["filename"])
    filename = item["filename"]
    min_bytes = item["min_bytes"]

    if os.path.islink(target_path):
        if not os.path.exists(target_path):
            os.remove(target_path)
        else:
            size = os.path.getsize(target_path)
            if size >= min_bytes:
                print(f"[✓] {filename} verified ({size / (1024*1024):.1f} MB).")
                return target_path

    if os.path.exists(target_path) and not force:
        size = os.path.getsize(target_path)
        if size >= min_bytes:
            print(f"[✓] {filename} verified ({size / (1024*1024):.1f} MB).")
            return target_path

    print(f"[Fleece Beta] Fetching {filename}...")

    # 1. Primary Origin (Dedicated GDrive File Link)
    try:
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        url = f"https://drive.google.com/uc?export=download&id={item['gdrive_id']}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = opener.open(req)

        content_type = resp.headers.get("Content-Type", "")
        if "octet-stream" in content_type:
            total_size = int(resp.headers.get("Content-Length", 0))
            return _download_stream(resp, target_path, total_size, filename, min_bytes)
        else:
            html = resp.read().decode("utf-8", errors="ignore")
            form_match = re.search(
                r"<form[^>]*action=[\x27\x22]([^\x27\x22]+)[\x27\x22][^>]*>(.*?)</form>",
                html,
                re.DOTALL,
            )
            if form_match:
                action, body = form_match.groups()
                inputs = dict(
                    re.findall(
                        r"<input[^>]*name=[\x27\x22]([^\x27\x22]+)[\x27\x22][^>]*value=[\x27\x22]([^\x27\x22]*)[\x27\x22]",
                        body,
                    )
                )
                download_url = action + "?" + urllib.parse.urlencode(inputs)
                resp2 = opener.open(urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"}))
                total_size = int(resp2.headers.get("Content-Length", 0))
                return _download_stream(resp2, target_path, total_size, filename, min_bytes)
    except Exception:
        pass

    # 2. Secondary Fallback Mirror
    try:
        req = urllib.request.Request(item["fallback_url"], headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req)
        total_size = int(resp.headers.get("Content-Length", 0))
        return _download_stream(resp, target_path, total_size, filename, min_bytes)
    except Exception as exc:
        print(f"\n[✗] Failed to fetch {filename}: {exc}")
        raise


def download_models(force: bool = False, include_mimir_vlm: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for item in MODELS:
        download_file(item, force=force)
    if include_mimir_vlm:
        for item in MIMIR_OCR_MODELS:
            download_file(item, force=force)


if __name__ == "__main__":
    download_models(include_mimir_vlm=True)
