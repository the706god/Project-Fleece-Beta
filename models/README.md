# Model Directory: Project Fleece Beta

This folder holds the local AI model brain and vision files for Project Fleece Beta.

Once these files are downloaded to your computer, Fleece Beta runs completely offline. It never sends your pictures, scans, or screenshots to the cloud.

---

## What Is in This Folder

1. Core Vision-Language Model (`model.gguf`)
   - Brain: Qwen 3.5 4B Vision Instruct.
   - Precision: Full Q8_0 precision (high accuracy for text and pictures).
   - What It Does: Understands what pictures look like and reads text within context.
   - Memory Speed: Uses TurboQuant 4-bit memory packing (turbo4/turbo4) to hold 65,536 tokens of words and image tokens together. If TurboQuant is not supported on your system, it automatically steps down to standard q8_0 memory so it never crashes.

2. Hardware Vision Projector (`mmproj.gguf`)
   - Lens File: Qwen 3.5 4B mmproj-F16.
   - What It Does: Translates raw pixels from your images into language tokens using the MTMDChatHandler so the brain can understand them.

3. Optical Sieve Specialists (Mimir OCR Pass 3)
   - Specialist Brain: `chandra-ocr-2.Q6_K.gguf`
   - Specialist Eyes: `chandra-ocr-2.mmproj-f16.gguf`
   - What They Do: These files sleep in the background. If a scanned document has messy handwriting, water stains, or tilted text that passes 1 and 2 cannot verify, this specialist wakes up, reads the bad crop, and goes back to sleep.

---

## How to Download and Verify the Models

You can download and verify all the necessary model files with one command from the project root folder:

```bash
python setup_models.py
```

This script checks whether the models are already on your computer. If they are missing or damaged, it downloads them automatically and tests their file size.

---

## Model Specifications

- Core Brain File: `model.gguf`
- Architecture: Qwen 3.5 Vision (4 Billion Parameters)
- Quantization: Q8_0 (8-bit quantization)
- Vision Projector File: `mmproj.gguf` (mmproj-F16)
- Vision Interface: Native MTMDChatHandler
- KV Cache Mode: turbo4/turbo4 (with automatic q8_0 fallback)
- Context Window: 65,536 tokens
- Default Source: Hugging Face (`unsloth/Qwen3.5-4B-GGUF`)
- OCR Fallback Model: `chandra-ocr-2.Q6_K.gguf` + `chandra-ocr-2.mmproj-f16.gguf`
