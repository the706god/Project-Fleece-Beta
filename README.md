# Project: Fleece Beta 🐑👁️

**The "Crate Motor" SLM build (Multimodal Edition)** — a small, self-contained AI model equipped with **Mimir OCR** that sees and reads, running **completely offline on your own machine**. No internet needed after the first setup.

**Core model:** Qwen 3.5 4B Vision Instruct (`Q4_K_M` GGUF + `mmproj-F16` GGUF)
**Ingestion & Memory Stack:** **Mimir OCR** (3-Pass Progressive Optical Sieve with Coordinate Delta-Masking + Two-Tier Multimodal Vectorless Memory)

---

## What "local agent" means here

Fleece Beta is a **tool for offloading work from frontier (cloud) agents so it can live locally**. You give it a task — including degraded scans, PDFs, diagrams, and images — and Mimir OCR extracts coordinate-grounded `[x0, y0, x1, y1]` blocks while the VLM sees visual inputs directly without calling a server.

- **No internet required.** Everything runs locally.
- **No cloud API keys.** No `OPENAI_API_KEY`, no AWS Textract, no external calls.
- **Mimir OCR + Direct Visual Projection.** Degraded documents pass through the 3-Pass Progressive Optical Sieve with Coordinate Delta-Masking (`sieve_ocr.py`), while diagrams and schematics are fed directly via `mmproj`.
- **No chunking, no truncation.** Text and visuals are treated as complete wholes.
- **Self-contained.** Everything the project needs sits in one folder.

---

## What's inside the crate

1. **Multimodal VLM engine (`engine/vlm_engine.py`)** — the local vision-language model with hardware acceleration (Metal / CUDA / CPU).
2. **Mimir OCR (`sieve_ocr.py` & `memory/`)**
   - **3-Pass Progressive Optical Sieve with Coordinate Delta-Masking (`engine/sieve_ocr.py`):**
     - **Pass 1 (OS-Native OCR):** Apple Vision / Windows OCR clears ~85% of clean text in `<150ms/page`.
     - **Coordinate Delta-Masking:** Locks in `>= 0.85` confidence blocks and crops only `< 0.85` bounding boxes (`12px` margin at `200 DPI`).
     - **Pass 2 (Targeted `RapidOCR` ONNX):** Runs on delta-masked sub-crops (~12%: faded print, stamps, tables) and projects coordinates back to full-page `[0.0000, 1.0000]` space.
     - **Pass 3 (In-Process Local VLM via `MiniCPM-V 2.6`):** Escalates stubborn cursive/marginalia crops (~3%) at `temperature=0.0` with Gate 3 quarantine (`needs_human_review = True` if `< 0.85` or `[?]`).
   - **Tier 1 — Mimir Living Dossier (`DOSSIER.md`):** a fast-path cache. Frequently asked questions and answers are served instantly without rescanning.
   - **Tier 2 — Mimir Vectorless PageIndex (`PAGE_INDEX.json`):** a plain table-of-contents tree over the markdown knowledge files, plus a visual-asset index. Zero embeddings, zero chunking.
3. **Context safeguard (`memory/context_budget.py`)** — tracks text tokens and visual patch tokens so the model never runs out of context.
4. **Ingestion protocols (`ingestion/protocols.py`)** — three guarantees:
   - **Direct Visual & Mimir Anti-Parsing:** diagrams, scans, and PDFs preserve structure and coordinates.
   - **Anti-Chunking & Anti-Truncating:** structure and boundaries are preserved.

---

## Quickstart

### 1. Install the model (once)

```bash
python -m pip install -r requirements.txt
python setup_models.py
```

### 2. Run Mimir OCR Benchmark or Launch Shell

```bash
python test_sieve_ocr.py
./run.sh
```

### 3. Use inside the interactive shell

- `!ocr /path/to/scan.pdf` — ingest a PDF or image via the Mimir 3-Pass Progressive Optical Sieve.
- `!img /path/to/diagram.png Describe this diagram` — pass an image directly to the VLM.
- `!dossier` — view the Mimir living dossier.
- `!toc` — view the Mimir table-of-contents index.
- `!budget` — check context token usage (text + visual tokens).
- `!reset` — clear conversation turns, keep the model and memory.

### 4. Run as a Python crate motor

```python
from fleece import FleeceBetaCrateMotor

motor = FleeceBetaCrateMotor()
motor.ingest_document("scanned_ledger.pdf")
response = motor.execute("Extract all line items and their bounding boxes.")
print(response)
```
