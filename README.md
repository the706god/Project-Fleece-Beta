# Project: Fleece Beta 🐑👁️

**The "Crate Motor" SLM build (Multimodal Edition)** — a small, self-contained AI model that sees and reads, running **completely offline on your own machine**. No internet needed after the first setup.

**Core model:** Qwen 3.5 4B Vision Instruct (`Q4_K_M` GGUF + `mmproj-F16` GGUF)

---

## What "local agent" means here

Fleece Beta is a **tool for offloading work from frontier (cloud) agents so it can live locally**. You give it a task — including images, diagrams, and documents — and it stays on your machine, sees the visual input directly, and keeps working without calling a server.

- **No internet required.** Everything runs locally.
- **No cloud API keys.** No `OPENAI_API_KEY`, no external calls.
- **No vector databases, embeddings, or OCR.** Images are fed directly to the model via the multimodal projector (`mmproj`); documents are read as unbroken markdown.
- **No chunking, no truncation.** Text and visuals are treated as complete wholes.
- **Self-contained.** Everything the project needs sits in one folder.

---

## What's inside the crate

1. **Multimodal VLM engine (`engine/vlm_engine.py`)** — the local vision-language model with hardware acceleration (Metal / CUDA / CPU).
2. **Two-tier living memory (`memory/`)**
   - **Tier 1 — Living Dossier (`DOSSIER.md`):** a fast-path cache. Frequently asked questions and answers are served instantly without rescanning.
   - **Tier 2 — Vectorless PageIndex (`PAGE_INDEX.json`):** a plain-table of contents tree over the markdown knowledge files, plus a visual-asset index. Zero embeddings, zero chunking.
3. **Context safeguard (`memory/context_budget.py`)** — tracks text tokens and visual patch tokens so the model never runs out of context. It offloads old conversation first and keeps ingested content intact.
4. **Ingestion protocols (`ingestion/protocols.py`)** — three guarantees:
   - **Direct Visual Anti-Parsing:** diagrams, scans, and screenshots are fed to the model as images, bypassing brittle OCR.
   - **Anti-Chunking & Anti-Truncating:** structure and boundaries are preserved.

---

## Quickstart

### 1. Install the model (once)

```bash
python -m pip install -r requirements.txt
python setup_models.py
```

The first run downloads the model weights automatically if they are not present.

### 2. Run

```bash
./run.sh
```

### 3. Use inside the interactive shell

- `!img /path/to/diagram.png Describe this diagram` — pass an image to the VLM.
- `!dossier` — view the living dossier.
- `!toc` — view the table-of-contents index.
- `!budget` — check context token usage (text + visual tokens).
- `!reset` — clear conversation turns, keep the model and memory.

### 4. Run as a Python crate motor

```python
from fleece import FleeceBetaCrateMotor

motor = FleeceBetaCrateMotor()
motor.set_task("Document Inspector")
response = motor.execute("Extract all key bullet points", image_paths=["invoice.png"])
print(response)
```
