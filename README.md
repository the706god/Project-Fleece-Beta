# Project: Fleece Beta 🐑👁️
> **The 'Crate Motor' SLM Build (Multimodal Edition)**  
> **Core Model:** Qwen 3.5 4B Vision Instruct (`Q4_K_M` GGUF + `mmproj-F16` GGUF)

Project: Fleece Beta is an out-of-the-box multimodal Small Language Model build designed to act as a **crate motor** for Vision & Language tasks. It ships with direct image and document ingestion, context protection, anti-parsing safeguards, and a two-tier living markdown memory system.

---

## ⚡ What's in the Crate

1. **Multimodal VLM Engine (`engine/vlm_engine.py`)**
   - Pre-configured for `Qwen 3.5 4B Vision` + `mmproj.gguf`.
   - Uses native `Qwen25VLChatHandler` with hardware acceleration (Metal / CUDA / CPU).
2. **Two-Tier Living Memory Architecture (`memory/`)**
   - **Tier 1 — Living Dossier (`DOSSIER.md`):** Instant fast-path cache (~150 tokens) modeled after legacy markdown memory. Frequently requested facts and answers are cached here and served immediately without document rescanning.
   - **Continuous Auto-Updating:** Whenever a deep query resolves information from documents, the Living Dossier is automatically updated with the new Q&A pair and exact file/anchor citations.
   - **Tier 2 — Vectorless PageIndex (`PAGE_INDEX.json`):** Hierarchical Table of Contents tree navigation over markdown knowledge files in `storage/markdown/knowledge/`. Zero vector databases, zero embeddings, zero chunking.
3. **Context Limitation Safeguard (`memory/context_budget.py`)**
   - Accurately tracks both text tokens and visual patch tokens per ingested image.
   - Prevents context overflow crashes via automated rolling compaction and offloading to Markdown memory.
4. **Ingestion Protocols (`ingestion/protocols.py`)**
   - **Direct Visual Anti-Parsing:** Feed diagrams, scanned pages, blueprints, or screenshots directly to the model as images—completely bypassing fragile OCR or text parsers that break tables and formatting.
   - **Anti-Chunking & Anti-Truncating:** Guarantees structure and boundary preservation for long inputs.

---

## 🚀 Quickstart

### 1. Run via Turnkey Launcher
```bash
./run.sh
```

### 2. Interactive CLI Commands
Once inside the interactive shell, you can use built-in management commands:
- `!dossier` — View the live contents of `DOSSIER.md`.
- `!toc` — Inspect the vectorless Table of Contents knowledge index.
- `!img /path/to/diagram.png Describe this diagram` — Pass an image directly to the VLM.
- `!budget` — Check active context token utilization.
- `!reset` — Clear conversation turns while keeping model and memory intact.

### 3. CLI Direct Execution (Text + Image)
```bash
./run.sh --task "Document Inspector" --prompt "Extract all key bullet points" --image invoice.png
```

### 4. Use as a Python Crate Motor
```python
from fleece import FleeceBetaCrateMotor

motor = FleeceBetaCrateMotor()
motor.set_task("Technical Diagram Analyst")
response = motor.execute("Explain the system architecture shown here", image_paths=["architecture.png"])
print(response)
```
