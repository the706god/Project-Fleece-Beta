# Project: Fleece Beta

The Crate Motor AI Build (Multimodal Edition) — a private, eyes-on AI model that can see, read, and understand images and documents directly on your own computer. Once set up, it requires zero internet connection and sends zero pixels to the cloud.

Core Model: Qwen 3.5 4B Vision Instruct (Full Q8_0 Precision) + Qwen 3.5 4B Multimodal Projector (mmproj)
Vision Interface: Native MTMDChatHandler with Hardware Vision Projection
Memory & Acceleration: Native Metal TurboQuant (turbo4/turbo4) with 65,536-token context capacity
Document Stack: Mimir OCR (3-Pass Progressive Optical Sieve) + Two-Tier Multimodal Vectorless Memory

---

## Why Fleece Exists: Stop Paying Cloud Tolls for Every Picture You Scan

Frontier vision models like Claude 3.5 Sonnet and GPT-4o are impressive, but they charge you for every single pixel they look at. 

Every time you have a background agent look at a scanned invoice, check a screenshot, or inspect a photo, you are billed by the tile. If you process dozens of receipts, delivery slips, or visual charts every day, you are burning money on basic vision tasks that your own computer can do for free. 

Even worse, sending sensitive photos of legal forms, business receipts, medical documents, or proprietary software screens to cloud servers creates serious privacy and security risks.

Project Fleece Beta solves this. It is your local visual crate motor. It gives your agents eyes that run 100% on your own machine. You get unlimited visual inspections for zero dollars per month, zero token bills, zero rate limits, and 100% privacy.

---

## What Fleece Beta Is Designed For

Fleece Beta is the visual workhorse of the Fleece suite. It is designed to look at the real world: scanned pages, digital diagrams, photographs of paper, and computer screenshots.

It pairs deep optical text extraction with direct visual understanding. That means it doesn't just read the words on a page—it understands how things look, where arrows point, what boxes contain, and how layouts are arranged.

### Updated Technical Specs

- Brain: Qwen 3.5 4B Vision Instruct with full Q8_0 precision weights.
- Vision Bridge: Modern MTMDChatHandler paired with the Qwen 3.5 4B mmproj projector, providing direct hardware image understanding.
- TurboQuant KV Cache: Uses high-efficiency 4-bit memory packing (turbo4/turbo4). This cuts memory usage by nearly four times and unlocks a huge 65,536-token workspace that holds both long conversations and dense visual image tokens.
- Memory Fallback Guard: If run on an older machine without TurboQuant, it smoothly steps down to standard q8_0 memory instead of crashing.
- Zero-Destruction Guarantee: Fleece strictly protects your visual documents:
  1. Anti-Parsing: Visual layouts and bounding boxes are preserved exactly as they appear.
  2. Anti-Chunking: Multi-page forms and complex diagrams are kept whole.
  3. Anti-Truncating: Information is never chopped off mid-page.

---

## How Fleece Remembers: Two-Tier Vectorless Memory

Most AI vision tools split pages into pieces and shove them into vector databases, losing the spatial layout of where things were located. Fleece Beta uses an unbroken two-tier memory system:

- Tier 1 Memory (Fast Visual Working Memory — The Living Dossier): A live, automatically updated notepad. When an agent identifies line items, customer names, or visual issues, it records them here so future questions are answered instantly.
- Tier 2 Memory (PageIndex Tree & Visual Asset Map): Keeps a structured table-of-contents tree for all text and an exact coordinate map of every image. The AI knows not just what was said, but the exact physical rectangle on the page where it was found.
- Cold Backup (Secondary Markdown Store): Every document and visual report is archived safely on your local drive as clean, permanent Markdown.

---

## Bonus Feature: Mimir OCR

Most local vision tools leave you wiring up your own PDF readers and cloud OCR APIs just to extract text from a scan. We packaged **Mimir OCR**—the proprietary OCR stack from our upcoming **MimirOS** release—directly into Fleece Beta so you get a complete, multi-purpose local AI tool right out of the box.

How you use it is completely up to you. Whether your workflow touches medical, legal, financial, administrative, or everyday files, Mimir OCR is built to read whatever documents and forms you feed it with fidelity and accuracy—offline and under your control.

Under the hood, it runs a 3-pass pipeline (OS-native reading first, RapidOCR on low-confidence boxes, and a sleeping Chandra-OCR-2 fallback only for tough scans) without chopping or cutting off your pages. Combined with Fleece Beta's portable vision engine, Mimir OCR pulls exact text and box coordinates while Beta inspects the layout itself—bringing real-world local AI to your machine anywhere in the world.

---

## Real-World Possibilities (Industry-Agnostic)

Fleece Beta brings immediate practical value to any workflow dealing with physical paper or digital imagery:

- Paper Receipts, Invoices, and Delivery Slips: Snap photos of crumpled paper or scanned receipts. Beta extracts every line item, price, tax amount, and vendor name into clean data without uploading anything to the internet.
- Diagram and Chart Inspection: Feed in engineering flowcharts, floor plans, network maps, or business organization charts. Ask Beta to explain connections, find bottlenecks, or verify that required steps are included.
- Screenshot Troubleshooting: Drop in a screenshot of a computer error, a software bug, or a dashboard display. Beta inspects the screen and tells you what went wrong and how to fix it.
- Forms with Checkboxes and Signatures: Check whether physical agreements or intake forms have been signed, which checkboxes were marked, and whether required fields are blank.

---

## Ideal System Specifications

### Mac (Apple Silicon)
- Best: Mac with M1 Pro, M2 Pro, M3 Pro, M4, or better (16 GB to 32 GB Unified Memory).
- Minimum: M1 / M2 / M3 Mac with 16 GB Unified Memory.
- Operating System: macOS 13 (Ventura) or later.

### Windows PC
- Best: 8-core CPU with 32 GB system RAM and an NVIDIA GPU (RTX 3060, 4060, 4070 or better with 8 GB+ VRAM).
- Minimum: 6-core CPU with 16 GB RAM and a 6 GB VRAM GPU.
- Operating System: Windows 10 or 11 (64-bit).

### Linux
- Best: Modern workstation with 32 GB RAM and an NVIDIA GPU with 8 GB+ VRAM (CUDA enabled).
- Minimum: 6-core modern CPU with 16 GB RAM and 6 GB GPU VRAM.

---

## What to Expect for Response Speeds

- Text Generation Speed: 25 to 45 words per second on Apple Silicon (M1 Pro or better) and modern NVIDIA GPUs.
- Image Processing Latency: Takes only 1 to 2 seconds to inspect a high-resolution photo or document scan.
- Optical Reading Speed (Mimir OCR): Under 1 second per page for over 85% of standard visual documents.
- Resource Impact: Runs quietly in the background without locking up your desktop or making fans scream.

---

## Quickstart Guide

### 1. Set Up the Environment

```bash
python -m pip install -r requirements.txt
python setup_models.py
```

### 2. Test Document Reading or Start the Shell

```bash
# Run the built-in document reading test:
python test_sieve_ocr.py

# Launch the interactive terminal:
./run.sh
```

### 3. Useful Commands Inside the Shell

- `!img /path/to/diagram.png What does this diagram show?` — Ask a question about an image directly.
- `!ocr /path/to/scan.pdf` — Process a multi-page document through the 3-Pass Optical Sieve.
- `!audit /path/to/skill` — Audit an outside agent or prompt for Fleece compatibility.
- `!dossier` — Open your active working notepad.
- `!toc` — View the document table of contents.
- `!budget` — Check your context token usage (including image token consumption).
- `!reset` — Clear the conversation while keeping loaded images and documents intact.

### 4. Use Inside Python Scripts

```python
from fleece import FleeceBetaCrateMotor

# 1. Start the visual engine
motor = FleeceBetaCrateMotor()

# 2. Ingest a document scan
motor.ingest_document("delivery_slip.pdf")

# 3. Ask your question with full visual grounding
answer = motor.execute("Extract all line items, totals, and check if the signature box is signed.")
print(answer)
```

---

## Auditing Outside Agents and Skills

Fleece Beta includes the automated compatibility auditor. It inspects prompts, vision tools, and agent workflows written for cloud systems (like OpenAI Vision or Claude) to confirm whether they can run sovereignly on local hardware.

```bash
./audit_agent.py /path/to/my_skill/
```
