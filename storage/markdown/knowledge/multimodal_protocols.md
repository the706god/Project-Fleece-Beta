# Multimodal Engine Protocols
date: 2026-10-09
tags: multimodal, vision, architecture

## Direct Visual Ingestion Protocol
In Project Fleece Beta, documents with tables, diagrams, and complex typography are converted directly to visual images.
These images bypass the text parsing pipeline entirely and enter the model via the multimodal projector (mmproj).
This prevents table breakage, text chunking distortion, and OCR hallucination.

## Visual Token Allocation
Each ingested image or document render is assigned a fixed budget of 576 tokens.
The multimodal context manager accounts for visual patches before accepting user text prompts.
When context usage crosses 85%, text history is offloaded to the secondary markdown ledger while preserving visual anchors.
