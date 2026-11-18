# Project Fleece Domain Glossary

Project Fleece is a suite of offline, local Small Language Model (SLM) and Vision-Language Model (VLM) runtime builds equipped with optical document ingestion and two-tier vectorless memory.

## Language

**Crate Motor**:
A self-contained local inference build that packages model weights, native acceleration libraries, optical document ingestion, and two-tier memory behind a single drop-in interface.
_Avoid_: Wrapper, microservice, cloud agent

**Mimir OCR**:
The local document-reading subsystem that converts scanned PDFs and raster images into coordinate-grounded Markdown blocks without external APIs.
_Avoid_: Cloud OCR, PDF parser

**Optical Sieve**:
The three-pass escalation pipeline inside Mimir OCR that routes clean text through OS-native recognition first, low-confidence crops through RapidOCR second, and stubborn crops through an isolated vision processor third.
_Avoid_: Single-pass OCR, batch scraper

**Coordinate Delta-Masking**:
The spatial filtering step that locks in high-confidence text blocks from earlier OCR passes and crops only low-confidence bounding boxes for deeper passes.
_Avoid_: Page re-scanning, full-page fallback

**Quarantined Block**:
An extracted text block whose confidence remains below the verification gate (`0.85`) or contains unreadable markers (`[?]`) after escalation, flagged for human review.
_Avoid_: Failed block, dropped text

**Living Dossier**:
The Tier 1 (L1) fast-path Markdown cache (`DOSSIER.md`) that stores verified answers and core facts for instant recall without re-scanning source documents.
_Avoid_: Prompt cache, chat log

**PageIndex**:
The Tier 2 (L2) hierarchical table-of-contents tree (`PAGE_INDEX.json`) that indexes document headings, section summaries, and token counts for vectorless retrieval.
_Avoid_: Vector store, embedding index, chunk database

**Protected Ingestion**:
The unbroken document retention rule guaranteeing that ingested reference sections are never destructively parsed, chunked, or truncated inside the active context window.
_Avoid_: Text chunking, sliding window, token slicing

**Secondary Ledger**:
The cold-storage Markdown archive (`MEMORY_LEDGER.md` and `sessions/`) that receives older conversation turns evicted when protecting ingested documents under the context budget.
_Avoid_: Trash, swap file
