# Multimodal Engine Protocols
date: 2026-11-18
tags: multimodal, vision, architecture

## Direct Visual & Mimir Sieve Ingestion Protocol
In Project Fleece Beta, documents with tables, diagrams, and complex typography are processed through the Mimir 3-Pass Progressive Optical Sieve and direct visual projection via MTMDChatHandler (Qwen 3.5 4B Vision Instruct Q8_0 + mmproj-F16).
This preserves spatial bounding boxes, prevents table breakage, and avoids text chunking distortion.

## Visual & Context Token Allocation
The working context capacity is set to 65,536 tokens via Native Metal TurboQuant (turbo4/turbo4 KV cache, with automatic q8_0 fallback).
Each direct image render is budgeted at 576 visual tokens. When context usage crosses 85%, dialogue history is offloaded to the secondary markdown ledger while preserving visual and document anchors.
