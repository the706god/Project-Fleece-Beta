#!/usr/bin/env python3
"""
Mimir OCR — Benchmark & Verification CLI for the 3-Pass Progressive Optical Sieve.
Runs ProgressiveOpticalSieve on any input PDF or scanned image, or generates
a diagnostic multi-tier test document if no path is provided.
"""

import argparse
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF

from sieve_ocr import ProgressiveOpticalSieve


def generate_synthetic_test_pdf(out_path: str = "/tmp/mimir_sieve_test_sample.pdf") -> str:
    """Generates a synthetic PDF containing clean text, table rows, and marginal notes."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Standard Letter 8.5x11 inches

    # Pass 1: Clean printed heading and vitals
    page.insert_text(
        fitz.Point(72, 72),
        "MIMIR OCR // SOVEREIGN DOCUMENT INGESTION RECORD",
        fontsize=15,
        color=(0, 0, 0),
    )
    page.insert_text(
        fitz.Point(72, 100),
        "RECORD ID: 2026-MIMIR-0892 | CLASSIFICATION: UNBROKEN RETENTION",
        fontsize=11,
        color=(0, 0, 0),
    )
    page.insert_text(
        fitz.Point(72, 120),
        "PIPELINE: 3-Pass Progressive Optical Sieve with Coordinate Delta-Masking",
        fontsize=10,
        color=(0, 0, 0),
    )

    # Pass 2 target: Faded/degraded table headers and currency lines
    page.insert_text(
        fitz.Point(72, 160),
        "QUALITY GATES: GATE 1 (SHA-256) | GATE 2 (200 DPI) | GATE 3 (>=0.85) | GATE 5 (BBOX)",
        fontsize=9,
        color=(0.3, 0.3, 0.3),
    )
    page.insert_text(
        fitz.Point(72, 180),
        "Pass 1 OS-Native Target: ~85% of Clean Text at <150ms/page",
        fontsize=9,
        color=(0.4, 0.4, 0.4),
    )
    page.insert_text(
        fitz.Point(72, 200),
        "Pass 2 RapidOCR ONNX Target: ~12% Delta-Masked Sub-Crops",
        fontsize=9,
        color=(0.35, 0.35, 0.35),
    )

    # Pass 3 target: Marginal note
    page.insert_text(
        fitz.Point(72, 240),
        "Pass 3 Local VLM Fallback: MiniCPM-V 2.6 GGUF on Stubborn Crops ( ~3% )",
        fontsize=9,
        color=(0.1, 0.1, 0.5),
    )

    doc.save(out_path)
    doc.close()
    return out_path


def run_benchmark(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    print("=" * 80)
    print("MIMIR OCR: 3-PASS PROGRESSIVE OPTICAL SIEVE // BENCHMARK EXECUTION")
    print("=" * 80)
    print(f"Target Document : {path.resolve()}")
    print(f"File Size       : {path.stat().st_size:,} bytes")

    sieve = ProgressiveOpticalSieve(
        confidence_threshold=0.85,
        raster_dpi=200,
        crop_padding_px=12,
    )

    print("\nExecuting Mimir 3-Pass Progressive Optical Sieve across document pages...")
    t0 = time.perf_counter()
    doc_result = sieve.process_document(path)
    elapsed_total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    print("\n--- [DOCUMENT SUMMARY] ---")
    print(f"SHA-256 Hash   : {doc_result.sha256_hash}")
    print(f"Total Pages    : {doc_result.page_count}")
    print(f"Total Latency  : {doc_result.total_latency_ms:.2f} ms ({elapsed_total_ms / 1000.0:.2f} s)")
    print(f"Quarantined    : {'YES (Needs Human Review)' if doc_result.quarantined else 'NO (All Clean)'}")

    for page in doc_result.pages:
        print("\n" + "-" * 80)
        print(f"PAGE {page.page_number} REPORT ({page.width_px}x{page.height_px} px @ {page.dpi} DPI):")
        print(f"  Latency                : {page.total_latency_ms:.2f} ms")
        print(f"  Total Extracted Blocks : {len(page.blocks)}")
        print(f"  Pass 1 (OS-Native)     : {page.pass_1_count} blocks")
        print(f"  Pass 2 (RapidOCR Crop) : {page.pass_2_escalated_count} blocks")
        print(f"  Pass 3 (Local VLM Crop): {page.pass_3_escalated_count} blocks")
        print(f"  Gate 3 Quarantined     : {page.quarantined_block_count} blocks")

        print("\n  [SAMPLE EXTRACTED BLOCKS (TOP 5)]:")
        for idx, blk in enumerate(page.blocks[:5], 1):
            flag = f" [QUARANTINE: {blk.quarantine_reason}]" if blk.needs_human_review else ""
            coords = f"[{blk.bbox_x0:.4f}, {blk.bbox_y0:.4f}, {blk.bbox_x1:.4f}, {blk.bbox_y1:.4f}]"
            src = blk.pass_source.value if hasattr(blk.pass_source, "value") else blk.pass_source
            print(f"    {idx}. [{src}] Conf: {blk.confidence:.2f} | BBox: {coords}{flag}")
            print(f"       Text: \"{blk.text}\"")

    print("\n" + "=" * 80)
    print("MIMIR OCR BENCHMARK COMPLETED SUCCESSFULLY.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Mimir OCR: 3-Pass Progressive Optical Sieve Benchmark")
    parser.add_argument("file_path", nargs="?", default=None, help="Path to PDF or image file to process")
    args = parser.parse_args()

    target = args.file_path
    if not target:
        print("[INFO] No file specified. Generating synthetic Mimir OCR test document...")
        target = generate_synthetic_test_pdf()

    run_benchmark(target)


if __name__ == "__main__":
    main()
