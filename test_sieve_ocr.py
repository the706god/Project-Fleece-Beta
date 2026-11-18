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
from PIL import Image, ImageDraw

import pymupdf as fitz

from engine.sieve_ocr import (
    OCRBlock,
    PassSource,
    ProgressiveOpticalSieve,
    _parse_and_clean_vlm_response,
)


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
        "Pass 3 Local VLM Fallback: Qwen2.5-VL-7B-Instruct GGUF (Non-Chat Optical Processor) on Stubborn Crops ( ~3% )",
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
    return doc_result


def run_regression_checks():
    """Verifies Pass 1 geometry preservation, Pass 2 RapidOCR, Pass 3 cleanup/routing, and rotation."""
    print("\n--- [MIMIR OCR REGRESSION CHECKS] ---")

    # 1. Clean digital PDF must not be warped/resized or falsely escalated
    sample_pdf = generate_synthetic_test_pdf()
    sieve = ProgressiveOpticalSieve(confidence_threshold=0.85, raster_dpi=200)
    doc_res = sieve.process_document(sample_pdf, use_idempotency_cache=False)
    p1 = doc_res.pages[0]
    assert (p1.width_px, p1.height_px) == (1700, 2200), f"Unexpected warp: {(p1.width_px, p1.height_px)}"
    assert "SOVEREIGN DOCUMENT INGESTION" in p1.full_page_text
    print("[PASS] Digital PDF geometry preserved (1700x2200) with zero OpenCV warp.")

    # 2. 90-degree rotated page must auto-rotate back upright
    with fitz.open(sample_pdf) as d:
        pix = d[0].get_pixmap(dpi=200, alpha=False)
        upright = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    rotated = upright.rotate(90, expand=True)
    rot_res = sieve.process_page_image(rotated, page_number=1)
    assert rot_res.applied_rotation_deg in (90, 270), f"Rotation not recovered: {rot_res.applied_rotation_deg}"
    assert "MIMIR OCR" in rot_res.full_page_text
    print(f"[PASS] Rotated page recovered ({rot_res.applied_rotation_deg} deg CCW).")

    # 3. Pass 2 RapidOCR crop extraction must read clean printed crop
    crop_img = Image.new("RGB", (520, 120), "white")
    ImageDraw.Draw(crop_img).text((24, 42), "VERIFIED AMOUNT 450.00", fill="black")
    p2_blocks = sieve._run_pass2_on_crop(crop_img, 0, 0, 520, 120, 1)
    assert any("450" in b.text for b in p2_blocks), f"Pass 2 missed crop text: {p2_blocks}"
    print("[PASS] Pass 2 RapidOCR crop recognition verified.")

    # 4. Pass 3 escalation + _parse_and_clean_vlm_response must preserve checklist steps & quotes
    txt1, _ = _parse_and_clean_vlm_response("1. Check all valves", crop_height=80)
    txt2, _ = _parse_and_clean_vlm_response('"Paid in full"', crop_height=80)
    assert txt1 == "1. Check all valves", f"Checklist truncated: {txt1!r}"
    assert txt2 == '"Paid in full"', f"Quotes stripped: {txt2!r}"

    p3_sieve = ProgressiveOpticalSieve(
        confidence_threshold=0.85,
        raster_dpi=200,
        vlm_fallback_fn=lambda img: ("Handwritten Margin Note", 0.91),
    )
    low_blk = OCRBlock(
        page_number=1,
        text="blurry",
        confidence=0.40,
        pass_source=PassSource.PASS_2_RAPIDOCR,
        bbox_x0=0.05,
        bbox_y0=0.05,
        bbox_x1=0.95,
        bbox_y1=0.95,
    )
    escalated = p3_sieve._evaluate_and_escalate_pass3([low_blk], crop_img, 520, 120, 1)
    assert len(escalated) == 1 and escalated[0].pass_source == PassSource.PASS_3_LOCAL_VLM
    assert escalated[0].text == "Handwritten Margin Note"
    print("[PASS] Pass 3 escalation & output sanitization verified.")


def main():
    parser = argparse.ArgumentParser(description="Mimir OCR: 3-Pass Progressive Optical Sieve Benchmark")
    parser.add_argument("file_path", nargs="?", default=None, help="Path to PDF or image file to process")
    args = parser.parse_args()

    target = args.file_path
    if not target:
        print("[INFO] No file specified. Generating synthetic Mimir OCR test document...")
        target = generate_synthetic_test_pdf()

    run_benchmark(target)
    if not args.file_path:
        run_regression_checks()


if __name__ == "__main__":
    main()
