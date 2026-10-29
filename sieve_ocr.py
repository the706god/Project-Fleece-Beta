"""
Mimir OCR — Top-level module re-export for ProgressiveOpticalSieve.
"""

from engine.sieve_ocr import (
    PassSource,
    OCRBlock,
    PageOCRResult,
    DocumentOCRResult,
    ProgressiveOpticalSieve,
    MimirOpticalSieve,
    create_local_vlm_fallback,
)

__all__ = [
    "PassSource",
    "OCRBlock",
    "PageOCRResult",
    "DocumentOCRResult",
    "ProgressiveOpticalSieve",
    "MimirOpticalSieve",
    "create_local_vlm_fallback",
]
