"""
Mimir OCR — Sovereign 3-Pass Progressive Optical Sieve with Coordinate Delta-Masking.
Part of the Mimir OCR & Vectorless Memory Stack for Project: Fleece.

Pass 1: OS-Native Zero-Footprint OCR (Apple Vision on macOS, Windows.Media.Ocr on Windows)
Pass 2: Targeted RapidOCR (ONNX Runtime CPU) on delta-masked low-confidence crops (< 0.85)
Pass 3: In-Process Local VLM fallback (MiniCPM-V 2.6 GGUF via llama-cpp-python) on cursive/stubborn crops
"""

import base64
import hashlib
import io
import json
import logging
import os
import sys
import time
import urllib.request
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger("MimirOCR")

# Platform-specific OS-Native OCR imports
HAS_APPLE_VISION = False
HAS_WINOCR = False

if sys.platform == "darwin":
    try:
        import Quartz
        import Vision
        from Foundation import NSData
        HAS_APPLE_VISION = True
    except ImportError:
        logger.warning("pyobjc Vision/Quartz not found; Apple Vision Pass 1 disabled.")
elif sys.platform == "win32":
    try:
        import winocr
        HAS_WINOCR = True
    except ImportError:
        logger.warning("winocr not found; Windows OCR Pass 1 disabled.")


class PassSource(str, Enum):
    PASS_1_OS_NATIVE = "PASS_1_OS_NATIVE"
    PASS_2_RAPIDOCR = "PASS_2_RAPIDOCR"
    PASS_3_LOCAL_VLM = "PASS_3_LOCAL_VLM"


class OCRBlock(BaseModel):
    block_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    page_number: int = Field(ge=1)
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    pass_source: PassSource
    # Normalized top-left origin coordinates [0.0000 to 1.0000]
    bbox_x0: float = Field(ge=0.0, le=1.0)
    bbox_y0: float = Field(ge=0.0, le=1.0)
    bbox_x1: float = Field(ge=0.0, le=1.0)
    bbox_y1: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool = False
    quarantine_reason: Optional[str] = None
    latency_ms: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def clamp_input_bbox(cls, data: Any):
        if isinstance(data, dict):
            for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1"):
                if k in data and isinstance(data[k], (int, float)):
                    data[k] = round(max(0.0, min(1.0, float(data[k]))), 4)
        return data

    @model_validator(mode="after")
    def validate_geometry(self):
        # Gate 5: Enforce valid normalized bounding box geometry
        self.bbox_x0 = round(max(0.0, min(1.0, float(self.bbox_x0))), 4)
        self.bbox_y0 = round(max(0.0, min(1.0, float(self.bbox_y0))), 4)
        self.bbox_x1 = round(max(0.0, min(1.0, float(self.bbox_x1))), 4)
        self.bbox_y1 = round(max(0.0, min(1.0, float(self.bbox_y1))), 4)
        if self.bbox_x1 <= self.bbox_x0:
            self.bbox_x1 = min(1.0, round(self.bbox_x0 + 0.001, 4))
        if self.bbox_y1 <= self.bbox_y0:
            self.bbox_y1 = min(1.0, round(self.bbox_y0 + 0.001, 4))
        return self


class PageOCRResult(BaseModel):
    page_number: int
    width_px: int
    height_px: int
    dpi: int
    full_page_text: str
    blocks: List[OCRBlock]
    pass_1_count: int
    pass_2_escalated_count: int
    pass_3_escalated_count: int
    quarantined_block_count: int
    total_latency_ms: float


class DocumentOCRResult(BaseModel):
    file_path: str
    sha256_hash: str
    page_count: int
    pages: List[PageOCRResult]
    quarantined: bool
    total_latency_ms: float


def create_local_vlm_fallback(
    model_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    lazy_load: bool = True,
) -> Callable[[Image.Image], Tuple[str, float]]:
    """
    Builds a Pass 3 Local VLM fallback callback powered by MiniCPM-V 2.6 GGUF
    via in-process llama-cpp-python (or local Ollama fallback).
    Loads weights lazily on first Pass 3 escalation so Pass 1/Pass 2 runs consume zero VLM RAM.
    """
    project_root = Path(__file__).resolve().parent.parent
    state: Dict[str, Any] = {"llama": None, "initialized": False}

    def _resolve_paths() -> Tuple[Optional[str], Optional[str]]:
        resolved_model = model_path
        resolved_mmproj = mmproj_path

        if not resolved_model:
            for c in [
                project_root / "models" / "MiniCPM-V-2.6-Q4_0.gguf",
                project_root / "models" / "MiniCPM-V.gguf",
                Path("/Users/ATLAS/Desktop/CODING/Tools/models/MiniCPM-V-2.6-Q4_0.gguf"),
            ]:
                if c.exists():
                    resolved_model = str(c)
                    break

        if not resolved_mmproj:
            for c in [
                project_root / "models" / "MiniCPM-V-2.6-mmproj-f16.gguf",
                project_root / "models" / "MiniCPM-V-mmproj.gguf",
                Path("/Users/ATLAS/Desktop/CODING/Tools/models/MiniCPM-V-2.6-mmproj-f16.gguf"),
            ]:
                if c.exists():
                    resolved_mmproj = str(c)
                    break

        return resolved_model, resolved_mmproj

    def _init_vlm():
        if state["initialized"]:
            return state["llama"]
        state["initialized"] = True
        rmodel, rmmproj = _resolve_paths()
        if rmodel and rmmproj and os.path.exists(rmodel) and os.path.exists(rmmproj):
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

                chat_handler = MiniCPMv26ChatHandler(clip_model_path=rmmproj, verbose=False)
                state["llama"] = Llama(
                    model_path=rmodel,
                    chat_handler=chat_handler,
                    n_ctx=2048,
                    n_gpu_layers=-1,
                    verbose=False,
                )
                logger.info("Mimir OCR Pass 3: In-process MiniCPM-V 2.6 GGUF initialized.")
            except Exception as e:
                logger.warning("Could not initialize in-process MiniCPM-V 2.6: %s", e)
                state["llama"] = None
        return state["llama"]

    if not lazy_load:
        _init_vlm()

    def vlm_transcribe_crop(crop_img: Image.Image) -> Tuple[str, float]:
        prompt = (
            "Transcribe the exact visible text in this cropped document image verbatim. "
            "Do not guess or hallucinate illegible words; mark any unreadable characters with [?]. "
            'Return JSON only: {"text": "...", "confidence": 0.0-1.0}'
        )

        buf = io.BytesIO()
        crop_img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        llama_instance = _init_vlm()
        if llama_instance is not None:
            try:
                uri = f"data:image/png;base64,{b64_img}"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": uri}},
                        ],
                    }
                ]
                resp = llama_instance.create_chat_completion(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=150,
                )
                raw_out = resp["choices"][0]["message"]["content"].strip()
                try:
                    data = json.loads(raw_out)
                    return str(data.get("text", "")).strip(), float(data.get("confidence", 0.90))
                except Exception:
                    clean_text = raw_out.strip('`"\'').replace("\n", " ").strip()
                    return clean_text, 0.90
            except Exception as exc:
                logger.warning("In-process MiniCPM-V Pass 3 inference error: %s", exc)

        return "", 0.0

    return vlm_transcribe_crop


class ProgressiveOpticalSieve:
    """
    Mimir OCR: 3-Pass Progressive Optical Sieve with Coordinate Delta-Masking.
    Pass 1: OS-Native OCR (Apple Vision / Windows.Media.Ocr)
    Pass 2: RapidOCR (ONNX Runtime CPU) on delta-masked low-confidence bounding boxes
    Pass 3: In-Process Local VLM callback (MiniCPM-V 2.6 GGUF) on cursive/stubborn crops
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        raster_dpi: int = 200,
        crop_padding_px: int = 12,
        vlm_fallback_fn: Optional[Callable[[Image.Image], Tuple[str, float]]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.raster_dpi = raster_dpi
        self.crop_padding_px = crop_padding_px
        self.vlm_fallback_fn = vlm_fallback_fn or create_local_vlm_fallback(lazy_load=True)
        self._rapid_ocr: Optional[RapidOCR] = None
        self._seen_hashes: Dict[str, DocumentOCRResult] = {}

    @property
    def rapid_ocr(self) -> RapidOCR:
        if self._rapid_ocr is None:
            self._rapid_ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)
        return self._rapid_ocr

    @staticmethod
    def compute_sha256(file_path: Union[str, Path]) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def process_document(self, file_path: Union[str, Path], use_idempotency_cache: bool = True) -> DocumentOCRResult:
        """
        Step 0: Pre-Flight SHA-256 (Gate 1) & 200 DPI Page Rasterization (Gate 2),
        followed by the 3-Pass Progressive Optical Sieve per page.
        """
        t0 = time.perf_counter()
        path = Path(file_path)
        sha256_hash = self.compute_sha256(path)

        # Gate 1: SHA-256 Idempotency Check
        if use_idempotency_cache and sha256_hash in self._seen_hashes:
            return self._seen_hashes[sha256_hash]

        ext = path.suffix.lower()
        pages_out: List[PageOCRResult] = []
        any_quarantined = False

        if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"):
            img = Image.open(str(path)).convert("RGB")
            page_res = self.process_page_image(img, page_number=1)
            if page_res.quarantined_block_count > 0:
                any_quarantined = True
            pages_out.append(page_res)
        else:
            doc = fitz.open(str(path))
            for idx in range(len(doc)):
                page_num = idx + 1
                page = doc[idx]
                pix = page.get_pixmap(dpi=self.raster_dpi, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_res = self.process_page_image(img, page_number=page_num)
                if page_res.quarantined_block_count > 0:
                    any_quarantined = True
                pages_out.append(page_res)
            doc.close()

        total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        result = DocumentOCRResult(
            file_path=str(path),
            sha256_hash=sha256_hash,
            page_count=len(pages_out),
            pages=pages_out,
            quarantined=any_quarantined,
            total_latency_ms=total_ms,
        )
        self._seen_hashes[sha256_hash] = result
        return result

    def process_page_image(self, image: Image.Image, page_number: int = 1) -> PageOCRResult:
        page_t0 = time.perf_counter()
        if image.mode != "RGB":
            image = image.convert("RGB")
        width_px, height_px = image.size

        # Gate 2: Resolution & Geometry Check (< 150 effective DPI check on standard 8.5x11 page)
        if width_px < 1275 or height_px < 1650:
            logger.debug(
                "Gate 2 Warning: Page %d resolution (%dx%d px) is below 150 DPI equivalent.",
                page_number,
                width_px,
                height_px,
            )

        # --- PASS 1: OS-Native OCR ---
        pass1_blocks = self._run_pass1_os_native(image, page_number)

        # Fallback: If OS-Native OCR is unavailable or returned zero blocks, run Pass 2 full-page
        if not pass1_blocks:
            full_p2_blocks = self._run_pass2_on_crop(
                crop_img=image,
                crop_x0=0,
                crop_y0=0,
                page_w=width_px,
                page_h=height_px,
                page_number=page_number,
            )
            final_blocks = self._evaluate_and_escalate_pass3(
                blocks=full_p2_blocks,
                full_image=image,
                page_w=width_px,
                page_h=height_px,
                page_number=page_number,
            )
            sorted_blocks, full_text = self._sort_and_assemble(final_blocks)
            p3_count = sum(1 for b in sorted_blocks if b.pass_source == PassSource.PASS_3_LOCAL_VLM)
            q_count = sum(1 for b in sorted_blocks if b.needs_human_review)
            return PageOCRResult(
                page_number=page_number,
                width_px=width_px,
                height_px=height_px,
                dpi=self.raster_dpi,
                full_page_text=full_text,
                blocks=sorted_blocks,
                pass_1_count=0,
                pass_2_escalated_count=len(full_p2_blocks),
                pass_3_escalated_count=p3_count,
                quarantined_block_count=q_count,
                total_latency_ms=round((time.perf_counter() - page_t0) * 1000.0, 2),
            )

        # --- COORDINATE DELTA-MASKING ---
        accepted_blocks: List[OCRBlock] = []
        escalated_to_p2: List[OCRBlock] = []

        for blk in pass1_blocks:
            if blk.confidence >= self.confidence_threshold and "[?]" not in blk.text:
                accepted_blocks.append(blk)
            else:
                escalated_to_p2.append(blk)

        pass1_accepted_count = len(accepted_blocks)
        pass2_escalated_count = len(escalated_to_p2)
        pass3_escalated_count = 0

        # --- PASS 2: RapidOCR on Cropped Low-Confidence Regions ---
        for low_blk in escalated_to_p2:
            crop_x0 = max(0, int(low_blk.bbox_x0 * width_px) - self.crop_padding_px)
            crop_y0 = max(0, int(low_blk.bbox_y0 * height_px) - self.crop_padding_px)
            crop_x1 = min(width_px, int(low_blk.bbox_x1 * width_px) + self.crop_padding_px)
            crop_y1 = min(height_px, int(low_blk.bbox_y1 * height_px) + self.crop_padding_px)

            crop_img = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            p2_sub_blocks = self._run_pass2_on_crop(
                crop_img=crop_img,
                crop_x0=crop_x0,
                crop_y0=crop_y0,
                page_w=width_px,
                page_h=height_px,
                page_number=page_number,
            )

            if p2_sub_blocks and all(b.confidence >= self.confidence_threshold for b in p2_sub_blocks):
                accepted_blocks.extend(p2_sub_blocks)
            else:
                # --- PASS 3: Local VLM Fallback on the Stubborn Crop ---
                pass3_escalated_count += 1
                best_candidate = p2_sub_blocks[0] if p2_sub_blocks else low_blk
                p3_blk = self._run_pass3_vlm_on_crop(
                    crop_img=crop_img,
                    fallback_block=best_candidate,
                    orig_bbox=(low_blk.bbox_x0, low_blk.bbox_y0, low_blk.bbox_x1, low_blk.bbox_y1),
                    page_number=page_number,
                )
                accepted_blocks.append(p3_blk)

        sorted_blocks, full_text = self._sort_and_assemble(accepted_blocks)
        q_count = sum(1 for b in sorted_blocks if b.needs_human_review)

        return PageOCRResult(
            page_number=page_number,
            width_px=width_px,
            height_px=height_px,
            dpi=self.raster_dpi,
            full_page_text=full_text,
            blocks=sorted_blocks,
            pass_1_count=pass1_accepted_count,
            pass_2_escalated_count=pass2_escalated_count,
            pass_3_escalated_count=pass3_escalated_count,
            quarantined_block_count=q_count,
            total_latency_ms=round((time.perf_counter() - page_t0) * 1000.0, 2),
        )

    def _run_pass1_os_native(self, image: Image.Image, page_number: int) -> List[OCRBlock]:
        t0 = time.perf_counter()
        blocks: List[OCRBlock] = []

        if HAS_APPLE_VISION:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            ns_data = NSData.dataWithBytes_length_(buf.getvalue(), len(buf.getvalue()))

            request = Vision.VNRecognizeTextRequest.alloc().init()
            request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
            request.setUsesLanguageCorrection_(True)

            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(ns_data, None)
            success, _ = handler.performRequests_error_([request], None)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            if success and request.results():
                for obs in request.results():
                    candidates = obs.topCandidates_(1)
                    if not candidates:
                        continue
                    top = candidates[0]
                    text = str(top.string()).strip()
                    if not text:
                        continue
                    conf = float(top.confidence())
                    bbox = obs.boundingBox()
                    # Convert Apple Vision bottom-left origin to top-left normalized [x0, y0, x1, y1]
                    x0 = max(0.0, min(1.0, float(bbox.origin.x)))
                    y0 = max(0.0, min(1.0, 1.0 - float(bbox.origin.y + bbox.size.height)))
                    x1 = max(0.0, min(1.0, float(bbox.origin.x + bbox.size.width)))
                    y1 = max(0.0, min(1.0, 1.0 - float(bbox.origin.y)))

                    blocks.append(
                        OCRBlock(
                            page_number=page_number,
                            text=text,
                            confidence=round(conf, 4),
                            pass_source=PassSource.PASS_1_OS_NATIVE,
                            bbox_x0=x0,
                            bbox_y0=y0,
                            bbox_x1=x1,
                            bbox_y1=y1,
                            latency_ms=elapsed_ms,
                        )
                    )
        elif HAS_WINOCR:
            res = winocr.recognize_pil_sync(image, lang="en")
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            w, h = image.size
            for line in res.lines:
                text = line.text.strip()
                if not text:
                    continue
                rect = line.words[0].bounding_rect if line.words else None
                if rect:
                    x0 = min(wd.bounding_rect.x for wd in line.words) / w
                    y0 = min(wd.bounding_rect.y for wd in line.words) / h
                    x1 = max(wd.bounding_rect.x + wd.bounding_rect.width for wd in line.words) / w
                    y1 = max(wd.bounding_rect.y + wd.bounding_rect.height for wd in line.words) / h
                else:
                    x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
                blocks.append(
                    OCRBlock(
                        page_number=page_number,
                        text=text,
                        confidence=0.90,
                        pass_source=PassSource.PASS_1_OS_NATIVE,
                        bbox_x0=x0,
                        bbox_y0=y0,
                        bbox_x1=x1,
                        bbox_y1=y1,
                        latency_ms=elapsed_ms,
                    )
                )

        return blocks

    def _run_pass2_on_crop(
        self,
        crop_img: Image.Image,
        crop_x0: int,
        crop_y0: int,
        page_w: int,
        page_h: int,
        page_number: int,
    ) -> List[OCRBlock]:
        t0 = time.perf_counter()
        crop_np = np.array(crop_img)
        result, _ = self.rapid_ocr(crop_np)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        blocks: List[OCRBlock] = []
        if not result:
            return blocks

        for item in result:
            poly, text, conf = item[0], str(item[1]).strip(), float(item[2])
            if not text:
                continue
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]
            # Project sub-crop pixel coordinates back onto full-page normalized [0.0, 1.0] space
            gx0 = max(0.0, min(1.0, (crop_x0 + min(xs)) / page_w))
            gy0 = max(0.0, min(1.0, (crop_y0 + min(ys)) / page_h))
            gx1 = max(0.0, min(1.0, (crop_x0 + max(xs)) / page_w))
            gy1 = max(0.0, min(1.0, (crop_y0 + max(ys)) / page_h))

            blocks.append(
                OCRBlock(
                    page_number=page_number,
                    text=text,
                    confidence=round(conf, 4),
                    pass_source=PassSource.PASS_2_RAPIDOCR,
                    bbox_x0=gx0,
                    bbox_y0=gy0,
                    bbox_x1=gx1,
                    bbox_y1=gy1,
                    latency_ms=elapsed_ms,
                )
            )
        return blocks

    def _run_pass3_vlm_on_crop(
        self,
        crop_img: Image.Image,
        fallback_block: OCRBlock,
        orig_bbox: Tuple[float, float, float, float],
        page_number: int,
    ) -> OCRBlock:
        t0 = time.perf_counter()
        if self.vlm_fallback_fn is not None:
            text, conf = self.vlm_fallback_fn(crop_img)
            source = PassSource.PASS_3_LOCAL_VLM
        else:
            text, conf = fallback_block.text, fallback_block.confidence
            source = fallback_block.pass_source

        if not text:
            text, conf = fallback_block.text, fallback_block.confidence
            source = fallback_block.pass_source

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        needs_review = conf < self.confidence_threshold or "[?]" in text
        reason = "LOW_CONFIDENCE_GATE_3" if needs_review else None

        return OCRBlock(
            page_number=page_number,
            text=text,
            confidence=round(conf, 4),
            pass_source=source,
            bbox_x0=orig_bbox[0],
            bbox_y0=orig_bbox[1],
            bbox_x1=orig_bbox[2],
            bbox_y1=orig_bbox[3],
            needs_human_review=needs_review,
            quarantine_reason=reason,
            latency_ms=elapsed_ms,
        )

    def _evaluate_and_escalate_pass3(
        self,
        blocks: List[OCRBlock],
        full_image: Image.Image,
        page_w: int,
        page_h: int,
        page_number: int,
    ) -> List[OCRBlock]:
        out: List[OCRBlock] = []
        for blk in blocks:
            if blk.confidence >= self.confidence_threshold and "[?]" not in blk.text:
                out.append(blk)
            else:
                cx0 = max(0, int(blk.bbox_x0 * page_w) - self.crop_padding_px)
                cy0 = max(0, int(blk.bbox_y0 * page_h) - self.crop_padding_px)
                cx1 = min(page_w, int(blk.bbox_x1 * page_w) + self.crop_padding_px)
                cy1 = min(page_h, int(blk.bbox_y1 * page_h) + self.crop_padding_px)
                crop_img = full_image.crop((cx0, cy0, cx1, cy1))
                out.append(
                    self._run_pass3_vlm_on_crop(
                        crop_img=crop_img,
                        fallback_block=blk,
                        orig_bbox=(blk.bbox_x0, blk.bbox_y0, blk.bbox_x1, blk.bbox_y1),
                        page_number=page_number,
                    )
                )
        return out

    @staticmethod
    def _sort_and_assemble(blocks: List[OCRBlock]) -> Tuple[List[OCRBlock], str]:
        if not blocks:
            return [], ""
        # Group into horizontal reading bands (~1.2% of page height) then sort left-to-right
        sorted_blocks = sorted(blocks, key=lambda b: (round(b.bbox_y0 / 0.012), b.bbox_x0))
        lines: List[str] = []
        current_band: Optional[int] = None
        current_line_tokens: List[str] = []

        for blk in sorted_blocks:
            band = round(blk.bbox_y0 / 0.012)
            if current_band is None or band == current_band:
                current_line_tokens.append(blk.text)
                current_band = band
            else:
                lines.append(" ".join(current_line_tokens))
                current_line_tokens = [blk.text]
                current_band = band

        if current_line_tokens:
            lines.append(" ".join(current_line_tokens))

        return sorted_blocks, "\n".join(lines)

    @staticmethod
    def format_as_mimir_markdown(doc_result: DocumentOCRResult) -> str:
        """
        Formats a DocumentOCRResult into an unbroken, coordinate-grounded Markdown document
        for direct storage in the Mimir OCR Vectorless Memory System (PageIndex + Dossier).
        """
        lines = [
            f"## Mimir OCR Document Ingestion Report",
            f"- **Source File:** `{doc_result.file_path}`",
            f"- **SHA-256 (Gate 1):** `{doc_result.sha256_hash}`",
            f"- **Pages:** {doc_result.page_count} | **Total Latency:** {doc_result.total_latency_ms:.2f} ms",
            f"- **Gate 3 Quarantine Status:** {'QUARANTINED (Needs Human Review)' if doc_result.quarantined else 'CLEAN (Verified >= 0.85)'}",
            "",
        ]
        for page in doc_result.pages:
            lines.append(
                f"## Page {page.page_number} ({page.width_px}x{page.height_px} px @ {page.dpi} DPI)"
            )
            lines.append(page.full_page_text)
            lines.append("")
            lines.append(f"### Page {page.page_number} Coordinate-Grounded Blocks")
            for blk in page.blocks:
                q_tag = f" [QUARANTINE: {blk.quarantine_reason}]" if blk.needs_human_review else ""
                bbox_str = f"[{blk.bbox_x0:.4f}, {blk.bbox_y0:.4f}, {blk.bbox_x1:.4f}, {blk.bbox_y1:.4f}]"
                lines.append(
                    f"- `{bbox_str}` ({blk.pass_source.value}, conf={blk.confidence:.2f}){q_tag}: {blk.text}"
                )
            lines.append("")
        return "\n".join(lines)


# Official Mimir OCR alias
MimirOpticalSieve = ProgressiveOpticalSieve
