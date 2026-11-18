"""
Mimir OCR — Sovereign 3-Pass Progressive Optical Sieve with Coordinate Delta-Masking.
Unified across Project-Fleece Alpha, Beta, and Gamma.

Layer 1: OpenCV DPI normalization (low-res raster), Hough deskewing, border cleanup
Pass 1 : OS-Native Zero-Footprint OCR (Apple Vision / WinOCR) + 90/180/270 auto-rotation
Pass 2 : Targeted RapidOCR (ONNX Runtime CPU) + geometric missed-ink gap recovery
Pass 3 : Isolated Subprocess Stateless Optical Processor (Qwen2.5-VL-7B-Instruct-Q4_K_M)
         with strict non-chat transcription prompting and clean Metal teardown.
"""

from __future__ import annotations

import base64
from collections import deque
import hashlib
import io
import json
import logging
import os
import re
import selectors
import subprocess
import sys
import threading
import time
import uuid
import weakref
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import pymupdf as fitz
import cv2
import math
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger("MimirOCR")

# Platform-specific OS-Native OCR imports
HAS_APPLE_VISION = False
HAS_WINOCR = False

if sys.platform == "darwin":
    try:
        import Quartz  # noqa: F401
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


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Converts a PIL Image to an OpenCV BGR numpy array."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Converts an OpenCV BGR or grayscale numpy array to a PIL Image."""
    if len(image_bgr.shape) == 2:
        return Image.fromarray(image_bgr)
    return Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def normalize_dpi(
    image_cv: np.ndarray,
    target_dpi: int = 300,
    source_dpi: Optional[int] = None,
) -> np.ndarray:
    """
    Upscales genuinely low-resolution raster inputs (e.g., <1500px on the long edge)
    using bicubic interpolation without ballooning standard 200-300 DPI PDF renders.
    """
    h, w = image_cv.shape[:2]
    long_edge = max(h, w)

    if source_dpi and source_dpi > 0 and source_dpi < target_dpi:
        scale = min(3.0, float(target_dpi) / float(source_dpi))
    elif long_edge < 1500:
        target_long_edge = 11.0 * float(target_dpi)
        scale = min(3.0, target_long_edge / float(long_edge))
    else:
        return image_cv

    if scale > 1.1:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return cv2.resize(image_cv, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image_cv


def detect_skew_angle(image_cv: np.ndarray) -> float:
    """
    Calculates document tilt in degrees using long structural Hough lines.
    Includes 0-degree lines in the median so straight digital pages never warp.
    """
    h, w = image_cv.shape[:2]
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    min_len = max(120, int(w * 0.15))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=min_len, maxLineGap=10
    )
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        pts = np.squeeze(line)
        if pts.shape != (4,):
            continue
        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(dy, dx))
        if abs(angle) <= 15.0:
            angles.append(angle)

    if len(angles) < 5:
        return 0.0

    return float(np.median(angles))


def deskew_image(image_cv: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Straightens tilted scans based on detected Hough line angles.
    Expands canvas bounds so corners are never clipped and fills borders with white.
    """
    angle = detect_skew_angle(image_cv)
    if abs(angle) < 0.5 or abs(angle) > max_angle:
        return image_cv

    h, w = image_cv.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        image_cv,
        rot_mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def clean_scanner_borders(image_cv: np.ndarray, border_margin_pct: float = 0.02) -> np.ndarray:
    """
    Removes black scanner edge lines and hole-punch shadows along outer 2% margins.
    Replaces dark edge pixels (gray < 60) with white background (255).
    """
    h, w = image_cv.shape[:2]
    out = image_cv.copy()

    margin_x = int(w * border_margin_pct)
    margin_y = int(h * border_margin_pct)

    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY) if len(image_cv.shape) == 3 else image_cv
    dark_mask = gray < 60

    out[0:h, 0:margin_x][dark_mask[0:h, 0:margin_x]] = 255
    out[0:h, w - margin_x : w][dark_mask[0:h, w - margin_x : w]] = 255
    out[0:margin_y, 0:w][dark_mask[0:margin_y, 0:w]] = 255
    out[h - margin_y : h, 0:w][dark_mask[h - margin_y : h, 0:w]] = 255

    return out


def preprocess_document_image(
    image: Image.Image,
    target_dpi: int = 300,
    deskew: bool = True,
    clean_borders: bool = True,
) -> Image.Image:
    """
    Full Layer 1 CV preprocessing pipeline for a document page.
    """
    cv_img = pil_to_cv2(image)
    cv_img = normalize_dpi(cv_img, target_dpi=target_dpi)
    if deskew:
        cv_img = deskew_image(cv_img)
    if clean_borders:
        cv_img = clean_scanner_borders(cv_img)
    return cv2_to_pil(cv_img)


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
    applied_rotation_deg: int = 0


class DocumentOCRResult(BaseModel):
    file_path: str
    sha256_hash: str
    page_count: int
    pages: List[PageOCRResult]
    quarantined: bool
    total_latency_ms: float


def _is_garbage_vlm_text(txt: str) -> bool:
    """Returns True if VLM output consists mostly of unreadable '?' repetition or empty chatter."""
    s = (txt or "").strip()
    if not s:
        return True
    if "????" in s or s.count("[?]") >= 4:
        return True
    non_ws = re.sub(r"\s+", "", s)
    if non_ws and (non_ws.count("?") / len(non_ws)) > 0.35:
        return True
    return False


def _parse_and_clean_vlm_response(raw_out: str, crop_height: int = 0) -> Tuple[str, float]:
    """
    Sanitizes Pass 3 Qwen2.5-VL verbatim optical output:
    - Strips accidental markdown code fences.
    - Rejects empty tokens, AI refusals, image descriptions, and '?' loops.
    - Preserves valid punctuation, quotes, numbered checklist items, and dense crop lines.
    """
    if not raw_out or not raw_out.strip():
        return "", 0.0

    raw_nofence = re.sub(
        r"^```(?:markdown|text)?\s*", "", raw_out.strip(), flags=re.IGNORECASE | re.MULTILINE
    )
    raw_nofence = re.sub(r"\s*```\s*$", "", raw_nofence, flags=re.MULTILINE).strip()

    no_text_tokens = {"no_text", "[no text]", "none", "n/a", "null", "empty"}
    if raw_nofence.lower() in no_text_tokens:
        return "", 0.0

    refusal_or_caption_markers = (
        "no additional body text",
        "no readable text",
        "cannot read the text",
        "unable to transcribe",
        "as an ai",
        "i cannot transcribe",
        "i'm sorry, i",
        "there is no text",
        "this image shows",
        "this is an image of",
        "the image depicts",
        "looking at the image",
        "no text found",
        "does not contain any text",
    )
    lower_raw = raw_nofence.lower()
    if any(m in lower_raw for m in refusal_or_caption_markers):
        return "", 0.0

    clean_text = re.sub(
        r"^(?:(?:sure[!,\.\s]*)?(?:the\s+(?:handwritten\s+)?text\s+(?:in\s+the\s+image\s+)?reads|transcribed\s+text|according to (?:the )?(?:given|provided)?\s*image|here is (?:the )?(?:extracted|transcribed)?\s*(?:text|information)))\s*:?\s*",
        "",
        raw_nofence,
        flags=re.IGNORECASE,
    ).strip()

    if crop_height and crop_height >= 160 and "\n" in clean_text:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in clean_text.splitlines()]
        clean_text = "\n".join(ln for ln in lines if ln).strip(" `")
    else:
        clean_text = re.sub(r"\s+", " ", clean_text).strip(" `")

    if clean_text.lower() in no_text_tokens or _is_garbage_vlm_text(clean_text):
        return "", 0.0

    conf = 0.78 if "[?]" in clean_text else (0.92 if "\n" in clean_text or crop_height >= 160 else 0.86)
    return clean_text, (conf if clean_text else 0.0)


def _terminate_worker_refs(
    proc_box: List[Optional[subprocess.Popen]],
    timer_box: List[Optional[threading.Timer]],
) -> None:
    """Cleanup target used by weakref.finalize and explicit close()."""
    timer = timer_box[0]
    timer_box[0] = None
    if timer is not None:
        timer.cancel()

    proc = proc_box[0]
    proc_box[0] = None
    if proc is not None and proc.poll() is None:
        try:
            if proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=4)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


class _SubprocessVLMProxy:
    """
    Manages a short-lived Pass 3 Qwen2.5-VL-7B worker subprocess with an idle reaper
    and bounded I/O timeouts so Metal buffers stay isolated and callers never hang.
    """

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        idle_timeout_s: float = 20.0,
        startup_timeout_s: float = 60.0,
        inference_timeout_s: float = 45.0,
    ):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.idle_timeout_s = idle_timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.inference_timeout_s = inference_timeout_s
        self._proc_box: List[Optional[subprocess.Popen]] = [None]
        self._timer_box: List[Optional[threading.Timer]] = [None]
        self._stderr_tail: Deque[str] = deque(maxlen=25)
        self._lock = threading.Lock()
        self._worker_script = str(Path(os.path.realpath(__file__)).resolve().parent / "vlm_ocr_worker.py")
        self._finalizer = weakref.finalize(
            self, _terminate_worker_refs, self._proc_box, self._timer_box
        )

    def _drain_stderr(self, pipe) -> None:
        try:
            for line in pipe:
                s = line.strip()
                if s:
                    self._stderr_tail.append(s)
        except Exception:
            pass

    def _read_json_line(self, pipe, timeout_s: float) -> Optional[Dict[str, Any]]:
        """Reads lines from pipe up to timeout_s total until valid JSON arrives or EOF/timeout."""
        deadline = time.monotonic() + timeout_s
        sel = selectors.DefaultSelector()
        try:
            sel.register(pipe, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"VLM worker timed out after {timeout_s:.1f}s")
                events = sel.select(timeout=remaining)
                if not events:
                    raise TimeoutError(f"VLM worker timed out after {timeout_s:.1f}s")
                line = pipe.readline()
                if not line:
                    return None
                try:
                    return json.loads(line.strip())
                except Exception:
                    continue
        finally:
            sel.close()

    def _reset_idle_timer(self) -> None:
        if self._timer_box[0] is not None:
            self._timer_box[0].cancel()
        if self.idle_timeout_s > 0:
            timer = threading.Timer(self.idle_timeout_s, self.close)
            timer.daemon = True
            self._timer_box[0] = timer
            timer.start()

    def _ensure_started(self) -> bool:
        proc = self._proc_box[0]
        if proc is not None and proc.poll() is None:
            return True
        if not (os.path.exists(self.model_path) and os.path.exists(self.mmproj_path)):
            return False

        self._stderr_tail.clear()
        cmd = [sys.executable, self._worker_script, "--model", self.model_path, "--mmproj", self.mmproj_path]
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._proc_box[0] = proc
            if proc.stderr is not None:
                threading.Thread(
                    target=self._drain_stderr, args=(proc.stderr,), daemon=True
                ).start()

            assert proc.stdout is not None
            deadline = time.monotonic() + self.startup_timeout_s
            while True:
                remaining = max(0.1, deadline - time.monotonic())
                msg = self._read_json_line(proc.stdout, timeout_s=remaining)
                if msg is None:
                    logger.warning(
                        "VLM worker exited during startup. stderr tail: %s",
                        " | ".join(self._stderr_tail),
                    )
                    self.close()
                    return False
                if msg.get("status") == "ready":
                    return True
                if msg.get("status") == "error":
                    logger.warning("VLM worker startup error: %s", msg.get("error"))
                    self.close()
                    return False
        except Exception as exc:
            logger.warning(
                "Failed to start isolated VLM worker (%s). stderr tail: %s",
                exc,
                " | ".join(self._stderr_tail),
            )
            self.close()
            return False

    def transcribe(self, crop_img: Image.Image) -> Tuple[str, float]:
        # Cap crop width to keep ViT patch count fast & within n_ctx=4096
        img_to_send = crop_img
        if img_to_send.width > 1024:
            new_h = max(1, int(round(img_to_send.height * (1024.0 / img_to_send.width))))
            img_to_send = img_to_send.resize((1024, new_h), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img_to_send.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        req_id = uuid.uuid4().hex[:8]
        max_toks = 400 if (img_to_send.height >= 240 or img_to_send.width >= 800) else 220

        with self._lock:
            if not self._ensure_started():
                return "", 0.0
            proc = self._proc_box[0]
            try:
                assert proc is not None and proc.stdin is not None and proc.stdout is not None
                payload = json.dumps({"id": req_id, "image_b64": b64_img, "max_tokens": max_toks})
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()

                deadline = time.monotonic() + self.inference_timeout_s
                while True:
                    remaining = max(0.1, deadline - time.monotonic())
                    resp = self._read_json_line(proc.stdout, timeout_s=remaining)
                    if resp is None:
                        self.close()
                        return "", 0.0
                    if resp.get("id") == req_id:
                        self._reset_idle_timer()
                        raw_out = resp.get("raw_text", "")
                        return _parse_and_clean_vlm_response(raw_out, crop_height=crop_img.height)
            except Exception as exc:
                logger.warning("Isolated Pass 3 VLM worker error: %s", exc)
                self.close()
                return "", 0.0

    def close(self) -> None:
        _terminate_worker_refs(self._proc_box, self._timer_box)


def create_local_vlm_fallback(
    model_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    lazy_load: bool = True,
) -> Callable[[Image.Image], Tuple[str, float]]:
    """
    Builds a Pass 3 Local VLM optical processor powered by Qwen2.5-VL-7B-Instruct
    inside an isolated worker subprocess (stateless, non-chat transcription only).
    Resolves weights strictly relative to this repository's models/ directory.
    """
    project_root = Path(os.path.realpath(__file__)).resolve().parent.parent
    state: Dict[str, Any] = {"proxy": None, "initialized": False}

    def _resolve_paths() -> Tuple[Optional[str], Optional[str]]:
        if model_path and mmproj_path and os.path.exists(model_path) and os.path.exists(mmproj_path):
            return model_path, mmproj_path

        q_model = project_root / "models" / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
        q_mmproj = project_root / "models" / "Qwen2.5-VL-7B-Instruct-mmproj-F16.gguf"
        if q_model.exists() and q_mmproj.exists():
            return str(q_model), str(q_mmproj)
        return None, None

    def _get_proxy() -> Optional[_SubprocessVLMProxy]:
        if state["proxy"] is None and not state["initialized"]:
            state["initialized"] = True
            rmodel, rmmproj = _resolve_paths()
            if rmodel and rmmproj:
                state["proxy"] = _SubprocessVLMProxy(rmodel, rmmproj)
                logger.info("Mimir OCR Pass 3: Isolated Qwen2.5-VL-7B processor ready (%s).", os.path.basename(rmodel))
        return state["proxy"]

    if not lazy_load:
        p = _get_proxy()
        if p is not None:
            p._ensure_started()

    def vlm_transcribe_crop(crop_img: Image.Image) -> Tuple[str, float]:
        proxy = _get_proxy()
        if proxy is not None:
            return proxy.transcribe(crop_img)
        return "", 0.0

    def _close_worker() -> None:
        if state["proxy"] is not None:
            state["proxy"].close()

    vlm_transcribe_crop.close = _close_worker  # type: ignore[attr-defined]
    return vlm_transcribe_crop


class ProgressiveOpticalSieve:
    """
    Mimir OCR: 3-Pass Progressive Optical Sieve with Coordinate Delta-Masking.
    Pass 1: OS-Native OCR (Apple Vision / Windows.Media.Ocr)
    Pass 2: RapidOCR (ONNX Runtime CPU) on low-confidence crops + geometric gap scan
    Pass 3: Stateless Subprocess Qwen2.5-VL-7B Optical Processor on stubborn/cursive crops
    """

    _IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    _TEXT_EXTENSIONS = (".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml", ".html", ".log")
    _MEDIA_EXTENSIONS = (
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".aiff",
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
    )

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        raster_dpi: int = 300,
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

    def _build_single_block_page(self, page_number: int, text: str) -> PageOCRResult:
        """Builds a single-block PageOCRResult for direct text or media passthrough files."""
        blocks = [
            OCRBlock(
                page_number=page_number,
                text=text,
                confidence=1.0,
                pass_source=PassSource.PASS_1_OS_NATIVE,
                bbox_x0=0.0,
                bbox_y0=0.0,
                bbox_x1=1.0,
                bbox_y1=1.0,
            )
        ]
        return PageOCRResult(
            page_number=page_number,
            width_px=1000,
            height_px=1000,
            dpi=self.raster_dpi,
            full_page_text=text,
            blocks=blocks,
            pass_1_count=1,
            pass_2_escalated_count=0,
            pass_3_escalated_count=0,
            quarantined_block_count=0,
            total_latency_ms=0.0,
        )

    def process_document(
        self, file_path: Union[str, Path], use_idempotency_cache: bool = True
    ) -> DocumentOCRResult:
        """Pre-flight SHA-256 cache check, CV preprocessing, and 3-pass OCR per page."""
        t0 = time.perf_counter()
        path = Path(file_path)
        sha256_hash = self.compute_sha256(path)

        if use_idempotency_cache and sha256_hash in self._seen_hashes:
            return self._seen_hashes[sha256_hash]

        ext = path.suffix.lower()
        pages_out: List[PageOCRResult] = []
        any_quarantined = False

        if ext in self._IMAGE_EXTENSIONS:
            with Image.open(str(path)) as raw_img:
                img = raw_img.convert("RGB")
            page_res = self.process_page_image(img, page_number=1)
            any_quarantined = page_res.quarantined_block_count > 0
            pages_out.append(page_res)
        elif ext in self._TEXT_EXTENSIONS:
            try:
                with open(str(path), "r", encoding="utf-8", errors="replace") as tf:
                    text_content = tf.read()
            except Exception:
                text_content = path.read_bytes().decode("utf-8", errors="replace")
            pages_out.append(self._build_single_block_page(1, text_content))
        elif ext in self._MEDIA_EXTENSIONS:
            size_b = os.path.getsize(str(path))
            desc = f"Media Stream: {path.name} ({ext.lstrip('.').upper()})\nSize: {size_b} bytes\nPath: {path}"
            pages_out.append(self._build_single_block_page(1, desc))
        else:
            with fitz.open(str(path)) as doc:
                for idx in range(len(doc)):
                    page_num = idx + 1
                    pix = doc[idx].get_pixmap(dpi=self.raster_dpi, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_res = self.process_page_image(img, page_number=page_num)
                    if page_res.quarantined_block_count > 0:
                        any_quarantined = True
                    pages_out.append(page_res)

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

    def _dispatch_pass3_crops(
        self,
        pending_pass3: List[Tuple[Image.Image, OCRBlock, Tuple[float, float, float, float]]],
        page_number: int,
    ) -> List[OCRBlock]:
        """Runs Pass 3 optical transcription across queued low-confidence crops."""
        return [
            self._run_pass3_vlm_on_crop(
                crop_img=p3_crop,
                fallback_block=fallback_blk,
                orig_bbox=orig_bbox,
                page_number=page_number,
            )
            for p3_crop, fallback_blk, orig_bbox in pending_pass3
        ]

    def process_page_image(self, image: Image.Image, page_number: int = 1) -> PageOCRResult:
        """Processes a single page image through Layer 1 CV + 3-pass progressive optical sieve."""
        page_t0 = time.perf_counter()
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            image = preprocess_document_image(
                image, target_dpi=self.raster_dpi, deskew=True, clean_borders=True
            )
        except Exception as cv_err:
            logger.debug("Computer Vision Preprocessing skipped: %s", cv_err)

        # Pass 1: OS-Native OCR + 90/180/270 Auto-Orientation
        pass1_blocks = self._run_pass1_os_native(image, page_number)
        image, pass1_blocks, applied_rot = self._auto_orient_if_rotated(image, pass1_blocks, page_number)
        width_px, height_px = image.size

        # Fallback: If Pass 1 returned zero blocks, run Pass 2 full-page
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
                applied_rotation_deg=applied_rot,
            )

        # Coordinate Delta-Masking
        accepted_blocks: List[OCRBlock] = []
        escalated_to_p2: List[OCRBlock] = []

        for blk in pass1_blocks:
            if blk.confidence >= self.confidence_threshold and "[?]" not in blk.text:
                accepted_blocks.append(blk)
            else:
                escalated_to_p2.append(blk)

        pass1_accepted_count = len(accepted_blocks)
        pass2_escalated_count = len(escalated_to_p2)

        # Pass 2 on cropped low-confidence Pass 1 regions
        pending_pass3: List[Tuple[Image.Image, OCRBlock, Tuple[float, float, float, float]]] = []
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
                best_candidate = p2_sub_blocks[0] if p2_sub_blocks else low_blk
                orig_bbox = (low_blk.bbox_x0, low_blk.bbox_y0, low_blk.bbox_x1, low_blk.bbox_y1)
                p3_crop = self._make_pass3_crop(image, orig_bbox, width_px, height_px)
                if self._should_skip_pass3_crop(p3_crop, best_candidate):
                    accepted_blocks.append(best_candidate)
                else:
                    pending_pass3.append((p3_crop, best_candidate, orig_bbox))

        pass3_escalated_count = len(pending_pass3)
        if pending_pass3:
            accepted_blocks.extend(self._dispatch_pass3_crops(pending_pass3, page_number))

        # Geometric Pass 2 + Pass 3 recovery for inked regions skipped completely by Pass 1
        extra_gap_blocks = self._recover_missed_ink_gaps(
            accepted_blocks=accepted_blocks,
            full_image=image,
            page_w=width_px,
            page_h=height_px,
            page_number=page_number,
        )
        if extra_gap_blocks:
            pass3_escalated_count += sum(
                1 for b in extra_gap_blocks if b.pass_source == PassSource.PASS_3_LOCAL_VLM
            )
            accepted_blocks.extend(extra_gap_blocks)

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
            applied_rotation_deg=applied_rot,
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

    @staticmethod
    def _score_orientation_blocks(blocks: List[OCRBlock]) -> Tuple[float, int, float]:
        """Scores OCR blocks by horizontal orientation ratio, readability, and confidence."""
        if not blocks:
            return 0.0, 0, 0.0
        horiz_count = 0
        readable_words = 0
        total_conf = 0.0
        for b in blocks:
            bw = max(0.001, b.bbox_x1 - b.bbox_x0)
            bh = max(0.001, b.bbox_y1 - b.bbox_y0)
            if len(b.text) >= 4 and bw >= bh * 0.9:
                horiz_count += 1
            for w in re.findall(r"[A-Za-z]{2,}", b.text):
                if re.search(r"[aeiouyAEIOUY]", w):
                    readable_words += 1
            total_conf += b.confidence
        horiz_ratio = horiz_count / max(1, len(blocks))
        mean_conf = total_conf / max(1, len(blocks))
        score = readable_words * mean_conf * (0.4 + 0.6 * horiz_ratio)
        return score, readable_words, horiz_ratio

    def _auto_orient_if_rotated(
        self,
        image: Image.Image,
        base_blocks: List[OCRBlock],
        page_number: int,
    ) -> Tuple[Image.Image, List[OCRBlock], int]:
        """Detects and corrects 90/180/270-degree rotated raster scans when Pass 1 looks sideways or garbled."""
        if not (HAS_APPLE_VISION or HAS_WINOCR):
            return image, base_blocks, 0

        base_score, base_words, base_horiz = self._score_orientation_blocks(base_blocks)
        mean_conf = (sum(b.confidence for b in base_blocks) / len(base_blocks)) if base_blocks else 0.0
        is_suspicious = (
            base_words < 5
            or mean_conf < 0.68
            or (len(base_blocks) >= 3 and base_horiz < 0.45)
        )
        if not is_suspicious:
            return image, base_blocks, 0

        w, h = image.size
        long_edge = max(w, h)
        if long_edge > 1400:
            scale = 1400.0 / float(long_edge)
            probe_base = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
        else:
            probe_base = image

        best_angle = 0
        best_score = base_score
        min_accept_score = base_score * 1.25
        for angle in (90, 180, 270):
            cand_img = probe_base.rotate(angle, expand=True)
            cand_blocks = self._run_pass1_os_native(cand_img, page_number)
            c_score, c_words, c_horiz = self._score_orientation_blocks(cand_blocks)
            if c_words >= 3 and c_horiz >= 0.45 and c_score > min_accept_score and c_score > best_score:
                best_score = c_score
                best_angle = angle

        if best_angle == 0:
            return image, base_blocks, 0

        logger.info("Mimir Sieve auto-rotated page %d by %d degrees CCW (score %.1f -> %.1f)", page_number, best_angle, base_score, best_score)
        rotated_full = image.rotate(best_angle, expand=True)
        rotated_blocks = self._run_pass1_os_native(rotated_full, page_number)
        return rotated_full, rotated_blocks, best_angle

    @staticmethod
    def _extract_gap_ink_subbands(
        full_image: Image.Image,
        band_y0: float,
        band_y1: float,
        page_w: int,
        page_h: int,
    ) -> List[Tuple[float, float]]:
        """Finds tight inked row sub-bands inside an uncovered vertical gap via horizontal projection."""
        x0 = int(0.04 * page_w)
        x1 = int(0.96 * page_w)
        y0 = max(0, int(band_y0 * page_h))
        y1 = min(page_h, int(band_y1 * page_h))
        if y1 - y0 < 12 or x1 - x0 < 50:
            return []
        gray = np.asarray(full_image.crop((x0, y0, x1, y1)).convert("L"))
        row_ink = (gray < 200).mean(axis=1)
        active_rows = np.where((row_ink >= 0.002) & (row_ink <= 0.70))[0]
        if active_rows.size < 4:
            return []

        bands: List[Tuple[int, int]] = []
        s_row = int(active_rows[0])
        prev_row = s_row
        for r in active_rows[1:]:
            ri = int(r)
            if ri - prev_row <= 28:
                prev_row = ri
            else:
                if prev_row - s_row >= 3:
                    bands.append((s_row, prev_row))
                s_row = ri
                prev_row = ri
        if prev_row - s_row >= 3:
            bands.append((s_row, prev_row))

        pad_px = 18
        out: List[Tuple[float, float]] = []
        for by0, by1 in bands[:5]:
            ny0 = max(0.0, (y0 + max(0, by0 - pad_px)) / page_h)
            ny1 = min(1.0, (y0 + min(y1 - y0, by1 + pad_px)) / page_h)
            if (ny1 - ny0) * page_h >= 18:
                out.append((ny0, ny1))
        return out

    @staticmethod
    def _bbox_overlap_ratio(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        ix0, iy0 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        ix1, iy1 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = (ix1 - ix0) * (iy1 - iy0)
        area_a = max(1e-6, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        return inter / area_a

    def _recover_missed_ink_gaps(
        self,
        accepted_blocks: List[OCRBlock],
        full_image: Image.Image,
        page_w: int,
        page_h: int,
        page_number: int,
    ) -> List[OCRBlock]:
        """
        Geometric gap recovery (domain-keyword agnostic):
        Merges overlapping Y-intervals, scans uncovered vertical gaps (>2.8% page height)
        for tight inked sub-bands, runs RapidOCR only on those cropped strips, accepts
        clear Pass 2 text, and escalates low-confidence ink bands to Pass 3 VLM.
        """
        if len(accepted_blocks) < 2 or page_w < 200 or page_h < 200:
            return []

        merged_intervals: List[List[float]] = []
        for y0, y1 in sorted((b.bbox_y0, b.bbox_y1) for b in accepted_blocks):
            if not merged_intervals or y0 > merged_intervals[-1][1]:
                merged_intervals.append([y0, y1])
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], y1)

        ink_gaps: List[Tuple[float, float]] = []
        for i in range(len(merged_intervals) - 1):
            gy0 = merged_intervals[i][1]
            gy1 = merged_intervals[i + 1][0]
            if (gy1 - gy0) >= 0.028 and len(ink_gaps) < 5:
                subbands = self._extract_gap_ink_subbands(
                    full_image, gy0 + 0.004, gy1 - 0.004, page_w, page_h
                )
                for sb in subbands:
                    if len(ink_gaps) < 5:
                        ink_gaps.append(sb)

        if not ink_gaps:
            return []

        acc_boxes = [(b.bbox_x0, b.bbox_y0, b.bbox_x1, b.bbox_y1) for b in accepted_blocks]
        recovered: List[OCRBlock] = []

        for band_y0, band_y1 in ink_gaps:
            bbox = (0.04, band_y0, 0.96, band_y1)
            cx0 = int(bbox[0] * page_w)
            cy0 = int(bbox[1] * page_h)
            cx1 = int(bbox[2] * page_w)
            cy1 = int(bbox[3] * page_h)
            crop_img = full_image.crop((cx0, cy0, cx1, cy1))

            p2_blocks = self._run_pass2_on_crop(
                crop_img=crop_img,
                crop_x0=cx0,
                crop_y0=cy0,
                page_w=page_w,
                page_h=page_h,
                page_number=page_number,
            )

            gap_resolved = False
            for p2 in p2_blocks:
                p2_box = (p2.bbox_x0, p2.bbox_y0, p2.bbox_x1, p2.bbox_y1)
                if any(self._bbox_overlap_ratio(p2_box, ab) > 0.25 for ab in acc_boxes):
                    continue
                if p2.confidence >= self.confidence_threshold and re.search(r"[A-Za-z0-9]", p2.text):
                    recovered.append(p2)
                    acc_boxes.append(p2_box)
                    gap_resolved = True
                else:
                    p3_crop = self._make_pass3_crop(full_image, p2_box, page_w, page_h)
                    if not self._should_skip_pass3_crop(p3_crop, p2):
                        vlm_blk = self._run_pass3_vlm_on_crop(
                            crop_img=p3_crop,
                            fallback_block=p2,
                            orig_bbox=p2_box,
                            page_number=page_number,
                        )
                        if vlm_blk.text.strip():
                            recovered.append(vlm_blk)
                            acc_boxes.append(p2_box)
                            gap_resolved = True

            if not gap_resolved and not p2_blocks:
                stub = OCRBlock(
                    page_number=page_number,
                    text="",
                    confidence=0.0,
                    pass_source=PassSource.PASS_1_OS_NATIVE,
                    bbox_x0=bbox[0],
                    bbox_y0=bbox[1],
                    bbox_x1=bbox[2],
                    bbox_y1=bbox[3],
                )
                if not self._should_skip_pass3_crop(crop_img, stub):
                    vlm_blk = self._run_pass3_vlm_on_crop(
                        crop_img=crop_img,
                        fallback_block=stub,
                        orig_bbox=bbox,
                        page_number=page_number,
                    )
                    if vlm_blk.text and vlm_blk.text.strip():
                        recovered.append(vlm_blk)

        return recovered

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

    def _make_pass3_crop(
        self,
        full_image: Image.Image,
        bbox: Tuple[float, float, float, float],
        page_w: int,
        page_h: int,
    ) -> Image.Image:
        """Expands crop context so Pass 3 optical processor sees adjacent printed field labels."""
        pad_x = max(28, int(0.30 * page_w), self.crop_padding_px * 2)
        pad_y = max(96, self.crop_padding_px * 8)
        cx0 = max(0, int(bbox[0] * page_w) - pad_x)
        cy0 = max(0, int(bbox[1] * page_h) - pad_y)
        cx1 = min(page_w, int(bbox[2] * page_w) + pad_x)
        cy1 = min(page_h, int(bbox[3] * page_h) + pad_y)
        return full_image.crop((cx0, cy0, cx1, cy1))

    @staticmethod
    def _should_skip_pass3_crop(crop_img: Image.Image, fallback_block: OCRBlock) -> bool:
        """Skips blank/border slivers and lone punctuation that provoke VLM hallucinations."""
        w, h = crop_img.size
        if w < 28 or h < 18:
            return True
        txt = (fallback_block.text or "").strip()
        if txt and not re.search(r"[A-Za-z0-9]", txt) and len(txt) <= 4:
            return True
        gray = np.asarray(crop_img.convert("L"))
        if gray.size == 0 or float(np.std(gray)) < 6.0:
            return True
        dark_ratio = float(np.mean(gray < 210))
        if dark_ratio < 0.002:
            return True
        return False

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
        pending_pass3: List[Tuple[Image.Image, OCRBlock, Tuple[float, float, float, float]]] = []
        for blk in blocks:
            if blk.confidence >= self.confidence_threshold and "[?]" not in blk.text:
                out.append(blk)
            else:
                orig_bbox = (blk.bbox_x0, blk.bbox_y0, blk.bbox_x1, blk.bbox_y1)
                crop_img = self._make_pass3_crop(full_image, orig_bbox, page_w, page_h)
                if self._should_skip_pass3_crop(crop_img, blk):
                    out.append(blk)
                else:
                    pending_pass3.append((crop_img, blk, orig_bbox))

        if pending_pass3:
            out.extend(self._dispatch_pass3_crops(pending_pass3, page_number))
        return out

    @staticmethod
    def _sort_and_assemble(blocks: List[OCRBlock]) -> Tuple[List[OCRBlock], str]:
        if not blocks:
            return [], ""
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
        """Formats a DocumentOCRResult into a coordinate-grounded Markdown report."""
        lines = [
            "## Mimir OCR Document Ingestion Report",
            f"- **Source File:** `{doc_result.file_path}`",
            f"- **SHA-256:** `{doc_result.sha256_hash}`",
            f"- **Pages:** {doc_result.page_count} | **Total Latency:** {doc_result.total_latency_ms:.2f} ms",
            f"- **Quarantine Status:** {'QUARANTINED (Needs Human Review)' if doc_result.quarantined else 'CLEAN (Verified >= 0.85)'}",
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
