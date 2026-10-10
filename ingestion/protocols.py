import os
import re
import logging
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger("BetaIngestionProtocols")

class MultimodalIngestionProtocols:
    """
    Multimodal Ingestion Protocols for Project: Fleece Beta:
    1. Anti-Parsing Protocol:
       - Direct Visual Ingestion: Feed raw images/diagrams/PDFs directly to the VLM,
         completely bypassing brittle OCR/text parsers that damage tables or formatting.
       - Zero-Destruction Text: Retain 100% of indentation, tables, and raw characters.
    2. Anti-Chunking Protocol:
       - Rejects arbitrary slicing. Treats documents and visual assets as complete semantic wholes.
    3. Anti-Truncating Protocol:
       - Rejects abrupt text cutting. Retains complete sections.
    """

    def __init__(self, max_block_tokens: int = 8192, token_counter: Optional[Callable[[str], int]] = None):
        self.max_block_tokens = max_block_tokens
        self.token_counter = token_counter or (lambda s: len(s.split()) * 4 // 3)
        self.supported_image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def is_image_path(self, path: str) -> bool:
        if not isinstance(path, str):
            return False
        _, ext = os.path.splitext(path.strip())
        return ext.lower() in self.supported_image_exts and os.path.exists(path.strip())

    def anti_parse_text(self, raw_input: str) -> str:
        if not raw_input:
            return ""
        clean = raw_input.replace("\r\n", "\n").replace("\r", "\n")
        fences = clean.count("```")
        if fences % 2 != 0:
            clean += "\n```\n"
        return clean

    def package_multimodal_payload(
        self,
        text_prompt: str,
        image_paths: Optional[List[str]] = None,
        encode_func: Optional[Callable[[str], str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Package input as OpenAI-compatible multimodal content.
        Bypasses OCR parsing by directly binding raw images into visual tensors.
        """
        content_items = []
        clean_text = self.anti_parse_text(text_prompt)
        if clean_text:
            content_items.append({"type": "text", "text": clean_text})

        if image_paths and encode_func:
            for img_p in image_paths:
                if self.is_image_path(img_p):
                    data_uri = encode_func(img_p)
                    content_items.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })

        return content_items

    def anti_chunk(self, content: str) -> List[str]:
        clean = self.anti_parse_text(content)
        pattern = r"(?=(?:^|\n)#{1,3}\s+)"
        sections = [s.strip() for s in re.split(pattern, clean, flags=re.MULTILINE) if s.strip()]
        if not sections:
            return [clean]
        return sections

    def verify_no_truncation(self, text: str, available_tokens: int) -> str:
        current_tokens = self.token_counter(text)
        if current_tokens > available_tokens:
            logger.warning(
                "Ingested document size (%d tokens) approaches context headroom (%d tokens). "
                "Retaining full document intact; conversational history will be evicted to accommodate.",
                current_tokens, available_tokens
            )
        return text
