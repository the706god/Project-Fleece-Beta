import os
import re
import datetime
from typing import List, Dict, Any, Optional, Tuple

class LivingDossierEngine:
    """
    Multimodal Living Dossier Engine for Project: Fleece Beta.
    Manages L1 Fast-Path Lookups and auto-updates for both text and visual discoveries.
    Strips internal thinking traces (<think>...</think>) to ensure pristine cached answers.
    """

    def __init__(self, storage_root: str):
        self.storage_root = os.path.abspath(storage_root)
        self.dossier_path = os.path.join(self.storage_root, "DOSSIER.md")
        self._ensure_dossier()

    def _ensure_dossier(self):
        if not os.path.exists(self.dossier_path):
            header = f"""# Project: Fleece Beta - Multimodal Living Dossier
Initialized: {datetime.datetime.now().isoformat()}
Last Updated: {datetime.datetime.now().isoformat()}

## Executive Facts & Core Entities
- [Engine]: Project Fleece Beta Multimodal Crate Motor.
- [Ingestion Mode]: Direct Visual Ingestion + Living L1/L2 Memory Architecture.

## Living FAQ & Fast-Path Knowledge Bank
"""
            with open(self.dossier_path, "w", encoding="utf-8") as f:
                f.write(header)

    def check_fast_path(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        if not os.path.exists(self.dossier_path):
            return False, None, None

        try:
            with open(self.dossier_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return False, None, None

        query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
        if not query_terms:
            return False, None, None

        faq_blocks = re.findall(r"### Q:\s*(.+?)\n\*\*A:\*\*\s*(.+?)(?=\n### Q:|\n## |\Z)", content, flags=re.DOTALL)
        for question, answer in faq_blocks:
            q_clean = question.strip().lower()
            matches = sum(1 for term in query_terms if term in q_clean)
            if matches >= max(1, len(query_terms) * 0.6):
                return True, answer.strip(), f"DOSSIER.md#faq-{matches}"

        facts_block = re.search(r"## Executive Facts & Core Entities\n(.*?)(?=\n## |\Z)", content, flags=re.DOTALL)
        if facts_block:
            bullets = [b.strip() for b in facts_block.group(1).split("\n") if b.strip().startswith("-")]
            for b in bullets:
                b_clean = b.lower()
                matches = sum(1 for term in query_terms if term in b_clean)
                if matches >= max(2, len(query_terms) * 0.7):
                    return True, b.strip("- []"), "DOSSIER.md#executive-facts"

        return False, None, None

    def record_discovery(self, query: str, response: str, source_refs: List[str]):
        timestamp = datetime.datetime.now().isoformat()
        clean_q = query.strip().replace("\n", " ")

        # Strip internal thinking traces if present
        sanitized_res = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if not sanitized_res:
            sanitized_res = response.strip()

        paragraphs = [p.strip() for p in sanitized_res.split("\n\n") if p.strip()]
        clean_ans = " ".join(paragraphs[:2]) if paragraphs else sanitized_res
        clean_ans = clean_ans.replace("\n", " ")
        sources = ", ".join(source_refs) if source_refs else "Deep Multimodal Retrieval"

        entry = f"\n### Q: {clean_q}\n**A:** {clean_ans}\n**Source:** {sources} *(Updated: {timestamp})*\n"

        try:
            with open(self.dossier_path, "r", encoding="utf-8") as f:
                content = f.read()

            if clean_q.lower() in content.lower():
                return

            updated_content = re.sub(
                r"Last Updated:\s*.*",
                f"Last Updated: {timestamp}",
                content
            )

            with open(self.dossier_path, "w", encoding="utf-8") as f:
                f.write(updated_content + entry)

        except Exception:
            with open(self.dossier_path, "a", encoding="utf-8") as f:
                f.write(entry)
