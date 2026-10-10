import os
import re
import json
import glob
from typing import List, Dict, Any, Optional, Callable

class MultimodalPageIndexEngine:
    """
    Multimodal Vectorless RAG Engine for Project: Fleece Beta.
    Indexes both Markdown documents and visual assets (diagrams, screenshots, page renders)
    into a unified hierarchical TOC tree without vector databases or embedding models.
    """

    def __init__(
        self,
        storage_root: str,
        token_counter: Optional[Callable[[str], int]] = None
    ):
        self.storage_root = os.path.abspath(storage_root)
        self.knowledge_dir = os.path.join(self.storage_root, "knowledge")
        self.tasks_dir = os.path.join(self.storage_root, "tasks")
        self.images_dir = os.path.join(self.storage_root, "images")
        self.index_file = os.path.join(self.storage_root, "PAGE_INDEX.json")
        self.token_counter = token_counter or (lambda s: len(s.split()) * 4 // 3)
        self.tree: List[Dict[str, Any]] = []

        os.makedirs(self.images_dir, exist_ok=True)
        self.rebuild_index()

    def rebuild_index(self):
        self.tree = []
        search_dirs = [self.knowledge_dir, self.tasks_dir]

        # 1. Index Markdown documents
        for sdir in search_dirs:
            if not os.path.exists(sdir):
                continue
            for filepath in glob.glob(os.path.join(sdir, "**", "*.md"), recursive=True):
                self._index_file(filepath)

        # 2. Index visual files
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            for img_path in glob.glob(os.path.join(self.images_dir, "**", ext), recursive=True):
                rel_path = os.path.relpath(img_path, self.storage_root)
                img_name = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ").title()
                self.tree.append({
                    "id": f"{rel_path}#visual",
                    "title": f"[Visual Asset] {img_name}",
                    "file": rel_path,
                    "level": 1,
                    "heading_path": ["Images", img_name],
                    "content_lines": [f"[Image File: {rel_path}]"],
                    "summary": f"Visual asset: {img_name}",
                    "is_visual": True,
                    "image_path": img_path,
                    "tokens": 576
                })

        self._save_index()

    def _index_file(self, filepath: str):
        rel_path = os.path.relpath(filepath, self.storage_root)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return

        lines = content.split("\n")
        current_section = {
            "id": f"{rel_path}#root",
            "title": os.path.splitext(os.path.basename(filepath))[0].replace("_", " ").title(),
            "file": rel_path,
            "level": 0,
            "heading_path": [os.path.basename(filepath)],
            "content_lines": [],
            "summary": "",
            "is_visual": False
        }

        file_sections = []
        heading_stack = [(0, current_section["title"])]

        for line in lines:
            m = re.match(r"^(#{1,4})\s+(.+)$", line)
            if m:
                if current_section["content_lines"]:
                    text_block = "\n".join(current_section["content_lines"]).strip()
                    current_section["tokens"] = self.token_counter(text_block)
                    current_section["summary"] = text_block[:160].replace("\n", " ") + ("..." if len(text_block) > 160 else "")
                    file_sections.append(current_section)

                level = len(m.group(1))
                title = m.group(2).strip()

                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                heading_path = [h[1] for h in heading_stack]

                slug = re.sub(r"[^a-zA-Z0-9_-]", "-", title.lower())
                current_section = {
                    "id": f"{rel_path}#{slug}",
                    "title": title,
                    "file": rel_path,
                    "level": level,
                    "heading_path": heading_path,
                    "content_lines": [],
                    "summary": "",
                    "is_visual": False
                }
            else:
                current_section["content_lines"].append(line)

        if current_section["content_lines"]:
            text_block = "\n".join(current_section["content_lines"]).strip()
            current_section["tokens"] = self.token_counter(text_block)
            current_section["summary"] = text_block[:160].replace("\n", " ") + ("..." if len(text_block) > 160 else "")
            file_sections.append(current_section)

        self.tree.extend(file_sections)

    def _save_index(self):
        try:
            lightweight = [
                {
                    "id": node["id"],
                    "title": node["title"],
                    "file": node["file"],
                    "level": node["level"],
                    "heading_path": " > ".join(node["heading_path"]),
                    "summary": node["summary"],
                    "is_visual": node.get("is_visual", False),
                    "tokens": node.get("tokens", 0)
                }
                for node in self.tree
            ]
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(lightweight, f, indent=2)
        except Exception:
            pass

    def get_navigation_map(self, max_nodes: int = 20) -> str:
        if not self.tree:
            return "No stored documents in PageIndex."

        lines = ["=== Multimodal TOC Map (PageIndex) ==="]
        for node in self.tree[:max_nodes]:
            indent = "  " * node["level"]
            path_str = " > ".join(node["heading_path"])
            tag = "[VISUAL]" if node.get("is_visual") else "[DOC]"
            lines.append(f"{indent}- {tag} [{node['id']}] {path_str} (~{node.get('tokens', 0)} tok)")
            if node["summary"]:
                lines.append(f"{indent}    Summary: {node['summary']}")
        return "\n".join(lines)

    def navigate_and_retrieve(self, query: str, max_sections: int = 2) -> List[Dict[str, Any]]:
        if not self.tree:
            return []

        query_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
        if not query_terms:
            return []

        scored_nodes = []
        for node in self.tree:
            score = 0
            text_corpus = f"{node['title']} {' '.join(node['heading_path'])} {node['summary']}".lower()

            for term in query_terms:
                if term in node["title"].lower():
                    score += 5
                elif term in " ".join(node["heading_path"]).lower():
                    score += 3
                elif term in node["summary"].lower():
                    score += 1

            if score > 0:
                scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        top_nodes = [item[1] for item in scored_nodes[:max_sections]]

        retrieved = []
        for node in top_nodes:
            if node.get("is_visual"):
                retrieved.append({
                    "id": node["id"],
                    "title": node["title"],
                    "file": node["file"],
                    "is_visual": True,
                    "image_path": node.get("image_path"),
                    "tokens": 576,
                    "unbroken_content": f"[Visual Diagram: {node['title']}]"
                })
            else:
                full_file_path = os.path.join(self.storage_root, node["file"])
                try:
                    with open(full_file_path, "r", encoding="utf-8") as f:
                        file_text = f.read()
                    retrieved.append({
                        "id": node["id"],
                        "title": node["title"],
                        "file": node["file"],
                        "is_visual": False,
                        "tokens": node.get("tokens", 0),
                        "unbroken_content": "\n".join(node.get("content_lines", [])) or file_text
                    })
                except Exception:
                    continue

        return retrieved
