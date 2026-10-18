import os
import glob
import datetime
from typing import List, Dict, Any, Optional

class LegacyMarkdownStore:
    """
    Secondary persistent memory bank powered by markdown files.
    Retains text and multimodal interaction logs, knowledge records,
    and task registries.
    """

    def __init__(self, root_dir: str = "storage/markdown"):
        self.root_dir = os.path.abspath(root_dir)
        self.knowledge_dir = os.path.join(self.root_dir, "knowledge")
        self.tasks_dir = os.path.join(self.root_dir, "tasks")
        self.sessions_dir = os.path.join(self.root_dir, "sessions")

        for d in [self.knowledge_dir, self.tasks_dir, self.sessions_dir]:
            os.makedirs(d, exist_ok=True)

        self.ledger_file = os.path.join(self.root_dir, "MEMORY_LEDGER.md")
        if not os.path.exists(self.ledger_file):
            self._init_ledger()

    def _init_ledger(self):
        with open(self.ledger_file, "w", encoding="utf-8") as f:
            f.write("# Project: Fleece Beta - Secondary Memory Ledger\n")
            f.write("Initialized: " + datetime.datetime.now().isoformat() + "\n\n")
            f.write("## Index & Reference Registry\n\n")

    def archive_evicted_context(self, turns: List[Dict[str, Any]]):
        date_str = datetime.date.today().isoformat()
        session_file = os.path.join(self.sessions_dir, f"session_{date_str}.md")
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        with open(session_file, "a", encoding="utf-8") as f:
            f.write(f"\n### Evicted Turns Snapshot [{timestamp}]\n\n")
            for t in turns:
                role = t.get("role", "unknown").upper()
                content = t.get("content", "")
                if isinstance(content, list):
                    f.write(f"**{role} (Multimodal)**:\n")
                    for item in content:
                        if item.get("type") == "text":
                            f.write(f"- Text: {item.get('text', '')}\n")
                        elif item.get("type") == "image_url":
                            f.write("- [Embedded Image Payload]\n")
                    f.write("\n")
                else:
                    f.write(f"**{role}**:\n```\n{content}\n```\n\n")

        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(f"- [{timestamp}] Archived {len(turns)} turns to `{os.path.basename(session_file)}`\n")

    def archive_record(self, record: str):
        """Append an archived user/assistant exchange to the secondary memory ledger."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(f"\n### Archived Exchange [{timestamp}]\n\n{record}\n")

    def store_knowledge(self, topic: str, content: str, tags: Optional[List[str]] = None) -> str:
        safe_topic = "".join([c if c.isalnum() or c in "-_" else "_" for c in topic.lower()])
        filepath = os.path.join(self.knowledge_dir, f"{safe_topic}.md")
        tag_line = f"tags: {', '.join(tags)}\n" if tags else ""
        data = f"""# {topic.title()}
date: {datetime.datetime.now().isoformat()}
{tag_line}
---

{content}
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)

        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(f"- Stored knowledge topic `{topic}` in `{os.path.basename(filepath)}`\n")
        return filepath

    def query(self, query_term: str, max_results: int = 3) -> List[Dict[str, str]]:
        results = []
        pattern = os.path.join(self.root_dir, "**", "*.md")
        query_lower = query_term.lower()

        for filepath in glob.glob(pattern, recursive=True):
            if filepath == self.ledger_file:
                continue
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                if query_lower in content.lower():
                    idx = content.lower().find(query_lower)
                    start = max(0, idx - 150)
                    end = min(len(content), idx + 350)
                    snippet = content[start:end].strip()
                    results.append({
                        "file": os.path.relpath(filepath, self.root_dir),
                        "snippet": f"...{snippet}..."
                    })
                    if len(results) >= max_results:
                        break
            except Exception:
                continue
        return results
