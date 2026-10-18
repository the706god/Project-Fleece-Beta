import logging
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger("BetaContextBudget")

class MultimodalContextBudgetManager:
    """
    Multimodal Context Budget Manager with Protected Ingestion Accommodation.
    Guarantees that ingested documents, tables, and images remain 100% UNPARSED, UNCHUNKED, and UNTRUNCATED.
    
    If context headroom becomes constrained, the manager actively sacrifices
    and offloads conversational dialogue to the secondary markdown ledger FIRST,
    protecting the full structural integrity of ingested reference materials and visual tokens.
    """

    def __init__(
        self,
        max_context_tokens: int = 16384,
        reserved_completion_tokens: int = 2048,
        system_anchor_tokens: int = 1200,
        image_token_cost: int = 576,
        compaction_threshold: float = 0.85,
        token_counter: Optional[Callable[[str], int]] = None
    ):
        self.max_context = max_context_tokens
        self.reserved_completion = reserved_completion_tokens
        self.image_token_cost = image_token_cost
        self.compaction_threshold = compaction_threshold
        self.token_counter = token_counter or (lambda s: len(s.split()) * 4 // 3)

        self.system_anchor: str = ""
        self.active_task: str = ""
        self.protected_ingested_content: str = ""  # NEVER chunked or truncated
        self.working_history: List[Dict[str, Any]] = []
        self.active_summary: str = ""

    def set_system_anchor(self, text: str):
        self.system_anchor = text

    def set_active_task(self, text: str):
        self.active_task = text

    def set_protected_ingestion(self, unbroken_content: str):
        """Store complete unbroken ingested information, shielded from truncation."""
        self.protected_ingested_content = unbroken_content

    def clear_protected_ingestion(self):
        self.protected_ingested_content = ""

    def add_interaction(self, role: str, content: Any):
        self.working_history.append({"role": role, "content": content})

    def _estimate_message_tokens(self, msg: Dict[str, Any]) -> int:
        content = msg.get("content", "")
        if isinstance(content, str):
            return self.token_counter(content)
        elif isinstance(content, list):
            tokens = 0
            for item in content:
                if item.get("type") == "text":
                    tokens += self.token_counter(item.get("text", ""))
                elif item.get("type") == "image_url":
                    tokens += self.image_token_cost
            return tokens
        return 0

    def get_token_allocation(self) -> Dict[str, int]:
        system_tokens = self.token_counter(self.system_anchor)
        task_tokens = self.token_counter(self.active_task)
        ingested_tokens = self.token_counter(self.protected_ingested_content)
        summary_tokens = self.token_counter(self.active_summary)
        
        history_tokens = sum(self._estimate_message_tokens(m) for m in self.working_history)
        total_input = system_tokens + task_tokens + ingested_tokens + summary_tokens + history_tokens
        ceiling = self.max_context - self.reserved_completion

        return {
            "system_tokens": system_tokens,
            "task_tokens": task_tokens,
            "ingested_tokens": ingested_tokens,
            "summary_tokens": summary_tokens,
            "history_tokens": history_tokens,
            "total_input_tokens": total_input,
            "available_input_ceiling": ceiling,
            "remaining_headroom": max(0, ceiling - total_input)
        }

    def get_budget_allocation(self) -> Dict[str, Any]:
        alloc = self.get_token_allocation()
        pct = (alloc["total_input_tokens"] / self.max_context) * 100.0 if self.max_context else 0.0
        alloc["total_tokens"] = alloc["total_input_tokens"]
        alloc["max_context"] = self.max_context
        alloc["percent_full"] = pct
        return alloc

    def enforce_budget(self, offload_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None) -> bool:
        """
        Accommodates unchunked ingested content by evicting conversational history
        rather than ever truncating ingested document or visual tokens.
        """
        allocation = self.get_token_allocation()
        total_input = allocation["total_input_tokens"]
        ceiling = allocation["available_input_ceiling"]

        if total_input <= ceiling * self.compaction_threshold:
            return False

        logger.warning(
            "VLM Context threshold reached (%d / %d tokens). "
            "Repartitioning context to shield unchunked documents and visual tokens.",
            total_input, ceiling
        )

        if self.working_history:
            evict_count = max(1, len(self.working_history) - 1)
            evicted_turns = self.working_history[:evict_count]
            self.working_history = self.working_history[evict_count:]

            if offload_callback:
                try:
                    offload_callback(evicted_turns)
                    logger.info("Evicted %d conversational turns to preserve unbroken document context.", len(evicted_turns))
                except Exception as e:
                    logger.error("Failed to archive evicted turns: %s", e)

            summaries = []
            for m in evicted_turns:
                c = m.get("content", "")
                text_part = c if isinstance(c, str) else "[Visual/Multimodal Content]"
                summaries.append(f"[{m['role']}] {str(text_part)[:80]}")
            self.active_summary = f"[Conversation Archive: {' '.join(summaries)}]"
            return True

        return False

    def build_safe_context(self) -> List[Dict[str, Any]]:
        messages = []
        if self.system_anchor:
            messages.append({"role": "system", "content": self.system_anchor})

        if self.active_task:
            messages.append({"role": "system", "content": f"CURRENT ACTIVE TASK:\n{self.active_task}"})

        # Inject complete, unbroken ingested information block
        if self.protected_ingested_content:
            messages.append({
                "role": "system",
                "content": f"=== UNPARSED & UNCHUNKED INGESTED KNOWLEDGE (PROTECTED) ===\n{self.protected_ingested_content}"
            })

        if self.active_summary:
            messages.append({"role": "system", "content": f"SESSION CONTINUITY:\n{self.active_summary}"})

        messages.extend(self.working_history)
        return messages
