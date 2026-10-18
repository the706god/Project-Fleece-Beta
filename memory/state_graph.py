import time
import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger("FleeceBetaStateGraph")

class MultimodalGraphState:
    def __init__(self, query: str, image_paths: Optional[List[str]] = None, task: Optional[str] = None):
        self.query: str = query
        self.image_paths: List[str] = image_paths or []
        self.task: Optional[str] = task
        self.clean_query: str = ""
        self.is_fast_path_hit: bool = False
        self.fast_path_answer: Optional[str] = None
        self.fast_path_source: Optional[str] = None
        self.index_map: str = ""
        self.matched_sections: List[Dict[str, Any]] = []
        self.unbroken_content_block: str = ""
        self.payload: List[Dict[str, Any]] = []
        self.allocation_before: Dict[str, int] = {}
        self.compaction_occurred: bool = False
        self.response: str = ""
        self.execution_trace: List[Dict[str, Any]] = []

    def trace(self, node_name: str, details: str):
        self.execution_trace.append({
            "node": node_name,
            "timestamp": time.time(),
            "details": details
        })

class MultimodalStateGraphMemoryEngine:
    """
    Multimodal Two-Tier Stateful Graph Memory Engine for Project: Fleece Beta.
    Tier 1: Living Dossier Fast-Path (L1 Cache)
    Tier 2: Deep Vectorless PageIndex (L2 Unbroken Docs & Visuals)
    Self-Updating: Constantly registers new discoveries into DOSSIER.md.
    """

    def __init__(
        self,
        engine,
        memory_budget,
        markdown_store,
        page_index,
        ingestion_protocols,
        dossier
    ):
        self.engine = engine
        self.budget = memory_budget
        self.markdown = markdown_store
        self.page_index = page_index
        self.ingestion = ingestion_protocols
        self.dossier = dossier

    def run(
        self,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        task_context: Optional[str] = None,
        stream: bool = True
    ) -> MultimodalGraphState:
        state = MultimodalGraphState(query=user_prompt, image_paths=image_paths, task=task_context)

        # 1. INGEST (Anti-Parsing Protocol)
        self._node_ingest(state)

        # 2. CHECK_DOSSIER_L1 (Fast-Path Check)
        self._node_check_dossier_l1(state)

        if not state.is_fast_path_hit:
            # 3. NAVIGATE_PAGEINDEX_L2 (Multimodal PageIndex Navigation)
            self._node_navigate_pageindex(state)

            # 4. RETRIEVE_UNBROKEN (Zero-Chunking Extraction)
            self._node_retrieve_unbroken(state)

        # 5. BUDGET_ENFORCE (Shield Ingested Content / Evict History)
        self._node_budget_enforce(state)

        # 6. EXECUTE (VLM Inference with Complete Ingested Material)
        self._node_execute(state, stream=stream)

        # 7. AUTO_UPDATE_DOSSIER (Consistently Update Dossier with New Discoveries)
        if not state.is_fast_path_hit and state.matched_sections:
            self._node_auto_update_dossier(state)

        # 8. ARCHIVE
        self._node_archive(state)

        return state

    def _node_ingest(self, state: MultimodalGraphState):
        state.clean_query = self.ingestion.anti_parse_text(state.query)
        img_count = len(state.image_paths)
        state.trace("INGEST", f"Input sanitized with 100% fidelity. Text: {len(state.clean_query)} chars. Attached images: {img_count}.")

    def _node_check_dossier_l1(self, state: MultimodalGraphState):
        # Only check text fast path if no raw external user images were supplied
        if not state.image_paths:
            is_hit, ans, src = self.dossier.check_fast_path(state.clean_query)
            if is_hit and ans:
                state.is_fast_path_hit = True
                state.fast_path_answer = ans
                state.fast_path_source = src
                dossier_block = f"### [Source: {src}]\n{ans}"
                self.budget.set_protected_ingestion(dossier_block)

                payload = self.ingestion.package_multimodal_payload(
                    text_prompt=state.clean_query,
                    image_paths=None,
                    encode_func=self.engine.encode_image_uri
                )
                state.payload = payload
                self.budget.add_interaction(role="user", content=state.clean_query)
                state.trace("CHECK_DOSSIER_L1", f"Fast-Path HIT! Found verified entry in DOSSIER.md ({src}). Sparing deep file scan.")
                return

        state.trace("CHECK_DOSSIER_L1", "Fast-Path MISS. Escalating to L2 PageIndex for deep unbroken retrieval.")

    def _node_navigate_pageindex(self, state: MultimodalGraphState):
        state.index_map = self.page_index.get_navigation_map()
        state.trace("NAVIGATE_PAGEINDEX_L2", f"PageIndex evaluated with {len(self.page_index.tree)} multimodal nodes.")

    def _node_retrieve_unbroken(self, state: MultimodalGraphState):
        matched = self.page_index.navigate_and_retrieve(state.clean_query, max_sections=2)
        state.matched_sections = matched

        active_images = list(state.image_paths)
        seen_images = set(active_images)

        if matched:
            citations = []
            for sec in matched:
                if sec.get("is_visual") and sec.get("image_path"):
                    if sec["image_path"] not in seen_images:
                        active_images.append(sec["image_path"])
                        seen_images.add(sec["image_path"])
                    citations.append(f"### [Cited Visual Asset: {sec['id']}]")
                else:
                    citations.append(f"### [Source: {sec['id']}]\n{sec['unbroken_content']}")
            unbroken_block = "\n\n".join(citations)
            state.unbroken_content_block = unbroken_block
            self.budget.set_protected_ingestion(unbroken_block)
            state.trace("RETRIEVE_UNBROKEN", f"Loaded {len(matched)} unbroken sections/visuals into protected memory.")
        else:
            self.budget.clear_protected_ingestion()
            state.trace("RETRIEVE_UNBROKEN", "No external document sections required for this turn.")

        payload = self.ingestion.package_multimodal_payload(
            text_prompt=state.clean_query,
            image_paths=active_images if active_images else None,
            encode_func=self.engine.encode_image_uri
        )
        state.payload = payload
        self.budget.add_interaction(role="user", content=payload if active_images else state.clean_query)

    def _node_budget_enforce(self, state: MultimodalGraphState):
        state.allocation_before = self.budget.get_token_allocation()
        compaction = self.budget.enforce_budget(offload_callback=self.markdown.archive_evicted_context)
        state.compaction_occurred = compaction

        allocation_after = self.budget.get_token_allocation()
        headroom = allocation_after["remaining_headroom"]

        if compaction:
            state.trace("BUDGET_ENFORCE", "Context repartitioned: evicted conversation turns to shield unchunked document.")
        else:
            state.trace("BUDGET_ENFORCE", f"Context safe. Protected doc: {allocation_after['ingested_tokens']} tok. Available headroom: {headroom} tok.")

    def _node_execute(self, state: MultimodalGraphState, stream: bool = True):
        messages = self.budget.build_safe_context()
        if stream:
            full_res = []
            print("\n[Fleece Beta Output]:\n", flush=True)
            for chunk in self.engine.generate_chat(messages, stream=True):
                import sys
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_res.append(chunk)
            print("\n")
            state.response = "".join(full_res)
        else:
            state.response = self.engine.generate_chat(messages, stream=False)

        self.budget.add_interaction(role="assistant", content=state.response)
        state.trace("EXECUTE", f"Multimodal response generated. Tokens: ~{self.engine.count_tokens(state.response)}.")

    def _node_auto_update_dossier(self, state: MultimodalGraphState):
        source_ids = [s["id"] for s in state.matched_sections]
        try:
            self.dossier.record_discovery(
                query=state.clean_query,
                response=state.response,
                source_refs=source_ids
            )
            self.page_index.rebuild_index()
            state.trace("AUTO_UPDATE_DOSSIER", "Consistently updated DOSSIER.md with newly verified fact for instant future recall.")
        except Exception as e:
            logger.error("Failed to update dossier: %s", e)

    def _node_archive(self, state: MultimodalGraphState):
        record = f"**User:** {state.clean_query[:100]}...\n**Assistant:** {state.response[:100]}...\n"
        if state.is_fast_path_hit:
            record += f"**Cache Hit:** {state.fast_path_source}\n"
        elif state.matched_sections:
            record += f"**Cited Sections:** {', '.join([s['id'] for s in state.matched_sections])}\n"
        self.markdown.archive_record(record)
        state.trace("ARCHIVE", "Multimodal state committed to secondary markdown ledger.")
