#!/usr/bin/env python3
"""
Project: Fleece Beta - The 'Crate Motor' SLM Build (Multimodal Edition)
Model: Qwen 3.5 4B Vision Instruct
Equipped with:
- Living Multimodal Dossier (L1 Fast-Path Cache, consistently auto-updating)
- Multimodal Vectorless RAG (L2 PageIndex Tree Navigation)
- Stateful Graph Memory Engine
- Explicit Anti-Parsing, Anti-Chunking, Anti-Truncating Protocols
- Secondary Legacy Markdown Memory Ledger
"""

import os
import argparse
from typing import Optional, List

from engine.vlm_engine import FleeceVLMEngine
from memory.context_budget import MultimodalContextBudgetManager
from memory.markdown_store import LegacyMarkdownStore
from memory.page_index import MultimodalPageIndexEngine
from memory.dossier import LivingDossierEngine
from memory.state_graph import MultimodalStateGraphMemoryEngine
from ingestion.protocols import MultimodalIngestionProtocols

SYSTEM_DIRECTIVE_BETA = """You are an AI engine powered by Project: Fleece Beta.

MANDATORY INGESTION & FIDELITY DIRECTIVES:
1. ZERO-DESTRUCTION FIDELITY: Any and all information ingested (text, document renders, or images) must NEVER be parsed, chunked, or truncated. All ingested materials are processed as unified, complete wholes.
2. DIRECT VISUAL INGESTION: Complex documents with tables, charts, or blueprints are processed directly through the visual pipeline, bypassing lossy text parsing.
3. HOLISTIC CONTEXT UTILIZATION: Evaluate ingested materials as unified wholes. Never guess or fabricate based on truncated fragments. If information is not documented, state that clearly.
4. CITATION OF UNBROKEN SOURCES: Always cite the exact source file and section anchor provided by the PageIndex or Dossier memory system.
5. ABSOLUTE FORMAT INTEGRITY: Maintain exact indentation, code syntax, and tabular structures without destructive reformatting."""

class FleeceBetaCrateMotor:
    """
    Plug-and-play Multimodal AI Crate Motor.
    Encapsulates Qwen 3.5 4B VLM inference, multimodal token budget management,
    Two-Tier memory (L1 Living Dossier + L2 Multimodal PageIndex), and stateful graph memory.
    """

    def __init__(self, config_path: str = "fleece_config.json"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_config = os.path.join(base_dir, config_path)

        self.engine = FleeceVLMEngine(resolved_config)
        markdown_root = os.path.join(
            base_dir, self.engine.config.get("storage", {}).get("markdown_root", "storage/markdown")
        )
        self.markdown_store = LegacyMarkdownStore(markdown_root)
        self.page_index = MultimodalPageIndexEngine(markdown_root, token_counter=self.engine.count_tokens)
        self.dossier = LivingDossierEngine(markdown_root)

        mem_cfg = self.engine.config.get("memory", {})
        self.memory = MultimodalContextBudgetManager(
            max_context_tokens=mem_cfg.get("max_context_tokens", 16384),
            reserved_completion_tokens=mem_cfg.get("reserved_completion_tokens", 2048),
            system_anchor_tokens=mem_cfg.get("system_anchor_tokens", 1200),
            image_token_cost=mem_cfg.get("image_token_cost", 576),
            compaction_threshold=mem_cfg.get("memory_compaction_threshold", 0.85),
            token_counter=self.engine.count_tokens
        )

        ingest_cfg = self.engine.config.get("ingestion", {})
        self.ingestion = MultimodalIngestionProtocols(
            max_block_tokens=ingest_cfg.get("max_block_tokens", 8192),
            token_counter=self.engine.count_tokens
        )

        # Two-Tier Multimodal Graph Memory Machine
        self.graph = MultimodalStateGraphMemoryEngine(
            engine=self.engine,
            memory_budget=self.memory,
            markdown_store=self.markdown_store,
            page_index=self.page_index,
            ingestion_protocols=self.ingestion,
            dossier=self.dossier
        )

        # Baked-in immutable system anchor
        self.memory.set_system_anchor(SYSTEM_DIRECTIVE_BETA)

    def set_task(self, task_description: str):
        sanitized = self.ingestion.anti_parse_text(task_description)
        self.memory.set_active_task(sanitized)
        self.markdown_store.store_knowledge(
            topic="active_task",
            content=sanitized,
            tags=["task", "beta_objective"]
        )
        self.page_index.rebuild_index()

    def store_knowledge(self, topic: str, content: str, tags: Optional[List[str]] = None) -> str:
        clean_content = self.ingestion.anti_parse_text(content)
        path = self.markdown_store.store_knowledge(topic, clean_content, tags)
        self.page_index.rebuild_index()
        return path

    def execute(
        self,
        user_prompt: str,
        image_paths: Optional[List[str]] = None,
        stream: bool = True,
        verbose: bool = False
    ) -> str:
        state = self.graph.run(user_prompt, image_paths=image_paths, stream=stream)
        if verbose:
            print("\n--- [Multimodal Memory Graph Trace] ---")
            for t in state.execution_trace:
                print(f"[{t['node']}] -> {t['details']}")
            print("---------------------------------------\n")
        return state.response

def cli():
    parser = argparse.ArgumentParser(description="Project: Fleece Beta - Multimodal SLM Crate Motor")
    parser.add_argument("--task", type=str, help="Initial task objective for the crate motor")
    parser.add_argument("--prompt", type=str, help="Prompt text")
    parser.add_argument("--image", nargs="*", help="Optional path(s) to image/document files")
    parser.add_argument("--trace", action="store_true", help="Print execution trace from memory graph")
    args = parser.parse_args()

    motor = FleeceBetaCrateMotor()
    if args.task:
        motor.set_task(args.task)
        print(f"[Task Initialized]: {args.task}")

    if args.prompt:
        motor.execute(args.prompt, image_paths=args.image, stream=True, verbose=args.trace)
        return

    print("=" * 60)
    print(" Project: Fleece Beta - Ready (Two-Tier Multimodal Memory Active)")
    print(" L1 Living Dossier (Auto-Updating) + L2 Vectorless PageIndex")
    print(" Explicit Directives: Zero Parsing | Zero Chunking | Zero Truncating")
    print(" Type 'exit' or 'quit' to terminate.")
    print(" Type '!toc' to display current PageIndex document map.")
    print(" Type '!dossier' to view current Living Dossier cache.")
    print(" Attach an image: '!img <path> <question>'")
    print("=" * 60)

    while True:
        try:
            user_in = input("\n[You] > ").strip()
            if not user_in:
                continue
            if user_in.lower() in ["exit", "quit"]:
                print("Shutting down Fleece Beta.")
                break
            if user_in == "!toc":
                print("\n" + motor.page_index.get_navigation_map() + "\n")
                continue
            if user_in == "!dossier":
                if os.path.exists(motor.dossier.dossier_path):
                    with open(motor.dossier.dossier_path, "r", encoding="utf-8") as f:
                        print("\n" + f.read() + "\n")
                continue

            images = []
            prompt = user_in
            if user_in.startswith("!img "):
                parts = user_in[5:].split(" ", 1)
                img_path = parts[0].strip()
                prompt = parts[1].strip() if len(parts) > 1 else "Analyze this image in detail."
                if os.path.exists(img_path):
                    images.append(img_path)
                else:
                    print(f"[Warning] Image file not found: {img_path}")

            motor.execute(prompt, image_paths=images if images else None, stream=True, verbose=args.trace)
        except KeyboardInterrupt:
            print("\nSession aborted.")
            break

if __name__ == "__main__":
    cli()
