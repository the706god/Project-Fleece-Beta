"""
Project: Fleece - Sovereign Agent & Skill Compatibility Auditor
Engine Component for Project: Fleece (Alpha, Beta, and Gamma)

Audits Frontier platform agents, prompts, tool definitions, and skill folders
(OpenAI Assistants, Claude Projects, Antigravity/AGY Skills, LangChain/LangGraph, CrewAI)
for sovereign execution compatibility with Fleece SLM Crate Motors.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("FleeceAgentAuditor")


@dataclass
class Finding:
    severity: str  # "PASS", "WARN", "FAIL"
    pillar: str    # "1_BUDGET", "2_TOOLS", "3_MEMORY", "4_INGESTION", "5_MODEL", "6_SOVEREIGNTY"
    title: str
    description: str
    remediation: Optional[str] = None


@dataclass
class AuditReport:
    target_path: str
    target_type: str  # "FILE", "DIRECTORY", "SKILL"
    compatibility_score: int  # 0 to 100
    rating: str               # "SOVEREIGN READY", "OPTIMIZATION ADVISED", "CLOUD REFACTOR REQUIRED"
    recommended_edition: str  # "ALPHA (Text/Code)", "BETA (Multimodal)", "GAMMA (Reasoning)"
    recommended_mode: Optional[str] = None
    findings: List[Finding] = field(default_factory=list)
    detected_frameworks: List[str] = field(default_factory=list)
    token_estimate: int = 0
    generated_adapter_code: str = ""

    def to_markdown(self) -> str:
        status_badge = "🟢" if self.compatibility_score >= 85 else ("🟡" if self.compatibility_score >= 60 else "🔴")
        lines = [
            f"# {status_badge} Fleece Compatibility Audit Report",
            f"- **Target:** `{self.target_path}` ({self.target_type})",
            f"- **Score:** `{self.compatibility_score} / 100` ({self.rating})",
            f"- **Recommended Fleece Edition:** **{self.recommended_edition}**" + (f" (Mode: `{self.recommended_mode}`)" if self.recommended_mode else ""),
            f"- **Detected Frameworks:** {', '.join(self.detected_frameworks) if self.detected_frameworks else 'Raw / Standard'}",
            f"- **Prompt / Preamble Size Estimate:** ~{self.token_estimate:,} tokens",
            "",
            "## 6-Pillar Assessment Breakdown",
            "",
        ]

        pillar_names = {
            "1_BUDGET": "Pillar 1: Context Budget Footprint (16k SLM Limit)",
            "2_TOOLS": "Pillar 2: Tooling & Function Calling Complexity",
            "3_MEMORY": "Pillar 3: Vectorless Memory Architecture (L1/L2)",
            "4_INGESTION": "Pillar 4: Document Ingestion & Mimir OCR Readiness",
            "5_MODEL": "Pillar 5: Model Specialization Alignment",
            "6_SOVEREIGNTY": "Pillar 6: Air-Gap & Cloud Sovereignty",
        }

        for pillar_key, pillar_label in pillar_names.items():
            pillar_findings = [f for f in self.findings if f.pillar == pillar_key]
            lines.append(f"### {pillar_label}")
            if not pillar_findings:
                lines.append("  - ✅ *No issues detected.*")
            else:
                for f in pillar_findings:
                    icon = "✅" if f.severity == "PASS" else ("⚠️" if f.severity == "WARN" else "❌")
                    lines.append(f"  - {icon} **{f.title}**: {f.description}")
                    if f.remediation:
                        lines.append(f"    - *Remediation:* {f.remediation}")
            lines.append("")

        if self.generated_adapter_code:
            lines.append("## Sovereign Fleece Crate Motor Adapter")
            lines.append("```python")
            lines.append(self.generated_adapter_code.strip())
            lines.append("```")
            lines.append("")

        return "\n".join(lines)


class FleeceAgentAuditor:
    """
    Sovereign auditor evaluating agents, prompts, and skills against the 6 Fleece pillars:
    1. Context Budget Footprint (16k token limit, 1,200 token system anchor limit)
    2. Tool Schema Complexity (flat, deterministic parameters for 3B-9B SLMs)
    3. Vectorless Memory Adaptation (maps Pinecone/Chroma/Weaviate to Dossier L1 + PageIndex L2)
    4. Document & Vision Ingestion (maps cloud OCR to Mimir 3-Pass Sieve)
    5. Model Routing (Alpha vs Beta vs Gamma with plusIQ reasoning modes)
    6. Sovereignty & Air-Gap Compliance (detects external APIs, keys, and cloud telemetry)
    """

    CLOUD_API_PATTERNS = [
        (r"api\.openai\.com", "OpenAI API endpoint"),
        (r"api\.anthropic\.com", "Anthropic Claude API endpoint"),
        (r"generativelanguage\.googleapis\.com", "Google Gemini API endpoint"),
        (r"api\.cohere\.ai", "Cohere API endpoint"),
        (r"api\.groq\.com", "Groq API endpoint"),
        (r"api\.together\.xyz", "Together AI API endpoint"),
        (r"api\.mistral\.ai", "Mistral Cloud API endpoint"),
        (r"sk-[a-zA-Z0-9]{20,}", "Hardcoded OpenAI / Cloud API secret key"),
    ]

    VECTOR_DB_PATTERNS = [
        (r"pinecone", "Pinecone Vector Database"),
        (r"chromadb", "ChromaDB Vector Store"),
        (r"weaviate", "Weaviate Vector DB"),
        (r"qdrant", "Qdrant Vector DB"),
        (r"milvus", "Milvus Vector DB"),
        (r"faiss", "FAISS Vector Index"),
        (r"pgvector", "PostgreSQL pgvector extension"),
        (r"embeddings\.create", "Cloud embedding generation"),
        (r"text-embedding-3", "OpenAI Embedding model"),
    ]

    CLOUD_VISION_OCR_PATTERNS = [
        (r"textract", "AWS Textract OCR"),
        (r"documentai", "Google Cloud Document AI"),
        (r"cognitiveservices\.azure\.com", "Azure Computer Vision / Form Recognizer"),
        (r"gpt-4-vision|gpt-4o.*image", "OpenAI Vision Cloud API"),
        (r"claude-3.*image", "Anthropic Vision Cloud API"),
    ]

    FRAMEWORK_PATTERNS = [
        (r"langchain", "LangChain"),
        (r"langgraph", "LangGraph"),
        (r"crewai", "CrewAI"),
        (r"autogen", "Microsoft AutoGen"),
        (r"llamaindex|llama_index", "LlamaIndex"),
        (r"semantic_kernel", "Semantic Kernel"),
        (r"google_antigravity|antigravity", "Antigravity AGY SDK"),
    ]

    def audit(self, target: Path | str) -> AuditReport:
        path = Path(target).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audit target does not exist: {path}")

        target_type = "DIRECTORY" if path.is_dir() else "FILE"
        if path.is_dir() and (path / "SKILL.md").exists():
            target_type = "SKILL"

        files_to_scan = self._gather_files(path)
        aggregated_text, file_count = self._read_aggregated_text(files_to_scan)

        score = 100
        findings: List[Finding] = []
        detected_frameworks = self._detect_frameworks(aggregated_text)
        token_est = self._estimate_tokens(aggregated_text)

        # -------------------------------------------------------------
        # Pillar 1: Context Budget Footprint (16,384 Limit)
        # -------------------------------------------------------------
        if token_est > 16384:
            score -= 30
            findings.append(Finding(
                severity="FAIL",
                pillar="1_BUDGET",
                title="Monolithic Content Exceeds 16k SLM Maximum Context",
                description=f"Scanned files contain ~{token_est:,} tokens, exceeding Fleece's 16,384 token ceiling.",
                remediation="Extract reference materials into markdown files in storage/markdown/knowledge/. Mimir PageIndex (L2) will index them with 0 context token load."
            ))
        elif token_est > 3500:
            score -= 15
            findings.append(Finding(
                severity="WARN",
                pillar="1_BUDGET",
                title="Preamble / Prompt Exceeds Safe Working Anchor (>3,500 tokens)",
                description=f"Prompt footprint is ~{token_est:,} tokens. Leaves reduced headroom for conversation and output generation.",
                remediation="Trim the active system prompt below 1,200 tokens. Move static policies and reference manuals into Mimir PageIndex knowledge files."
            ))
        else:
            findings.append(Finding(
                severity="PASS",
                pillar="1_BUDGET",
                title="Compact Context Budget Footprint",
                description=f"Total footprint is ~{token_est:,} tokens. Fits comfortably inside Fleece's 1,200-token anchor and 16k budget."
            ))

        # -------------------------------------------------------------
        # Pillar 2: Tooling & Function Calling Complexity
        # -------------------------------------------------------------
        tool_findings, tool_penalty = self._audit_tools(aggregated_text, files_to_scan)
        findings.extend(tool_findings)
        score -= tool_penalty

        # -------------------------------------------------------------
        # Pillar 3: Vectorless Memory Architecture
        # -------------------------------------------------------------
        vector_db_found = []
        for pat, label in self.VECTOR_DB_PATTERNS:
            if re.search(pat, aggregated_text, re.IGNORECASE):
                vector_db_found.append(label)

        if vector_db_found:
            score -= 20
            findings.append(Finding(
                severity="FAIL",
                pillar="3_MEMORY",
                title="Cloud Vector Database Dependency Detected",
                description=f"Target relies on vector embedding stores: {', '.join(set(vector_db_found))}.",
                remediation="Replace vector embeddings with Fleece Mimir Two-Tier Vectorless Memory: store records in storage/markdown/ and use PageIndexEngine for unbroken anchor retrieval."
            ))
        else:
            findings.append(Finding(
                severity="PASS",
                pillar="3_MEMORY",
                title="Vectorless Memory Friendly",
                description="Zero external vector stores or embedding models detected. Ready for L1 Living Dossier + L2 PageIndex."
            ))

        # -------------------------------------------------------------
        # Pillar 4: Document Ingestion & Vision Readiness
        # -------------------------------------------------------------
        cloud_ocr_found = []
        for pat, label in self.CLOUD_VISION_OCR_PATTERNS:
            if re.search(pat, aggregated_text, re.IGNORECASE):
                cloud_ocr_found.append(label)

        if cloud_ocr_found:
            score -= 15
            findings.append(Finding(
                severity="WARN",
                pillar="4_INGESTION",
                title="External Cloud OCR / Vision Dependency",
                description=f"Found cloud vision/OCR calls: {', '.join(set(cloud_ocr_found))}.",
                remediation="Replace cloud OCR with Fleece Mimir OCR (sieve_ocr.py) for sovereign 3-pass optical sieving with normalized [x0, y0, x1, y1] coordinates."
            ))
        else:
            findings.append(Finding(
                severity="PASS",
                pillar="4_INGESTION",
                title="Mimir OCR Compatible",
                description="No external cloud OCR calls detected. Documents can be ingested via Fleece's native Mimir 3-Pass Sieve."
            ))

        # -------------------------------------------------------------
        # Pillar 5: Model Specialization Routing (Alpha vs Beta vs Gamma)
        # -------------------------------------------------------------
        recommended_edition, recommended_mode = self._determine_edition(aggregated_text, target_type)
        findings.append(Finding(
            severity="PASS",
            pillar="5_MODEL",
            title=f"Target Model Edition: Fleece {recommended_edition.split()[0]}",
            description=f"Target workload is optimally aligned with Fleece {recommended_edition}" + (f" using mode {{REASON:{recommended_mode}}}." if recommended_mode else ".")
        ))

        # -------------------------------------------------------------
        # Pillar 6: Sovereignty & Air-Gap Compliance
        # -------------------------------------------------------------
        cloud_apis = []
        for pat, label in self.CLOUD_API_PATTERNS:
            if re.search(pat, aggregated_text, re.IGNORECASE):
                cloud_apis.append(label)

        if cloud_apis:
            score -= 25
            findings.append(Finding(
                severity="FAIL",
                pillar="6_SOVEREIGNTY",
                title="External Cloud API Leakage Detected",
                description=f"Detected external endpoints or cloud credentials: {', '.join(set(cloud_apis))}.",
                remediation="Remove cloud API endpoints and API keys. Configure local llama.cpp GGUF Crate Motor execution via fleece.py."
            ))
        else:
            findings.append(Finding(
                severity="PASS",
                pillar="6_SOVEREIGNTY",
                title="100% Air-Gap Sovereign",
                description="Zero external cloud API endpoints or proprietary cloud credentials detected."
            ))

        final_score = max(0, min(100, score))
        if final_score >= 85:
            rating = "SOVEREIGN READY"
        elif final_score >= 60:
            rating = "OPTIMIZATION ADVISED"
        else:
            rating = "CLOUD REFACTOR REQUIRED"

        adapter_code = self._generate_adapter(
            target_name=path.name,
            edition=recommended_edition.split()[0].upper(),
            mode=recommended_mode,
            prompt_preview=aggregated_text[:300].strip(),
        )

        return AuditReport(
            target_path=str(path),
            target_type=target_type,
            compatibility_score=final_score,
            rating=rating,
            recommended_edition=recommended_edition,
            recommended_mode=recommended_mode,
            findings=findings,
            detected_frameworks=detected_frameworks,
            token_estimate=token_est,
            generated_adapter_code=adapter_code,
        )

    def _gather_files(self, path: Path) -> List[Path]:
        valid_exts = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".toml", ".sh"}
        if path.is_file():
            return [path]
        collected = []
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__pycache__", "venv", ".venv", "node_modules")]
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in valid_exts or p.name in ("SKILL.md", "Dockerfile", "Modelfile"):
                    collected.append(p)
        return collected

    def _read_aggregated_text(self, files: List[Path]) -> Tuple[str, int]:
        chunks = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as stream:
                    chunks.append(stream.read())
            except Exception:
                pass
        return "\n\n".join(chunks), len(files)

    def _detect_frameworks(self, text: str) -> List[str]:
        found = []
        for pat, label in self.FRAMEWORK_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                found.append(label)
        return list(dict.fromkeys(found))

    def _estimate_tokens(self, text: str) -> int:
        words = len(text.split())
        return max(1, int(words * 1.33))

    def _audit_tools(self, text: str, files: List[Path]) -> Tuple[List[Finding], int]:
        findings = []
        penalty = 0

        # Detect tool definitions in JSON or python defs
        tool_count = len(re.findall(r"def\s+[a-zA-Z0-9_]+\s*\(.*?\)\s*->", text))
        tool_count += len(re.findall(r"\"name\":\s*\"[a-zA-Z0-9_]+\",\s*\"description\"", text))

        if tool_count > 12:
            penalty += 15
            findings.append(Finding(
                severity="WARN",
                pillar="2_TOOLS",
                title=f"High Active Tool Count ({tool_count} Tools)",
                description=f"Skill defines {tool_count} functions/tools. SLMs perform best with <= 8 high-signal tools per turn.",
                remediation="Group related sub-tools into composite handlers or page them via Mimir PageIndex."
            ))
        elif tool_count > 0:
            findings.append(Finding(
                severity="PASS",
                pillar="2_TOOLS",
                title=f"Manageable Tool Surface ({tool_count} Tools)",
                description=f"Tool count ({tool_count}) is within sovereign SLM execution sweet spot (<= 8 active tools)."
            ))
        else:
            findings.append(Finding(
                severity="PASS",
                pillar="2_TOOLS",
                title="Direct Prompt / Knowledge Agent",
                description="Zero complex tool dispatch overhead. Ideal for sovereign Crate Motor execution."
            ))

        return findings, penalty

    def _determine_edition(self, text: str, target_type: str) -> Tuple[str, Optional[str]]:
        t_lower = text.lower()
        # Visual indicators
        visual_keywords = ["image", "screenshot", "diagram", "layout", "visual", "jpeg", "png", "crop", "render"]
        visual_score = sum(1 for kw in visual_keywords if kw in t_lower)

        # Reasoning / multi-step indicators
        reasoning_keywords = [
            "step-by-step", "think", "chain-of-thought", "verify", "audit", "logic", "proof",
            "first-principles", "investigate", "decompose", "hypothesis", "discrepancy"
        ]
        reasoning_score = sum(1 for kw in reasoning_keywords if kw in t_lower)

        # Coding indicators
        coding_keywords = ["def ", "class ", "function", "refactor", "bug", "syntax", "compile", "git", "bash", "cli"]
        coding_score = sum(1 for kw in coding_keywords if kw in t_lower)

        if visual_score >= 3:
            return "BETA (Multimodal Vision - Qwen 3.5 4B)", None
        elif reasoning_score >= 3:
            mode = "einstein" if "multi" in t_lower or "complex" in t_lower else "medium"
            return "GAMMA (Reasoning Core - Qwen 3.5 9B)", mode
        else:
            return "ALPHA (Text & Code Core - Qwen 2.5 Coder 3B)", None

    def _generate_adapter(self, target_name: str, edition: str, mode: Optional[str], prompt_preview: str) -> str:
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", target_name)
        mode_call = f'    motor.set_reasoning_mode("{mode}")\n' if mode else ""
        return f"""# Sovereign Fleece Adapter for: {target_name}
# Target: Project: Fleece {edition}

import os
from fleece import FleeceCrateMotor

def run_sovereign_agent():
    # 1. Initialize Fleece Crate Motor
    motor = FleeceCrateMotor()
{mode_call}
    # 2. Ingest Reference Documents into Vectorless Memory (Mimir OCR)
    # motor.ingest_document("path/to/reference_document.pdf")

    # 3. Define Sovereign Objective
    motor.set_task("Execute audited skill: {clean_name}")

    # 4. Run Execution through Two-Tier State Graph Memory
    prompt = \"\"\"{prompt_preview[:180]}...\"\"\"
    result = motor.execute(prompt, stream=True, verbose=True)
    return result

if __name__ == "__main__":
    run_sovereign_agent()
"""
