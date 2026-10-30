#!/usr/bin/env bash
"true" '''\'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [ -f "$DIR/.venv/bin/python" ]; then
    PY="$DIR/.venv/bin/python"
elif [ -f "/Users/ATLAS/Desktop/CODING/MSNFH INC/charon/.venv/bin/python" ]; then
    PY="/Users/ATLAS/Desktop/CODING/MSNFH INC/charon/.venv/bin/python"
else
    PY="python3"
fi
if [ "$(uname -s)" = "Darwin" ] && /usr/sbin/sysctl -n hw.optional.arm64 >/dev/null 2>&1; then
    exec arch -arm64 "$PY" "$0" "$@"
else
    exec "$PY" "$0" "$@"
fi
exit $?
'''
import os
import sys
import json
import argparse
from pathlib import Path
from dataclasses import asdict

# Insert local engine root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine.auditor import FleeceAgentAuditor, AuditReport


def colorize(text: str, color_code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{color_code}m{text}\033[0m"


def print_cli_report(report: AuditReport):
    score = report.compatibility_score
    if score >= 85:
        badge = colorize(f"[{score}/100 - SOVEREIGN READY]", "1;32")
    elif score >= 60:
        badge = colorize(f"[{score}/100 - OPTIMIZATION ADVISED]", "1;33")
    else:
        badge = colorize(f"[{score}/100 - CLOUD REFACTOR REQUIRED]", "1;31")

    print("=" * 76)
    print(colorize(" FLEECE AGENT & SKILL COMPATIBILITY AUDITOR", "1;36"))
    print("=" * 76)
    print(f" Target           : {report.target_path}")
    print(f" Target Type      : {report.target_type}")
    print(f" Compatibility    : {badge}")
    print(f" Recommended Core : {colorize(report.recommended_edition, '1;35')}")
    if report.recommended_mode:
        print(f" Reasoning Mode   : {{REASON:{report.recommended_mode}}}")
    print(f" Est. Token Size  : ~{report.token_estimate:,} tokens")
    if report.detected_frameworks:
        print(f" Frameworks Found : {', '.join(report.detected_frameworks)}")
    print("-" * 76)
    print(" 6-PILLAR AUDIT FINDINGS:")

    for f in report.findings:
        if f.severity == "PASS":
            icon = colorize(" [PASS] ", "1;32")
        elif f.severity == "WARN":
            icon = colorize(" [WARN] ", "1;33")
        else:
            icon = colorize(" [FAIL] ", "1;31")
        print(f"{icon} {f.title}")
        print(f"         {f.description}")
        if f.remediation:
            print(f"         {colorize('-> Action: ' + f.remediation, '0;36')}")

    print("-" * 76)
    print(" SOVEREIGN CRATE MOTOR INTEGRATION CODE:")
    print(colorize(report.generated_adapter_code.strip(), "0;37"))
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(
        description="Fleece Sovereign Agent & Skill Compatibility Auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./audit_agent.py path/to/SKILL.md
  ./audit_agent.py path/to/agent_repo/ --format markdown --output audit_report.md
  ./audit_agent.py path/to/openai_assistant.py --strict
        """
    )
    parser.add_argument("target", type=str, help="Path to agent prompt, skill folder, config, or Python file")
    parser.add_argument("--format", choices=["text", "json", "markdown"], default="text", help="Output report format")
    parser.add_argument("--output", type=str, help="Optional output path to write report")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 if score < 80 or any FAIL finding exists")

    args = parser.parse_args()
    target_path = Path(args.target).resolve()

    if not target_path.exists():
        print(colorize(f"Error: Target path does not exist: {target_path}", "1;31"), file=sys.stderr)
        sys.exit(1)

    auditor = FleeceAgentAuditor()
    report = auditor.audit(target_path)

    if args.format == "json":
        out_content = json.dumps(
            {
                "target_path": report.target_path,
                "target_type": report.target_type,
                "compatibility_score": report.compatibility_score,
                "rating": report.rating,
                "recommended_edition": report.recommended_edition,
                "recommended_mode": report.recommended_mode,
                "detected_frameworks": report.detected_frameworks,
                "token_estimate": report.token_estimate,
                "findings": [asdict(f) for f in report.findings],
                "adapter_code": report.generated_adapter_code,
            },
            indent=2
        )
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_content)
            print(f"Report written to {args.output}")
        else:
            print(out_content)
    elif args.format == "markdown":
        md = report.to_markdown()
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"Markdown report written to {args.output}")
        else:
            print(md)
    else:
        print_cli_report(report)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            print(f"Markdown report also saved to {args.output}")

    if args.strict:
        has_fail = any(f.severity == "FAIL" for f in report.findings)
        if report.compatibility_score < 80 or has_fail:
            sys.exit(1)


if __name__ == "__main__":
    main()
