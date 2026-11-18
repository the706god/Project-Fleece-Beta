"""
Isolated Subprocess Worker for Mimir OCR Pass 3 (Qwen2.5-VL-7B-Instruct-Q4_K_M).

Runs strictly as a stateless, non-chat optical transcription processor in its own
dedicated process so:
1. Main app processes never load/unload two competing Metal GGUF handlers
   (eliminating ggml_metal GGML_ASSERT residency crashes).
2. Unified RAM (~5.7 GB) and Metal buffers are released 100% cleanly on worker exit.
3. All conversational chatter / interactive chat behavior is suppressed at system,
   prompt-ordering (image_url first), and sampling layers.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path


def _load_local_turboquant():
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from engine.turboquant_loader import configure_turboquant_runtime

        configure_turboquant_runtime()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Mimir OCR Pass 3 Isolated VLM Worker")
    parser.add_argument("--model", required=True, help="Path to Qwen2.5-VL GGUF weights")
    parser.add_argument("--mmproj", required=True, help="Path to mmproj-F16 GGUF weights")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size for OCR crops")
    args = parser.parse_args()

    _load_local_turboquant()

    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler
    except Exception as exc:
        sys.stdout.write(json.dumps({"status": "error", "error": f"Import failed: {exc}"}) + "\n")
        sys.stdout.flush()
        return 1

    llm = None
    try:
        handler = Qwen25VLChatHandler(clip_model_path=args.mmproj, verbose=False)
        for init_kwargs in (
            {
                "n_ctx": args.n_ctx,
                "n_gpu_layers": -1,
                "type_k": 8,
                "type_v": 8,
                "flash_attn": True,
                "verbose": False,
            },
            {
                "n_ctx": args.n_ctx,
                "n_gpu_layers": -1,
                "verbose": False,
            },
        ):
            try:
                llm = Llama(model_path=args.model, chat_handler=handler, **init_kwargs)
                break
            except Exception:
                llm = None

        if llm is None:
            raise RuntimeError(f"Unable to initialize Qwen2.5-VL worker for {args.model}")

        sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
        sys.stdout.flush()

        system_prompt = (
            "Strict optical transcription processor. "
            "Output ONLY exact visible text verbatim. Zero commentary, zero greetings, zero explanations."
        )
        user_prompt = "Transcribe exact visible text verbatim. Mark unreadable characters with [?]."

        for line in sys.stdin:
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                req = json.loads(raw_line)
            except Exception as parse_err:
                sys.stdout.write(json.dumps({"id": None, "raw_text": "", "error": str(parse_err)}) + "\n")
                sys.stdout.flush()
                continue

            if req.get("cmd") == "shutdown":
                break

            req_id = req.get("id")
            img_b64 = req.get("image_b64", "")
            max_tokens = int(req.get("max_tokens", 320))
            if not img_b64:
                sys.stdout.write(json.dumps({"id": req_id, "raw_text": ""}) + "\n")
                sys.stdout.flush()
                continue

            uri = f"data:image/png;base64,{img_b64}"
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": uri}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            try:
                resp = llm.create_chat_completion(
                    messages=messages,
                    temperature=0.0,
                    repeat_penalty=1.05,
                    max_tokens=max_tokens,
                    stop=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
                )
                raw_out = (resp["choices"][0]["message"]["content"] or "").strip()
                sys.stdout.write(json.dumps({"id": req_id, "raw_text": raw_out}) + "\n")
            except Exception as inf_err:
                sys.stdout.write(json.dumps({"id": req_id, "raw_text": "", "error": str(inf_err)}) + "\n")
            sys.stdout.flush()

    except Exception as fatal_err:
        sys.stdout.write(
            json.dumps(
                {"status": "error", "error": f"{fatal_err}", "trace": traceback.format_exc()}
            )
            + "\n"
        )
        sys.stdout.flush()
        return 1
    finally:
        if llm is not None and hasattr(llm, "close"):
            try:
                llm.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
