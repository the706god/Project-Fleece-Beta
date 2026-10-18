import os
import sys
import json
import base64
import logging
from typing import Generator, List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format="[Fleece-Beta] %(levelname)s: %(message)s")
logger = logging.getLogger("FleeceBetaEngine")

class FleeceVLMEngine:
    """
    Multimodal VLM execution engine for Project: Fleece Beta.
    Supports joint text and visual (image/document render) processing using
    Qwen 3.5 4B + multimodal projector (mmproj).
    """

    def __init__(self, config_path: str = "fleece_config.json"):
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        raw_model_path = self.config.get("model_path", "models/model.gguf")
        raw_mmproj_path = self.config.get("mmproj_path", "models/mmproj.gguf")

        self.model_path = os.path.normpath(os.path.join(self.base_dir, raw_model_path))
        self.mmproj_path = os.path.normpath(os.path.join(self.base_dir, raw_mmproj_path))

        # Handle broken symlinks or missing files via automated download
        broken_model = os.path.islink(self.model_path) and not os.path.exists(self.model_path)
        broken_mmproj = os.path.islink(self.mmproj_path) and not os.path.exists(self.mmproj_path)
        missing_model = not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 1_000_000
        missing_mmproj = not os.path.exists(self.mmproj_path) or os.path.getsize(self.mmproj_path) < 1_000_000

        if broken_model or broken_mmproj or missing_model or missing_mmproj:
            logger.info("VLM models not found locally. Initiating automated first-run setup...")
            sys.path.insert(0, self.base_dir)
            from setup_models import download_models
            download_models()

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(f"Multimodal projector file not found at: {self.mmproj_path}")

        engine_cfg = self.config.get("engine", {})
        self.n_ctx = engine_cfg.get("n_ctx", 16384)
        self.n_threads = engine_cfg.get("n_threads", os.cpu_count() or 4)
        self.n_gpu_layers = engine_cfg.get("n_gpu_layers", -1)
        self.default_temp = engine_cfg.get("temperature", 0.2)
        self.default_top_p = engine_cfg.get("top_p", 0.95)

        self.llm = None
        self.chat_handler = None
        self._init_backend()

    def _init_backend(self):
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler

            logger.info("Initializing Qwen25VLChatHandler with mmproj: %s", self.mmproj_path)
            self.chat_handler = Qwen25VLChatHandler(clip_model_path=self.mmproj_path)

            logger.info("Initializing Llama backend with GGUF: %s", self.model_path)
            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=self.chat_handler,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            logger.info("VLM Engine loaded successfully. Context: %d tokens", self.n_ctx)
        except Exception as e:
            logger.error("Failed to initialize VLM backend: %s", e)
            raise

    def count_tokens(self, text: str) -> int:
        if not self.llm:
            return len(text.split()) * 4 // 3
        tokens = self.llm.tokenize(text.encode("utf-8"))
        return len(tokens)

    def encode_image_uri(self, image_path: str) -> str:
        """Read local image and format as data URI for multimodal ingestion."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        mime = f"image/{ext if ext != 'jpg' else 'jpeg'}"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def generate_chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> Any:
        """
        Execute OpenAI-compatible chat completion supporting both text and images.
        """
        temp = temperature if temperature is not None else self.default_temp
        if stream:
            return self._stream_chat(messages, max_tokens, temp)

        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=self.default_top_p
        )
        return response["choices"][0]["message"]["content"]

    def _stream_chat(self, messages, max_tokens, temp) -> Generator[str, None, None]:
        stream = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=self.default_top_p,
            stream=True
        )
        for chunk in stream:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
