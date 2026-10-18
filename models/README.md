# Multimodal Model Directory

This directory houses the GGUF model weights for Project: Fleece Beta.

### Automated Setup
The engine automatically downloads the verified GGUF weights on first launch.

You can also pre-download the weights manually at any time by running from the repository root:
```bash
python setup_models.py
```

### Model Specifications
1. **Core VLM:** Qwen 3.5 4B Vision Instruct
   - **Quantization:** Q4_K_M
   - **Filename:** `model.gguf`
   - **Source:** Hugging Face (`unsloth/Qwen3.5-4B-GGUF`)
2. **Multimodal Projector:** mmproj-F16
   - **Filename:** `mmproj.gguf`
   - **Source:** Hugging Face (`unsloth/Qwen3.5-4B-GGUF`)
