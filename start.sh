#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  ComfyUI + ID-LoRA-LTX2.3 — Koyeb Startup"
echo "============================================"

MODELS_DIR="${MODELS_PATH:-/app/models}"
mkdir -p "$MODELS_DIR"

# ---- Authenticate with HuggingFace if token is set ----
if [ -n "${HF_TOKEN:-}" ]; then
    echo ">>> Logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo ">>> WARNING: HF_TOKEN not set. Gated model downloads (Gemma) will fail."
    echo ">>>          Set HF_TOKEN in your Koyeb environment variables."
fi

# ---- Download models if not already present ----
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    echo ">>> Checking/downloading models to $MODELS_DIR ..."

    # LTX-2.3 base model (~44GB)
    if [ ! -f "$MODELS_DIR/ltx-2.3-22b-dev.safetensors" ]; then
        echo ">>> Downloading LTX-2.3 base model..."
        huggingface-cli download Lightricks/LTX-2.3 \
            ltx-2.3-22b-dev.safetensors --local-dir "$MODELS_DIR"
    else
        echo ">>> LTX-2.3 base model already present, skipping."
    fi

    # Gemma text encoder
    if [ ! -d "$MODELS_DIR/gemma-3-12b-it-qat-q4_0-unquantized" ]; then
        echo ">>> Downloading Gemma text encoder..."
        huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
            --local-dir "$MODELS_DIR/gemma-3-12b-it-qat-q4_0-unquantized"
    else
        echo ">>> Gemma text encoder already present, skipping."
    fi

    # Spatial upscaler (for two-stage)
    if [ ! -f "$MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" ]; then
        echo ">>> Downloading spatial upscaler..."
        huggingface-cli download Lightricks/LTX-2.3 \
            ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir "$MODELS_DIR"
    else
        echo ">>> Spatial upscaler already present, skipping."
    fi

    # Distilled LoRA (for two-stage)
    if [ ! -f "$MODELS_DIR/ltx-2.3-22b-distilled-lora-384.safetensors" ]; then
        echo ">>> Downloading distilled LoRA..."
        huggingface-cli download Lightricks/LTX-2.3 \
            ltx-2.3-22b-distilled-lora-384.safetensors --local-dir "$MODELS_DIR"
    else
        echo ">>> Distilled LoRA already present, skipping."
    fi

    # ID-LoRA CelebVHQ checkpoint
    if [ ! -d "$MODELS_DIR/id-lora-celebvhq-ltx2.3" ]; then
        echo ">>> Downloading ID-LoRA CelebVHQ checkpoint..."
        huggingface-cli download AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K \
            lora_weights.safetensors --local-dir "$MODELS_DIR/id-lora-celebvhq-ltx2.3"
    else
        echo ">>> ID-LoRA CelebVHQ checkpoint already present, skipping."
    fi

    # ID-LoRA TalkVid checkpoint
    if [ ! -d "$MODELS_DIR/id-lora-talkvid-ltx2.3" ]; then
        echo ">>> Downloading ID-LoRA TalkVid checkpoint..."
        huggingface-cli download AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K \
            lora_weights.safetensors --local-dir "$MODELS_DIR/id-lora-talkvid-ltx2.3"
    else
        echo ">>> ID-LoRA TalkVid checkpoint already present, skipping."
    fi

    echo ">>> All models ready."
else
    echo ">>> SKIP_MODEL_DOWNLOAD=1 — skipping model download."
fi

# ---- Start ComfyUI ----
echo ">>> Starting ComfyUI on 0.0.0.0:8188 ..."
exec python /app/ComfyUI/main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch
