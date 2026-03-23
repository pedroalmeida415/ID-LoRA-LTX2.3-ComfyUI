#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  ComfyUI + ID-LoRA-LTX2.3 — Koyeb Startup"
echo "============================================"

MODELS_DIR="${MODELS_PATH:-/app/models}"
mkdir -p "$MODELS_DIR"

# ---- HuggingFace auth ----
# HF_TOKEN env var is automatically used by huggingface-cli download.
# No explicit login needed.
if [ -z "${HF_TOKEN:-}" ]; then
    echo ">>> WARNING: HF_TOKEN not set. Gated model downloads (Gemma) will fail."
    echo ">>>          Set HF_TOKEN in your Koyeb environment variables."
fi

# ---- Download models in the background ----
download_models() {
    echo ">>> [background] Starting model downloads to $MODELS_DIR ..."

    # LTX-2.3 base model
    if [ ! -f "$MODELS_DIR/ltx-2.3-22b-dev.safetensors" ]; then
        echo ">>> [background] Downloading LTX-2.3 base model..."
        huggingface-cli download Lightricks/LTX-2.3 \
            ltx-2.3-22b-dev.safetensors --local-dir "$MODELS_DIR" 2>&1
    else
        echo ">>> [background] LTX-2.3 base model already present."
    fi

    # Gemma text encoder
    if [ ! -d "$MODELS_DIR/gemma-3-12b-it-qat-q4_0-unquantized" ] || \
       [ ! -f "$MODELS_DIR/gemma-3-12b-it-qat-q4_0-unquantized/config.json" ]; then
        echo ">>> [background] Downloading Gemma text encoder..."
        huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
            --local-dir "$MODELS_DIR/gemma-3-12b-it-qat-q4_0-unquantized" 2>&1
    else
        echo ">>> [background] Gemma text encoder already present."
    fi

    # Spatial upscaler (two-stage)
    if [ ! -f "$MODELS_DIR/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" ]; then
        echo ">>> [background] Downloading spatial upscaler..."
        huggingface-cli download Lightricks/LTX-2.3 \
            ltx-2.3-spatial-upscaler-x2-1.1.safetensors --local-dir "$MODELS_DIR" 2>&1
    else
        echo ">>> [background] Spatial upscaler already present."
    fi

    # Distilled LoRA (two-stage)
    if [ ! -f "$MODELS_DIR/ltx-2.3-22b-distilled-lora-384.safetensors" ]; then
        echo ">>> [background] Downloading distilled LoRA..."
        huggingface-cli download Lightricks/LTX-2.3 \
            ltx-2.3-22b-distilled-lora-384.safetensors --local-dir "$MODELS_DIR" 2>&1
    else
        echo ">>> [background] Distilled LoRA already present."
    fi

    # ID-LoRA CelebVHQ
    if [ ! -d "$MODELS_DIR/id-lora-celebvhq-ltx2.3" ]; then
        echo ">>> [background] Downloading ID-LoRA CelebVHQ..."
        huggingface-cli download AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K \
            lora_weights.safetensors --local-dir "$MODELS_DIR/id-lora-celebvhq-ltx2.3" 2>&1
    else
        echo ">>> [background] ID-LoRA CelebVHQ already present."
    fi

    # ID-LoRA TalkVid
    if [ ! -d "$MODELS_DIR/id-lora-talkvid-ltx2.3" ]; then
        echo ">>> [background] Downloading ID-LoRA TalkVid..."
        huggingface-cli download AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K \
            lora_weights.safetensors --local-dir "$MODELS_DIR/id-lora-talkvid-ltx2.3" 2>&1
    else
        echo ">>> [background] ID-LoRA TalkVid already present."
    fi

    echo "=========================================="
    echo ">>> [background] ALL MODELS READY ✅"
    echo "=========================================="
}

# Start model downloads in background
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
    download_models &
    DOWNLOAD_PID=$!
    echo ">>> Model download started in background (PID: $DOWNLOAD_PID)"
else
    echo ">>> SKIP_MODEL_DOWNLOAD=1 — skipping."
fi

# ---- Start ComfyUI immediately (health checks pass right away) ----
echo ">>> Starting ComfyUI on 0.0.0.0:8188 ..."
exec python /app/ComfyUI/main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --disable-auto-launch
