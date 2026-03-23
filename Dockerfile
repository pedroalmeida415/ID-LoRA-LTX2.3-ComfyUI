# ============================================================
# Dockerfile for deploying ComfyUI + ID-LoRA-LTX2.3 on Koyeb
# GPU-enabled, CUDA 12.4, Python 3.11
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ---- Set up working directory ----
WORKDIR /app

# ---- Install ComfyUI ----
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

# Install ComfyUI requirements
RUN pip install -r /app/ComfyUI/requirements.txt

# Pin transformers to 4.x (ComfyUI may install 5.x which is incompatible)
RUN pip install 'transformers>=4.52,<5'

# ---- Install ID-LoRA repo (for packages + model download script) ----
RUN git clone https://github.com/ID-LoRA/ID-LoRA.git /app/ID-LoRA

# Install ltx packages from ID-LoRA
RUN pip install -e /app/ID-LoRA/ID-LoRA-2.3/packages/ltx-core \
    && pip install -e /app/ID-LoRA/ID-LoRA-2.3/packages/ltx-pipelines \
    && pip install -e /app/ID-LoRA/ID-LoRA-2.3/packages/ltx-trainer

# ---- Install ComfyUI custom node: ID-LoRA-LTX2.3-ComfyUI ----
RUN git clone https://github.com/ID-LoRA/ID-LoRA-LTX2.3-ComfyUI.git \
    /app/ComfyUI/custom_nodes/comfyui-id-lora-ltx

# Install custom node requirements (torch, torchaudio, etc.)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install huggingface_hub CLI for model downloads
RUN pip install huggingface_hub[cli]

# ---- Copy example inputs into ComfyUI input dir ----
RUN mkdir -p /app/ComfyUI/input && \
    cp /app/ComfyUI/custom_nodes/comfyui-id-lora-ltx/example_inputs/poster_image.png /app/ComfyUI/input/ && \
    cp /app/ComfyUI/custom_nodes/comfyui-id-lora-ltx/example_inputs/reference.mp3 /app/ComfyUI/input/

# ---- Copy startup script ----
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ---- Expose ComfyUI port ----
EXPOSE 8188

# ---- Set environment variables ----
ENV COMFYUI_PATH=/app/ComfyUI
ENV MODELS_PATH=/app/models
# Set to 1 to skip model download (if models are pre-loaded via volume)
ENV SKIP_MODEL_DOWNLOAD=0
# HuggingFace token for gated models (Gemma). Set via Koyeb env vars.
ENV HF_TOKEN=""

# ---- Entrypoint ----
CMD ["/app/start.sh"]
