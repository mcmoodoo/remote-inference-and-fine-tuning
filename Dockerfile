# Lean Docker image for video captioning with LLaVA
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install PyTorch and core dependencies
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    runpod>=1.0.0 \
    opencv-python>=4.8.0 \
    pillow>=10.0.0 \
    numpy>=1.24.0 \
    requests>=2.31.0

# Copy application files
COPY handler.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
ENV HF_HOME=/workspace/.cache/huggingface

# Pre-download model components
RUN python3 -c "from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration; \
    LlavaNextProcessor.from_pretrained('${MODEL_NAME}'); \
    print('Model downloaded successfully')"

CMD ["python3", "-u", "handler.py"]