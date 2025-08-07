# Use PyTorch official image with CUDA support (smaller and pre-optimized)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Install system dependencies for video processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

# Fix NumPy compatibility and install dependencies
RUN pip install --no-cache-dir \
    "numpy>=2.0,<2.3" \
    transformers>=4.36.0 \
    accelerate>=0.25.0 \
    runpod>=1.0.0 \
    opencv-python>=4.8.0 \
    requests>=2.31.0 \
    && pip cache purge

# Copy application files
COPY handler.py .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# Pre-download model processor only (model will be downloaded at runtime)
RUN python -c "from transformers import LlavaNextProcessor; \
    LlavaNextProcessor.from_pretrained('${MODEL_NAME}'); \
    print('Processor downloaded successfully')" \
    && rm -rf ~/.cache/pip

CMD ["python", "-u", "handler.py"]