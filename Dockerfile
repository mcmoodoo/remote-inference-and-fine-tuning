# Use RunPod's official base image with CUDA support
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install VLLM with CUDA 12.1 support
RUN pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the handler and application code
COPY handler.py .
COPY video_captioner_vllm.py .

# Set environment variables for RunPod
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME="llava-hf/llava-v1.6-mistral-7b-hf"

# Pre-download the model to speed up cold starts
RUN python -c "from transformers import AutoModel, AutoProcessor; \
    AutoProcessor.from_pretrained('${MODEL_NAME}'); \
    print('Model downloaded successfully')"

# RunPod handler is the entrypoint
CMD ["python", "-u", "handler.py"]