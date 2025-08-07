# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a video captioning and analysis system using vision-language models (VLLMs) for automated video understanding. The project uses VLLM for high-performance inference on GPU infrastructure, with two deployment modes:
1. **Pod-based deployment**: Traditional GPU pods for development and testing
2. **Serverless deployment**: Auto-scaling serverless workers deployed via GitHub Actions

## Common Development Commands

### Serverless Deployment (Recommended)
```bash
# Build container with Podman
just build-serverless

# Test locally with Podman
just test-serverless-local

# Push to GitHub Container Registry
just push-to-ghcr latest

# Deploy to RunPod serverless
export RUNPOD_API_KEY="your-api-key"
just deploy-serverless video-captioner

# Test deployed endpoint
just test-serverless-endpoint <endpoint_id> https://example.com/video.mp4

# Check endpoint status
just check-serverless-status <endpoint_id>

# List all endpoints
just list-serverless-endpoints
```

### Pod-based Infrastructure Management
```bash
# Get running RunPod instance ID
just get-running-pod-id

# Connect to running GPU pod via SSH
just connect-to-running-pod

# Create new RunPod with RTX 5090
just create-pod

# Terminate running pod
just destroy-pod

# Download test videos from S3
just download-video-files-from-s3
```

### Running the Application
```bash
# Start VLLM server (required for server mode)
vllm serve llava-hf/llava-v1.6-mistral-7b-hf --max-model-len 4096

# Run video captioner in server mode (default)
python video_captioner_vllm.py <video_file>

# Run video captioner in offline mode (no server required)
python video_captioner_vllm.py <video_file> --offline
```

### Development Setup
```bash
# Install dependencies
pip install transformers torch pillow opencv-python accelerate vllm
```

## Architecture and Code Structure

### Core Architecture
The system supports multiple deployment and execution strategies:

#### Serverless Architecture (Production)
- **handler.py**: RunPod serverless handler that processes video analysis requests
- **Dockerfile**: Container image with VLLM and dependencies using Podman
- **GitHub Actions**: Automated CI/CD pipeline for building and deploying to RunPod
- Supports multiple analysis types: comprehensive, quick, action-focused, or custom

#### Local Execution Modes
1. **Server Mode** (`VLLMVideoCaptioner`): Communicates with VLLM server via OpenAI-compatible API on localhost:8000
2. **Offline Mode** (`VLLMOfflineVideoCaptioner`): Direct VLLM integration without server dependency

### Key Components

**Serverless Components**:
- `handler.py`: RunPod serverless worker handler with video processing logic
- `Dockerfile`: Container definition using Podman for serverless deployment
- `.github/workflows/deploy.yml`: GitHub Actions CI/CD pipeline
- `.github/tests.json`: Test cases for automated validation
- `requirements.txt`: Python dependencies for serverless environment

**Local Development Components**:
- `video_captioner_vllm.py`: Standalone application for local testing
- `extract_frames_uniform()`: Samples frames uniformly across video timeline
- `VLLMVideoCaptioner`: Server-based inference using HTTP API
- `VLLMOfflineVideoCaptioner`: Direct VLLM inference without server

**Infrastructure Files**:
- `Justfile`: Automation for both pod-based and serverless deployments
- `pyproject.toml`: Python 3.11+ dependency management
- `instructions.md`: Setup and usage documentation

### Design Patterns
- **Strategy Pattern**: Swappable captioner implementations for different inference backends
- **Template Method**: Common video processing workflow with variant inference steps
- **Multi-stage Analysis**: Three-prompt system for comprehensive video understanding

### Key Technical Considerations

1. **GPU Memory Management**: Uses `--gpu-memory-utilization 0.9` for optimal VLLM performance
2. **Frame Extraction**: Uniform temporal sampling for consistent video representation
3. **Base64 Encoding**: Images encoded for API compatibility in server mode
4. **Model**: Uses `llava-hf/llava-v1.6-mistral-7b-hf` vision-language model
5. **SSH Authentication**: Requires `~/.ssh/runpod_ed25519` key for pod access

### Development Workflows

#### Serverless Deployment Workflow (Recommended)
1. Make code changes to `handler.py` or dependencies
2. Test locally with `just test-serverless-local`
3. Push to GitHub - triggers automatic deployment via GitHub Actions
4. Monitor deployment in GitHub Actions tab
5. Test endpoint with `just test-serverless-endpoint`

#### GitHub Actions CI/CD
- **Trigger**: Push to `main` or `serverless` branches
- **Build**: Uses Podman to build container image
- **Registry**: Pushes to GitHub Container Registry (ghcr.io)
- **Deploy**: Automatically deploys to RunPod serverless
- **Testing**: Runs validation tests before deployment

#### Local Development Workflow
1. Create/connect to RunPod GPU instance using Justfile commands
2. Start VLLM server if using server mode
3. Process videos using either server or offline mode
4. Results are printed to console with frame-by-frame and overall analysis

### Important Notes

- **Container Runtime**: Uses Podman instead of Docker for all container operations
- **GPU Requirements**: RTX 5090 recommended for optimal performance
- **Python Version**: 3.11+ required (specified in `.python-version`)
- **API Authentication**: Requires `RUNPOD_API_KEY` environment variable for deployments
- **GitHub Secrets Required**:
  - `RUNPOD_API_KEY`: For deploying to RunPod
  - `GITHUB_TOKEN`: Automatically provided for registry access
- **Serverless Benefits**: Auto-scaling, pay-per-use, no idle costs
- Recent migration from Video-LLaVA to VLLM for performance improvements