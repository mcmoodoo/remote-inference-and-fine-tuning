# RunPod Serverless Video Captioner

Deploy a serverless video captioning API using VLLM and RunPod's GitHub integration, built with Podman containers.

## Features

- 🚀 **Serverless deployment** - Auto-scaling, pay-per-use
- 🔄 **GitHub Actions CI/CD** - Automatic deployment on push
- 🐳 **Podman-based** - Rootless container builds
- 🎥 **Multiple analysis modes**:
  - Comprehensive (overall, temporal, scene analysis)
  - Quick (single concise description)
  - Action-focused (movement and activity detection)
  - Custom prompts

## Quick Start

### 1. Fork/Clone Repository
```bash
git clone <your-repo>
cd remote-inference-and-fine-tuning
```

### 2. Set up GitHub Secrets
Add to your repository settings:
- `RUNPOD_API_KEY` - Your RunPod API key

### 3. Deploy via GitHub
```bash
git push origin serverless  # or main
```
GitHub Actions will automatically:
1. Build container with Podman
2. Push to GitHub Container Registry
3. Deploy to RunPod serverless

### 4. Manual Deployment (Alternative)
```bash
# Set API key
export RUNPOD_API_KEY="your-key"

# Build and deploy
just build-serverless
just push-to-ghcr latest
just deploy-serverless video-captioner
```

## API Usage

### Request Format
```json
{
  "input": {
    "video": "https://example.com/video.mp4",
    "analysis_type": "comprehensive",
    "num_frames": 8
  }
}
```

### Input Parameters
- `video` (required): Video URL, base64 data, or file path
- `analysis_type` (optional): 
  - `"comprehensive"` - Full analysis (default)
  - `"quick"` - Concise description
  - `"action"` - Focus on movements
  - Custom prompt string
- `num_frames` (optional): Number of frames to analyze (default: 8)

### Response Format
```json
{
  "status": "completed",
  "video_info": {
    "duration": 10.5,
    "frames_analyzed": 8,
    "frame_timestamps": [0, 1.5, 3.0, ...]
  },
  "model": "llava-hf/llava-v1.6-mistral-7b-hf",
  "analysis": {
    "overall_description": "...",
    "temporal_changes": "...",
    "scene_context": "..."
  }
}
```

## Testing

### Local Testing with Podman
```bash
# Build and run locally
just test-serverless-local

# In another terminal, test the endpoint
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"input": {"video": "test.mp4", "analysis_type": "quick"}}'
```

### Test Deployed Endpoint
```bash
# Get endpoint ID
just list-serverless-endpoints

# Test with sample video
just test-serverless-endpoint <endpoint_id> https://example.com/video.mp4
```

## Monitoring

```bash
# Check endpoint health
just check-serverless-status <endpoint_id>

# List all endpoints
just list-serverless-endpoints
```

## Architecture

```
GitHub Push → GitHub Actions → Podman Build → ghcr.io → RunPod Serverless
                                     ↓
                              Container with:
                              - VLLM Model
                              - Handler Logic
                              - Video Processing
```

## Cost Optimization

- **Min Workers**: 0 (scales to zero when idle)
- **Max Workers**: 3 (configurable in Justfile)
- **GPU Type**: RTX 5090 (optimal price/performance)
- **Model**: Pre-downloaded in container for faster cold starts

## Troubleshooting

### Container Build Issues
```bash
# Test build locally
podman build -t test:latest .

# Check logs
podman logs <container_id>
```

### Deployment Issues
```bash
# Verify API key
echo $RUNPOD_API_KEY

# Check GitHub Actions logs
# Go to GitHub → Actions tab
```

### Model Loading Issues
- Ensure sufficient GPU memory (RTX 5090 recommended)
- Check `gpu_memory_utilization` in handler.py (default: 0.9)

## Development

### Project Structure
```
.
├── handler.py              # Serverless handler logic
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── .github/
│   ├── workflows/
│   │   └── deploy.yml    # CI/CD pipeline
│   └── tests.json        # Test cases
└── Justfile              # Automation commands
```

### Adding Features
1. Modify `handler.py` for new analysis types
2. Update `tests.json` with test cases
3. Push to trigger deployment

## License

See LICENSE file in repository.