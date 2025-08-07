#!/usr/bin/env python3
"""
RunPod Serverless Handler for Video Captioning with VLLM
"""

import runpod
import cv2
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import os
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple
from vllm import LLM, SamplingParams
from vllm.multimodal.image import ImagePixelData

# Initialize VLLM model globally for reuse across requests
print("Initializing VLLM model...")
MODEL_NAME = os.environ.get("MODEL_NAME", "llava-hf/llava-v1.6-mistral-7b-hf")

llm = LLM(
    model=MODEL_NAME,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512
)

print(f"Model {MODEL_NAME} loaded successfully")


def download_video(video_url: str) -> str:
    """Download video from URL to temporary file."""
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    
    # Create temporary file
    suffix = Path(video_url).suffix or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            tmp_file.write(chunk)
        return tmp_file.name


def extract_frames_uniform(video_path: str, num_frames: int = 8) -> Tuple[List[Image.Image], List[float]]:
    """Extract frames uniformly from video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate frame indices to extract uniformly
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    frame_times = []
    
    for target_frame in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_times.append(target_frame / fps)
    
    cap.release()
    
    return frames, frame_times


def generate_description(frames: List[Image.Image], prompt: str) -> str:
    """Generate description using VLLM."""
    # Format prompt with image placeholders
    image_prompt = ""
    for i in range(len(frames)):
        image_prompt += f"<image_{i}>"
    
    full_prompt = f"{image_prompt}\n{prompt}"
    
    # Convert PIL images to format VLLM expects
    image_data = []
    for frame in frames:
        img_array = np.array(frame)
        image_data.append(ImagePixelData(img_array))
    
    # Generate with VLLM
    outputs = llm.generate(
        prompts=[full_prompt],
        multi_modal_data={"image": image_data},
        sampling_params=sampling_params
    )
    
    return outputs[0].outputs[0].text


def process_video(video_source: str, num_frames: int = 8, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """Process video and generate description based on analysis type."""
    
    # Handle video source (URL, base64, or local path)
    if video_source.startswith(('http://', 'https://')):
        video_path = download_video(video_source)
        cleanup_needed = True
    elif video_source.startswith('data:video'):
        # Handle base64 encoded video
        video_data = base64.b64decode(video_source.split(',')[1])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_data)
            video_path = tmp_file.name
        cleanup_needed = True
    else:
        video_path = video_source
        cleanup_needed = False
    
    try:
        # Extract frames
        frames, frame_times = extract_frames_uniform(video_path, num_frames)
        
        if not frames:
            return {
                "error": "Could not extract frames from video",
                "status": "failed"
            }
        
        # Generate descriptions based on analysis type
        results = {
            "video_info": {
                "duration": frame_times[-1] if frame_times else 0,
                "frames_analyzed": len(frames),
                "frame_timestamps": frame_times
            },
            "model": MODEL_NAME,
            "status": "completed"
        }
        
        if analysis_type == "comprehensive":
            # Generate multiple analyses
            overall = generate_description(
                frames,
                "You are analyzing multiple frames from a video in chronological order. "
                "Describe what happens in this video from beginning to end. "
                "Focus on the main subjects, actions, movements, and how the scene progresses over time."
            )
            
            temporal = generate_description(
                frames,
                "These frames are from a video shown in chronological order. "
                "Describe the specific changes and movements between frames. "
                "What actions occur? How do things move or change throughout the video?"
            )
            
            scene = generate_description(
                frames[:3],
                "Describe the setting, environment, and context of this video. "
                "Where does it take place? What objects are visible? What's the mood or atmosphere?"
            )
            
            results["analysis"] = {
                "overall_description": overall,
                "temporal_changes": temporal,
                "scene_context": scene
            }
            
        elif analysis_type == "quick":
            # Single quick analysis
            description = generate_description(
                frames,
                "Analyze these video frames and provide a concise description of what happens in the video."
            )
            results["analysis"] = {
                "description": description
            }
            
        elif analysis_type == "action":
            # Focus on actions and movements
            description = generate_description(
                frames,
                "Focus on identifying and describing all actions, movements, and activities happening in this video."
            )
            results["analysis"] = {
                "action_description": description
            }
            
        else:
            # Custom prompt provided
            description = generate_description(frames, analysis_type)
            results["analysis"] = {
                "custom_description": description
            }
        
        return results
        
    finally:
        # Cleanup temporary file if needed
        if cleanup_needed and os.path.exists(video_path):
            os.unlink(video_path)


def handler(job):
    """RunPod serverless handler function."""
    try:
        job_input = job["input"]
        
        # Validate input
        if "video" not in job_input:
            return {
                "error": "No video provided. Please provide 'video' as URL, base64, or path",
                "status": "failed"
            }
        
        # Get parameters
        video_source = job_input["video"]
        num_frames = job_input.get("num_frames", 8)
        analysis_type = job_input.get("analysis_type", "comprehensive")
        
        # Process video
        result = process_video(video_source, num_frames, analysis_type)
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }


# RunPod serverless entrypoint
runpod.serverless.start({
    "handler": handler
})