#!/usr/bin/env python3
"""
RunPod Serverless Handler for Video Captioning with LLaVA
"""

import runpod
import cv2
import base64
import numpy as np
from PIL import Image
import tempfile
import os
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# Initialize model globally for reuse across requests
print("Loading LLaVA model...")
MODEL_NAME = os.environ.get("MODEL_NAME", "llava-hf/llava-v1.6-mistral-7b-hf")

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = LlavaNextProcessor.from_pretrained(MODEL_NAME)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

print(f"Model {MODEL_NAME} loaded successfully on {device}")


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


def generate_caption(frames: List[Image.Image], prompt: str) -> str:
    """Generate caption using LLaVA model."""
    # Prepare conversation prompt
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ] + [{"type": "image"} for _ in frames]
        },
    ]
    
    # Format prompt
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Process inputs
    inputs = processor(
        text=text,
        images=frames,
        return_tensors="pt"
    ).to(device)
    
    # Generate caption
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
    
    # Decode output
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "assistant" in generated_text:
        response = generated_text.split("assistant")[-1].strip()
    else:
        response = generated_text.split(prompt)[-1].strip()
    
    return response


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
            "device": device,
            "status": "completed"
        }
        
        if analysis_type == "comprehensive":
            # Generate multiple analyses
            overall = generate_caption(
                frames,
                "These are frames from a video shown in chronological order. "
                "Describe what happens in this video from beginning to end. "
                "Focus on the main subjects, actions, and how the scene progresses."
            )
            
            temporal = generate_caption(
                frames,
                "Analyze these video frames and describe the specific changes and movements between frames. "
                "What actions occur? How do things move or change throughout?"
            )
            
            scene = generate_caption(
                frames[:3],
                "Describe the setting, environment, and context of this video. "
                "Where does it take place? What's the mood or atmosphere?"
            )
            
            results["analysis"] = {
                "overall_description": overall,
                "temporal_changes": temporal,
                "scene_context": scene
            }
            
        elif analysis_type == "quick":
            description = generate_caption(
                frames,
                "Provide a concise description of what happens in this video."
            )
            results["analysis"] = {
                "description": description
            }
            
        elif analysis_type == "action":
            description = generate_caption(
                frames,
                "Focus on identifying all actions, movements, and activities in this video."
            )
            results["analysis"] = {
                "action_description": description
            }
            
        else:
            # Custom prompt
            description = generate_caption(frames, analysis_type)
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