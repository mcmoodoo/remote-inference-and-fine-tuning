#!/usr/bin/env python3
"""
Video captioning using VLLM for fast inference with vision-language models.
Usage: python video_captioner_vllm.py video.mkv
"""

import sys
import cv2
import base64
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO
import json

class VLLMVideoCaptioner:
    def __init__(self, server_url="http://localhost:8000", model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
        """Initialize VLLM client for video understanding."""
        self.server_url = server_url
        self.model_name = model_name
        
        print("Connecting to VLLM server...")
        print(f"Server: {server_url}")
        print(f"Model: {model_name}")
        
        # Test connection
        try:
            response = requests.get(f"{server_url}/v1/models")
            if response.status_code == 200:
                print("✓ Connected to VLLM server")
            else:
                print(f"Warning: Server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("\n⚠️  VLLM server not running!")
            print("\nTo start VLLM server, run:")
            print(f"  vllm serve {model_name} --max-model-len 4096")
            print("\nOr with GPU memory optimization:")
            print(f"  vllm serve {model_name} --max-model-len 4096 --gpu-memory-utilization 0.9")
            sys.exit(1)
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def extract_video_frames(self, video_path, num_frames=8):
        """Extract frames uniformly from video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\nVideo info:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Extracting {num_frames} frames for analysis")
        
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
    
    def generate_description_vllm(self, frames, prompt):
        """Generate description using VLLM's OpenAI-compatible API."""
        
        # Convert frames to base64
        image_urls = []
        for frame in frames:
            b64_image = self.image_to_base64(frame)
            image_urls.append(f"data:image/png;base64,{b64_image}")
        
        # Build message content with multiple images
        content = [{"type": "text", "text": prompt}]
        for img_url in image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url}
            })
        
        # Create request
        request_data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }
        
        # Send request to VLLM
        try:
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=request_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error: Server returned {response.status_code}: {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_video(self, video_path):
        """Process video and generate comprehensive description."""
        print(f"\n{'='*50}")
        print(f"Processing: {video_path}")
        print('='*50)
        
        # Extract frames
        frames, frame_times = self.extract_video_frames(video_path, num_frames=8)
        
        if not frames:
            return "Error: Could not extract frames from video"
        
        # Generate overall description
        print("\n1. Generating overall video description...")
        overall_description = self.generate_description_vllm(
            frames,
            "You are analyzing multiple frames from a video in chronological order. "
            "Describe what happens in this video from beginning to end. "
            "Focus on the main subjects, actions, movements, and how the scene progresses over time."
        )
        
        # Generate detailed temporal analysis
        print("\n2. Analyzing temporal changes...")
        temporal_description = self.generate_description_vllm(
            frames,
            "These frames are from a video shown in chronological order. "
            "Describe the specific changes and movements between frames. "
            "What actions occur? How do things move or change throughout the video?"
        )
        
        # Generate scene analysis
        print("\n3. Analyzing environment and context...")
        scene_description = self.generate_description_vllm(
            frames[:3],  # Use fewer frames for scene analysis
            "Describe the setting, environment, and context of this video. "
            "Where does it take place? What objects are visible? What's the mood or atmosphere?"
        )
        
        # Compile final description
        final_description = f"""
{'='*60}
VIDEO ANALYSIS REPORT (VLLM)
{'='*60}

FILE: {video_path.name}
DURATION: {frame_times[-1]:.1f} seconds
FRAMES ANALYZED: {len(frames)}
MODEL: {self.model_name}

{'='*60}
OVERALL VIDEO DESCRIPTION:
{'='*60}
{overall_description}

{'='*60}
TEMPORAL CHANGES & MOVEMENTS:
{'='*60}
{temporal_description}

{'='*60}
SCENE & ENVIRONMENT:
{'='*60}
{scene_description}

{'='*60}
FRAME TIMESTAMPS:
{'='*60}"""
        
        for i, time in enumerate(frame_times):
            final_description += f"\nFrame {i+1}: {time:.2f}s"
        
        return final_description


class VLLMOfflineVideoCaptioner:
    """Alternative: Use VLLM offline mode (no server required)."""
    
    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf"):
        """Initialize VLLM in offline mode."""
        print("Loading VLLM in offline mode...")
        print(f"Model: {model_name}")
        
        try:
            from vllm import LLM, SamplingParams
            from vllm.multimodal.image import ImagePixelData
            
            # Initialize VLLM
            self.llm = LLM(
                model=model_name,
                max_model_len=4096,
                gpu_memory_utilization=0.9,
                trust_remote_code=True
            )
            
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=512
            )
            
            print("✓ VLLM loaded successfully")
            
        except ImportError:
            print("\n⚠️  VLLM not installed!")
            print("\nInstall with:")
            print("  pip install vllm")
            print("\nFor specific CUDA versions:")
            print("  pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118")
            sys.exit(1)
    
    def extract_video_frames(self, video_path, num_frames=8):
        """Extract frames uniformly from video."""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\nVideo info:")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Extracting {num_frames} frames")
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        frame_times = []
        
        for target_frame in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
                frame_times.append(target_frame / fps)
        
        cap.release()
        return frames, frame_times
    
    def generate_description(self, frames, prompt):
        """Generate description using VLLM offline mode."""
        
        # Format prompt with image placeholders
        image_prompt = ""
        for i in range(len(frames)):
            image_prompt += f"<image_{i}>"
        
        full_prompt = f"{image_prompt}\n{prompt}"
        
        # Convert PIL images to format VLLM expects
        image_data = []
        for frame in frames:
            # Convert to numpy array
            img_array = np.array(frame)
            image_data.append(ImagePixelData(img_array))
        
        # Generate with VLLM
        outputs = self.llm.generate(
            prompts=[full_prompt],
            multi_modal_data={"image": image_data},
            sampling_params=self.sampling_params
        )
        
        return outputs[0].outputs[0].text
    
    def process_video(self, video_path):
        """Process video with offline VLLM."""
        print(f"\n{'='*50}")
        print(f"Processing: {video_path}")
        print('='*50)
        
        frames, frame_times = self.extract_video_frames(video_path)
        
        if not frames:
            return "Error: Could not extract frames"
        
        print("\nGenerating video analysis...")
        description = self.generate_description(
            frames,
            "Analyze these video frames in order and describe what happens in the video. "
            "Include details about actions, movements, and how the scene changes over time."
        )
        
        return f"""
{'='*60}
VIDEO ANALYSIS (VLLM Offline)
{'='*60}

FILE: {video_path.name}
DURATION: {frame_times[-1]:.1f}s
FRAMES: {len(frames)}

{'='*60}
DESCRIPTION:
{'='*60}
{description}
"""


def main():
    if len(sys.argv) < 2:
        print("Usage: python video_captioner_vllm.py <video_file> [--offline]")
        print("\nExamples:")
        print("  python video_captioner_vllm.py movie.mkv")
        print("  python video_captioner_vllm.py movie.mkv --offline")
        print("\nStart VLLM server with:")
        print("  vllm serve llava-hf/llava-v1.6-mistral-7b-hf --max-model-len 4096")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)
    
    try:
        # Check if offline mode requested
        if "--offline" in sys.argv:
            print("Using VLLM offline mode (no server required)")
            captioner = VLLMOfflineVideoCaptioner()
        else:
            # Use VLLM server mode
            captioner = VLLMVideoCaptioner()
        
        # Process video
        description = captioner.process_video(video_path)
        
        # Print results
        print(description)
        
        # Save to file
        output_file = video_path.with_suffix('.txt')
        with open(output_file, 'w') as f:
            f.write(description)
        print(f"\n{'='*60}")
        print(f"Analysis saved to: {output_file}")
        print('='*60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()