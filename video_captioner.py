#!/usr/bin/env python3
"""
Video captioning script that reads MKV videos and generates vivid descriptions.
Usage: python video_captioner.py video.mkv
"""

import sys
import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np
from pathlib import Path

class VideoCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-large"):
        """Initialize the video captioning model."""
        print("Loading BLIP captioning model...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model loaded on {self.device}")
    
    def extract_frames(self, video_path, frame_interval=30):
        """Extract frames from video at specified intervals."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        
        frames = []
        frame_times = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_times.append(frame_count / fps)
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames for analysis")
        return frames, frame_times
    
    def caption_frame(self, frame, prompt=None):
        """Generate a caption for a single frame."""
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        if prompt:
            # Conditional captioning with prompt
            inputs = self.processor(image, prompt, return_tensors="pt")
        else:
            # Unconditional captioning
            inputs = self.processor(image, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=0.8
            )
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def generate_vivid_description(self, frame):
        """Generate multiple detailed captions for a frame to create a vivid description."""
        prompts = [
            None,  # General caption
            "a photo of",  # What is in the image
            "the scene shows",  # Scene description
            "the mood is",  # Emotional/atmospheric description
        ]
        
        descriptions = []
        for prompt in prompts:
            caption = self.caption_frame(frame, prompt)
            if caption and not caption.startswith("the mood is") and not caption.startswith("the scene shows"):
                descriptions.append(caption)
        
        return descriptions
    
    def process_video(self, video_path, frame_interval=30):
        """Process entire video and generate comprehensive description."""
        print(f"\nProcessing video: {video_path}")
        
        # Extract frames
        frames, frame_times = self.extract_frames(video_path, frame_interval)
        
        if not frames:
            return "Could not extract frames from video"
        
        print("\nGenerating captions for extracted frames...")
        video_description = []
        
        for i, (frame, time) in enumerate(zip(frames, frame_times)):
            print(f"Processing frame {i+1}/{len(frames)} at {time:.2f}s...")
            
            # Generate vivid descriptions for this frame
            descriptions = self.generate_vivid_description(frame)
            
            # Format the description with timestamp
            frame_desc = f"\n[{time:.1f}s] "
            if descriptions:
                # Combine descriptions into a coherent narrative
                main_desc = descriptions[0]
                if len(descriptions) > 1:
                    # Add additional details
                    details = [d for d in descriptions[1:] if d not in main_desc]
                    if details:
                        main_desc += ". " + ". ".join(details)
                frame_desc += main_desc.capitalize()
            
            video_description.append(frame_desc)
        
        # Create overall video summary
        summary = "\n=== VIDEO DESCRIPTION ===\n"
        summary += f"Duration: {frame_times[-1]:.1f} seconds\n"
        summary += f"Analyzed {len(frames)} key frames\n"
        summary += "\n=== SCENE BY SCENE DESCRIPTION ===" 
        summary += "".join(video_description)
        
        return summary

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_captioner.py video.mkv")
        print("\nExample: python video_captioner.py movie.mkv")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)
    
    if not video_path.suffix.lower() in ['.mkv', '.mp4', '.avi', '.mov']:
        print(f"Warning: File extension '{video_path.suffix}' might not be a video file")
    
    try:
        # Initialize captioner
        captioner = VideoCaptioner()
        
        # Process video
        description = captioner.process_video(video_path)
        
        # Print results
        print("\n" + "="*50)
        print(description)
        print("="*50)
        
        # Optionally save to file
        output_file = video_path.with_suffix('.txt')
        with open(output_file, 'w') as f:
            f.write(description)
        print(f"\nDescription saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have installed the required packages:")
        print("  pip install transformers torch pillow opencv-python")
        print("\nNote: First run will download the BLIP model (~1.8GB)")
        sys.exit(1)

if __name__ == "__main__":
    main()