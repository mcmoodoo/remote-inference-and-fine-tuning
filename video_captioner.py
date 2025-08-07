#!/usr/bin/env python3
"""
Video captioning script using Video-LLaVA for temporal understanding.
Usage: python video_captioner.py video.mkv
"""

import sys
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

class VideoLLaVACaptioner:
    def __init__(self, model_name="LanguageBind/Video-LLaVA-7B"):
        """Initialize Video-LLaVA model for video understanding."""
        print("Loading Video-LLaVA model...")
        print("This may take a while on first run (~15GB download)...")
        
        try:
            # Load processor and tokenizer
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                print("Warning: Running on CPU will be slow. GPU recommended.")
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying alternative loading method...")
            self._load_alternative()
    
    def _load_alternative(self):
        """Alternative loading method for Video-LLaVA."""
        from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
        
        model_name = "LanguageBind/Video-LLaVA-7B"
        self.processor = VideoLlavaProcessor.from_pretrained(model_name)
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def extract_video_frames(self, video_path, num_frames=8):
        """
        Extract frames uniformly from video for Video-LLaVA processing.
        Video-LLaVA typically works best with 8-16 frames.
        """
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
    
    def generate_video_description(self, frames, prompt="Describe this video in detail. What actions occur? What is the setting? Describe the progression of events."):
        """
        Generate comprehensive video description using Video-LLaVA.
        """
        print("\nGenerating video description...")
        
        # Prepare the conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image"} for _ in frames]
                ]
            }
        ]
        
        # Process inputs
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt_text,
            images=frames,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        # Decode response
        response = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()
        elif "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response
    
    def analyze_video_segments(self, video_path, segment_frames=8):
        """
        Analyze video in segments for longer videos.
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        cap.release()
        
        # For videos longer than 30 seconds, analyze in segments
        if duration > 30:
            print(f"\nLong video detected ({duration:.1f}s). Analyzing in segments...")
            segments = []
            segment_duration = 10  # 10-second segments
            num_segments = int(duration / segment_duration) + 1
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                
                print(f"\nProcessing segment {i+1}/{num_segments} ({start_time:.1f}s - {end_time:.1f}s)")
                
                # Extract frames from this segment
                segment_frames = self._extract_segment_frames(
                    video_path, start_time, end_time, segment_frames
                )
                
                if segment_frames:
                    # Generate description for this segment
                    segment_desc = self.generate_video_description(
                        segment_frames,
                        f"Describe what happens in this video segment. Focus on actions, changes, and key events."
                    )
                    segments.append(f"[{start_time:.1f}s - {end_time:.1f}s]: {segment_desc}")
            
            return segments
        else:
            return None
    
    def _extract_segment_frames(self, video_path, start_time, end_time, num_frames):
        """Extract frames from a specific time segment."""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        
        frames = []
        for target_frame in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames
    
    def process_video(self, video_path):
        """
        Main method to process video and generate comprehensive description.
        """
        print(f"\n{'='*50}")
        print(f"Processing: {video_path}")
        print('='*50)
        
        # Extract frames for overall understanding
        frames, frame_times = self.extract_video_frames(video_path, num_frames=8)
        
        if not frames:
            return "Error: Could not extract frames from video"
        
        # Generate overall description
        print("\n1. Generating overall video description...")
        overall_description = self.generate_video_description(
            frames,
            "Provide a detailed description of this entire video. What is the main subject? What actions take place? Describe the setting, mood, and any important details you observe throughout the video."
        )
        
        # Generate action-focused description
        print("\n2. Analyzing actions and movements...")
        action_description = self.generate_video_description(
            frames,
            "Focus on the actions and movements in this video. What is happening? Describe any motion, gestures, or activities taking place."
        )
        
        # Generate scene analysis
        print("\n3. Analyzing scene and environment...")
        scene_description = self.generate_video_description(
            frames,
            "Describe the setting and environment of this video. Where does it take place? What objects are visible? What is the lighting and atmosphere like?"
        )
        
        # Check if we need segment analysis for long videos
        segment_descriptions = self.analyze_video_segments(video_path)
        
        # Compile final description
        final_description = f"""
{'='*60}
VIDEO ANALYSIS REPORT
{'='*60}

FILE: {video_path.name}
DURATION: {frame_times[-1]:.1f} seconds
FRAMES ANALYZED: {len(frames)}

{'='*60}
OVERALL DESCRIPTION:
{'='*60}
{overall_description}

{'='*60}
ACTIONS & MOVEMENTS:
{'='*60}
{action_description}

{'='*60}
SCENE & ENVIRONMENT:
{'='*60}
{scene_description}
"""
        
        if segment_descriptions:
            final_description += f"""
{'='*60}
DETAILED TIMELINE:
{'='*60}
"""
            for segment in segment_descriptions:
                final_description += f"\n{segment}\n"
        
        return final_description

def main():
    if len(sys.argv) < 2:
        print("Usage: python video_captioner.py <video_file>")
        print("\nExample: python video_captioner.py movie.mkv")
        print("\nSupported formats: MKV, MP4, AVI, MOV, etc.")
        sys.exit(1)
    
    video_path = Path(sys.argv[1])
    
    if not video_path.exists():
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)
    
    try:
        # Initialize Video-LLaVA captioner
        captioner = VideoLLaVACaptioner()
        
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
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Install required packages:")
        print("   pip install transformers torch accelerate pillow opencv-python")
        print("\n2. For Video-LLaVA specifically:")
        print("   pip install git+https://github.com/huggingface/transformers.git")
        print("\n3. Ensure you have enough disk space (~15GB for model)")
        print("4. GPU with 16GB+ VRAM recommended")
        sys.exit(1)

if __name__ == "__main__":
    main()