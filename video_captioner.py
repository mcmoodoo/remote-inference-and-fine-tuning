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
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
import warnings
warnings.filterwarnings("ignore")

class VideoLLaVACaptioner:
    def __init__(self):
        """Initialize Video-LLaVA model for video understanding."""
        print("Loading Video-LLaVA model...")
        print("This may take a while on first run...")
        
        # Use the correct Video-LLaVA Next model
        model_name = "llava-hf/LLaVA-NeXT-Video-7B-hf"
        
        try:
            # Load processor and model
            self.processor = LlavaNextVideoProcessor.from_pretrained(model_name)
            self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
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
            print(f"Error with LLaVA-NeXT-Video: {e}")
            print("\nFalling back to alternative video understanding model...")
            self._load_alternative()
    
    def _load_alternative(self):
        """Alternative: Use standard LLaVA with video frames."""
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        # Use standard LLaVA-NeXT which can process multiple images
        model_name = "llava-hf/llava-v1.6-mistral-7b-hf"
        
        print(f"Loading {model_name}...")
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Alternative model loaded on {self.device}")
    
    def extract_video_frames(self, video_path, num_frames=8):
        """
        Extract frames uniformly from video for processing.
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
    
    def generate_video_description(self, frames, prompt="Describe this video in detail. What actions occur? What is the setting? Describe the progression of events from beginning to end."):
        """
        Generate comprehensive video description.
        """
        print("\nGenerating video description...")
        
        # Format prompt for multi-frame understanding
        full_prompt = f"USER: <video>\n{prompt}\nASSISTANT:"
        
        # Process inputs
        inputs = self.processor(
            text=full_prompt,
            videos=frames,  # Pass frames as video
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
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()
        
        return response
    
    def generate_frame_sequence_description(self, frames):
        """
        Alternative method: Describe frames as a sequence.
        """
        print("\nAnalyzing frame sequence...")
        
        descriptions = []
        
        # Analyze pairs of frames for temporal understanding
        for i in range(len(frames) - 1):
            prompt = f"USER: <image><image>Compare these two consecutive frames from a video. What changes or movements occur between them?\nASSISTANT:"
            
            inputs = self.processor(
                text=prompt,
                images=[frames[i], frames[i+1]],
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.processor.decode(output[0], skip_special_tokens=True)
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            descriptions.append(f"Frame {i+1} to {i+2}: {response}")
        
        return descriptions
    
    def process_video(self, video_path):
        """
        Main method to process video and generate comprehensive description.
        """
        print(f"\n{'='*50}")
        print(f"Processing: {video_path}")
        print('='*50)
        
        # Extract frames for analysis
        frames, frame_times = self.extract_video_frames(video_path, num_frames=8)
        
        if not frames:
            return "Error: Could not extract frames from video"
        
        try:
            # Try to process as video first
            print("\n1. Generating overall video description...")
            overall_description = self.generate_video_description(
                frames,
                "Analyze this video sequence. Describe the main subject, actions that take place over time, the setting, and how the scene progresses from beginning to end."
            )
            
            print("\n2. Analyzing actions and movements...")
            action_description = self.generate_video_description(
                frames,
                "Focus on the temporal aspects: What movements and actions occur throughout this video? Describe how things change over time."
            )
            
        except Exception as e:
            print(f"Video processing error: {e}")
            print("Falling back to frame-by-frame analysis...")
            
            # Fallback: Analyze individual frames
            overall_description = self._analyze_frames_individually(frames)
            
            # Analyze temporal changes
            print("\n2. Analyzing temporal changes...")
            action_description = "\n".join(self.generate_frame_sequence_description(frames))
        
        # Compile final description
        final_description = f"""
{'='*60}
VIDEO ANALYSIS REPORT
{'='*60}

FILE: {video_path.name}
DURATION: {frame_times[-1]:.1f} seconds
FRAMES ANALYZED: {len(frames)}

{'='*60}
VIDEO DESCRIPTION:
{'='*60}
{overall_description}

{'='*60}
TEMPORAL ANALYSIS:
{'='*60}
{action_description}

{'='*60}
FRAME TIMESTAMPS:
{'='*60}"""
        
        for i, time in enumerate(frame_times):
            final_description += f"\nFrame {i+1}: {time:.2f}s"
        
        return final_description
    
    def _analyze_frames_individually(self, frames):
        """Fallback: Analyze frames individually."""
        descriptions = []
        for i, frame in enumerate(frames):
            prompt = f"USER: <image>Describe what you see in this frame from a video.\nASSISTANT:"
            
            inputs = self.processor(
                text=prompt,
                images=frame,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True
                )
            
            response = self.processor.decode(output[0], skip_special_tokens=True)
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
            
            descriptions.append(f"Frame {i+1}: {response}")
        
        return "\n".join(descriptions)

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
        # Initialize captioner
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
        print("\n2. Ensure you have enough disk space for model download")
        print("3. GPU with 16GB+ VRAM recommended for best performance")
        sys.exit(1)

if __name__ == "__main__":
    main()