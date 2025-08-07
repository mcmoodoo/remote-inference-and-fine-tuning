#!/usr/bin/env python3
"""
Simple script to run Llama-based models with a given prompt using Transformers.
Usage: python llama_chat.py "Your prompt here"
"""

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_llama(prompt, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Run Llama-based model with the given prompt."""
    print("Loading model and tokenizer...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format the prompt as a chat message (TinyLlama uses ChatML format)
    chat_format = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    
    # Tokenize input
    inputs = tokenizer(chat_format, return_tensors="pt", padding=True)
    
    # Move to same device as model
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    print("Generating response...")
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and extract only the new tokens (response)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    
    return response.strip()

def main():
    # Check if prompt is provided
    if len(sys.argv) < 2:
        print("Usage: python llama_chat.py \"Your prompt here\"")
        print("\nExample: python llama_chat.py \"What is the capital of France?\"")
        sys.exit(1)
    
    # Get prompt from command line arguments
    prompt = ' '.join(sys.argv[1:])
    
    try:
        # Run model and print response
        response = run_llama(prompt)
        print("\nResponse:")
        print("-" * 50)
        print(response)
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nMake sure you have installed the required packages:")
        print("  pip install transformers torch accelerate")
        print("\nNote: First run will download the model (~550MB)")
        sys.exit(1)

if __name__ == "__main__":
    main()