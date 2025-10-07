# Configuration file for AgentSynth
# Set environment variables or modify this file to configure model usage

import os

# Model Configuration
# Set USE_LOCAL_LLAVA=true to use local LLaVa model, false for OpenAI
USE_LOCAL_LLAVA = os.getenv('USE_LOCAL_LLAVA', 'false').lower() == 'true'

# Local LLaVa Configuration
LOCAL_LLAVA_MODEL_PATH = os.getenv('LOCAL_LLAVA_MODEL_PATH', 'llava-hf/llava-1.5-7b-hf')
LOCAL_LLAVA_CHECKPOINT_PATH = os.getenv('LOCAL_LLAVA_CHECKPOINT_PATH', None)  # Path to LoRA checkpoint
LOCAL_LLAVA_DEVICE = os.getenv('LOCAL_LLAVA_DEVICE', 'auto')  # 'auto', 'cuda', 'cpu'

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Example configurations:

# For local LLaVa usage:
# export USE_LOCAL_LLAVA=true
# export LOCAL_LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf  # Base model from HF
# export LOCAL_LLAVA_CHECKPOINT_PATH=/path/to/checkpoint-13  # Optional: for LoRA checkpoints
# export LOCAL_LLAVA_DEVICE=cuda  # or 'cpu' if no GPU

# For OpenAI usage:
# export USE_LOCAL_LLAVA=false
# export OPENAI_API_KEY=your_api_key_here

def print_config():
    """Print current configuration"""
    print("=" * 50)
    print("AgentSynth Configuration:")
    print(f"Use Local LLaVa: {USE_LOCAL_LLAVA}")
    print(f"Local LLaVa Model Path: {LOCAL_LLAVA_MODEL_PATH}")
    print(f"Local LLaVa Checkpoint Path: {LOCAL_LLAVA_CHECKPOINT_PATH or 'None'}")
    print(f"Local LLaVa Device: {LOCAL_LLAVA_DEVICE}")
    print(f"OpenAI API Key Set: {'Yes' if OPENAI_API_KEY else 'No'}")
    print("=" * 50)

if __name__ == "__main__":
    print_config()
