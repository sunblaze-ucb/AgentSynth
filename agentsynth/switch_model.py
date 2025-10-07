#!/usr/bin/env python3
"""
Script to help switch between local LLaVa and OpenAI models
"""

import os
import sys

def set_local_llava():
    """Set environment to use local LLaVa"""
    os.environ['USE_LOCAL_LLAVA'] = 'true'
    print("Switched to local LLaVa model")
    print("Make sure you have the required dependencies installed:")
    print("pip install torch transformers")

def set_openai():
    """Set environment to use OpenAI"""
    os.environ['USE_LOCAL_LLAVA'] = 'false'
    print("Switched to OpenAI model")
    print("Make sure you have OPENAI_API_KEY set in your environment")

def print_current_config():
    """Print current configuration"""
    use_local = os.getenv('USE_LOCAL_LLAVA', 'false').lower() == 'true'
    model_path = os.getenv('LOCAL_LLAVA_MODEL_PATH', 'llava-hf/llava-1.5-7b-hf')
    device = os.getenv('LOCAL_LLAVA_DEVICE', 'auto')
    api_key = os.getenv('OPENAI_API_KEY')
    
    print("Current Configuration:")
    print(f"  Use Local LLaVa: {use_local}")
    if use_local:
        print(f"  Model Path: {model_path}")
        print(f"  Device: {device}")
    else:
        print(f"  OpenAI API Key: {'Set' if api_key else 'Not Set'}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python switch_model.py [local|openai|status]")
        print("  local   - Switch to local LLaVa model")
        print("  openai  - Switch to OpenAI model")
        print("  status  - Show current configuration")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'local':
        set_local_llava()
    elif command == 'openai':
        set_openai()
    elif command == 'status':
        print_current_config()
    else:
        print(f"Unknown command: {command}")
        print("Use 'local', 'openai', or 'status'")

if __name__ == "__main__":
    main()
