#!/usr/bin/env python3
"""
Simple runner script for OSWorld evaluation

This script provides an easy interface to run evaluations with different configurations.
"""

import os
import sys
import subprocess
from pathlib import Path

def run_evaluation_with_config(model_type='auto', sample_tasks=True, verbose=True, custom_task_file=None, dataset_file=None, agentsynth_hf=False):
    """Run evaluation with specific configuration"""
    
    # Set environment variables based on model type
    if model_type == 'local':
        os.environ['USE_LOCAL_LLAVA'] = 'true'
        print("Using local LLaVa model")
    elif model_type == 'openai':
        os.environ['USE_LOCAL_LLAVA'] = 'false'
        print("Using OpenAI model")
    else:
        print("Using auto-detected model")
    
    # Build command
    cmd = [sys.executable, 'evaluate_osworld.py']
    
    if sample_tasks:
        cmd.append('--sample')
    elif custom_task_file:
        cmd.extend(['--tasks', custom_task_file])
    elif dataset_file:
        cmd.extend(['--dataset', dataset_file])
    elif agentsynth_hf:
        cmd.append('--agentsynth-hf')
    
    if verbose:
        cmd.append('--verbose')
    
    # Run evaluation
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        print("Evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"Error running evaluation: {e}")
        return False

def main():
    """Main function with interactive menu"""
    print("OSWorld Evaluation Runner")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not Path('evaluate_osworld.py').exists():
        print("Error: evaluate_osworld.py not found in current directory")
        print("Please run this script from the agentsynth directory")
        return
    
    # Model selection
    print("\nSelect model type:")
    print("1. Auto-detect (use environment settings)")
    print("2. Local LLaVa")
    print("3. OpenAI")
    
    choice = input("Enter choice (1-3): ").strip()
    
    model_type = 'auto'
    if choice == '2':
        model_type = 'local'
    elif choice == '3':
        model_type = 'openai'
    
    # Task selection
    print("\nSelect tasks:")
    print("1. Sample tasks (from config)")
    print("2. Custom task file")
    print("3. Dataset tasks (JSONL format)")
    print("4. AgentSynth Hugging Face dataset (official)")
    
    task_choice = input("Enter choice (1-4): ").strip()
    
    sample_tasks = task_choice == '1'
    custom_task_file = None
    dataset_file = None
    agentsynth_hf = task_choice == '4'
    
    if task_choice == '2':
        custom_task_file = input("Enter path to task JSON file: ").strip()
        if not Path(custom_task_file).exists():
            print(f"Error: Task file {custom_task_file} not found")
            return
    elif task_choice == '3':
        dataset_file = input("Enter path to dataset JSONL file: ").strip()
        if not Path(dataset_file).exists():
            print(f"Error: Dataset file {dataset_file} not found")
            return
    elif task_choice == '4':
        print("Using official AgentSynth dataset from Hugging Face")
        print("Note: This will use extended time limits (6x normal steps) for complex tasks")
    
    # Verbose logging
    verbose = input("Enable verbose logging? (y/n): ").strip().lower() == 'y'
    
    # Run evaluation
    print(f"\nStarting evaluation...")
    print(f"Model: {model_type}")
    if sample_tasks:
        print(f"Tasks: Sample tasks")
    elif custom_task_file:
        print(f"Tasks: {custom_task_file}")
    elif dataset_file:
        print(f"Tasks: Dataset {dataset_file}")
    elif agentsynth_hf:
        print(f"Tasks: AgentSynth Hugging Face dataset (official)")
    print(f"Verbose: {verbose}")
    print("-" * 30)
    
    success = run_evaluation_with_config(model_type, sample_tasks, verbose, custom_task_file, dataset_file, agentsynth_hf)
    
    if success:
        print("\n✓ Evaluation completed successfully!")
        print("Check the output directory for results.")
    else:
        print("\n✗ Evaluation failed!")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()
