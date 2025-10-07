#!/usr/bin/env python3
"""
Run the evaluation function generator on AgentSynth dataset.

This script demonstrates how to use the LLM-powered evaluation function generator
to create verifiable evaluation functions for AgentSynth tasks.
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from generate_evaluation_functions import EvaluationFunctionGenerator

def main():
    """Run evaluation function generation on AgentSynth dataset."""
    
    # Paths
    dataset_path = "../insta_data/summarized_task_sequences.jsonl"
    output_path = "generated_evaluation_functions.py"
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the AgentSynth dataset is available.")
        return
    
    print("AgentSynth Evaluation Function Generator")
    print("=" * 50)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print()
    
    # Initialize generator
    generator = EvaluationFunctionGenerator(model_name='gpt-4o')
    
    # Process a small subset first (5 tasks) for demonstration
    print("Processing first 5 tasks for demonstration...")
    results = generator.process_agentsynth_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        max_tasks=5
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("GENERATION RESULTS")
    print("=" * 50)
    print(f"Processed tasks: {results['processed_tasks']}")
    print(f"Successful generations: {results['successful_generations']}")
    print(f"Failed generations: {results['failed_generations']}")
    
    if results['successful_generations'] > 0:
        print(f"\nSuccess rate: {results['successful_generations']/results['processed_tasks']:.2%}")
        print(f"\nGenerated evaluation functions saved to: {output_path}")
        
        # Show example of generated function
        if results['generated_functions']:
            first_task_id = list(results['generated_functions'].keys())[0]
            first_function = results['generated_functions'][first_task_id]
            
            print(f"\nExample generated function for task: {first_task_id}")
            print(f"Task description: {first_function['task_description'][:100]}...")
            print(f"Task type: {first_function['analysis_result'].get('task_type', 'unknown')}")
            print(f"Verification methods: {first_function['analysis_result'].get('verification_methods', [])}")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\nTo process more tasks, run:")
    print(f"python generate_evaluation_functions.py --dataset {dataset_path} --output {output_path} --max-tasks 50")

if __name__ == "__main__":
    main()
