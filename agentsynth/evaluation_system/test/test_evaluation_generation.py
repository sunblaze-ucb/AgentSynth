#!/usr/bin/env python3
"""
Test script to demonstrate evaluation function generation on a small sample.

This script processes just a few tasks from the AgentSynth dataset to show
how the evaluation function generation works.
"""

import os
import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
# Add core evaluation modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

from generate_evaluation_functions import EvaluationFunctionGenerator

def test_small_sample():
    """Test evaluation function generation on a small sample."""
    
    print("Testing AgentSynth Evaluation Function Generation")
    print("=" * 60)
    
    # Check if we have the dataset files
    dataset_dir = Path("../../oai_data_files")
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return
    
    # Find the first non-empty JSONL file
    jsonl_files = list(dataset_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("No JSONL files found in dataset directory")
        return
    
    # Use the first file (skip the empty one)
    test_file = None
    for file_path in jsonl_files:
        if file_path.stat().st_size > 0:
            test_file = file_path
            break
    
    if not test_file:
        print("No non-empty JSONL files found")
        return
    
    print(f"Using test file: {test_file.name}")
    print(f"File size: {test_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Create output directory
    output_dir = Path("../test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    print("\nInitializing evaluation function generator...")
    generator = EvaluationFunctionGenerator(model_name='gpt-4o')
    
    # Process just 2 tasks for testing
    print("Processing 2 tasks for testing...")
    try:
        results = generator.process_agentsynth_dataset(
            dataset_path=str(test_file),
            output_path=str(output_dir / "test_evaluation_functions.py"),
            max_tasks=2
        )
        
        print("\nTest Results:")
        print("-" * 30)
        print(f"Processed tasks: {results['processed_tasks']}")
        print(f"Successful generations: {results['successful_generations']}")
        print(f"Failed generations: {results['failed_generations']}")
        
        if results['processed_tasks'] > 0:
            success_rate = results['successful_generations'] / results['processed_tasks']
            print(f"Success rate: {success_rate:.2%}")
        
        print(f"\nGenerated evaluation functions saved to: {output_dir}")
        
        # Show a sample of the generated code
        generated_file = output_dir / "test_evaluation_functions.py"
        if generated_file.exists():
            print(f"\nSample of generated evaluation functions:")
            print("-" * 50)
            with open(generated_file, 'r') as f:
                content = f.read()
                # Show first 20 lines
                lines = content.split('\n')[:20]
                for line in lines:
                    print(line)
                if len(content.split('\n')) > 20:
                    print("... (truncated)")
        
        return results
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_usage_examples():
    """Show examples of how to use the generated evaluation functions."""
    
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
1. Run the test script:
   python test_evaluation_generation.py

2. Process a single file with more tasks:
   python run_agentsynth_evaluation.py --single-file /path/to/file.jsonl --max-tasks 10

3. Process all files in the dataset directory:
   python run_agentsynth_evaluation.py --max-tasks 5

4. Run comprehensive evaluation with verifiable methods:
   python run_agentsynth_evaluation.py --comprehensive --max-tasks 3

5. Use a different model:
   python run_agentsynth_evaluation.py --model gpt-4o-mini --max-tasks 5

The generated evaluation functions will be saved in the output directory and can be
imported and used to evaluate agent performance without relying on LLM judges.
""")

if __name__ == "__main__":
    # Run the test
    results = test_small_sample()
    
    # Show usage examples
    show_usage_examples()
    
    if results:
        print(f"\n✓ Test completed successfully!")
        print(f"✓ Generated {results['successful_generations']} evaluation functions")
    else:
        print(f"\n✗ Test failed - check the error messages above")
