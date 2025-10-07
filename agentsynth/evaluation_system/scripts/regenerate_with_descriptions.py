#!/usr/bin/env python3
"""
Regenerate evaluation functions with proper task descriptions.

This script fixes the issue where task descriptions show as "..." instead of
the actual task content.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
# Add core evaluation modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

from generate_evaluation_functions import EvaluationFunctionGenerator

def regenerate_evaluation_functions():
    """Regenerate evaluation functions with proper task descriptions."""
    
    print("Regenerating evaluation functions with proper task descriptions...")
    print("=" * 70)
    
    # Initialize the generator
    generator = EvaluationFunctionGenerator(model_name='gpt-4o')
    
    # Find the first non-empty JSONL file
    dataset_dir = Path("../../oai_data_files")
    jsonl_files = list(dataset_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in dataset directory")
        return
    
    # Use the first non-empty file
    test_file = None
    for file_path in jsonl_files:
        if file_path.stat().st_size > 0:
            test_file = file_path
            break
    
    if not test_file:
        print("No non-empty JSONL files found")
        return
    
    print(f"Using dataset file: {test_file.name}")
    print(f"File size: {test_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Create output directory
    output_dir = Path("../generated")
    output_dir.mkdir(exist_ok=True)
    
    # Process just 3 tasks to show the fix
    print("\nProcessing 3 tasks to demonstrate proper task descriptions...")
    try:
        results = generator.process_agentsynth_dataset(
            dataset_path=str(test_file),
            output_path=str(output_dir / "regenerated_evaluation_functions.py"),
            max_tasks=3
        )
        
        print("\nRegeneration Results:")
        print("-" * 40)
        print(f"Processed tasks: {results['processed_tasks']}")
        print(f"Successful generations: {results['successful_generations']}")
        print(f"Failed generations: {results['failed_generations']}")
        
        if results['processed_tasks'] > 0:
            success_rate = results['successful_generations'] / results['processed_tasks']
            print(f"Success rate: {success_rate:.2%}")
        
        print(f"\nRegenerated evaluation functions saved to: {output_dir}")
        
        # Show a sample of the regenerated code
        generated_file = output_dir / "regenerated_evaluation_functions.py"
        if generated_file.exists():
            print(f"\nSample of regenerated evaluation functions:")
            print("-" * 60)
            with open(generated_file, 'r') as f:
                content = f.read()
                # Show first 30 lines to see the task descriptions
                lines = content.split('\n')[:30]
                for line in lines:
                    print(line)
                if len(content.split('\n')) > 30:
                    print("... (truncated)")
        
        return results
        
    except Exception as e:
        print(f"Error during regeneration: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = regenerate_evaluation_functions()
    
    if results:
        print(f"\n✓ Regeneration completed successfully!")
        print(f"✓ Generated {results['successful_generations']} evaluation functions with proper task descriptions")
        print(f"\nThe regenerated functions now show actual task descriptions instead of '...'")
    else:
        print(f"\n✗ Regeneration failed - check the error messages above")
