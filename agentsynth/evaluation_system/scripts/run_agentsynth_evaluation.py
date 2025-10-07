#!/usr/bin/env python3
"""
Runner script for generating evaluation functions from AgentSynth dataset.

This script processes the large AgentSynth dataset files and generates
verifiable evaluation functions using LLM analysis.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datetime import datetime, timezone

# Add current directory to path
sys.path.append(str(Path(__file__).parent))
# Add core evaluation modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

from generate_evaluation_functions import EvaluationFunctionGenerator
from integrated_verifiable_evaluation import IntegratedVerifiableEvaluationSystem

def process_agentsynth_file(
    file_path: str, 
    output_dir: str, 
    max_tasks: int = 10,
    model_name: str = 'gpt-4o'
) -> Dict[str, Any]:
    """
    Process a single AgentSynth dataset file and generate evaluation functions.
    
    Args:
        file_path: Path to the AgentSynth JSONL file
        output_dir: Directory to save generated evaluation functions
        max_tasks: Maximum number of tasks to process
        model_name: LLM model to use for analysis
        
    Returns:
        Processing results
    """
    
    print(f"Processing AgentSynth file: {file_path}")
    print(f"Output directory: {output_dir}")
    print(f"Max tasks: {max_tasks}")
    print(f"Model: {model_name}")
    print("-" * 60)
    
    # Initialize the generator
    generator = EvaluationFunctionGenerator(model_name=model_name)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    time_label = datetime.now(timezone.utc).isoformat()
    results = generator.process_agentsynth_dataset(
        dataset_path=file_path,
        output_path=str(output_path / f"generated_evaluation_functions_{time_label}.py"),
        max_tasks=max_tasks
    )
    
    return results

def run_comprehensive_evaluation(
    dataset_dir: str,
    output_dir: str,
    max_tasks_per_file: int = 5,
    model_name: str = 'gpt-4o'
) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on multiple AgentSynth dataset files.
    
    Args:
        dataset_dir: Directory containing AgentSynth dataset files
        output_dir: Directory to save results
        max_tasks_per_file: Maximum tasks to process per file
        model_name: LLM model to use
        
    Returns:
        Comprehensive results
    """
    
    print("AgentSynth Comprehensive Evaluation")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max tasks per file: {max_tasks_per_file}")
    print(f"Model: {model_name}")
    print()
    
    # Initialize the integrated system
    system = IntegratedVerifiableEvaluationSystem(model_name=model_name)
    
    # Find all JSONL files in the dataset directory
    dataset_path = Path(dataset_dir)
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {dataset_dir}")
        return {'error': 'No dataset files found'}
    
    print(f"Found {len(jsonl_files)} dataset files")
    
    all_results = {
        'files_processed': 0,
        'total_tasks': 0,
        'successful_generations': 0,
        'failed_generations': 0,
        'file_results': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Process each file
    for i, file_path in enumerate(jsonl_files):
        print(f"\nProcessing file {i+1}/{len(jsonl_files)}: {file_path.name}")
        
        try:
            # Process this file
            file_results = process_agentsynth_file(
                str(file_path),
                output_dir,
                max_tasks_per_file,
                model_name
            )
            
            all_results['file_results'][file_path.name] = file_results
            all_results['files_processed'] += 1
            all_results['total_tasks'] += file_results.get('processed_tasks', 0)
            all_results['successful_generations'] += file_results.get('successful_generations', 0)
            all_results['failed_generations'] += file_results.get('failed_generations', 0)
            
            print(f"  ✓ Processed {file_results.get('processed_tasks', 0)} tasks")
            print(f"  ✓ Generated {file_results.get('successful_generations', 0)} evaluation functions")
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
            all_results['file_results'][file_path.name] = {'error': str(e)}
    
    # Calculate overall success rate
    if all_results['total_tasks'] > 0:
        all_results['overall_success_rate'] = all_results['successful_generations'] / all_results['total_tasks']
    else:
        all_results['overall_success_rate'] = 0.0
    
    # Save comprehensive results
    results_file = Path(output_dir) / f"comprehensive_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComprehensive results saved to: {results_file}")
    
    return all_results

def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(description='Generate evaluation functions from AgentSynth dataset')
    parser.add_argument('--dataset-dir', type=str, 
                       default='../../oai_data_files',
                       help='Directory containing AgentSynth dataset files')
    parser.add_argument('--output-dir', type=str, 
                       default='../generated',
                       help='Directory to save generated evaluation functions')
    parser.add_argument('--max-tasks', type=int, default=5,
                       help='Maximum number of tasks to process per file')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='LLM model to use for analysis')
    parser.add_argument('--single-file', type=str,
                       help='Process only a single file instead of all files')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive evaluation with verifiable methods')
    
    args = parser.parse_args()
    
    # Check if dataset directory exists
    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.comprehensive:
        # Run comprehensive evaluation
        results = run_comprehensive_evaluation(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            max_tasks_per_file=args.max_tasks,
            model_name=args.model
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("COMPREHENSIVE EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Files processed: {results['files_processed']}")
        print(f"Total tasks: {results['total_tasks']}")
        print(f"Successful generations: {results['successful_generations']}")
        print(f"Failed generations: {results['failed_generations']}")
        print(f"Overall success rate: {results['overall_success_rate']:.2%}")
        
    elif args.single_file:
        # Process single file
        if not os.path.exists(args.single_file):
            print(f"Error: File not found: {args.single_file}")
            return
        
        results = process_agentsynth_file(
            file_path=args.single_file,
            output_dir=args.output_dir,
            max_tasks=args.max_tasks,
            model_name=args.model
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("SINGLE FILE PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Processed tasks: {results['processed_tasks']}")
        print(f"Successful generations: {results['successful_generations']}")
        print(f"Failed generations: {results['failed_generations']}")
        if results['processed_tasks'] > 0:
            print(f"Success rate: {results['successful_generations']/results['processed_tasks']:.2%}")
    
    else:
        # Process all files in directory
        dataset_path = Path(args.dataset_dir)
        jsonl_files = list(dataset_path.glob("*.jsonl"))
        
        if not jsonl_files:
            print(f"No JSONL files found in {args.dataset_dir}")
            return
        
        print(f"Found {len(jsonl_files)} files to process")
        
        total_processed = 0
        total_successful = 0
        total_failed = 0
        
        for i, file_path in enumerate(jsonl_files):
            print(f"\nProcessing file {i+1}/{len(jsonl_files)}: {file_path.name}")
            
            try:
                results = process_agentsynth_file(
                    str(file_path),
                    args.output_dir,
                    args.max_tasks,
                    args.model
                )
                
                total_processed += results.get('processed_tasks', 0)
                total_successful += results.get('successful_generations', 0)
                total_failed += results.get('failed_generations', 0)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        
        # Print overall summary
        print("\n" + "=" * 60)
        print("OVERALL PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Files processed: {len(jsonl_files)}")
        print(f"Total tasks: {total_processed}")
        print(f"Successful generations: {total_successful}")
        print(f"Failed generations: {total_failed}")
        if total_processed > 0:
            print(f"Overall success rate: {total_successful/total_processed:.2%}")
    
    print(f"\nGenerated evaluation functions saved to: {args.output_dir}")
    print("\nTo use the generated functions:")
    print("1. Import the generated_evaluation_functions.py file")
    print("2. Use the evaluate_task() function with task data and agent trajectory")
    print("3. The functions provide verifiable evaluation without LLM judges")

if __name__ == "__main__":
    main()
