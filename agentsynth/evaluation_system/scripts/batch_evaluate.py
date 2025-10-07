#!/usr/bin/env python3
"""
Batch evaluation script for comparing different models on OSWorld tasks

This script runs evaluations with different model configurations and compares results.
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
import subprocess

def run_single_evaluation(model_config, output_suffix=""):
    """Run a single evaluation with specific model configuration"""
    
    # Set environment variables
    for key, value in model_config.items():
        os.environ[key] = str(value)
    
    # Create output directory with timestamp and suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"batch_results_{timestamp}{output_suffix}"
    
    # Build command
    cmd = [
        sys.executable, 'evaluate_osworld.py',
        '--sample',
        '--output', output_dir,
        '--verbose'
    ]
    
    print(f"Running evaluation with config: {model_config}")
    print(f"Output directory: {output_dir}")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True, 
                              capture_output=True, text=True)
        print("✓ Evaluation completed successfully")
        return output_dir, True
    except subprocess.CalledProcessError as e:
        print(f"✗ Evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return output_dir, False

def compare_results(result_dirs):
    """Compare results from multiple evaluations"""
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    comparison_data = []
    
    for result_dir in result_dirs:
        if not Path(result_dir).exists():
            continue
            
        # Find the latest result file
        result_files = list(Path(result_dir).glob("osworld_evaluation_*.json"))
        if not result_files:
            continue
            
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            
            summary = data.get('summary', {})
            config = data.get('config', {})
            
            comparison_data.append({
                'model': config.get('model', {}).get('name', 'unknown'),
                'local_llava': config.get('model', {}).get('use_local_llava', False),
                'success_rate': summary.get('success_rate', 0),
                'total_tasks': summary.get('total_tasks', 0),
                'successful_tasks': summary.get('successful_tasks', 0),
                'average_steps': summary.get('average_steps', 0),
                'result_dir': result_dir
            })
            
        except Exception as e:
            print(f"Error reading results from {result_dir}: {e}")
    
    # Print comparison table
    if comparison_data:
        print(f"{'Model':<20} {'Local LLaVa':<12} {'Success Rate':<12} {'Tasks':<8} {'Avg Steps':<10}")
        print("-" * 70)
        
        for data in comparison_data:
            model_name = data['model']
            local_llava = "Yes" if data['local_llava'] else "No"
            success_rate = f"{data['success_rate']:.2%}"
            tasks = f"{data['successful_tasks']}/{data['total_tasks']}"
            avg_steps = f"{data['average_steps']:.1f}"
            
            print(f"{model_name:<20} {local_llava:<12} {success_rate:<12} {tasks:<8} {avg_steps:<10}")
        
        # Find best performing model
        best_model = max(comparison_data, key=lambda x: x['success_rate'])
        print(f"\nBest performing model: {best_model['model']} ({best_model['success_rate']:.2%} success rate)")
    
    return comparison_data

def main():
    """Main batch evaluation function"""
    
    print("OSWorld Batch Evaluation")
    print("=" * 30)
    
    # Check if we're in the right directory
    if not Path('evaluate_osworld.py').exists():
        print("Error: evaluate_osworld.py not found in current directory")
        return
    
    # Define model configurations to test
    model_configs = [
        {
            'name': 'OpenAI GPT-4',
            'config': {
                'USE_LOCAL_LLAVA': 'false',
                'EVALUATION_MODEL': 'gpt-4.1'
            }
        },
        {
            'name': 'Local LLaVa',
            'config': {
                'USE_LOCAL_LLAVA': 'true',
                'EVALUATION_MODEL': 'local-llava',
                'LOCAL_LLAVA_MODEL_PATH': 'llava-hf/llava-1.5-7b-hf',
                'LOCAL_LLAVA_DEVICE': 'auto'
            }
        }
    ]
    
    # Ask user which configurations to run
    print("\nAvailable model configurations:")
    for i, config in enumerate(model_configs):
        print(f"{i+1}. {config['name']}")
    
    print("0. Run all configurations")
    
    choice = input("\nSelect configurations to run (comma-separated, e.g., 1,2 or 0 for all): ").strip()
    
    if choice == '0':
        selected_configs = model_configs
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_configs = [model_configs[i] for i in indices if 0 <= i < len(model_configs)]
        except (ValueError, IndexError):
            print("Invalid selection. Running all configurations.")
            selected_configs = model_configs
    
    if not selected_configs:
        print("No valid configurations selected.")
        return
    
    print(f"\nRunning {len(selected_configs)} evaluation(s)...")
    
    # Run evaluations
    result_dirs = []
    successful_runs = 0
    
    for i, config in enumerate(selected_configs):
        print(f"\n--- Evaluation {i+1}/{len(selected_configs)}: {config['name']} ---")
        
        output_dir, success = run_single_evaluation(
            config['config'], 
            f"_{config['name'].replace(' ', '_').lower()}"
        )
        
        result_dirs.append(output_dir)
        if success:
            successful_runs += 1
        
        # Small delay between runs
        if i < len(selected_configs) - 1:
            time.sleep(2)
    
    print(f"\nCompleted {successful_runs}/{len(selected_configs)} evaluations successfully")
    
    # Compare results
    if successful_runs > 1:
        comparison_data = compare_results(result_dirs)
        
        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"batch_comparison_{timestamp}.json"
        
        with open(comparison_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'comparison_data': comparison_data,
                'result_dirs': result_dirs
            }, f, indent=2)
        
        print(f"\nComparison results saved to: {comparison_file}")
    
    print("\nBatch evaluation completed!")

if __name__ == "__main__":
    main()
