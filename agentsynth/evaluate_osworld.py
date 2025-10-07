#!/usr/bin/env python3
"""
OSWorld Evaluation Script using AgentSynth utils

This script evaluates models on OSWorld tasks using the methods from utils.py
while minimizing redundancies and leveraging existing functionality.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import traceback

# Import existing utilities
from utils import (
    call_llms, call_gpt, call_computer_use_preview, parse_computer_use_preview,
    generate_action, generate_computer_use_action, generate_verifier,
    generate_verifier_verdict_key_info, generate_key_info, generate_summary,
    encode_image_from_variable, decode_image_from_variable, parse_json,
    USE_LOCAL_LLAVA, LOCAL_LLAVA_AVAILABLE
)

# Import configuration
from osworld_config import (
    get_evaluation_config, print_config, SAMPLE_TASKS,
    DEFAULT_MAX_STEPS, DEFAULT_TIMEOUT, DEFAULT_OUTPUT_DIR,
    ENABLE_DETAILED_METRICS, SAVE_SCREENSHOTS, SAVE_ACTIONS,
    VERBOSE_LOGGING, TASK_CATEGORIES, SKIP_TASK_TYPES
)

# OSWorld imports (if available)
try:
    from desktop_env.desktop_env import DesktopEnv
    OSWORLD_AVAILABLE = True
except ImportError:
    OSWORLD_AVAILABLE = False
    print("Warning: OSWorld not available. Install desktop-env to use OSWorld evaluation.")

# Hugging Face datasets import
try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    print("Warning: Hugging Face datasets not available. Install datasets to load AgentSynth dataset.")

class OSWorldEvaluator:
    """OSWorld evaluation class using AgentSynth utilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = []
        self.output_dir = Path(config['evaluation']['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize environment if OSWorld is available
        self.env = None
        if OSWORLD_AVAILABLE:
            self._init_environment()
    
    def _init_environment(self):
        """Initialize the desktop environment"""
        try:
            self.env = DesktopEnv(
                action_space="pyautogui",
                provider_name='docker',
                os_type='Ubuntu',
                require_a11y_tree=False
            )
            if VERBOSE_LOGGING:
                print("Desktop environment initialized successfully")
        except Exception as e:
            print(f"Failed to initialize desktop environment: {e}")
            self.env = None
    
    def _create_mock_screenshot(self) -> bytes:
        """Create a simple mock screenshot for testing when OSWorld is not available"""
        try:
            from PIL import Image
            import io
            
            # Create a simple 800x600 RGB image with a gradient
            width, height = 800, 600
            image = Image.new('RGB', (width, height), color='white')
            
            # Add some simple content to make it look like a desktop
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # Draw a simple desktop-like interface
            draw.rectangle([0, 0, width, 50], fill='#2c3e50')  # Top bar
            draw.rectangle([0, height-50, width, height], fill='#34495e')  # Bottom bar
            
            # Add some text
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((10, 10), "Mock Desktop Environment", fill='white', font=font)
            draw.text((10, height-40), "Task: Mock evaluation without OSWorld", fill='white', font=font)
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()
            
        except ImportError:
            # Fallback: create a minimal PNG header if PIL is not available
            # This is a minimal 1x1 PNG in bytes
            return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xdd\x8d\xb4\x1c\x00\x00\x00\x00IEND\xaeB`\x82'
    
    def load_task(self, task_path: str) -> Dict[str, Any]:
        """Load a task from JSON file"""
        try:
            with open(task_path, 'r') as f:
                task = json.load(f)
            return task
        except Exception as e:
            print(f"Error loading task {task_path}: {e}")
            return None
    
    def load_hf_agentsynth_dataset(self, max_tasks: int = None) -> List[Dict[str, Any]]:
        """Load tasks from the official AgentSynth Hugging Face dataset"""
        tasks = []
        
        if not HF_DATASETS_AVAILABLE:
            print("Hugging Face datasets not available. Install with: pip install datasets")
            return tasks
        
        try:
            print("Loading AgentSynth dataset from Hugging Face...")
            dataset = load_dataset("sunblaze-ucb/AgentSynth", split="train")
            
            # Filter for highest level tasks only (summarized_task_sequences)
            # Based on the HF page, we want the summarized version for highest level tasks
            for i, item in enumerate(dataset):
                if max_tasks and i >= max_tasks:
                    break
                
                # Convert HF dataset format to evaluation format
                # Use the highest level task (task_level_6) as the main instruction
                instruction = item.get('task_level_6', item.get('task', item.get('instruction', item.get('text', ''))))
                
                task = {
                    'id': f"agentsynth_hf_{i+1}",
                    'instruction': instruction,
                    'website': item.get('website', ''),
                    'config': {
                        'applications': ['browser'],  # Default to browser for web tasks
                        'setup': []
                    },
                    'evaluator': {
                        'type': 'answer_check',
                        'expected_answer': item.get('action_sequence', [{}])[-1].get('action_kwargs', {}).get('answer', '')
                    },
                    'dataset_info': {
                        'action_sequence': item.get('action_sequence', []),
                        'thoughts_sequence': item.get('thoughts_sequence', []),
                        'webpage_text': item.get('webpage_text', [])
                    },
                    'source': 'huggingface_agentsynth'
                }
                
                tasks.append(task)
                
            print(f"Loaded {len(tasks)} tasks from AgentSynth Hugging Face dataset")
            
        except Exception as e:
            print(f"Error loading AgentSynth dataset from Hugging Face: {e}")
            
        return tasks
    
    def load_dataset_tasks(self, dataset_path: str, max_tasks: int = None) -> List[Dict[str, Any]]:
        """Load tasks from local dataset files (JSONL format)"""
        tasks = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_tasks and i >= max_tasks:
                        break
                    
                    try:
                        task_data = json.loads(line.strip())
                        
                        # Convert dataset format to evaluation format
                        task = {
                            'id': f"dataset_task_{i+1}",
                            'instruction': task_data.get('task', ''),
                            'website': task_data.get('website', ''),
                            'config': {
                                'applications': ['browser'],  # Default to browser for web tasks
                                'setup': []
                            },
                            'evaluator': {
                                'type': 'answer_check',
                                'expected_answer': task_data.get('action_sequence', [{}])[-1].get('action_kwargs', {}).get('answer', '')
                            },
                            'dataset_info': {
                                'action_sequence': task_data.get('action_sequence', []),
                                'thoughts_sequence': task_data.get('thoughts_sequence', []),
                                'webpage_text': task_data.get('webpage_text', [])
                            },
                            'source': 'local_dataset'
                        }
                        
                        tasks.append(task)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i+1}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"Dataset file not found: {dataset_path}")
        except Exception as e:
            print(f"Error loading dataset {dataset_path}: {e}")
            
        return tasks
    
    def load_sample_tasks(self) -> List[Dict[str, Any]]:
        """Load sample tasks from configuration"""
        tasks = []
        for category, task_config in SAMPLE_TASKS.items():
            if category not in SKIP_TASK_TYPES:
                tasks.append(task_config)
        return tasks
    
    def execute_task(self, task: Dict[str, Any], max_steps: int = None) -> Dict[str, Any]:
        """Execute a single task using AgentSynth methods"""
        if max_steps is None:
            max_steps = self.config['evaluation']['max_steps']
        
        task_id = task.get('id', 'unknown')
        instruction = task.get('instruction', '')
        website = task.get('website', '')
        source = task.get('source', 'unknown')
        
        # Use extended time limits for AgentSynth tasks (highest level tasks)
        if source == 'huggingface_agentsynth':
            # Give 6x the normal steps for complex AgentSynth tasks
            max_steps = max_steps * 6
            if VERBOSE_LOGGING:
                print(f"Using extended time limit ({max_steps} steps) for AgentSynth task")
        
        if VERBOSE_LOGGING:
            print(f"Executing task: {task_id}")
            print(f"Instruction: {instruction}")
            if website:
                print(f"Website: {website}")
            print(f"Source: {source}")
            print(f"Max steps: {max_steps}")
        
        # Initialize tracking variables
        thoughts_history = []
        action_history = []
        command_history = []
        screenshot_history = []
        info_history = []
        
        # Get initial screenshot
        if self.env:
            try:
                obs = self.env.reset(task_config=task.get('config', {}))
                base64_image = encode_image_from_variable(obs['screenshot'])
                screenshot_history.append(base64_image)
            except Exception as e:
                print(f"Failed to get initial screenshot: {e}")
                return self._create_failure_result(task_id, str(e))
        else:
            # Mock environment for testing - create a simple mock screenshot
            mock_screenshot = self._create_mock_screenshot()
            base64_image = encode_image_from_variable(mock_screenshot)
            screenshot_history.append(base64_image)
        
        # Execute task steps
        for step in range(max_steps):
            try:
                # Generate action using existing utils
                action, thoughts = generate_action(
                    instruction, thoughts_history, action_history, 
                    info_history, base64_image
                )
                
                thoughts_history.append(thoughts)
                action_history.append(action)
                
                if VERBOSE_LOGGING:
                    print(f"Step {step}: {action}")
                
                # Check if task is complete
                if action == 'DONE':
                    break
                
                # Generate computer use action
                python_command = generate_computer_use_action(
                    instruction, action, command_history, base64_image
                )
                command_history.append(python_command)
                
                # Execute command if environment is available
                if self.env and python_command:
                    try:
                        python_command += '; time.sleep(2)'
                        obs, reward, done, info = self.env.step(python_command)
                        base64_image = encode_image_from_variable(obs['screenshot'])
                        screenshot_history.append(base64_image)
                    except Exception as e:
                        print(f"Error executing command: {e}")
                        continue
                else:
                    # Mock execution
                    time.sleep(1)
                    mock_screenshot = self._create_mock_screenshot()
                    mock_base64_image = encode_image_from_variable(mock_screenshot)
                    screenshot_history.append(mock_base64_image)
                
            except Exception as e:
                print(f"Error in step {step}: {e}")
                if VERBOSE_LOGGING:
                    traceback.print_exc()
                continue
        
        # Generate key information using existing utils
        if screenshot_history:
            key_info = generate_key_info(instruction, thoughts_history, screenshot_history[-1])
        else:
            # Fallback if no screenshots available
            key_info = "No screenshots available for analysis"
        info_history.append(key_info)
        
        # Create result
        result = {
            'task_id': task_id,
            'instruction': instruction,
            'website': website,
            'success': action == 'DONE',
            'steps_taken': len(action_history),
            'thoughts_history': thoughts_history,
            'action_history': action_history,
            'command_history': command_history,
            'screenshot_count': len(screenshot_history),
            'key_info': key_info,
            'dataset_info': task.get('dataset_info', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def evaluate_task_success(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task success using existing verifier methods"""
        task_id = task.get('id', 'unknown')
        evaluator_config = task.get('evaluator', {})
        
        # Use existing verifier methods
        try:
            if result['screenshot_count'] > 0:
                # Use the comprehensive verifier
                success_rate, success, thoughts, necessary = generate_verifier_verdict_key_info(
                    task['instruction'], 
                    [result.get('final_screenshot', 'mock_screenshot')],
                    model=self.config['model']['name']
                )
                
                evaluation = {
                    'success': success,
                    'success_rate': success_rate,
                    'verifier_thoughts': thoughts,
                    'necessary_screenshots': necessary,
                    'evaluation_method': 'verifier_verdict_key_info'
                }
            else:
                # Fallback evaluation
                evaluation = {
                    'success': result['success'],
                    'success_rate': 1.0 if result['success'] else 0.0,
                    'verifier_thoughts': 'No screenshots available for evaluation',
                    'evaluation_method': 'fallback'
                }
        except Exception as e:
            print(f"Error in evaluation: {e}")
            evaluation = {
                'success': False,
                'success_rate': 0.0,
                'verifier_thoughts': f'Evaluation error: {str(e)}',
                'evaluation_method': 'error'
            }
        
        # Add dataset-specific evaluation if available
        if 'dataset_info' in task:
            dataset_info = task['dataset_info']
            expected_answer = evaluator_config.get('expected_answer', '')
            
            # Simple answer comparison for dataset tasks
            if expected_answer and result.get('key_info'):
                # Check if the expected answer appears in the key info or final thoughts
                thoughts_history = result.get('thoughts_history', [])
                final_thoughts = thoughts_history[-1] if thoughts_history else ''
                key_info = result.get('key_info', '')
                
                # Simple string matching (can be improved with more sophisticated comparison)
                answer_found = (
                    expected_answer.lower() in key_info.lower() or 
                    expected_answer.lower() in final_thoughts.lower()
                )
                
                evaluation['dataset_evaluation'] = {
                    'expected_answer': expected_answer,
                    'answer_found': answer_found,
                    'key_info': key_info,
                    'final_thoughts': final_thoughts
                }
                
                # Override success if we have dataset evaluation
                if answer_found:
                    evaluation['success'] = True
                    evaluation['success_rate'] = 1.0
        
        return evaluation
    
    def _create_failure_result(self, task_id: str, error: str) -> Dict[str, Any]:
        """Create a failure result"""
        return {
            'task_id': task_id,
            'success': False,
            'error': error,
            'steps_taken': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """Save evaluation results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"osworld_evaluation_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare results with configuration
        output_data = {
            'config': self.config,
            'results': results,
            'summary': self._generate_summary(results),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate evaluation summary"""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.get('success', False))
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        avg_steps = sum(r.get('steps_taken', 0) for r in results) / total_tasks if total_tasks > 0 else 0
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': success_rate,
            'average_steps': avg_steps,
            'model_used': self.config['model']['name'],
            'local_llava_used': self.config['model']['use_local_llava']
        }
    
    def run_evaluation(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run evaluation on a list of tasks"""
        print(f"Starting evaluation of {len(tasks)} tasks...")
        print(f"Model: {self.config['model']['name']}")
        print(f"Local LLaVa: {self.config['model']['use_local_llava']}")
        print("-" * 50)
        
        results = []
        
        for i, task in enumerate(tasks):
            print(f"\nTask {i+1}/{len(tasks)}: {task.get('id', 'unknown')}")
            
            try:
                # Execute task
                result = self.execute_task(task)
                
                # Evaluate success
                evaluation = self.evaluate_task_success(task, result)
                result.update(evaluation)
                
                results.append(result)
                
                # Print progress
                status = "✓" if result.get('success', False) else "✗"
                print(f"{status} Task completed in {result.get('steps_taken', 0)} steps")
                
            except Exception as e:
                print(f"✗ Task failed with error: {e}")
                results.append(self._create_failure_result(task.get('id', 'unknown'), str(e)))
        
        return results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate models on OSWorld tasks')
    parser.add_argument('--tasks', type=str, help='Path to task JSON file or directory')
    parser.add_argument('--dataset', type=str, help='Path to dataset JSONL file')
    parser.add_argument('--agentsynth-hf', action='store_true', help='Load from official AgentSynth Hugging Face dataset')
    parser.add_argument('--sample', action='store_true', help='Use sample tasks from config')
    parser.add_argument('--output', type=str, help='Output directory for results')
    parser.add_argument('--max-steps', type=int, help='Maximum steps per task')
    parser.add_argument('--max-tasks', type=int, help='Maximum number of tasks to evaluate')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_evaluation_config()
    
    # Override config with command line arguments
    if args.output:
        config['evaluation']['output_dir'] = args.output
    if args.max_steps:
        config['evaluation']['max_steps'] = args.max_steps
    if args.verbose:
        config['logging']['verbose'] = True
    
    # Print configuration
    print_config()
    
    # Initialize evaluator
    evaluator = OSWorldEvaluator(config)
    
    # Load tasks
    tasks = []
    
    if args.agentsynth_hf:
        tasks = evaluator.load_hf_agentsynth_dataset(args.max_tasks)
        print(f"Loaded {len(tasks)} tasks from AgentSynth Hugging Face dataset")
    elif args.dataset:
        tasks = evaluator.load_dataset_tasks(args.dataset, args.max_tasks)
        print(f"Loaded {len(tasks)} tasks from dataset: {args.dataset}")
    elif args.sample:
        tasks = evaluator.load_sample_tasks()
        print(f"Loaded {len(tasks)} sample tasks")
    elif args.tasks:
        if os.path.isfile(args.tasks):
            task = evaluator.load_task(args.tasks)
            if task:
                tasks = [task]
        elif os.path.isdir(args.tasks):
            for task_file in Path(args.tasks).glob('*.json'):
                task = evaluator.load_task(str(task_file))
                if task:
                    tasks.append(task)
        print(f"Loaded {len(tasks)} tasks from {args.tasks}")
    else:
        # Default: try to load from AgentSynth HF dataset first, then local datasets
        print("No specific dataset specified. Trying to load from available sources...")
        
        # Try AgentSynth Hugging Face dataset first
        if HF_DATASETS_AVAILABLE:
            tasks = evaluator.load_hf_agentsynth_dataset(args.max_tasks or 5)
            if tasks:
                print(f"Loaded {len(tasks)} tasks from AgentSynth Hugging Face dataset")
        
        # Fallback to local datasets if HF dataset not available
        if not tasks:
            dataset_paths = [
                '../insta_data/task_sequences.jsonl',
                '../insta_data/summarized_task_sequences.jsonl',
                'insta_data/task_sequences.jsonl',
                'insta_data/summarized_task_sequences.jsonl'
            ]
            
            for dataset_path in dataset_paths:
                if os.path.exists(dataset_path):
                    tasks = evaluator.load_dataset_tasks(dataset_path, args.max_tasks or 5)
                    print(f"Loaded {len(tasks)} tasks from dataset: {dataset_path}")
                    break
        
        # Final fallback to sample tasks
        if not tasks:
            tasks = evaluator.load_sample_tasks()
            print(f"Using {len(tasks)} sample tasks (use --agentsynth-hf for official dataset)")
    
    if not tasks:
        print("No tasks to evaluate!")
        return
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation(tasks)
        
        # Save results
        output_file = evaluator.save_results(results)
        
        # Print summary
        summary = evaluator._generate_summary(results)
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Tasks: {summary['total_tasks']}")
        print(f"Successful: {summary['successful_tasks']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Average Steps: {summary['average_steps']:.1f}")
        print(f"Model Used: {summary['model_used']}")
        print(f"Results saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        if VERBOSE_LOGGING:
            traceback.print_exc()

if __name__ == "__main__":
    main()