#!/usr/bin/env python3
"""
Integrated Verifiable Evaluation System for AgentSynth

This script demonstrates the complete pipeline:
1. Load AgentSynth tasks
2. Generate evaluation functions using LLM analysis
3. Use verifiable evaluation methods (no LLM judges)
4. Provide comprehensive evaluation results
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from datetime import datetime

# Import our modules
from generate_evaluation_functions import EvaluationFunctionGenerator
from verification_tools import AdvancedVerificationTools
from verifiable_evaluator import VerifiableTaskEvaluator
from example_verifiable_evaluation import AgentSynthVerifiableEvaluator

class IntegratedVerifiableEvaluationSystem:
    """
    Complete integrated system for verifiable evaluation of AgentSynth tasks.
    """
    
    def __init__(self, model_name: str = 'gpt-4o'):
        self.model_name = model_name
        self.function_generator = EvaluationFunctionGenerator(model_name)
        self.verification_tools = AdvancedVerificationTools()
        self.verifiable_evaluator = VerifiableTaskEvaluator()
        self.agentsynth_evaluator = AgentSynthVerifiableEvaluator()
        self.generated_functions = {}
    
    def load_agentsynth_dataset(self, dataset_path: str, max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load tasks from AgentSynth dataset."""
        
        tasks = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_tasks and i >= max_tasks:
                        break
                    
                    try:
                        task_data = json.loads(line.strip())
                        task_data['id'] = f"task_{i+1}"
                        tasks.append(task_data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping line {i+1} due to JSON error: {e}")
                        continue
            
            print(f"Loaded {len(tasks)} tasks from {dataset_path}")
            return tasks
            
        except FileNotFoundError:
            print(f"Error: Dataset file not found: {dataset_path}")
            return []
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def generate_evaluation_functions(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate evaluation functions for tasks using LLM analysis."""
        
        print(f"Generating evaluation functions for {len(tasks)} tasks...")
        
        generated_functions = {}
        successful_generations = 0
        failed_generations = 0
        
        for i, task_data in enumerate(tasks):
            print(f"Processing task {i+1}/{len(tasks)}: {task_data.get('task', '')[:50]}...")
            
            try:
                evaluation_result = self.function_generator.analyze_task_and_generate_evaluator(task_data)
                
                if 'error' not in evaluation_result:
                    generated_functions[evaluation_result['task_id']] = evaluation_result
                    successful_generations += 1
                else:
                    failed_generations += 1
                    print(f"  Failed: {evaluation_result['error']}")
                
            except Exception as e:
                failed_generations += 1
                print(f"  Error: {str(e)}")
        
        self.generated_functions = generated_functions
        
        print(f"Generation complete: {successful_generations} successful, {failed_generations} failed")
        return {
            'generated_functions': generated_functions,
            'successful_generations': successful_generations,
            'failed_generations': failed_generations
        }
    
    def evaluate_task_with_verifiable_methods(
        self, 
        task_data: Dict[str, Any], 
        agent_trajectory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a task using verifiable methods (no LLM judges).
        
        Args:
            task_data: Task data from AgentSynth dataset
            agent_trajectory: Agent execution trajectory
            
        Returns:
            Comprehensive evaluation results
        """
        
        task_id = task_data.get('id', 'unknown')
        
        # Method 1: Use generated evaluation function if available
        if task_id in self.generated_functions:
            try:
                # Execute the generated function
                function_data = self.generated_functions[task_id]
                function_code = function_data['evaluation_function']
                
                # Create a temporary module to execute the function
                import types
                temp_module = types.ModuleType('temp_evaluator')
                exec(function_code, temp_module.__dict__)
                
                # Find the evaluation function
                eval_func = None
                for name in dir(temp_module):
                    if name.startswith('evaluate_') and callable(getattr(temp_module, name)):
                        eval_func = getattr(temp_module, name)
                        break
                
                if eval_func:
                    generated_result = eval_func(task_data, agent_trajectory, self.verification_tools)
                    generated_result['evaluation_method'] = 'generated_function'
                    return generated_result
                    
            except Exception as e:
                print(f"Warning: Generated function failed for {task_id}: {e}")
        
        # Method 2: Use AgentSynth-specific evaluator
        try:
            agentsynth_result = self.agentsynth_evaluator.evaluate_agentsynth_task(task_data, agent_trajectory)
            agentsynth_result['evaluation_method'] = 'agentsynth_evaluator'
            return agentsynth_result
            
        except Exception as e:
            print(f"Warning: AgentSynth evaluator failed for {task_id}: {e}")
        
        # Method 3: Use generic verifiable evaluator
        try:
            task_description = task_data.get('task', '')
            task_type = self._classify_task_type(task_description)
            
            expected_outcome = self._create_expected_outcome(task_data, task_type)
            
            generic_result = self.verifiable_evaluator.evaluate_task_completion(
                task_description=task_description,
                task_type=task_type,
                expected_outcome=expected_outcome,
                system_state_before=agent_trajectory.get('system_state_before'),
                system_state_after=agent_trajectory.get('system_state_after')
            )
            generic_result['evaluation_method'] = 'generic_verifiable'
            return generic_result
            
        except Exception as e:
            print(f"Error: All evaluation methods failed for {task_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'evaluation_method': 'failed'
            }
    
    def _classify_task_type(self, task_description: str) -> str:
        """Classify task type based on description."""
        
        task_lower = task_description.lower()
        
        if any(keyword in task_lower for keyword in ['navigate', 'visit', 'go to', 'find on']):
            return 'web_navigation'
        elif any(keyword in task_lower for keyword in ['find', 'extract', 'locate', 'get information']):
            return 'data_extraction'
        elif any(keyword in task_lower for keyword in ['create', 'save', 'download', 'file']):
            return 'file_operation'
        elif any(keyword in task_lower for keyword in ['launch', 'open', 'start', 'application']):
            return 'application_launch'
        else:
            return 'web_navigation'  # Default for most AgentSynth tasks
    
    def _create_expected_outcome(self, task_data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Create expected outcome based on task data."""
        
        website = task_data.get('website', '')
        action_sequence = task_data.get('action_sequence', [])
        
        expected_outcome = {}
        
        if task_type == 'web_navigation':
            expected_outcome = {
                'expected_url': f'https://{website}',
                'expected_content': self._extract_expected_content(action_sequence)
            }
        elif task_type == 'data_extraction':
            expected_outcome = {
                'expected_data': self._extract_expected_data(action_sequence),
                'expected_url': f'https://{website}'
            }
        elif task_type == 'file_operation':
            expected_outcome = {
                'file_path': '/tmp/task_output.txt',
                'expected_content': self._extract_expected_content(action_sequence)
            }
        
        return expected_outcome
    
    def _extract_expected_content(self, action_sequence: List[Dict[str, Any]]) -> List[str]:
        """Extract expected content from action sequence."""
        
        expected_content = []
        
        for action in action_sequence:
            if action.get('action_key') == 'stop' and 'answer' in action.get('action_kwargs', {}):
                answer = action['action_kwargs']['answer']
                expected_content.append(answer)
        
        return expected_content
    
    def _extract_expected_data(self, action_sequence: List[Dict[str, Any]]) -> List[str]:
        """Extract expected data from action sequence."""
        
        expected_data = []
        
        for action in action_sequence:
            if action.get('action_key') == 'stop' and 'answer' in action.get('action_kwargs', {}):
                answer = action['action_kwargs']['answer']
                
                # Extract numbers, dates, and key terms
                import re
                numbers = re.findall(r'\d+\.?\d*', answer)
                dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', answer)
                quoted = re.findall(r'"([^"]*)"', answer)
                
                expected_data.extend(numbers + dates + quoted)
        
        return list(set(expected_data))  # Remove duplicates
    
    def run_comprehensive_evaluation(
        self, 
        dataset_path: str, 
        max_tasks: Optional[int] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation pipeline.
        
        Args:
            dataset_path: Path to AgentSynth dataset
            max_tasks: Maximum number of tasks to process
            save_results: Whether to save results to file
            
        Returns:
            Comprehensive evaluation results
        """
        
        print("AgentSynth Integrated Verifiable Evaluation System")
        print("=" * 60)
        print(f"Dataset: {dataset_path}")
        print(f"Max tasks: {max_tasks or 'All'}")
        print(f"Model: {self.model_name}")
        print()
        
        # Step 1: Load dataset
        print("Step 1: Loading AgentSynth dataset...")
        tasks = self.load_agentsynth_dataset(dataset_path, max_tasks)
        
        if not tasks:
            return {'error': 'No tasks loaded'}
        
        # Step 2: Generate evaluation functions
        print("\nStep 2: Generating evaluation functions...")
        generation_results = self.generate_evaluation_functions(tasks)
        
        # Step 3: Evaluate tasks with verifiable methods
        print("\nStep 3: Evaluating tasks with verifiable methods...")
        evaluation_results = []
        
        for i, task_data in enumerate(tasks):
            print(f"Evaluating task {i+1}/{len(tasks)}: {task_data.get('task', '')[:50]}...")
            
            # Create mock agent trajectory for demonstration
            agent_trajectory = self._create_mock_trajectory(task_data)
            
            # Evaluate task
            evaluation_result = self.evaluate_task_with_verifiable_methods(task_data, agent_trajectory)
            evaluation_result['task_id'] = task_data.get('id', f'task_{i+1}')
            evaluation_result['task_description'] = task_data.get('task', '')
            
            evaluation_results.append(evaluation_result)
        
        # Step 4: Compile results
        print("\nStep 4: Compiling results...")
        
        total_tasks = len(evaluation_results)
        successful_tasks = sum(1 for result in evaluation_results if result.get('success', False))
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        # Count evaluation methods used
        method_counts = {}
        for result in evaluation_results:
            method = result.get('evaluation_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        comprehensive_results = {
            'evaluation_summary': {
                'total_tasks': total_tasks,
                'successful_tasks': successful_tasks,
                'success_rate': success_rate,
                'evaluation_methods_used': method_counts
            },
            'generation_results': generation_results,
            'evaluation_results': evaluation_results,
            'timestamp': datetime.now().isoformat(),
            'model_used': self.model_name
        }
        
        # Step 5: Save results
        if save_results:
            output_file = f"agentsynth_verifiable_evaluation_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        return comprehensive_results
    
    def _create_mock_trajectory(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock agent trajectory for demonstration."""
        
        action_sequence = task_data.get('action_sequence', [])
        
        # Convert action sequence to action history
        action_history = []
        for action in action_sequence:
            action_key = action.get('action_key', '')
            if action_key == 'click':
                action_history.append(f"Click on element {action.get('target_element_id', 'unknown')}")
            elif action_key == 'fill':
                value = action.get('action_kwargs', {}).get('value', '')
                action_history.append(f"Fill field with: {value}")
            elif action_key == 'stop':
                action_history.append("Task completed")
        
        # Create mock system states
        system_state_before = {
            'files': {},
            'processes': [],
            'timestamp': time.time() - 100
        }
        
        system_state_after = {
            'files': {},
            'processes': [],
            'timestamp': time.time()
        }
        
        return {
            'action_history': action_history,
            'screenshot_history': ['mock_screenshot.png'],
            'system_state_before': system_state_before,
            'system_state_after': system_state_after,
            'final_answer': self._extract_final_answer(action_sequence)
        }
    
    def _extract_final_answer(self, action_sequence: List[Dict[str, Any]]) -> str:
        """Extract final answer from action sequence."""
        
        for action in action_sequence:
            if action.get('action_key') == 'stop' and 'answer' in action.get('action_kwargs', {}):
                return action['action_kwargs']['answer']
        
        return ""

def main():
    """Main function for command-line usage."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Run integrated verifiable evaluation on AgentSynth dataset')
    parser.add_argument('--dataset', type=str, default='../insta_data/summarized_task_sequences.jsonl', help='Path to AgentSynth dataset')
    parser.add_argument('--max-tasks', type=int, default=10, help='Maximum number of tasks to process')
    parser.add_argument('--model', type=str, default='gpt-4o', help='LLM model for function generation')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Initialize system
    system = IntegratedVerifiableEvaluationSystem(model_name=args.model)
    
    # Run comprehensive evaluation
    results = system.run_comprehensive_evaluation(
        dataset_path=args.dataset,
        max_tasks=args.max_tasks,
        save_results=not args.no_save
    )
    
    # Print summary
    if 'evaluation_summary' in results:
        summary = results['evaluation_summary']
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"Successful tasks: {summary['successful_tasks']}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Evaluation methods used: {summary['evaluation_methods_used']}")
        
        if 'generation_results' in results:
            gen_results = results['generation_results']
            print(f"Generated functions: {gen_results['successful_generations']}")
            print(f"Generation success rate: {gen_results['successful_generations']/max(1, gen_results['successful_generations'] + gen_results['failed_generations']):.2%}")
    
    print("\nVerifiable evaluation complete!")
    print("Key benefits:")
    print("✓ No LLM judges - objective system state verification")
    print("✓ Fast execution - no API calls during evaluation")
    print("✓ 100% reproducible - deterministic results")
    print("✓ Cost-effective - minimal LLM usage only for function generation")

if __name__ == "__main__":
    main()
