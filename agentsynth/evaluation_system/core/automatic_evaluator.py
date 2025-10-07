#!/usr/bin/env python3
"""
Comprehensive Automatic Evaluation Function for AgentSynth Tasks

This module provides a multi-layered automatic evaluation system that combines
LLM-based verification with programmatic checks to determine task completion success.
"""

import json
import base64
import time
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import subprocess
import re
from PIL import Image
import io
import sys

# Import existing AgentSynth utilities
# Add parent directory to path for utils import
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils import (
    generate_verifier_verdict_key_info, generate_verifier, generate_key_info,
    encode_image_from_variable, decode_image_from_variable, parse_json,
    call_llms, USE_LOCAL_LLAVA, LOCAL_LLAVA_AVAILABLE
)

class AutomaticTaskEvaluator:
    """
    Comprehensive automatic evaluation system for AgentSynth tasks.
    
    Combines multiple verification methods:
    1. LLM-based semantic verification
    2. Programmatic state checking
    3. Screenshot analysis
    4. Action sequence validation
    5. Expected outcome matching
    """
    
    def __init__(self, model_name: str = 'gpt-4o', confidence_threshold: float = 0.7):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.evaluation_cache = {}
        
    def evaluate_task_completion(
        self, 
        task_description: str,
        action_history: List[str],
        screenshot_history: List[Union[str, bytes]],
        expected_outcome: Optional[Dict[str, Any]] = None,
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive task completion evaluation.
        
        Args:
            task_description: The original task instruction
            action_history: List of actions taken by the agent
            screenshot_history: List of screenshots (base64 strings or bytes)
            expected_outcome: Expected result if known
            task_metadata: Additional task information
            
        Returns:
            Dictionary with evaluation results including success, confidence, and detailed analysis
        """
        
        evaluation_results = {
            'task_description': task_description,
            'timestamp': time.time(),
            'evaluation_methods': [],
            'overall_success': False,
            'confidence_score': 0.0,
            'detailed_analysis': {}
        }
        
        # Method 1: LLM-based Semantic Verification
        llm_evaluation = self._llm_semantic_verification(
            task_description, action_history, screenshot_history
        )
        evaluation_results['evaluation_methods'].append('llm_semantic')
        evaluation_results['detailed_analysis']['llm_semantic'] = llm_evaluation
        
        # Method 2: Programmatic State Verification
        if expected_outcome:
            programmatic_evaluation = self._programmatic_verification(
                expected_outcome, action_history, screenshot_history
            )
            evaluation_results['evaluation_methods'].append('programmatic')
            evaluation_results['detailed_analysis']['programmatic'] = programmatic_evaluation
        
        # Method 3: Screenshot Analysis
        screenshot_evaluation = self._screenshot_analysis(
            task_description, screenshot_history
        )
        evaluation_results['evaluation_methods'].append('screenshot_analysis')
        evaluation_results['detailed_analysis']['screenshot_analysis'] = screenshot_evaluation
        
        # Method 4: Action Sequence Validation
        action_evaluation = self._action_sequence_validation(
            task_description, action_history
        )
        evaluation_results['evaluation_methods'].append('action_validation')
        evaluation_results['detailed_analysis']['action_validation'] = action_evaluation
        
        # Method 5: Task-Specific Verification
        if task_metadata:
            task_specific_evaluation = self._task_specific_verification(
                task_description, task_metadata, action_history, screenshot_history
            )
            evaluation_results['evaluation_methods'].append('task_specific')
            evaluation_results['detailed_analysis']['task_specific'] = task_specific_evaluation
        
        # Combine results for final decision
        final_evaluation = self._combine_evaluation_results(evaluation_results)
        evaluation_results.update(final_evaluation)
        
        return evaluation_results
    
    def _llm_semantic_verification(
        self, 
        task_description: str, 
        action_history: List[str], 
        screenshot_history: List[Union[str, bytes]]
    ) -> Dict[str, Any]:
        """Use AgentSynth's existing LLM-based verification methods."""
        
        try:
            # Convert screenshots to base64 strings if needed
            processed_screenshots = []
            for screenshot in screenshot_history:
                if isinstance(screenshot, bytes):
                    processed_screenshots.append(encode_image_from_variable(screenshot))
                else:
                    processed_screenshots.append(screenshot)
            
            # Use the comprehensive verifier from AgentSynth
            success_rate, success, thoughts, necessary = generate_verifier_verdict_key_info(
                task_description, processed_screenshots, model=self.model_name
            )
            
            return {
                'success': success,
                'success_rate': success_rate,
                'verifier_thoughts': thoughts,
                'necessary_screenshots': necessary,
                'method': 'agentsynth_verifier'
            }
            
        except Exception as e:
            return {
                'success': False,
                'success_rate': 0.0,
                'error': str(e),
                'method': 'agentsynth_verifier'
            }
    
    def _programmatic_verification(
        self, 
        expected_outcome: Dict[str, Any], 
        action_history: List[str], 
        screenshot_history: List[Union[str, bytes]]
    ) -> Dict[str, Any]:
        """Programmatic verification based on expected outcomes."""
        
        verification_results = {
            'success': True,
            'checks_passed': 0,
            'total_checks': 0,
            'failed_checks': [],
            'method': 'programmatic'
        }
        
        # Check 1: File existence
        if 'file_path' in expected_outcome:
            verification_results['total_checks'] += 1
            file_path = expected_outcome['file_path']
            if os.path.exists(file_path):
                verification_results['checks_passed'] += 1
            else:
                verification_results['failed_checks'].append(f"File not found: {file_path}")
                verification_results['success'] = False
        
        # Check 2: URL navigation
        if 'expected_url' in expected_outcome:
            verification_results['total_checks'] += 1
            expected_url = expected_outcome['expected_url']
            # This would require browser state checking - simplified for example
            url_found = any(expected_url in action for action in action_history)
            if url_found:
                verification_results['checks_passed'] += 1
            else:
                verification_results['failed_checks'].append(f"Expected URL not found: {expected_url}")
                verification_results['success'] = False
        
        # Check 3: Text content verification
        if 'expected_text' in expected_outcome:
            verification_results['total_checks'] += 1
            expected_text = expected_outcome['expected_text']
            # Check if expected text appears in final screenshot
            if screenshot_history:
                try:
                    final_screenshot = screenshot_history[-1]
                    if isinstance(final_screenshot, bytes):
                        # Use OCR or text extraction here
                        text_found = self._extract_text_from_screenshot(final_screenshot)
                        if expected_text.lower() in text_found.lower():
                            verification_results['checks_passed'] += 1
                        else:
                            verification_results['failed_checks'].append(f"Expected text not found: {expected_text}")
                            verification_results['success'] = False
                except Exception as e:
                    verification_results['failed_checks'].append(f"Text extraction error: {str(e)}")
                    verification_results['success'] = False
        
        return verification_results
    
    def _screenshot_analysis(
        self, 
        task_description: str, 
        screenshot_history: List[Union[str, bytes]]
    ) -> Dict[str, Any]:
        """Analyze screenshots for task completion indicators."""
        
        if not screenshot_history:
            return {
                'success': False,
                'analysis': 'No screenshots available',
                'method': 'screenshot_analysis'
            }
        
        try:
            # Get final screenshot
            final_screenshot = screenshot_history[-1]
            if isinstance(final_screenshot, bytes):
                final_screenshot_b64 = encode_image_from_variable(final_screenshot)
            else:
                final_screenshot_b64 = final_screenshot
            
            # Use LLM to analyze the final screenshot
            analysis_prompt = f"""
            Analyze this screenshot to determine if the task has been completed successfully.
            Task: {task_description}
            
            Look for:
            1. Visual indicators of task completion
            2. Success messages or confirmations
            3. Expected UI elements or states
            4. Error messages or failure indicators
            
            Provide a detailed analysis and success determination.
            """
            
            # Use existing LLM call function
            response = call_llms(
                "You are an expert at analyzing computer screenshots for task completion.",
                analysis_prompt,
                final_screenshot_b64,
                model=self.model_name
            )
            
            # Parse response (assuming JSON format)
            try:
                analysis_result = parse_json(response)
                return {
                    'success': analysis_result.get('success', False),
                    'confidence': analysis_result.get('confidence', 0.5),
                    'analysis': analysis_result.get('analysis', response),
                    'method': 'screenshot_analysis'
                }
            except:
                # Fallback to text analysis
                success_indicators = ['success', 'completed', 'done', 'finished', 'saved', 'submitted']
                failure_indicators = ['error', 'failed', 'invalid', 'not found', 'denied']
                
                response_lower = response.lower()
                success_score = sum(1 for indicator in success_indicators if indicator in response_lower)
                failure_score = sum(1 for indicator in failure_indicators if indicator in response_lower)
                
                return {
                    'success': success_score > failure_score,
                    'confidence': min(0.9, (success_score - failure_score) / 5.0 + 0.5),
                    'analysis': response,
                    'method': 'screenshot_analysis'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'method': 'screenshot_analysis'
            }
    
    def _action_sequence_validation(
        self, 
        task_description: str, 
        action_history: List[str]
    ) -> Dict[str, Any]:
        """Validate that the action sequence makes sense for the task."""
        
        validation_results = {
            'success': True,
            'action_quality_score': 0.0,
            'issues_found': [],
            'method': 'action_validation'
        }
        
        # Check for common issues
        issues = []
        
        # Check 1: Too many repeated actions (potential loops)
        action_counts = {}
        for action in action_history:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        repeated_actions = {action: count for action, count in action_counts.items() if count > 3}
        if repeated_actions:
            issues.append(f"Repeated actions detected: {repeated_actions}")
            validation_results['success'] = False
        
        # Check 2: Action sequence length (too short or too long)
        if len(action_history) < 2:
            issues.append("Action sequence too short - task may not be properly attempted")
            validation_results['success'] = False
        elif len(action_history) > 50:
            issues.append("Action sequence too long - possible inefficiency or loops")
            validation_results['success'] = False
        
        # Check 3: Task completion indicators in actions
        completion_actions = ['save', 'submit', 'done', 'finish', 'complete', 'confirm']
        has_completion_action = any(
            any(completion in action.lower() for completion in completion_actions)
            for action in action_history
        )
        
        if not has_completion_action:
            issues.append("No clear completion action found in sequence")
        
        # Calculate quality score
        quality_score = 1.0
        quality_score -= len(issues) * 0.2
        quality_score = max(0.0, quality_score)
        
        validation_results['action_quality_score'] = quality_score
        validation_results['issues_found'] = issues
        
        return validation_results
    
    def _task_specific_verification(
        self, 
        task_description: str, 
        task_metadata: Dict[str, Any], 
        action_history: List[str], 
        screenshot_history: List[Union[str, bytes]]
    ) -> Dict[str, Any]:
        """Task-specific verification based on metadata."""
        
        verification_results = {
            'success': True,
            'method': 'task_specific',
            'checks': {}
        }
        
        # Check application-specific requirements
        if 'applications' in task_metadata:
            required_apps = task_metadata['applications']
            for app in required_apps:
                app_used = any(app.lower() in action.lower() for action in action_history)
                verification_results['checks'][f'app_{app}'] = app_used
                if not app_used:
                    verification_results['success'] = False
        
        # Check website navigation
        if 'website' in task_metadata:
            website = task_metadata['website']
            website_visited = any(website in action for action in action_history)
            verification_results['checks']['website_visited'] = website_visited
            if not website_visited:
                verification_results['success'] = False
        
        return verification_results
    
    def _extract_text_from_screenshot(self, screenshot_bytes: bytes) -> str:
        """Extract text from screenshot using OCR (placeholder implementation)."""
        # This would typically use OCR libraries like pytesseract
        # For now, return empty string as placeholder
        return ""
    
    def _combine_evaluation_results(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from all evaluation methods into final decision."""
        
        methods = evaluation_results['detailed_analysis']
        success_scores = []
        confidence_scores = []
        
        # Collect scores from each method
        for method_name, method_result in methods.items():
            if 'success' in method_result:
                success_scores.append(1.0 if method_result['success'] else 0.0)
            
            if 'success_rate' in method_result:
                confidence_scores.append(method_result['success_rate'] / 100.0)
            elif 'confidence' in method_result:
                confidence_scores.append(method_result['confidence'])
            elif 'action_quality_score' in method_result:
                confidence_scores.append(method_result['action_quality_score'])
        
        # Calculate weighted averages
        if success_scores:
            overall_success = sum(success_scores) / len(success_scores) >= self.confidence_threshold
        else:
            overall_success = False
        
        if confidence_scores:
            overall_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            overall_confidence = 0.0
        
        return {
            'overall_success': overall_success,
            'confidence_score': overall_confidence,
            'method_scores': {
                method: result.get('success_rate', result.get('confidence', 0.0))
                for method, result in methods.items()
            }
        }

# Example usage function
def evaluate_agentsynth_task(
    task_data: Dict[str, Any],
    agent_trajectory: Dict[str, Any],
    model_name: str = 'gpt-4o'
) -> Dict[str, Any]:
    """
    Evaluate a single AgentSynth task completion.
    
    Args:
        task_data: Task information from AgentSynth dataset
        agent_trajectory: Agent's execution trajectory
        model_name: LLM model to use for evaluation
        
    Returns:
        Comprehensive evaluation results
    """
    
    evaluator = AutomaticTaskEvaluator(model_name=model_name)
    
    # Extract information from task data
    task_description = task_data.get('instruction', task_data.get('task', ''))
    expected_outcome = task_data.get('evaluator', {})
    task_metadata = {
        'applications': task_data.get('config', {}).get('applications', []),
        'website': task_data.get('website', ''),
        'dataset_info': task_data.get('dataset_info', {})
    }
    
    # Extract information from trajectory
    action_history = agent_trajectory.get('action_history', [])
    screenshot_history = agent_trajectory.get('screenshot_history', [])
    
    # Run evaluation
    results = evaluator.evaluate_task_completion(
        task_description=task_description,
        action_history=action_history,
        screenshot_history=screenshot_history,
        expected_outcome=expected_outcome,
        task_metadata=task_metadata
    )
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_task = {
        "instruction": "Create a new document in LibreOffice Writer titled 'Test Document'",
        "config": {"applications": ["libreoffice"]},
        "evaluator": {"type": "file_exists", "file_path": "/tmp/Test Document.odt"}
    }
    
    sample_trajectory = {
        "action_history": [
            "Open LibreOffice Writer",
            "Create new document",
            "Set title to 'Test Document'",
            "Save document"
        ],
        "screenshot_history": ["screenshot1.png", "screenshot2.png"]
    }
    
    results = evaluate_agentsynth_task(sample_task, sample_trajectory)
    print(json.dumps(results, indent=2))
