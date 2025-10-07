#!/usr/bin/env python3
"""
OSWorld Agent Implementation

This module implements the OSWorld agent interface as specified in:
https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/README.md

The agent uses AgentSynth utilities for action generation and execution.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import AgentSynth utilities
from utils import (
    generate_action, generate_computer_use_action, generate_key_info,
    call_llms, call_gpt, call_computer_use_preview, parse_computer_use_preview,
    encode_image_from_variable, decode_image_from_variable, parse_json,
    USE_LOCAL_LLAVA, LOCAL_LLAVA_AVAILABLE
)

class OSWorldAgent:
    """
    OSWorld Agent implementation using AgentSynth utilities.
    
    This agent follows the OSWorld agent interface requirements:
    - Takes screenshots as input
    - Generates actions based on task instructions
    - Executes actions in the desktop environment
    - Returns action results
    """
    
    def __init__(self, model_name: str = None, max_steps: int = 15):
        """
        Initialize the OSWorld agent.
        
        Args:
            model_name: Name of the model to use (optional)
            max_steps: Maximum number of steps per task
        """
        self.model_name = model_name or ('local-llava' if USE_LOCAL_LLAVA else 'gpt-4o')
        self.max_steps = max_steps
        self.step_count = 0
        self.task_instruction = ""
        self.thoughts_history = []
        self.action_history = []
        self.command_history = []
        self.info_history = []
        
        print(f"Initialized OSWorld Agent with model: {self.model_name}")
        print(f"Local LLaVa available: {LOCAL_LLAVA_AVAILABLE}")
        print(f"Using local LLaVa: {USE_LOCAL_LLAVA}")
    
    def reset(self, task_config: Dict[str, Any]) -> None:
        """
        Reset the agent for a new task.
        
        Args:
            task_config: Task configuration containing instruction and setup
        """
        self.step_count = 0
        self.task_instruction = task_config.get('instruction', '')
        self.thoughts_history = []
        self.action_history = []
        self.command_history = []
        self.info_history = []
        
        print(f"Agent reset for new task: {self.task_instruction[:100]}...")
    
    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step of the agent.
        
        Args:
            observation: Dictionary containing 'screenshot' and other observation data
            
        Returns:
            Dictionary containing action results
        """
        if self.step_count >= self.max_steps:
            return {
                'action': 'stop',
                'action_kwargs': {'reason': 'max_steps_reached'},
                'success': False
            }
        
        try:
            # Get screenshot
            screenshot = observation.get('screenshot')
            if screenshot is None:
                return {
                    'action': 'stop',
                    'action_kwargs': {'reason': 'no_screenshot'},
                    'success': False
                }
            
            # Encode screenshot for model processing
            base64_image = encode_image_from_variable(screenshot)
            
            # Generate action using AgentSynth utilities
            action, thoughts = generate_action(
                self.task_instruction,
                self.thoughts_history,
                self.action_history,
                self.info_history,
                base64_image
            )
            
            # Store thoughts and action
            self.thoughts_history.append(thoughts)
            self.action_history.append(action)
            
            print(f"Step {self.step_count + 1}: {action}")
            
            # Check if task is complete
            if action == 'DONE':
                return {
                    'action': 'stop',
                    'action_kwargs': {'reason': 'task_complete'},
                    'success': True
                }
            
            # Generate computer use action
            python_command = generate_computer_use_action(
                self.task_instruction,
                action,
                self.command_history,
                base64_image
            )
            
            self.command_history.append(python_command)
            
            # Prepare action result
            action_result = {
                'action': 'python',
                'action_kwargs': {'code': python_command},
                'success': True,
                'step': self.step_count + 1,
                'thoughts': thoughts
            }
            
            self.step_count += 1
            return action_result
            
        except Exception as e:
            print(f"Error in agent step: {e}")
            return {
                'action': 'stop',
                'action_kwargs': {'reason': f'error: {str(e)}'},
                'success': False
            }
    
    def get_task_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current task execution.
        
        Returns:
            Dictionary containing task summary
        """
        return {
            'task_instruction': self.task_instruction,
            'steps_taken': self.step_count,
            'max_steps': self.max_steps,
            'thoughts_history': self.thoughts_history,
            'action_history': self.action_history,
            'command_history': self.command_history,
            'model_used': self.model_name,
            'timestamp': datetime.now().isoformat()
        }

class OSWorldAgentWithVerification(OSWorldAgent):
    """
    Extended OSWorld Agent with verification capabilities.
    
    This agent includes additional verification steps and can
    generate key information summaries.
    """
    
    def __init__(self, model_name: str = None, max_steps: int = 15, enable_verification: bool = True):
        """
        Initialize the extended OSWorld agent.
        
        Args:
            model_name: Name of the model to use
            max_steps: Maximum number of steps per task
            enable_verification: Whether to enable verification steps
        """
        super().__init__(model_name, max_steps)
        self.enable_verification = enable_verification
        self.screenshot_history = []
        
        if enable_verification:
            print("Verification enabled - agent will generate key information summaries")
    
    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step with verification capabilities.
        """
        # Store screenshot for verification
        screenshot = observation.get('screenshot')
        if screenshot:
            base64_image = encode_image_from_variable(screenshot)
            self.screenshot_history.append(base64_image)
        
        # Call parent step method
        result = super().step(observation)
        
        # Add verification if enabled and task is complete
        if (self.enable_verification and 
            result.get('action') == 'stop' and 
            result.get('action_kwargs', {}).get('reason') == 'task_complete' and
            self.screenshot_history):
            
            try:
                # Generate key information summary
                key_info = generate_key_info(
                    self.task_instruction,
                    self.thoughts_history,
                    self.screenshot_history[-1]
                )
                
                result['verification'] = {
                    'key_info': key_info,
                    'screenshots_analyzed': len(self.screenshot_history)
                }
                
            except Exception as e:
                print(f"Verification failed: {e}")
                result['verification'] = {'error': str(e)}
        
        return result
    
    def reset(self, task_config: Dict[str, Any]) -> None:
        """Reset with verification data."""
        super().reset(task_config)
        self.screenshot_history = []

def create_osworld_agent(model_name: str = None, max_steps: int = 15, enable_verification: bool = False) -> OSWorldAgent:
    """
    Factory function to create an OSWorld agent.
    
    Args:
        model_name: Name of the model to use
        max_steps: Maximum number of steps per task
        enable_verification: Whether to enable verification
        
    Returns:
        OSWorldAgent instance
    """
    if enable_verification:
        return OSWorldAgentWithVerification(model_name, max_steps, enable_verification)
    else:
        return OSWorldAgent(model_name, max_steps)

# Example usage for OSWorld integration
if __name__ == "__main__":
    # Create agent
    agent = create_osworld_agent(
        model_name='local-llava' if USE_LOCAL_LLAVA else 'gpt-4o',
        max_steps=15,
        enable_verification=True
    )
    
    # Example task configuration
    task_config = {
        'instruction': 'Navigate to google.com and search for "machine learning"',
        'setup': []
    }
    
    # Reset agent for new task
    agent.reset(task_config)
    
    # Example observation (would come from OSWorld environment)
    observation = {
        'screenshot': b'mock_screenshot_data'  # In real usage, this would be actual screenshot
    }
    
    # Execute step
    result = agent.step(observation)
    print("Agent step result:", result)
    
    # Get task summary
    summary = agent.get_task_summary()
    print("Task summary:", summary)
