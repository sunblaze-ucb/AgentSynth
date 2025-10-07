#!/usr/bin/env python3
"""
LLM-Powered Evaluation Function Generator for AgentSynth Tasks

This script uses an LLM to analyze AgentSynth tasks and automatically generate
or select appropriate verifiable evaluation functions from verification_tools.py.
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import argparse
import time
from datetime import datetime
import re
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated

# Add parent directory to path for utils import
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import existing utilities
from utils import call_llms, parse_json, encode_image_from_variable
from verification_tools import AdvancedVerificationTools

# Pydantic models for structured outputs
class UrlVerification(BaseModel):
    expected_url: Optional[str] = Field(None, description="Expected URL to verify")
    expected_url_pattern: Optional[str] = Field(None, description="Expected URL pattern to match")
    expected_content: Optional[List[str]] = Field(None, description="Expected content to find on the page")
    
    class Config:
        extra = "forbid"

class UIElement(BaseModel):
    type: str = Field(..., description="Type of UI element (text, button, etc.)")
    text: Optional[str] = Field(None, description="Text content of the element")
    selector: Optional[str] = Field(None, description="CSS selector for the element")
    
    class Config:
        extra = "forbid"

class ExpectedOutcome(BaseModel):
    url_verification: Optional[UrlVerification] = Field(None, description="URL verification requirements")
    ui_elements: Optional[List[UIElement]] = Field(None, description="UI elements to verify")
    
    class Config:
        extra = "forbid"

class TaskAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    task_type: str = Field(..., description="The type of computer use task", 
                          pattern="^(web_navigation|file_operation|application_launch|data_extraction|form_filling|document_creation|application_interaction|generic)$")
    verification_methods: List[str] = Field(..., description="List of verification methods to use")
    expected_outcome: ExpectedOutcome = Field(..., description="Expected outcomes to verify")
    verification_parameters: Optional[Dict[str, str]] = Field(default_factory=dict, description="Parameters for verification methods")
    success_criteria: Optional[Dict[str, str]] = Field(default_factory=dict, description="Criteria that define task success")
    custom_logic: str = Field(..., description="Any custom verification logic needed")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the analysis (0.0-1.0)")
    reasoning: str = Field(..., description="Explanation of the analysis approach")
    
    @classmethod
    def model_json_schema(cls, **kwargs):
        """Override to ensure additionalProperties: false for Dict fields"""
        schema = super().model_json_schema(**kwargs)
        
        # Fix the verification_parameters and success_criteria to have additionalProperties: false
        if 'properties' in schema:
            if 'verification_parameters' in schema['properties']:
                schema['properties']['verification_parameters']['additionalProperties'] = False
            if 'success_criteria' in schema['properties']:
                schema['properties']['success_criteria']['additionalProperties'] = False
        
        return schema

class EvaluationFunctionGenerator:
    """
    Uses LLM to analyze tasks and generate verifiable evaluation functions.
    """
    
    def __init__(self, model_name: str = 'gpt-4o'):
        self.model_name = model_name
        self.verification_tools = AdvancedVerificationTools()
        self.generated_functions = {}
        
    def analyze_task_and_generate_evaluator(
        self, 
        task_data: Dict[str, Any],
        screenshot_b64: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a task and generate appropriate evaluation function.
        
        Args:
            task_data: Task data from AgentSynth dataset
            screenshot_b64: Optional base64-encoded screenshot data
            
        Returns:
            Generated evaluation function and metadata
        """
        
        # Extract task information from AgentSynth dataset format
        task_description = 'Task description not found'
        
        # Check if this is AgentSynth format with messages
        if 'messages' in task_data and len(task_data['messages']) > 1:
            try:
                # Get the task text from the first user message
                user_message = task_data['messages'][1]
                if 'content' in user_message and len(user_message['content']) > 0:
                    task_text = user_message['content'][0].get('text', '')
                    # Extract the actual task from 'Given the task: ...'
                    if 'Given the task:' in task_text:
                        task_description = task_text.split('Given the task:')[1].split('.')[0].strip()
                    else:
                        task_description = task_text[:200] + '...' if len(task_text) > 200 else task_text
            except (KeyError, IndexError, AttributeError):
                pass
        
        # Fallback to other possible keys
        if task_description == 'Task description not found':
            task_description = (
                task_data.get('task', '') or 
                task_data.get('instruction', '') or 
                task_data.get('description', '') or 
                task_data.get('goal', '') or
                task_data.get('objective', '') or
                'Task description not found'
            )
        website = task_data.get('website', '') or task_data.get('url', '')
        action_sequence = task_data.get('action_sequence', []) or task_data.get('actions', [])
        
        # Prepare task analysis prompt
        analysis_prompt = self._create_task_analysis_prompt(
            task_description, website, action_sequence
        )
        
        # Use provided screenshot or extract from task data
        if not screenshot_b64 and 'messages' in task_data and len(task_data['messages']) > 1:
            user_message = task_data['messages'][1]
            if 'content' in user_message and len(user_message['content']) > 1:
                for content_item in user_message['content']:
                    if content_item.get('type') == 'input_image':
                        # Check for image_url (AgentSynth format)
                        if 'image_url' in content_item:
                            image_url = content_item['image_url']
                            if image_url.startswith('data:image/'):
                                # Extract base64 data from data URL
                                screenshot_b64 = image_url.split(',', 1)[1]
                                break
                        # Check for source.data (alternative format)
                        elif 'source' in content_item and 'data' in content_item['source']:
                            screenshot_b64 = content_item['source']['data']
                            break
        
        # Analyze task with LLM using structured outputs
        try:
            print(f"Analyzing task with {self.model_name} (Pydantic structured outputs)...")
            
            # Use Pydantic structured outputs for reliable parsing
            analysis_result = self._call_llms_with_pydantic(
                "You are an expert at analyzing computer use tasks and determining the best verification methods. Analyze the task and provide a structured response.",
                analysis_prompt,
                screenshot_b64 if screenshot_b64 else [],
                model=self.model_name
            )
            
            print(f"✅ Pydantic structured analysis completed: {analysis_result.task_type} task")
            print(f"   Verification methods: {analysis_result.verification_methods}")
            print(f"   Confidence: {analysis_result.confidence:.2f}")
            
            # Convert Pydantic model to dict for compatibility
            analysis_result = analysis_result.model_dump()
            
        except Exception as structured_error:
            print(f"⚠️  Pydantic structured outputs failed: {structured_error}")
            print("Falling back to regular LLM call with JSON parsing...")
            
            # Fallback to regular LLM call with improved JSON parsing
            try:
                if screenshot_b64:
                    analysis_response = call_llms(
                        "You are an expert at analyzing computer use tasks and determining the best verification methods.",
                        analysis_prompt,
                        screenshot_b64,
                        model=self.model_name
                    )
                else:
                    analysis_response = call_llms(
                        "You are an expert at analyzing computer use tasks and determining the best verification methods.",
                        analysis_prompt,
                        [],
                        model=self.model_name
                    )
                
                print(f"LLM response length: {len(analysis_response)} characters")
                
                # Parse LLM response with improved error handling
                analysis_result = self._parse_llm_response(analysis_response)
                print(f"✅ Fallback analysis completed: {analysis_result.get('task_type', 'unknown')} task")
                
            except Exception as fallback_error:
                print(f"❌ Both structured and fallback parsing failed: {fallback_error}")
                analysis_result = self._fallback_parse_analysis("Fallback due to parsing errors")
        
        # Generate evaluation function based on analysis
        evaluation_function = self._generate_evaluation_function(
            task_data, analysis_result
        )
        
        return {
            'task_id': task_data.get('id', f"task_{int(time.time())}"),
            'task_description': task_description,
            'analysis_result': analysis_result,
            'evaluation_function': evaluation_function,
            'generated_at': datetime.now().isoformat()
        }
    
    def _create_task_analysis_prompt(
        self, 
        task_description: str, 
        website: str, 
        action_sequence: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for LLM to analyze the task."""
        
        # Truncate action sequence if too long
        action_summary = action_sequence[:5] if len(action_sequence) > 5 else action_sequence
        
        return f"""
Analyze this computer use task and determine the best verifiable evaluation approach.

TASK: {task_description[:200]}{'...' if len(task_description) > 200 else ''}
WEBSITE: {website}
ACTIONS: {len(action_sequence)} actions (showing first {len(action_summary)})

Available verification methods:
1. analyze_screenshot_for_elements - Computer vision analysis of UI elements
2. verify_web_page_state - Selenium-based web page verification
3. verify_file_system_changes - File system state verification
4. verify_process_changes - Process monitoring verification
5. verify_database_state - Database state verification
6. verify_network_state - Network/API verification

Based on the task description and action sequence, determine:
- What type of task this is
- Which verification methods would be most appropriate
- What specific, verifiable outcomes should be checked
- What parameters should be used for verification
- What constitutes success for this task
- Any custom verification logic needed beyond standard methods

Provide a structured analysis following the required schema.
"""
    
    def _call_llms_with_pydantic(self, system_prompt: str, user_prompt: str, img, model: str = 'gpt-4o') -> TaskAnalysis:
        """Call LLM with Pydantic structured outputs using client.chat.completions.parse"""
        
        try:
            import openai
            from openai import OpenAI
            
            # Check if we have API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
            client = OpenAI(api_key=api_key)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Add image if provided
            if img:
                if isinstance(img, list):
                    for item in img:
                        messages[1]["content"] = [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{item}"}}
                        ]
                else:
                    messages[1]["content"] = [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    ]
            
            # Use client.chat.completions.parse with Pydantic model
            completion = client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=TaskAnalysis,
                temperature=0.1  # Lower temperature for more consistent structured output
            )
            
            # Return the parsed Pydantic model
            return completion.choices[0].message.parsed
            
        except Exception as e:
            raise Exception(f"Pydantic structured outputs failed: {e}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response with robust JSON handling."""
        
        try:
            # First try standard JSON parsing
            result = parse_json(response)
            if result:
                return result
        except Exception as e:
            print(f"Standard JSON parsing failed: {e}")
        
        # Try to extract JSON from markdown code blocks
        try:
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                if result:
                    return result
        except Exception as e:
            print(f"Markdown JSON extraction failed: {e}")
        
        # Try to find and fix common JSON issues
        try:
            # Remove any text before the first {
            start_idx = response.find('{')
            if start_idx != -1:
                json_str = response[start_idx:]
                
                # Try to find the matching closing brace
                brace_count = 0
                end_idx = -1
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx != -1:
                    json_str = json_str[:end_idx]
                    
                    # Try to fix common issues
                    json_str = self._fix_common_json_issues(json_str)
                    
                    result = json.loads(json_str)
                    if result:
                        return result
        except Exception as e:
            print(f"JSON repair attempt failed: {e}")
        
        # Fallback to keyword-based analysis
        print("Using fallback analysis due to JSON parsing failure")
        return self._fallback_parse_analysis(response)
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix missing commas between object properties (more comprehensive)
        # Pattern: "key": value "next_key" -> "key": value, "next_key"
        json_str = re.sub(r'("\s*:\s*[^,}]+)\s+(")', r'\1, \2', json_str)
        
        # Fix missing commas between array elements
        json_str = re.sub(r'(\])\s*(\[)', r'\1, \2', json_str)
        
        # Fix missing commas between object properties on new lines
        json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
        
        # Fix missing commas after values before closing braces
        json_str = re.sub(r'([^,}\]])\s*(\n\s*[}\]])', r'\1,\2', json_str)
        
        return json_str
    
    def _fallback_parse_analysis(self, response: str) -> Dict[str, Any]:
        """Fallback parsing if JSON parsing fails."""
        
        # Simple keyword-based analysis
        task_type = "generic"
        verification_methods = ["analyze_screenshot_for_elements"]
        
        if "web" in response.lower() or "navigate" in response.lower():
            task_type = "web_navigation"
            verification_methods = ["verify_web_page_state"]
        elif "file" in response.lower() or "create" in response.lower():
            task_type = "file_operation"
            verification_methods = ["verify_file_system_changes"]
        elif "application" in response.lower() or "launch" in response.lower():
            task_type = "application_launch"
            verification_methods = ["verify_process_changes"]
        
        return {
            "task_type": task_type,
            "verification_methods": verification_methods,
            "expected_outcome": {},
            "verification_parameters": {},
            "success_criteria": {"basic_success": True},
            "custom_logic": "Basic verification",
            "confidence": 0.5,
            "reasoning": "Fallback analysis due to parsing error"
        }
    
    def _generate_evaluation_function(
        self, 
        task_data: Dict[str, Any], 
        analysis_result: Dict[str, Any]
    ) -> str:
        """Generate Python evaluation function based on analysis."""
        
        task_type = analysis_result.get('task_type', 'generic')
        verification_methods = analysis_result.get('verification_methods', [])
        expected_outcome = analysis_result.get('expected_outcome', {})
        verification_parameters = analysis_result.get('verification_parameters', {})
        success_criteria = analysis_result.get('success_criteria', {})
        
        # Generate function based on task type
        if task_type == 'web_navigation':
            return self._generate_web_navigation_evaluator(
                task_data, verification_parameters, success_criteria
            )
        elif task_type == 'file_operation':
            return self._generate_file_operation_evaluator(
                task_data, verification_parameters, success_criteria
            )
        elif task_type == 'application_launch':
            return self._generate_application_launch_evaluator(
                task_data, verification_parameters, success_criteria
            )
        elif task_type == 'data_extraction':
            return self._generate_data_extraction_evaluator(
                task_data, verification_parameters, success_criteria
            )
        else:
            return self._generate_generic_evaluator(
                task_data, verification_parameters, success_criteria
            )
    
    def _generate_web_navigation_evaluator(
        self, 
        task_data: Dict[str, Any], 
        verification_parameters: Dict[str, Any], 
        success_criteria: Dict[str, Any]
    ) -> str:
        """Generate web navigation evaluation function."""
        
        website = task_data.get('website', '')
        expected_url = verification_parameters.get('url', f'https://{website}')
        expected_elements = verification_parameters.get('expected_elements', [])
        
        # Get task description for function docstring
        task_desc = 'Web navigation task'
        
        # Check if this is AgentSynth format with messages
        if 'messages' in task_data and len(task_data['messages']) > 1:
            try:
                user_message = task_data['messages'][1]
                if 'content' in user_message and len(user_message['content']) > 0:
                    task_text = user_message['content'][0].get('text', '')
                    if 'Given the task:' in task_text:
                        task_desc = task_text.split('Given the task:')[1].split('.')[0].strip()
                    else:
                        task_desc = task_text[:200] + '...' if len(task_text) > 200 else task_text
            except (KeyError, IndexError, AttributeError):
                pass
        
        # Fallback to other possible keys
        if task_desc == 'Web navigation task':
            task_desc = (
                task_data.get('task', '') or 
                task_data.get('instruction', '') or 
                task_data.get('description', '') or 
                task_data.get('goal', '') or
                task_data.get('objective', '') or
                'Web navigation task'
            )
        
        function_code = f'''
def evaluate_web_navigation_task(task_data, agent_trajectory, verification_tools):
    """
    Evaluate web navigation task: {task_desc}
    Generated by LLM analysis.
    """
    
    results = {{
        'task_type': 'web_navigation',
        'success': False,
        'verification_methods': [],
        'details': {{}}
    }}
    
    try:
        # Web page state verification
        web_verification = verification_tools.verify_web_page_state(
            url='{expected_url}',
            expected_elements={expected_elements}
        )
        
        results['verification_methods'].append('web_page_state')
        results['details']['web_verification'] = web_verification
        
        # Screenshot analysis if available
        if agent_trajectory.get('screenshot_history'):
            final_screenshot = agent_trajectory['screenshot_history'][-1]
            if isinstance(final_screenshot, str) and os.path.exists(final_screenshot):
                screenshot_analysis = verification_tools.analyze_screenshot_for_elements(
                    final_screenshot,
                    {expected_elements}
                )
                results['verification_methods'].append('screenshot_analysis')
                results['details']['screenshot_analysis'] = screenshot_analysis
        
        # Determine overall success
        results['success'] = web_verification.get('overall_success', False)
        
        # Additional success criteria
        if '{website}' in web_verification.get('page_title', '').lower():
            results['details']['website_confirmed'] = True
        else:
            results['details']['website_confirmed'] = False
            results['success'] = False
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        return results
'''
        
        return function_code
    
    def _generate_file_operation_evaluator(
        self, 
        task_data: Dict[str, Any], 
        verification_parameters: Dict[str, Any], 
        success_criteria: Dict[str, Any]
    ) -> str:
        """Generate file operation evaluation function."""
        
        function_code = f'''
def evaluate_file_operation_task(task_data, agent_trajectory, verification_tools):
    """
    Evaluate file operation task: {task_data.get('task', '')}
    Generated by LLM analysis.
    """
    
    results = {{
        'task_type': 'file_operation',
        'success': False,
        'verification_methods': [],
        'details': {{}}
    }}
    
    try:
        # File system changes verification
        before_state = agent_trajectory.get('system_state_before', {{}})
        after_state = agent_trajectory.get('system_state_after', {{}})
        
        if before_state and after_state:
            file_verification = verification_tools.verify_file_system_changes(
                before_state=before_state,
                after_state=after_state,
                expected_changes=verification_parameters.get('expected_changes', [])
            )
            
            results['verification_methods'].append('file_system_changes')
            results['details']['file_verification'] = file_verification
            results['success'] = file_verification.get('overall_success', False)
        
        # Check for specific file operations in action history
        action_history = agent_trajectory.get('action_history', [])
        file_operations = [action for action in action_history if 'file' in action.lower() or 'save' in action.lower()]
        results['details']['file_operations_detected'] = len(file_operations) > 0
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        return results
'''
        
        return function_code
    
    def _generate_application_launch_evaluator(
        self, 
        task_data: Dict[str, Any], 
        verification_parameters: Dict[str, Any], 
        success_criteria: Dict[str, Any]
    ) -> str:
        """Generate application launch evaluation function."""
        
        function_code = f'''
def evaluate_application_launch_task(task_data, agent_trajectory, verification_tools):
    """
    Evaluate application launch task: {task_data.get('task', '')}
    Generated by LLM analysis.
    """
    
    results = {{
        'task_type': 'application_launch',
        'success': False,
        'verification_methods': [],
        'details': {{}}
    }}
    
    try:
        # Process changes verification
        before_processes = agent_trajectory.get('system_state_before', {{}}).get('processes', [])
        after_processes = agent_trajectory.get('system_state_after', {{}}).get('processes', [])
        
        if before_processes and after_processes:
            process_verification = verification_tools.verify_process_changes(
                before_processes=before_processes,
                after_processes=after_processes,
                expected_processes=verification_parameters.get('expected_processes', [])
            )
            
            results['verification_methods'].append('process_changes')
            results['details']['process_verification'] = process_verification
            results['success'] = process_verification.get('overall_success', False)
        
        # Check action history for application launch indicators
        action_history = agent_trajectory.get('action_history', [])
        launch_actions = [action for action in action_history if any(app in action.lower() for app in ['launch', 'open', 'start', 'run'])]
        results['details']['launch_actions_detected'] = len(launch_actions) > 0
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        return results
'''
        
        return function_code
    
    def _generate_data_extraction_evaluator(
        self, 
        task_data: Dict[str, Any], 
        verification_parameters: Dict[str, Any], 
        success_criteria: Dict[str, Any]
    ) -> str:
        """Generate data extraction evaluation function."""
        
        # Get task description for function docstring
        task_desc = 'Data extraction task'
        
        # Check if this is AgentSynth format with messages
        if 'messages' in task_data and len(task_data['messages']) > 1:
            try:
                user_message = task_data['messages'][1]
                if 'content' in user_message and len(user_message['content']) > 0:
                    task_text = user_message['content'][0].get('text', '')
                    if 'Given the task:' in task_text:
                        task_desc = task_text.split('Given the task:')[1].split('.')[0].strip()
                    else:
                        task_desc = task_text[:200] + '...' if len(task_text) > 200 else task_text
            except (KeyError, IndexError, AttributeError):
                pass
        
        # Fallback to other possible keys
        if task_desc == 'Data extraction task':
            task_desc = (
                task_data.get('task', '') or 
                task_data.get('instruction', '') or 
                task_data.get('description', '') or 
                task_data.get('goal', '') or
                task_data.get('objective', '') or
                'Data extraction task'
            )
        
        function_code = f'''
def evaluate_data_extraction_task(task_data, agent_trajectory, verification_tools):
    """
    Evaluate data extraction task: {task_desc}
    Generated by LLM analysis.
    """
    
    results = {{
        'task_type': 'data_extraction',
        'success': False,
        'verification_methods': [],
        'details': {{}}
    }}
    
    try:
        # Check if expected data was extracted
        expected_data = verification_parameters.get('expected_data', [])
        action_history = agent_trajectory.get('action_history', [])
        
        # Look for data extraction indicators in actions
        extraction_actions = [action for action in action_history if any(keyword in action.lower() for keyword in ['extract', 'find', 'search', 'locate', 'get'])]
        results['details']['extraction_actions'] = extraction_actions
        
        # Check final answer/result
        final_answer = agent_trajectory.get('final_answer', '')
        if final_answer:
            # Simple keyword matching for expected data
            data_found = 0
            for expected_item in expected_data:
                if expected_item.lower() in final_answer.lower():
                    data_found += 1
            
            results['details']['data_found_count'] = data_found
            results['details']['expected_data_count'] = len(expected_data)
            results['success'] = data_found >= len(expected_data) * 0.8  # 80% threshold
        
        # Screenshot analysis for data verification
        if agent_trajectory.get('screenshot_history'):
            final_screenshot = agent_trajectory['screenshot_history'][-1]
            if isinstance(final_screenshot, str) and os.path.exists(final_screenshot):
                screenshot_analysis = verification_tools.analyze_screenshot_for_elements(
                    final_screenshot,
                    [{{"type": "text", "text": item}} for item in expected_data]
                )
                results['verification_methods'].append('screenshot_analysis')
                results['details']['screenshot_analysis'] = screenshot_analysis
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        return results
'''
        
        return function_code
    
    def _generate_generic_evaluator(
        self, 
        task_data: Dict[str, Any], 
        verification_parameters: Dict[str, Any], 
        success_criteria: Dict[str, Any]
    ) -> str:
        """Generate generic evaluation function."""
        
        # Get task description for function docstring
        task_desc = 'Generic task'
        
        # Check if this is AgentSynth format with messages
        if 'messages' in task_data and len(task_data['messages']) > 1:
            try:
                user_message = task_data['messages'][1]
                if 'content' in user_message and len(user_message['content']) > 0:
                    task_text = user_message['content'][0].get('text', '')
                    if 'Given the task:' in task_text:
                        task_desc = task_text.split('Given the task:')[1].split('.')[0].strip()
                    else:
                        task_desc = task_text[:200] + '...' if len(task_text) > 200 else task_text
            except (KeyError, IndexError, AttributeError):
                pass
        
        # Fallback to other possible keys
        if task_desc == 'Generic task':
            task_desc = (
                task_data.get('task', '') or 
                task_data.get('instruction', '') or 
                task_data.get('description', '') or 
                task_data.get('goal', '') or
                task_data.get('objective', '') or
                'Generic task'
            )
        
        function_code = f'''
def evaluate_generic_task(task_data, agent_trajectory, verification_tools):
    """
    Evaluate generic task: {task_desc}
    Generated by LLM analysis.
    """
    
    results = {{
        'task_type': 'generic',
        'success': False,
        'verification_methods': [],
        'details': {{}}
    }}
    
    try:
        # Basic action sequence verification
        action_history = agent_trajectory.get('action_history', [])
        results['details']['action_count'] = len(action_history)
        results['details']['has_actions'] = len(action_history) > 0
        
        # Check for completion indicators
        completion_indicators = ['done', 'complete', 'finish', 'success', 'submit', 'save']
        completion_found = any(indicator in ' '.join(action_history).lower() for indicator in completion_indicators)
        results['details']['completion_indicators'] = completion_found
        
        # Screenshot analysis if available
        if agent_trajectory.get('screenshot_history'):
            final_screenshot = agent_trajectory['screenshot_history'][-1]
            if isinstance(final_screenshot, str) and os.path.exists(final_screenshot):
                screenshot_analysis = verification_tools.analyze_screenshot_for_elements(
                    final_screenshot,
                    verification_parameters.get('expected_elements', [])
                )
                results['verification_methods'].append('screenshot_analysis')
                results['details']['screenshot_analysis'] = screenshot_analysis
                results['success'] = screenshot_analysis.get('overall_success', False)
        
        # Basic success criteria
        if not results['success']:
            results['success'] = len(action_history) > 0 and completion_found
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        return results
'''
        
        return function_code
    
    def _create_fallback_evaluator(self, task_data: Dict[str, Any]) -> str:
        """Create a simple fallback evaluator."""
        
        return f'''
def evaluate_fallback_task(task_data, agent_trajectory, verification_tools):
    """
    Fallback evaluator for task: {task_data.get('task', '')}
    """
    
    results = {{
        'task_type': 'fallback',
        'success': False,
        'verification_methods': ['basic_check'],
        'details': {{}}
    }}
    
    try:
        action_history = agent_trajectory.get('action_history', [])
        results['details']['action_count'] = len(action_history)
        results['success'] = len(action_history) > 0
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        results['success'] = False
        return results
'''
    
    def process_agentsynth_dataset(
        self, 
        dataset_path: str, 
        output_path: str, 
        max_tasks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process AgentSynth dataset and generate evaluation functions.
        
        Args:
            dataset_path: Path to AgentSynth dataset JSONL file
            output_path: Path to save generated evaluation functions
            max_tasks: Maximum number of tasks to process
            
        Returns:
            Summary of processing results
        """
        
        results = {
            'processed_tasks': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'generated_functions': {},
            'errors': []
        }
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                tasks = []
                for i, line in enumerate(f):
                    if max_tasks and i >= max_tasks:
                        break
                    
                    try:
                        task_data = json.loads(line.strip())
                        tasks.append(task_data)
                    except json.JSONDecodeError as e:
                        results['errors'].append(f"Line {i+1}: JSON decode error - {e}")
                        continue
            
            print(f"Processing {len(tasks)} tasks from {dataset_path}")
            
            # Generate evaluation functions for each task
            for i, task_data in enumerate(tasks):
                print(f"Processing task {i+1}/{len(tasks)}: ...")
                
                try:
                    # Extract screenshot from AgentSynth format if available
                    screenshot_b64 = None
                    if 'messages' in task_data and len(task_data['messages']) > 1:
                        user_message = task_data['messages'][1]
                        if 'content' in user_message and len(user_message['content']) > 1:
                            for content_item in user_message['content']:
                                if content_item.get('type') == 'input_image':
                                    # Check for image_url (AgentSynth format)
                                    if 'image_url' in content_item:
                                        image_url = content_item['image_url']
                                        if image_url.startswith('data:image/'):
                                            # Extract base64 data from data URL
                                            screenshot_b64 = image_url.split(',', 1)[1]
                                            break
                                    # Check for source.data (alternative format)
                                    elif 'source' in content_item and 'data' in content_item['source']:
                                        screenshot_b64 = content_item['source']['data']
                                        break
                    
                    evaluation_result = self.analyze_task_and_generate_evaluator(task_data, screenshot_b64)
                    
                    if 'error' not in evaluation_result:
                        results['successful_generations'] += 1
                        results['generated_functions'][evaluation_result['task_id']] = evaluation_result
                    else:
                        results['failed_generations'] += 1
                        results['errors'].append(f"Task {i+1}: {evaluation_result['error']}")
                    
                    results['processed_tasks'] += 1
                    
                except Exception as e:
                    results['failed_generations'] += 1
                    results['errors'].append(f"Task {i+1}: {str(e)}")
                    results['processed_tasks'] += 1
            
            # Save generated functions
            self._save_generated_functions(results['generated_functions'], output_path)
            
        except Exception as e:
            results['errors'].append(f"Dataset processing error: {str(e)}")
        
        return results
    
    def _save_generated_functions(
        self, 
        generated_functions: Dict[str, Any], 
        output_path: str
    ) -> None:
        """Save generated evaluation functions to file."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create Python file with all generated functions
        with open(output_file, 'w') as f:
            f.write('#!/usr/bin/env python3\n')
            f.write('"""\n')
            f.write('Generated Evaluation Functions for AgentSynth Tasks\n')
            f.write(f'Generated on: {datetime.now().isoformat()}\n')
            f.write('"""\n\n')
            f.write('import os\n')
            f.write('import json\n')
            f.write('from typing import Dict, List, Any\n')
            f.write('from verification_tools import AdvancedVerificationTools\n\n')
            
            # Add each generated function
            for task_id, function_data in generated_functions.items():
                task_desc = function_data.get("task_description", "No description available")
                if not task_desc or task_desc.strip() == "":
                    task_desc = f"Task ID: {task_id}"
                f.write(f'# Task: {task_desc[:100]}{"..." if len(task_desc) > 100 else ""}\n')
                f.write(function_data['evaluation_function'])
                f.write('\n\n')
            
            # Add function registry
            f.write('# Function Registry\n')
            f.write('EVALUATION_FUNCTIONS = {\n')
            for task_id, function_data in generated_functions.items():
                function_name = f"evaluate_{function_data['analysis_result'].get('task_type', 'generic')}_task"
                f.write(f'    "{task_id}": {function_name},\n')
            f.write('}\n\n')
            
            # Add helper function
            f.write('''
def evaluate_task(task_id: str, task_data: Dict[str, Any], agent_trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a task using the appropriate generated function.
    
    Args:
        task_id: Task identifier
        task_data: Original task data
        agent_trajectory: Agent execution trajectory
        
    Returns:
        Evaluation results
    """
    
    verification_tools = AdvancedVerificationTools()
    
    if task_id in EVALUATION_FUNCTIONS:
        evaluator_func = EVALUATION_FUNCTIONS[task_id]
        return evaluator_func(task_data, agent_trajectory, verification_tools)
    else:
        # Fallback to generic evaluation
        return evaluate_generic_task(task_data, agent_trajectory, verification_tools)
''')
        
        print(f"Generated evaluation functions saved to: {output_path}")

def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(description='Generate evaluation functions for AgentSynth tasks')
    parser.add_argument('--dataset', type=str, required=True, help='Path to AgentSynth dataset JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output path for generated evaluation functions')
    parser.add_argument('--max-tasks', type=int, help='Maximum number of tasks to process')
    parser.add_argument('--model', type=str, default='gpt-4o', help='LLM model to use for analysis')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = EvaluationFunctionGenerator(model_name=args.model)
    
    # Process dataset
    print(f"Starting evaluation function generation...")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Max tasks: {args.max_tasks or 'All'}")
    print("-" * 50)
    
    results = generator.process_agentsynth_dataset(
        dataset_path=args.dataset,
        output_path=args.output,
        max_tasks=args.max_tasks
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Processed tasks: {results['processed_tasks']}")
    print(f"Successful generations: {results['successful_generations']}")
    print(f"Failed generations: {results['failed_generations']}")
    print(f"Success rate: {results['successful_generations']/max(1, results['processed_tasks']):.2%}")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")

if __name__ == "__main__":
    main()
