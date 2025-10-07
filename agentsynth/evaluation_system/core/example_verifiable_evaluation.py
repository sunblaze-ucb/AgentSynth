#!/usr/bin/env python3
"""
Example: Using Generated Verifiable Evaluation Functions

This script demonstrates how to use the LLM-generated evaluation functions
to verify AgentSynth task completion without relying on LLM judges.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Import our verification tools
from verification_tools import AdvancedVerificationTools
from verifiable_evaluator import VerifiableTaskEvaluator

class AgentSynthVerifiableEvaluator:
    """
    Comprehensive verifiable evaluator for AgentSynth tasks.
    """
    
    def __init__(self):
        self.verification_tools = AdvancedVerificationTools()
        self.verifiable_evaluator = VerifiableTaskEvaluator()
    
    def evaluate_agentsynth_task(
        self, 
        task_data: Dict[str, Any], 
        agent_trajectory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate an AgentSynth task using verifiable methods.
        
        Args:
            task_data: Task data from AgentSynth dataset
            agent_trajectory: Agent execution trajectory
            
        Returns:
            Comprehensive evaluation results
        """
        
        task_description = task_data.get('task', '')
        website = task_data.get('website', '')
        action_sequence = task_data.get('action_sequence', [])
        
        # Determine task type and appropriate verification method
        task_type = self._classify_task_type(task_description, action_sequence)
        
        # Create expected outcome based on task analysis
        expected_outcome = self._create_expected_outcome(task_data, task_type)
        
        # Run verifiable evaluation
        evaluation_result = self.verifiable_evaluator.evaluate_task_completion(
            task_description=task_description,
            task_type=task_type,
            expected_outcome=expected_outcome,
            system_state_before=agent_trajectory.get('system_state_before'),
            system_state_after=agent_trajectory.get('system_state_after')
        )
        
        # Add task-specific analysis
        evaluation_result['task_analysis'] = {
            'task_type': task_type,
            'website': website,
            'action_count': len(action_sequence),
            'expected_outcome': expected_outcome
        }
        
        return evaluation_result
    
    def _classify_task_type(self, task_description: str, action_sequence: List[Dict[str, Any]]) -> str:
        """Classify task type based on description and actions."""
        
        task_lower = task_description.lower()
        action_text = ' '.join([str(action) for action in action_sequence]).lower()
        
        # Web navigation tasks
        if any(keyword in task_lower for keyword in ['navigate', 'visit', 'go to', 'find on', 'search on']):
            return 'web_navigation'
        
        # Data extraction tasks
        if any(keyword in task_lower for keyword in ['find', 'extract', 'locate', 'get information', 'list']):
            return 'data_extraction'
        
        # Form filling tasks
        if any(keyword in task_lower for keyword in ['fill', 'enter', 'submit', 'form']):
            return 'form_filling'
        
        # File operations
        if any(keyword in task_lower for keyword in ['create', 'save', 'download', 'file']):
            return 'file_operation'
        
        # Application tasks
        if any(keyword in task_lower for keyword in ['launch', 'open', 'start', 'application']):
            return 'application_launch'
        
        # Default to web navigation for most AgentSynth tasks
        return 'web_navigation'
    
    def _create_expected_outcome(self, task_data: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """Create expected outcome based on task data and type."""
        
        website = task_data.get('website', '')
        action_sequence = task_data.get('action_sequence', [])
        
        expected_outcome = {}
        
        if task_type == 'web_navigation':
            expected_outcome = {
                'expected_url': f'https://{website}',
                'expected_content': self._extract_expected_content_from_actions(action_sequence)
            }
        
        elif task_type == 'data_extraction':
            expected_outcome = {
                'expected_data': self._extract_expected_data_from_actions(action_sequence),
                'expected_url': f'https://{website}'
            }
        
        elif task_type == 'file_operation':
            expected_outcome = {
                'file_path': self._extract_expected_file_path(task_data),
                'expected_content': self._extract_expected_content_from_actions(action_sequence)
            }
        
        elif task_type == 'application_launch':
            expected_outcome = {
                'application_name': self._extract_expected_application(task_data),
                'expected_processes': [self._extract_expected_application(task_data)]
            }
        
        return expected_outcome
    
    def _extract_expected_content_from_actions(self, action_sequence: List[Dict[str, Any]]) -> List[str]:
        """Extract expected content from action sequence."""
        
        expected_content = []
        
        for action in action_sequence:
            if action.get('action_key') == 'stop' and 'answer' in action.get('action_kwargs', {}):
                answer = action['action_kwargs']['answer']
                # Extract key information from the answer
                if 'found' in answer.lower() or 'located' in answer.lower():
                    expected_content.append(answer)
        
        return expected_content
    
    def _extract_expected_data_from_actions(self, action_sequence: List[Dict[str, Any]]) -> List[str]:
        """Extract expected data from action sequence."""
        
        expected_data = []
        
        for action in action_sequence:
            if action.get('action_key') == 'stop' and 'answer' in action.get('action_kwargs', {}):
                answer = action['action_kwargs']['answer']
                # Look for specific data points (numbers, dates, names, etc.)
                import re
                
                # Extract numbers
                numbers = re.findall(r'\d+\.?\d*', answer)
                expected_data.extend(numbers)
                
                # Extract dates
                dates = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', answer)
                expected_data.extend(dates)
                
                # Extract quoted text
                quoted = re.findall(r'"([^"]*)"', answer)
                expected_data.extend(quoted)
        
        return list(set(expected_data))  # Remove duplicates
    
    def _extract_expected_file_path(self, task_data: Dict[str, Any]) -> str:
        """Extract expected file path from task data."""
        
        task_description = task_data.get('task', '')
        
        # Look for file names in task description
        import re
        file_matches = re.findall(r'["\']([^"\']*\.[a-zA-Z]{2,4})["\']', task_description)
        
        if file_matches:
            return f"/tmp/{file_matches[0]}"
        
        # Default file path
        return "/tmp/task_output.txt"
    
    def _extract_expected_application(self, task_data: Dict[str, Any]) -> str:
        """Extract expected application from task data."""
        
        task_description = task_data.get('task', '').lower()
        
        # Common applications
        if 'libreoffice' in task_description or 'writer' in task_description:
            return 'libreoffice'
        elif 'chrome' in task_description or 'browser' in task_description:
            return 'chrome'
        elif 'gimp' in task_description:
            return 'gimp'
        elif 'firefox' in task_description:
            return 'firefox'
        
        return 'unknown'

def demonstrate_verifiable_evaluation():
    """Demonstrate verifiable evaluation with example tasks."""
    
    evaluator = AgentSynthVerifiableEvaluator()
    
    # Example 1: Web Navigation Task
    web_navigation_task = {
        "task": "On joblo.com, find and watch the official trailer for the John Madden biopic featuring Nicolas Cage and Christian Bale.",
        "website": "joblo.com",
        "action_sequence": [
            {"action_key": "click", "action_kwargs": {}, "target_element_id": 204},
            {"action_key": "stop", "action_kwargs": {"answer": "The first look at Nicolas Cage and Christian Bale in the John Madden biopic has been viewed on joblo.com."}, "target_element_id": None}
        ]
    }
    
    web_trajectory = {
        "action_history": ["Navigate to joblo.com", "Click on biopic article", "View trailer"],
        "screenshot_history": ["screenshot1.png", "screenshot2.png"],
        "system_state_before": {"files": {}, "processes": []},
        "system_state_after": {"files": {}, "processes": []}
    }
    
    print("Example 1: Web Navigation Task")
    print("=" * 50)
    print(f"Task: {web_navigation_task['task']}")
    
    web_result = evaluator.evaluate_agentsynth_task(web_navigation_task, web_trajectory)
    print(f"Task Type: {web_result['task_analysis']['task_type']}")
    print(f"Success: {web_result['success']}")
    print(f"Verification Methods: {web_result['verification_methods']}")
    print()
    
    # Example 2: Data Extraction Task
    data_extraction_task = {
        "task": "On the Agweek website, compare the latest last prices of the Lean Hogs contracts for June '25 and July '25 to determine which contract has a higher price.",
        "website": "agweek.com",
        "action_sequence": [
            {"action_key": "click", "action_kwargs": {}, "target_element_id": 436},
            {"action_key": "stop", "action_kwargs": {"answer": "The Jul '25 Lean Hogs contract has the higher last price at 102.450s compared to the Jun '25 contract last price of 98.850s."}, "target_element_id": None}
        ]
    }
    
    data_trajectory = {
        "action_history": ["Navigate to agweek.com", "Click on Lean Hogs", "Compare prices"],
        "screenshot_history": ["screenshot1.png"],
        "system_state_before": {"files": {}, "processes": []},
        "system_state_after": {"files": {}, "processes": []}
    }
    
    print("Example 2: Data Extraction Task")
    print("=" * 50)
    print(f"Task: {data_extraction_task['task']}")
    
    data_result = evaluator.evaluate_agentsynth_task(data_extraction_task, data_trajectory)
    print(f"Task Type: {data_result['task_analysis']['task_type']}")
    print(f"Success: {data_result['success']}")
    print(f"Expected Data: {data_result['task_analysis']['expected_outcome'].get('expected_data', [])}")
    print()
    
    # Example 3: File Operation Task (simulated)
    file_task = {
        "task": "Create a new document in LibreOffice Writer titled 'Report.docx'",
        "website": "",
        "action_sequence": [
            {"action_key": "click", "action_kwargs": {}, "target_element_id": 1},
            {"action_key": "stop", "action_kwargs": {"answer": "Document 'Report.docx' created successfully"}, "target_element_id": None}
        ]
    }
    
    file_trajectory = {
        "action_history": ["Open LibreOffice Writer", "Create new document", "Save as Report.docx"],
        "screenshot_history": ["screenshot1.png"],
        "system_state_before": {"files": {}},
        "system_state_after": {"files": {"/tmp/Report.docx": {"size": 1024, "modified": 1234567890}}}
    }
    
    print("Example 3: File Operation Task")
    print("=" * 50)
    print(f"Task: {file_task['task']}")
    
    file_result = evaluator.evaluate_agentsynth_task(file_task, file_trajectory)
    print(f"Task Type: {file_result['task_analysis']['task_type']}")
    print(f"Success: {file_result['success']}")
    print(f"Verification Methods: {file_result['verification_methods']}")
    print()
    
    return {
        'web_navigation': web_result,
        'data_extraction': data_result,
        'file_operation': file_result
    }

def main():
    """Main function."""
    
    print("AgentSynth Verifiable Evaluation Demonstration")
    print("=" * 60)
    print()
    
    # Run demonstration
    results = demonstrate_verifiable_evaluation()
    
    # Summary
    print("SUMMARY")
    print("=" * 60)
    total_tasks = len(results)
    successful_tasks = sum(1 for result in results.values() if result['success'])
    
    print(f"Total tasks evaluated: {total_tasks}")
    print(f"Successful tasks: {successful_tasks}")
    print(f"Success rate: {successful_tasks/total_tasks:.2%}")
    print()
    
    print("Key advantages of verifiable evaluation:")
    print("✓ No LLM judges - objective system state verification")
    print("✓ Fast execution - no API calls or model inference")
    print("✓ 100% reproducible - same inputs always produce same outputs")
    print("✓ Cost-free - no LLM API costs")
    print("✓ Transparent - clear logic, debuggable results")
    print("✓ Scalable - can evaluate thousands of tasks quickly")

if __name__ == "__main__":
    main()
