#!/usr/bin/env python3
"""
Verifiable Evaluation Functions for AgentSynth Tasks

This module provides programmatic, objective evaluation methods that don't rely on LLM judges.
These functions can definitively determine task completion success through system state verification.
"""

import os
import json
import subprocess
import time
import re
import sqlite3
import psutil
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import requests
from urllib.parse import urlparse
import hashlib
import difflib

class VerifiableTaskEvaluator:
    """
    Programmatic evaluation system that verifies task completion through objective system checks.
    No LLM judges - only verifiable system state changes.
    """
    
    def __init__(self, base_dir: str = "/tmp/agentsynth_eval"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.evaluation_log = []
    
    def evaluate_task_completion(
        self, 
        task_description: str,
        task_type: str,
        expected_outcome: Dict[str, Any],
        system_state_before: Optional[Dict[str, Any]] = None,
        system_state_after: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate task completion using verifiable system state checks.
        
        Args:
            task_description: The task instruction
            task_type: Type of task (file_creation, web_navigation, etc.)
            expected_outcome: Expected system state changes
            system_state_before: System state before task execution
            system_state_after: System state after task execution
            
        Returns:
            Verifiable evaluation results
        """
        
        evaluation_result = {
            'task_description': task_description,
            'task_type': task_type,
            'timestamp': time.time(),
            'success': False,
            'verification_methods': [],
            'details': {}
        }
        
        # Route to appropriate verification method
        if task_type == 'file_creation':
            result = self._verify_file_creation(expected_outcome, system_state_before, system_state_after)
        elif task_type == 'file_modification':
            result = self._verify_file_modification(expected_outcome, system_state_before, system_state_after)
        elif task_type == 'web_navigation':
            result = self._verify_web_navigation(expected_outcome, system_state_before, system_state_after)
        elif task_type == 'application_launch':
            result = self._verify_application_launch(expected_outcome, system_state_before, system_state_after)
        elif task_type == 'database_operation':
            result = self._verify_database_operation(expected_outcome, system_state_before, system_state_after)
        elif task_type == 'system_configuration':
            result = self._verify_system_configuration(expected_outcome, system_state_before, system_state_after)
        else:
            result = self._verify_generic_task(expected_outcome, system_state_before, system_state_after)
        
        evaluation_result.update(result)
        self.evaluation_log.append(evaluation_result)
        
        return evaluation_result
    
    def _verify_file_creation(
        self, 
        expected_outcome: Dict[str, Any], 
        state_before: Optional[Dict[str, Any]], 
        state_after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify file creation tasks."""
        
        verification_result = {
            'success': True,
            'verification_methods': ['file_existence', 'file_content', 'file_metadata'],
            'details': {}
        }
        
        # Check 1: File existence
        file_path = expected_outcome.get('file_path')
        if file_path:
            file_exists = os.path.exists(file_path)
            verification_result['details']['file_exists'] = file_exists
            if not file_exists:
                verification_result['success'] = False
                return verification_result
        
        # Check 2: File content verification
        expected_content = expected_outcome.get('expected_content')
        if expected_content and file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    actual_content = f.read()
                
                content_matches = expected_content in actual_content
                verification_result['details']['content_matches'] = content_matches
                verification_result['details']['content_similarity'] = self._calculate_text_similarity(
                    expected_content, actual_content
                )
                
                if not content_matches:
                    verification_result['success'] = False
                    
            except Exception as e:
                verification_result['details']['content_error'] = str(e)
                verification_result['success'] = False
        
        # Check 3: File metadata verification
        expected_size = expected_outcome.get('expected_size')
        expected_extension = expected_outcome.get('expected_extension')
        
        if file_path and os.path.exists(file_path):
            file_stat = os.stat(file_path)
            verification_result['details']['file_size'] = file_stat.st_size
            
            if expected_size and file_stat.st_size != expected_size:
                verification_result['success'] = False
            
            if expected_extension and not file_path.endswith(expected_extension):
                verification_result['success'] = False
        
        return verification_result
    
    def _verify_file_modification(
        self, 
        expected_outcome: Dict[str, Any], 
        state_before: Optional[Dict[str, Any]], 
        state_after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify file modification tasks."""
        
        verification_result = {
            'success': True,
            'verification_methods': ['file_modification_time', 'content_changes', 'backup_creation'],
            'details': {}
        }
        
        file_path = expected_outcome.get('file_path')
        if not file_path or not os.path.exists(file_path):
            verification_result['success'] = False
            return verification_result
        
        # Check 1: Modification time
        if state_before and 'file_modification_times' in state_before:
            before_time = state_before['file_modification_times'].get(file_path)
            current_time = os.path.getmtime(file_path)
            
            if before_time and current_time > before_time:
                verification_result['details']['file_modified'] = True
            else:
                verification_result['details']['file_modified'] = False
                verification_result['success'] = False
        
        # Check 2: Content changes
        expected_changes = expected_outcome.get('expected_changes')
        if expected_changes:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_content = f.read()
                
                changes_found = []
                for change in expected_changes:
                    if change in current_content:
                        changes_found.append(change)
                
                verification_result['details']['changes_found'] = changes_found
                verification_result['details']['changes_missing'] = [
                    change for change in expected_changes if change not in changes_found
                ]
                
                if len(changes_found) < len(expected_changes):
                    verification_result['success'] = False
                    
            except Exception as e:
                verification_result['details']['content_error'] = str(e)
                verification_result['success'] = False
        
        return verification_result
    
    def _verify_web_navigation(
        self, 
        expected_outcome: Dict[str, Any], 
        state_before: Optional[Dict[str, Any]], 
        state_after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify web navigation tasks."""
        
        verification_result = {
            'success': True,
            'verification_methods': ['url_verification', 'page_content', 'browser_state'],
            'details': {}
        }
        
        # Check 1: URL verification
        expected_url = expected_outcome.get('expected_url')
        if expected_url:
            # This would require browser automation to get current URL
            # For now, we'll check if the URL is accessible
            try:
                response = requests.head(expected_url, timeout=10)
                verification_result['details']['url_accessible'] = response.status_code == 200
                verification_result['details']['response_code'] = response.status_code
            except Exception as e:
                verification_result['details']['url_error'] = str(e)
                verification_result['success'] = False
        
        # Check 2: Page content verification
        expected_content = expected_outcome.get('expected_content')
        if expected_content and expected_url:
            try:
                response = requests.get(expected_url, timeout=10)
                content_found = expected_content.lower() in response.text.lower()
                verification_result['details']['content_found'] = content_found
                
                if not content_found:
                    verification_result['success'] = False
                    
            except Exception as e:
                verification_result['details']['content_error'] = str(e)
                verification_result['success'] = False
        
        # Check 3: Browser state (if available)
        if state_after and 'browser_state' in state_after:
            browser_state = state_after['browser_state']
            verification_result['details']['browser_state'] = browser_state
        
        return verification_result
    
    def _verify_application_launch(
        self, 
        expected_outcome: Dict[str, Any], 
        state_before: Optional[Dict[str, Any]], 
        state_after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify application launch tasks."""
        
        verification_result = {
            'success': True,
            'verification_methods': ['process_check', 'window_detection', 'port_listening'],
            'details': {}
        }
        
        expected_app = expected_outcome.get('application_name')
        if not expected_app:
            verification_result['success'] = False
            return verification_result
        
        # Check 1: Process verification
        running_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if expected_app.lower() in proc.info['name'].lower():
                    running_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        verification_result['details']['running_processes'] = running_processes
        if not running_processes:
            verification_result['success'] = False
        
        # Check 2: Window detection (simplified)
        # This would require platform-specific window management
        verification_result['details']['window_detection'] = 'not_implemented'
        
        # Check 3: Port listening (for server applications)
        expected_port = expected_outcome.get('expected_port')
        if expected_port:
            port_open = self._check_port_listening(expected_port)
            verification_result['details']['port_listening'] = port_open
            if not port_open:
                verification_result['success'] = False
        
        return verification_result
    
    def _verify_database_operation(
        self, 
        expected_outcome: Dict[str, Any], 
        state_before: Optional[Dict[str, Any]], 
        state_after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify database operation tasks."""
        
        verification_result = {
            'success': True,
            'verification_methods': ['table_existence', 'data_verification', 'query_results'],
            'details': {}
        }
        
        db_path = expected_outcome.get('database_path')
        if not db_path or not os.path.exists(db_path):
            verification_result['success'] = False
            return verification_result
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check 1: Table existence
            expected_tables = expected_outcome.get('expected_tables', [])
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            verification_result['details']['existing_tables'] = existing_tables
            verification_result['details']['expected_tables'] = expected_tables
            
            missing_tables = [table for table in expected_tables if table not in existing_tables]
            if missing_tables:
                verification_result['success'] = False
                verification_result['details']['missing_tables'] = missing_tables
            
            # Check 2: Data verification
            expected_data = expected_outcome.get('expected_data')
            if expected_data:
                for table, expected_rows in expected_data.items():
                    if table in existing_tables:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        actual_count = cursor.fetchone()[0]
                        verification_result['details'][f'{table}_count'] = actual_count
                        
                        if actual_count < expected_rows:
                            verification_result['success'] = False
            
            # Check 3: Query results
            expected_queries = expected_outcome.get('expected_queries', [])
            for query_info in expected_queries:
                query = query_info['query']
                expected_result = query_info['expected_result']
                
                cursor.execute(query)
                actual_result = cursor.fetchall()
                
                if actual_result != expected_result:
                    verification_result['success'] = False
                    verification_result['details'][f'query_failed_{query[:20]}'] = {
                        'expected': expected_result,
                        'actual': actual_result
                    }
            
            conn.close()
            
        except Exception as e:
            verification_result['success'] = False
            verification_result['details']['database_error'] = str(e)
        
        return verification_result
    
    def _verify_system_configuration(
        self, 
        expected_outcome: Dict[str, Any], 
        state_before: Optional[Dict[str, Any]], 
        state_after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify system configuration tasks."""
        
        verification_result = {
            'success': True,
            'verification_methods': ['config_file_check', 'environment_variables', 'system_settings'],
            'details': {}
        }
        
        # Check 1: Configuration file changes
        config_files = expected_outcome.get('config_files', [])
        for config_info in config_files:
            file_path = config_info['path']
            expected_content = config_info.get('expected_content')
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        actual_content = f.read()
                    
                    if expected_content and expected_content not in actual_content:
                        verification_result['success'] = False
                        verification_result['details'][f'config_missing_{file_path}'] = expected_content
                        
                except Exception as e:
                    verification_result['details'][f'config_error_{file_path}'] = str(e)
                    verification_result['success'] = False
            else:
                verification_result['success'] = False
                verification_result['details'][f'config_missing_file_{file_path}'] = True
        
        # Check 2: Environment variables
        expected_env_vars = expected_outcome.get('environment_variables', {})
        for var_name, expected_value in expected_env_vars.items():
            actual_value = os.environ.get(var_name)
            verification_result['details'][f'env_{var_name}'] = actual_value
            
            if actual_value != expected_value:
                verification_result['success'] = False
        
        return verification_result
    
    def _verify_generic_task(
        self, 
        expected_outcome: Dict[str, Any], 
        state_before: Optional[Dict[str, Any]], 
        state_after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generic verification for tasks that don't fit specific categories."""
        
        verification_result = {
            'success': True,
            'verification_methods': ['generic_checks'],
            'details': {}
        }
        
        # Generic checks based on expected outcome
        if 'command_output' in expected_outcome:
            command = expected_outcome['command_output']['command']
            expected_output = expected_outcome['command_output']['expected_output']
            
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
                actual_output = result.stdout.strip()
                
                verification_result['details']['command_output'] = actual_output
                verification_result['details']['command_success'] = result.returncode == 0
                
                if expected_output not in actual_output:
                    verification_result['success'] = False
                    
            except Exception as e:
                verification_result['success'] = False
                verification_result['details']['command_error'] = str(e)
        
        return verification_result
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def _check_port_listening(self, port: int) -> bool:
        """Check if a port is listening."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
    
    def capture_system_state(self, task_id: str) -> Dict[str, Any]:
        """Capture current system state for before/after comparison."""
        
        state = {
            'timestamp': time.time(),
            'task_id': task_id,
            'files': {},
            'processes': [],
            'environment_variables': dict(os.environ),
            'file_modification_times': {}
        }
        
        # Capture file states in common directories
        common_dirs = ['/tmp', '/home', str(self.base_dir)]
        for directory in common_dirs:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            stat = os.stat(file_path)
                            state['files'][file_path] = {
                                'size': stat.st_size,
                                'modified': stat.st_mtime
                            }
                            state['file_modification_times'][file_path] = stat.st_mtime
                        except (OSError, PermissionError):
                            continue
        
        # Capture running processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                state['processes'].append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return state

# Example usage and test functions
def create_verifiable_evaluation_examples():
    """Create examples of verifiable evaluation functions for common AgentSynth tasks."""
    
    evaluator = VerifiableTaskEvaluator()
    
    # Example 1: File Creation Task
    file_creation_task = {
        'instruction': "Create a new text file called 'test.txt' with content 'Hello World'",
        'type': 'file_creation',
        'expected_outcome': {
            'file_path': '/tmp/test.txt',
            'expected_content': 'Hello World',
            'expected_extension': '.txt'
        }
    }
    
    # Example 2: Web Navigation Task
    web_navigation_task = {
        'instruction': "Navigate to example.com and verify the page loads",
        'type': 'web_navigation',
        'expected_outcome': {
            'expected_url': 'https://example.com',
            'expected_content': 'Example Domain'
        }
    }
    
    # Example 3: Application Launch Task
    app_launch_task = {
        'instruction': "Launch LibreOffice Writer",
        'type': 'application_launch',
        'expected_outcome': {
            'application_name': 'libreoffice',
            'expected_port': None  # Not a server app
        }
    }
    
    # Example 4: Database Operation Task
    db_task = {
        'instruction': "Create a SQLite database with a users table",
        'type': 'database_operation',
        'expected_outcome': {
            'database_path': '/tmp/test.db',
            'expected_tables': ['users'],
            'expected_data': {'users': 0}  # 0 rows initially
        }
    }
    
    return {
        'file_creation': file_creation_task,
        'web_navigation': web_navigation_task,
        'application_launch': app_launch_task,
        'database_operation': db_task
    }

if __name__ == "__main__":
    # Demonstrate verifiable evaluation
    evaluator = VerifiableTaskEvaluator()
    
    # Capture initial state
    initial_state = evaluator.capture_system_state("test_task")
    
    # Simulate a file creation task
    test_file_path = "/tmp/agentsynth_test.txt"
    with open(test_file_path, 'w') as f:
        f.write("Hello World - Task Completed!")
    
    # Capture final state
    final_state = evaluator.capture_system_state("test_task")
    
    # Evaluate the task
    task_result = evaluator.evaluate_task_completion(
        task_description="Create a text file with 'Hello World'",
        task_type="file_creation",
        expected_outcome={
            'file_path': test_file_path,
            'expected_content': 'Hello World',
            'expected_extension': '.txt'
        },
        system_state_before=initial_state,
        system_state_after=final_state
    )
    
    print("Verifiable Evaluation Result:")
    print(json.dumps(task_result, indent=2))
    
    # Clean up
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
