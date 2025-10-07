"""
OSWorld Evaluation Configuration

This file contains configuration options for OSWorld evaluation
using the AgentSynth framework.
"""

import os
from typing import Dict, Any, List

# Model Configuration
EVALUATION_MODEL = os.getenv('EVALUATION_MODEL', 'auto')  # 'auto', 'local-llava', 'gpt-4.1', etc.
USE_LOCAL_LLAVA = os.getenv('USE_LOCAL_LLAVA', 'false').lower() == 'true'

# Evaluation Settings
DEFAULT_MAX_STEPS = int(os.getenv('DEFAULT_MAX_STEPS', '10'))
DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '300'))  # 5 minutes per task
DEFAULT_OUTPUT_DIR = os.getenv('DEFAULT_OUTPUT_DIR', 'osworld_evaluation_results')

# OSWorld Specific Settings
OSWORLD_TASKS_DIR = os.getenv('OSWORLD_TASKS_DIR', 'osworld_tasks')
OSWORLD_BENCHMARK_VERSION = os.getenv('OSWORLD_BENCHMARK_VERSION', 'latest')

# Evaluation Metrics
ENABLE_DETAILED_METRICS = os.getenv('ENABLE_DETAILED_METRICS', 'true').lower() == 'true'
SAVE_SCREENSHOTS = os.getenv('SAVE_SCREENSHOTS', 'true').lower() == 'true'
SAVE_ACTIONS = os.getenv('SAVE_ACTIONS', 'true').lower() == 'true'

# Performance Settings
PARALLEL_EVALUATION = os.getenv('PARALLEL_EVALUATION', 'false').lower() == 'true'
MAX_PARALLEL_TASKS = int(os.getenv('MAX_PARALLEL_TASKS', '1'))

# Logging
VERBOSE_LOGGING = os.getenv('VERBOSE_LOGGING', 'true').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Task Categories to Evaluate
TASK_CATEGORIES = [
    'web_navigation',
    'file_management', 
    'text_editing',
    'email_management',
    'calendar_management',
    'browser_automation',
    'system_administration'
]

# Skip certain task types (if needed)
SKIP_TASK_TYPES = os.getenv('SKIP_TASK_TYPES', '').split(',')
SKIP_TASK_TYPES = [t.strip() for t in SKIP_TASK_TYPES if t.strip()]

# Evaluation Criteria
SUCCESS_THRESHOLD = float(os.getenv('SUCCESS_THRESHOLD', '0.8'))  # 80% success rate threshold
MIN_STEPS_FOR_SUCCESS = int(os.getenv('MIN_STEPS_FOR_SUCCESS', '1'))
MAX_STEPS_FOR_SUCCESS = int(os.getenv('MAX_STEPS_FOR_SUCCESS', '20'))

def get_evaluation_config() -> Dict[str, Any]:
    """Get complete evaluation configuration"""
    return {
        'model': {
            'name': EVALUATION_MODEL,
            'use_local_llava': USE_LOCAL_LLAVA
        },
        'evaluation': {
            'max_steps': DEFAULT_MAX_STEPS,
            'timeout': DEFAULT_TIMEOUT,
            'output_dir': DEFAULT_OUTPUT_DIR,
            'success_threshold': SUCCESS_THRESHOLD,
            'min_steps': MIN_STEPS_FOR_SUCCESS,
            'max_steps': MAX_STEPS_FOR_SUCCESS
        },
        'osworld': {
            'tasks_dir': OSWORLD_TASKS_DIR,
            'benchmark_version': OSWORLD_BENCHMARK_VERSION,
            'task_categories': TASK_CATEGORIES,
            'skip_types': SKIP_TASK_TYPES
        },
        'performance': {
            'parallel': PARALLEL_EVALUATION,
            'max_parallel': MAX_PARALLEL_TASKS
        },
        'logging': {
            'verbose': VERBOSE_LOGGING,
            'level': LOG_LEVEL,
            'save_screenshots': SAVE_SCREENSHOTS,
            'save_actions': SAVE_ACTIONS,
            'detailed_metrics': ENABLE_DETAILED_METRICS
        }
    }

def print_config():
    """Print current configuration"""
    config = get_evaluation_config()
    
    print("=" * 60)
    print("OSWorld Evaluation Configuration")
    print("=" * 60)
    
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    
    print("=" * 60)

# Example task configurations for different categories
SAMPLE_TASKS = {
    'web_navigation': {
        "id": "web_nav_001",
        "instruction": "Navigate to google.com and search for 'machine learning'",
        "config": {
            "applications": ["browser"],
            "setup": []
        },
        "evaluator": {
            "type": "url_check",
            "expected_url_contains": "google.com/search"
        }
    },
    
    'file_management': {
        "id": "file_mgmt_001", 
        "instruction": "Create a new folder called 'test_folder' on the desktop",
        "config": {
            "applications": ["file_manager"],
            "setup": []
        },
        "evaluator": {
            "type": "file_exists",
            "expected_path": "~/Desktop/test_folder"
        }
    },
    
    'text_editing': {
        "id": "text_edit_001",
        "instruction": "Open a text editor and write 'Hello OSWorld'",
        "config": {
            "applications": ["text_editor"],
            "setup": []
        },
        "evaluator": {
            "type": "file_content",
            "expected_content": "Hello OSWorld"
        }
    }
}

if __name__ == "__main__":
    print_config()
