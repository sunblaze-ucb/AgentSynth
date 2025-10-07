# AgentSynth Verifiable Evaluation Functions

This directory contains scripts to generate **verifiable evaluation functions** for AgentSynth tasks using LLM analysis. These functions provide objective, programmatic evaluation without relying on LLM judges.

## Overview

The system generates Python evaluation functions that use:
- **Computer Vision** (OpenCV, OCR) for screenshot analysis
- **Web Automation** (Selenium) for web page verification  
- **File System** checks for file operations
- **Process Monitoring** for application state
- **Database Verification** for data changes
- **Network Testing** for API calls

## Quick Start

### 1. Test with Small Sample
```bash
cd agentsynth
python test_evaluation_generation.py
```

This processes just 2 tasks to demonstrate the system.

### 2. Process Single File
```bash
python run_agentsynth_evaluation.py --single-file oai_data_files/openai_finetune_per_action_part_001.jsonl --max-tasks 10
```

### 3. Process All Dataset Files
```bash
python run_agentsynth_evaluation.py --max-tasks 5
```

### 4. Comprehensive Evaluation
```bash
python run_agentsynth_evaluation.py --comprehensive --max-tasks 3
```

## Command Line Options

```bash
python run_agentsynth_evaluation.py [OPTIONS]

Options:
  --dataset-dir PATH     Directory with AgentSynth files (default: oai_data_files/)
  --output-dir PATH      Where to save generated functions (default: generated_evaluations/)
  --max-tasks N          Max tasks per file (default: 5)
  --model MODEL          LLM model to use (default: gpt-4o)
  --single-file PATH     Process only one file
  --comprehensive        Run with verifiable evaluation methods
```

## Generated Output

The system creates:

1. **`generated_evaluation_functions.py`** - Python functions for each task
2. **`comprehensive_results.json`** - Processing statistics and results
3. **`verification_tools.py`** - Core verification utilities

## Example Generated Function

```python
def evaluate_task_001(task_data, agent_trajectory, before_state, after_state):
    """
    Evaluate: Navigate to Amazon and search for 'wireless headphones'
    """
    verifier = VerifiableEvaluator()
    
    # Check if Amazon page was loaded
    web_verifier = WebVerifier()
    if not web_verifier.check_page_loaded("amazon.com"):
        return False, "Amazon page not loaded"
    
    # Check if search was performed
    if not web_verifier.check_element_text_contains("input", "wireless headphones"):
        return False, "Search not performed correctly"
    
    # Check if results page loaded
    if not web_verifier.check_url_contains("s?k=wireless+headphones"):
        return False, "Search results not displayed"
    
    return True, "Task completed successfully"
```

## Using Generated Functions

```python
# Import the generated functions
from generated_evaluation_functions import evaluate_task_001

# Load task data and agent trajectory
task_data = {...}  # From AgentSynth dataset
agent_trajectory = {...}  # Agent's actions and screenshots
before_state = {...}  # System state before task
after_state = {...}  # System state after task

# Evaluate the task
success, message = evaluate_task_001(task_data, agent_trajectory, before_state, after_state)
print(f"Task success: {success}")
print(f"Message: {message}")
```

## File Structure

```
agentsynth/
├── run_agentsynth_evaluation.py      # Main runner script
├── test_evaluation_generation.py     # Test with small sample
├── generate_evaluation_functions.py  # Core generation logic
├── verification_tools.py             # Verification utilities
├── integrated_verifiable_evaluation.py # Complete system
├── oai_data_files/                   # AgentSynth dataset files
│   ├── openai_finetune_per_action_part_001.jsonl
│   ├── openai_finetune_per_action_part_002.jsonl
│   └── ...
└── generated_evaluations/            # Output directory
    ├── generated_evaluation_functions.py
    ├── comprehensive_results.json
    └── ...
```

## Requirements

- Python 3.8+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- Required packages: `opencv-python`, `pytesseract`, `selenium`, `psutil`, `requests`

## Installation

```bash
pip install opencv-python pytesseract selenium psutil requests openai
```

## Troubleshooting

### Large File Processing
The AgentSynth files are 500MB+ each. For testing, use `--max-tasks 2` to process just a few tasks.

### Memory Issues
If you encounter memory issues with large files:
```bash
python run_agentsynth_evaluation.py --max-tasks 1 --single-file oai_data_files/openai_finetune_per_action_part_001.jsonl
```

### API Rate Limits
If you hit OpenAI API rate limits, use a smaller model:
```bash
python run_agentsynth_evaluation.py --model gpt-4o-mini --max-tasks 3
```

## Performance

- **Small test (2 tasks)**: ~30 seconds
- **Single file (10 tasks)**: ~2-3 minutes  
- **All files (5 tasks each)**: ~10-15 minutes
- **Comprehensive evaluation**: ~20-30 minutes

## Output Examples

The system generates evaluation functions for tasks like:
- Web navigation and search
- File operations (create, edit, delete)
- Application interactions
- Form filling and submission
- Data extraction and processing

Each function provides objective, verifiable evaluation without subjective LLM judgments.
