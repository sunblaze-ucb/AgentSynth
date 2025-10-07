# AgentSynth Evaluation System

This directory contains the complete evaluation system for AgentSynth tasks, including verifiable evaluation functions, generation scripts, and documentation.

## Quick Setup

Since this is part of the main [AgentSynth repository](https://github.com/sunblaze-ucb/AgentSynth), you only need to install the additional dependencies:

```bash
# Install evaluation system dependencies
pip install -r evaluation_system/requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Generate Evaluation Functions for All 23 Files

```bash
cd evaluation_system/scripts
python3 run_agentsynth_evaluation.py --comprehensive --max-tasks 10
```

### Generate for a Single File

```bash
python3 run_agentsynth_evaluation.py --single-file ../../oai_data_files/openai_finetune_per_action_part_001.jsonl --max-tasks 5
```

## Directory Structure

```
evaluation_system/
├── core/                           # Core evaluation modules
│   ├── generate_evaluation_functions.py    # Main function generator
│   ├── integrated_verifiable_evaluation.py # Integrated evaluation system
│   ├── verification_tools.py               # Verifiable evaluation tools
│   ├── automatic_evaluator.py              # Automatic evaluation logic
│   ├── verifiable_evaluator.py             # Verifiable evaluator
│   └── example_verifiable_evaluation.py    # Example usage
├── scripts/                        # Utility scripts
│   ├── run_agentsynth_evaluation.py        # Main runner for all files
│   ├── run_evaluation_generator.py         # Evaluation generator
│   ├── run_evaluation.py                   # Single evaluation runner
│   ├── batch_evaluate.py                   # Batch evaluation
│   ├── fix_existing_descriptions.py        # Fix existing descriptions
│   ├── fix_task_descriptions.py            # Fix task descriptions
│   ├── comprehensive_fix_descriptions.py   # Comprehensive fix
│   └── regenerate_with_descriptions.py     # Regenerate with descriptions
├── docs/                           # Documentation
│   ├── EVALUATION_README.md                # Main evaluation documentation
│   ├── README_EVALUATION.md                # Evaluation guide
│   ├── LOCAL_LLAVA_README.md               # Local LLaVA setup
│   └── MISTRAL_UPLOAD_README.md            # Mistral upload guide
├── generated/                      # Generated evaluation functions
│   └── generated_evaluation_functions.py   # Generated functions
└── test/                          # Test files
    ├── test_evaluation_generation.py       # Test generation
    └── test_output/                        # Test outputs
```

## Key Features

- **Verifiable Evaluation**: Uses programmatic checks instead of LLM judges
- **Real Task Descriptions**: Extracts actual task descriptions from AgentSynth dataset
- **Comprehensive Coverage**: Processes all 23 dataset files
- **Multiple Verification Methods**: Screenshot analysis, web automation, file system checks, etc.
- **Batch Processing**: Handles large datasets efficiently

## Dependencies

The evaluation system uses the existing AgentSynth `utils.py` and adds these external packages:

- **Computer Vision**: OpenCV, pytesseract, Pillow, numpy
- **Web Automation**: Selenium
- **System Monitoring**: psutil
- **HTTP Requests**: requests
- **LLM Integration**: openai

## Usage Examples

### Basic Evaluation
```python
from evaluation_system.core.generate_evaluation_functions import EvaluationFunctionGenerator

generator = EvaluationFunctionGenerator()
result = generator.analyze_task_and_generate_evaluator(task_data, task_id)
```

### Using Generated Functions
```python
from evaluation_system.generated.generated_evaluation_functions import evaluate_task

result = evaluate_task(task_id, task_data, agent_trajectory)
print(f"Success: {result['success']}")
```

## Configuration

- **Dataset Directory**: `../../oai_data_files/` (23 JSONL files)
- **Output Directory**: `../generated/`
- **Default Model**: `gpt-4o`
- **Max Tasks per File**: 5 (configurable)

## Notes

- The system extracts real task descriptions from the AgentSynth dataset format
- Generated functions use verifiable methods instead of LLM judges
- All paths are relative to the evaluation_system directory
- The system handles the large AgentSynth dataset files efficiently
- Requires the main AgentSynth repository's `utils.py` for LLM integration