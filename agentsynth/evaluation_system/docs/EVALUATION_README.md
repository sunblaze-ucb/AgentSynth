# OSWorld Evaluation System

This directory contains a comprehensive evaluation system for testing models on OSWorld tasks using the AgentSynth framework. The system leverages existing utilities from `utils.py` to minimize redundancies and provide consistent evaluation across different models.

## Files Overview

### Core Evaluation Scripts
- **`evaluate_osworld.py`** - Main evaluation script that runs models on OSWorld tasks
- **`run_evaluation.py`** - Simple interactive runner for single evaluations
- **`batch_evaluate.py`** - Batch evaluation script for comparing multiple models
- **`task_generator.py`** - Tool for generating custom OSWorld tasks

### Configuration Files
- **`osworld_config.py`** - Configuration settings for OSWorld evaluation
- **`config.py`** - General model configuration (from previous work)
- **`switch_model.py`** - Model switching utility (from previous work)

### Original AgentSynth data file schema
```json
{'messages': [
    [
        {'role': 'system', 'content': [{
            'type': 'input_text', 'text': 'You are a computer agent...'
        }]},
        {'role': 'user', 'content': [
            {'type': 'input_text', 'text': <task description, past actions, etc>},
            {'type': 'input_image', 'text': <base64 screenshot>}
        ]},
        {'role': 'assistant', 'content': [{
            'type': 'input_text', 'text': '\{"thoughts": "[etc]"\}'
        }]}
    ]
]}
```

## Quick Start

### 1. Single Evaluation
```bash
# Interactive evaluation
python run_evaluation.py

# Direct evaluation with sample tasks
python evaluate_osworld.py --sample --verbose

# Evaluation with official AgentSynth Hugging Face dataset
python evaluate_osworld.py --agentsynth-hf --max-tasks 5 --verbose

# Evaluation with local dataset tasks (JSONL format)
python evaluate_osworld.py --dataset ../insta_data/task_sequences.jsonl --max-tasks 10

# Evaluation with custom tasks
python evaluate_osworld.py --tasks path/to/tasks.json --output results/
```

### 2. Batch Evaluation
```bash
# Compare multiple models
python batch_evaluate.py
```

### 3. Generate Custom Tasks
```bash
# Generate all task types
python task_generator.py --category all --output my_tasks.json

# Generate specific category
python task_generator.py --category web_navigation --output web_tasks.json
```

## Configuration

### Environment Variables

Set these environment variables to configure the evaluation:

```bash
# Model Configuration
export USE_LOCAL_LLAVA=true                    # Use local LLaVa instead of OpenAI
export LOCAL_LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf
export LOCAL_LLAVA_DEVICE=cuda                 # or 'cpu', 'auto'

# Evaluation Settings
export DEFAULT_MAX_STEPS=10                    # Maximum steps per task
export DEFAULT_TIMEOUT=300                     # Timeout per task (seconds)
export DEFAULT_OUTPUT_DIR=osworld_evaluation_results

# Performance
export PARALLEL_EVALUATION=false               # Enable parallel evaluation
export MAX_PARALLEL_TASKS=1                    # Number of parallel tasks

# Logging
export VERBOSE_LOGGING=true                    # Enable verbose output
export SAVE_SCREENSHOTS=true                   # Save screenshots
export SAVE_ACTIONS=true                       # Save action history
```

### Configuration File

Edit `osworld_config.py` to customize default settings:

```python
# Task categories to evaluate
TASK_CATEGORIES = [
    'web_navigation',
    'file_management', 
    'text_editing',
    'email_management',
    'calendar_management'
]

# Success criteria
SUCCESS_THRESHOLD = 0.8                        # 80% success rate threshold
MIN_STEPS_FOR_SUCCESS = 1
MAX_STEPS_FOR_SUCCESS = 20
```

## Task Format

### Custom Tasks (JSON format)
Tasks are defined in JSON format with the following structure:

```json
{
    "id": "task_001",
    "instruction": "Navigate to google.com and search for 'machine learning'",
    "config": {
        "applications": ["browser"],
        "setup": []
    },
    "evaluator": {
        "type": "url_check",
        "expected_url_contains": "google.com/search"
    }
}
```

### Dataset Tasks (JSONL format)
The evaluation system also supports tasks from the actual dataset in JSONL format:

```json
{
    "task": "On joblo.com, find and watch the official trailer for the John Madden biopic featuring Nicolas Cage and Christian Bale.",
    "website": "joblo.com",
    "action_sequence": [
        {"action_key": "click", "action_kwargs": {}, "target_element_id": 204},
        {"action_key": "stop", "action_kwargs": {"answer": "The first look at Nicolas Cage and Christian Bale in the John Madden biopic has been viewed on joblo.com."}, "target_element_id": null}
    ],
    "thoughts_sequence": ["We located the article headline...", "Now viewing the article..."],
    "webpage_text": ["JoBlo - Movie News, Latest Trailers, and More..."]
}
```

### Supported Evaluator Types
- **`url_check`** - Check if URL contains expected text
- **`file_exists`** - Check if file/folder exists at path
- **`file_content`** - Check if file contains expected content
- **`email_sent`** - Check if email was sent to recipient
- **`calendar_event`** - Check if calendar event was created
- **`answer_check`** - Check if expected answer appears in model output (for dataset tasks)

## Evaluation Process

The evaluation system follows this process:

1. **Task Loading** - Load tasks from JSON files or use sample tasks
2. **Environment Setup** - Initialize desktop environment (if available)
3. **Task Execution** - Run each task using AgentSynth methods:
   - `generate_action()` - Generate next action
   - `generate_computer_use_action()` - Convert to executable command
   - `generate_key_info()` - Extract key information
4. **Success Evaluation** - Use verifier methods:
   - `generate_verifier_verdict_key_info()` - Comprehensive evaluation
   - `generate_verifier()` - Basic success check
5. **Results Collection** - Save results with metrics and screenshots

## Output Format

Evaluation results are saved as JSON with the following structure:

```json
{
    "config": {
        "model": {"name": "local-llava", "use_local_llava": true},
        "evaluation": {"max_steps": 10, "timeout": 300}
    },
    "results": [
        {
            "task_id": "task_001",
            "instruction": "Navigate to google.com...",
            "success": true,
            "steps_taken": 5,
            "success_rate": 0.95,
            "verifier_thoughts": "Task completed successfully...",
            "thoughts_history": [...],
            "action_history": [...],
            "command_history": [...],
            "timestamp": "2024-01-01T12:00:00"
        }
    ],
    "summary": {
        "total_tasks": 10,
        "successful_tasks": 8,
        "success_rate": 0.8,
        "average_steps": 6.2,
        "model_used": "local-llava"
    }
}
```

## Model Support

### Local LLaVa Models
- **llava-hf/llava-1.5-7b-hf** (default)
- **llava-hf/llava-1.5-7b-hf**
- **llava-hf/llava-1.5-13b-hf**
- Any other LLaVa model from Hugging Face

### OpenAI Models
- **gpt-4.1** (default)
- **gpt-4o**
- **gpt-3.5-turbo**

## Performance Tips

### For Local LLaVa
- Use GPU if available (set `LOCAL_LLAVA_DEVICE=cuda`)
- Ensure sufficient VRAM (8GB+ recommended)
- Use smaller models for faster inference

### For OpenAI
- Monitor API usage and costs
- Use appropriate model for task complexity
- Consider rate limiting for batch evaluations

## Troubleshooting

### Common Issues

1. **Desktop Environment Not Available**
   - Install `desktop-env` package
   - Ensure Docker is running
   - Check system requirements

2. **Model Loading Errors**
   - Verify model path is correct
   - Check available disk space
   - Ensure sufficient memory/VRAM

3. **Task Execution Failures**
   - Check task JSON format
   - Verify evaluator configuration
   - Review error logs

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
export VERBOSE_LOGGING=true
python evaluate_osworld.py --sample --verbose
```

## Examples

### Example 1: Quick Test
```bash
# Test with sample tasks using local LLaVa
export USE_LOCAL_LLAVA=true
python evaluate_osworld.py --sample
```

### Example 2: Official AgentSynth Dataset Evaluation
```bash
# Evaluate on official AgentSynth Hugging Face dataset (highest level tasks)
python evaluate_osworld.py --agentsynth-hf --max-tasks 5 --verbose

# Note: AgentSynth tasks automatically get 6x extended time limits
```

### Example 3: Local Dataset Task Evaluation
```bash
# Evaluate on local dataset tasks
python evaluate_osworld.py --dataset ../insta_data/task_sequences.jsonl --max-tasks 5 --verbose

# Evaluate on summarized dataset
python evaluate_osworld.py --dataset ../insta_data/summarized_task_sequences.jsonl --max-tasks 10
```

### Example 4: Model Comparison
```bash
# Run batch evaluation comparing models
python batch_evaluate.py
```

### Example 4: Configuration Override
```bash
# Override default settings
python evaluate_osworld.py --sample --max-steps 15 --output custom_results/
```

## OSWorld Agent Interface

The system includes a complete OSWorld agent implementation (`osworld_agent.py`) that follows the [OSWorld agent interface requirements](https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/README.md):

### Agent Features
- **Standard OSWorld Interface**: Implements the required `reset()` and `step()` methods
- **AgentSynth Integration**: Uses existing utilities for action generation and execution
- **Verification Support**: Optional verification capabilities with key information generation
- **Flexible Configuration**: Supports different models and step limits

### Usage
```python
from osworld_agent import create_osworld_agent

# Create agent
agent = create_osworld_agent(
    model_name='local-llava',
    max_steps=15,
    enable_verification=True
)

# Reset for new task
agent.reset(task_config)

# Execute steps
result = agent.step(observation)
```

## Integration with Existing Code

The evaluation system is designed to work seamlessly with existing AgentSynth utilities:

- **Reuses existing functions** from `utils.py` to avoid code duplication
- **Maintains compatibility** with existing model configurations
- **Extends functionality** without breaking existing workflows
- **Provides consistent interface** across different model types
- **OSWorld Compliance**: Full compatibility with OSWorld agent interface requirements

## Contributing

To add new task types or evaluators:

1. **Add task template** in `task_generator.py`
2. **Implement evaluator logic** in `evaluate_osworld.py`
3. **Update configuration** in `osworld_config.py`
4. **Add tests** and documentation

## License

This evaluation system follows the same license as the main AgentSynth project.
