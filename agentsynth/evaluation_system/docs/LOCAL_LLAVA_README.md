# Local LLaVa Integration

This document explains how to use local LLaVa models instead of OpenAI API for AgentSynth.

## Prerequisites

1. Install required dependencies:
```bash
pip install torch transformers
```

2. Ensure you have sufficient GPU memory (recommended 8GB+ VRAM) or use CPU mode.

## Configuration

### Environment Variables

Set the following environment variables to use local LLaVa:

### For LoRA Checkpoint (Recommended)
```bash
export USE_LOCAL_LLAVA=true
export LOCAL_LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf  # Base model from HF
export LOCAL_LLAVA_CHECKPOINT_PATH=/path/to/checkpoint-13  # LoRA checkpoint directory
export LOCAL_LLAVA_DEVICE=cuda  # or 'cpu' if no GPU
```

### For Base Model Only
```bash
export USE_LOCAL_LLAVA=true
export LOCAL_LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf  # Base model from HF
export LOCAL_LLAVA_DEVICE=cuda  # or 'cpu' if no GPU
```

### Using the Switch Script

Use the provided script to easily switch between models:

```bash
# Switch to local LLaVa
python switch_model.py local

# Switch to OpenAI
python switch_model.py openai

# Check current configuration
python switch_model.py status
```

## Model Path Types

The system supports two types of model paths:

### 1. Local Model Directory
- **Path**: Absolute path to a local directory containing the model files
- **Example**: `/home/user/models/llava-1.6-mistral-7b-hf`
- **Use case**: When you have downloaded the model locally
- **Advantage**: Faster loading, no internet required

### 2. Hugging Face Model Name
- **Path**: Hugging Face model identifier
- **Example**: `llava-hf/llava-1.5-7b-hf`
- **Use case**: When you want to download from Hugging Face
- **Advantage**: Always get the latest version

## Supported Models

The following LLaVa models are supported:

- `llava-hf/llava-1.5-7b-hf` (default)
- `llava-hf/llava-1.5-7b-hf`
- `llava-hf/llava-1.5-13b-hf`
- Any other LLaVa model from Hugging Face
- Any local LLaVa model directory

### LoRA Checkpoints

The system also supports LoRA (Low-Rank Adaptation) checkpoints:

#### Example: Using checkpoint-13
```bash
# Set up your LoRA checkpoint (base model + adapter)
export USE_LOCAL_LLAVA=true
export LOCAL_LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf  # Base model from HF
export LOCAL_LLAVA_CHECKPOINT_PATH=out_llava_lora/checkpoint-13
export LOCAL_LLAVA_DEVICE=cuda

# Then run evaluation
python evaluate_osworld.py --agentsynth-hf --max-tasks 5 --verbose
```

- Set `LOCAL_LLAVA_MODEL_PATH` to the base model (e.g., `llava-hf/llava-1.5-7b-hf`)
- Set `LOCAL_LLAVA_CHECKPOINT_PATH` to your LoRA checkpoint directory (e.g., `/path/to/checkpoint-13`)
- The system will automatically load the base model and apply the LoRA adapters

## Device Configuration

- `auto`: Automatically detect CUDA availability
- `cuda`: Force GPU usage (requires CUDA)
- `cpu`: Force CPU usage (slower but works without GPU)

## Usage

Once configured, simply run your existing scripts:

```bash
python generate_and_save_traces_persona.py
```

The system will automatically use the local LLaVa model if `USE_LOCAL_LLAVA=true`.

## Performance Notes

- **GPU**: Much faster inference, requires sufficient VRAM
- **CPU**: Slower but works on any system
- First run will download the model (can take several GB)

## Limitations

1. **Computer Use Actions**: The `call_computer_use_preview` function falls back to regular text generation when using local LLaVa, as the specialized computer use API is only available through OpenAI.

2. **Model Size**: LLaVa models are large (7B+ parameters), ensure you have sufficient disk space and memory.

## Troubleshooting

### Out of Memory Errors
- Reduce batch size or use CPU mode
- Use a smaller model variant
- Close other GPU-intensive applications

### Model Loading Issues
- Check internet connection for initial download
- Verify model path is correct
- Ensure sufficient disk space

### Performance Issues
- Use GPU if available
- Consider using a smaller model
- Monitor system resources

## Example Configuration Files

### For GPU Usage
```bash
export USE_LOCAL_LLAVA=true
export LOCAL_LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf
export LOCAL_LLAVA_DEVICE=cuda
```

### For CPU Usage
```bash
export USE_LOCAL_LLAVA=true
export LOCAL_LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf
export LOCAL_LLAVA_DEVICE=cpu
```

### For LoRA Checkpoint Usage
```bash
export USE_LOCAL_LLAVA=true
export LOCAL_LLAVA_MODEL_PATH=llava-hf/llava-1.5-7b-hf
export LOCAL_LLAVA_CHECKPOINT_PATH=out_llava_lora/checkpoint-13
export LOCAL_LLAVA_DEVICE=cuda
```

### For OpenAI (default)
```bash
export USE_LOCAL_LLAVA=false
export OPENAI_API_KEY=your_api_key_here
```
