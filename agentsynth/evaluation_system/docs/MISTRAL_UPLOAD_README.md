# Mistral Fine-tuning Upload Guide

This guide explains how to use the modified scripts to create training files and upload them to Mistral API for fine-tuning.

## Overview

The scripts have been modified to:
1. **Split large datasets into 512MB chunks** to comply with Mistral's file size limits
2. **Upload files to Mistral API** for fine-tuning
3. **Monitor fine-tuning job progress**

## Files

- `convert_to_oai_training_file.py` - Modified to create multiple files under 512MB
- `upload_to_mistral.py` - Uploads training files to Mistral API
- `monitor_mistral_jobs.py` - Monitors fine-tuning job status

## Setup

1. **Set your Mistral API key:**
   ```bash
   export MISTRAL_API_KEY="your_mistral_api_key_here"
   ```

2. **Install required dependencies:**
   ```bash
   pip install requests tqdm huggingface_hub
   ```

## Usage

### Step 1: Generate Training Files

Run the modified conversion script to create training files:

```bash
python convert_to_oai_training_file.py
```

This will:
- Download the AgentSynth dataset from Hugging Face
- Convert it to OpenAI-compatible format
- Split into multiple files, each under 512MB
- Save files in the `oai_data_files/` directory

**Output example:**
```
oai_data_files/
├── openai_finetune_per_action_part_000.jsonl  (e.g., 480MB)
├── openai_finetune_per_action_part_001.jsonl  (e.g., 450MB)
├── openai_finetune_per_action_part_002.jsonl  (e.g., 320MB)
└── ...
```

### Step 2: Upload to Mistral

Upload the training files to Mistral API:

```bash
python upload_to_mistral.py
```

This will:
- Validate file sizes (must be ≤ 512MB)
- Upload each file to Mistral
- Create fine-tuning jobs for each file
- Save job information to `oai_data_files/mistral_jobs.json`

**Example output:**
```
Found 3 JSONL files:
  - oai_data_files/openai_finetune_per_action_part_000.jsonl
  - oai_data_files/openai_finetune_per_action_part_001.jsonl
  - oai_data_files/openai_finetune_per_action_part_002.jsonl

Validating file sizes...
✓ oai_data_files/openai_finetune_per_action_part_000.jsonl: 480.25 MB
✓ oai_data_files/openai_finetune_per_action_part_001.jsonl: 450.12 MB
✓ oai_data_files/openai_finetune_per_action_part_002.jsonl: 320.45 MB

Uploading 3 files...
✓ File uploaded successfully: file-abc123
✓ File uploaded successfully: file-def456
✓ File uploaded successfully: file-ghi789

✓ Successfully created 3 fine-tuning jobs
```

### Step 3: Monitor Jobs

Check the status of your fine-tuning jobs:

```bash
# Single status check
python monitor_mistral_jobs.py

# Continuous monitoring (checks every 60 seconds)
python monitor_mistral_jobs.py --continuous

# List all jobs from API
python monitor_mistral_jobs.py --list-all
```

## Configuration

### File Size Limits

The script automatically splits files to stay under 512MB. You can modify this in `convert_to_oai_training_file.py`:

```python
MAX_FILE_SIZE_BYTES = 512 * 1024 * 1024  # 512 MB
```

### Fine-tuning Parameters

You can customize fine-tuning parameters in `upload_to_mistral.py`:

```python
job_data = uploader.create_fine_tuning_job(
    training_file_id=file_info["file_id"],
    model="mistral-tiny",  # or "mistral-small", "mistral-medium"
    hyperparameters={
        "n_epochs": 3,           # Number of training epochs
        "learning_rate": 1e-5,   # Learning rate
    }
)
```

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   ❌ Error: MISTRAL_API_KEY environment variable not set!
   ```
   **Solution:** Set your API key: `export MISTRAL_API_KEY="your_key"`

2. **File Too Large**
   ```
   ⚠️  Warning: file.jsonl is 600.25 MB (exceeds 512 MB limit)
   ```
   **Solution:** The script should automatically handle this, but check your `MAX_FILE_SIZE_BYTES` setting.

3. **Upload Failures**
   ```
   ❌ Failed to upload file.jsonl: 401 Unauthorized
   ```
   **Solution:** Check your API key and ensure it has the correct permissions.

### Monitoring Jobs

- Jobs typically take several hours to complete
- Use `--continuous` flag to monitor progress automatically
- Check the `mistral_jobs.json` file for job IDs and status

### File Management

- Training files are large (several GB total)
- Consider cleaning up old files after successful uploads
- The `oai_data_files/` directory contains all generated files

## API Limits

- **File size limit:** 512MB per file
- **Rate limits:** Check Mistral's current API documentation
- **Concurrent jobs:** Mistral may limit concurrent fine-tuning jobs

## Next Steps

After fine-tuning completes:
1. Use the fine-tuned model ID from the job status
2. Test the model with your specific use cases
3. Consider creating additional training files if needed
4. Monitor model performance and iterate

## Support

For issues with:
- **Mistral API:** Check [Mistral's documentation](https://docs.mistral.ai/)
- **Script errors:** Check the error messages and ensure all dependencies are installed
- **File processing:** Verify you have sufficient disk space (>40GB recommended)
