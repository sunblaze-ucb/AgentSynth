# Installation Guide for AgentSynth Evaluation System

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Access to the main [AgentSynth repository](https://github.com/sunblaze-ucb/AgentSynth)

## System Dependencies

### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libopencv-dev \
    python3-opencv \
    firefox \
    chromium-browser

# Install Chrome/Chromium for Selenium
wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt-get update
sudo apt-get install -y google-chrome-stable
```

### macOS
```bash
# Install system dependencies using Homebrew
brew install tesseract opencv
brew install --cask google-chrome
```

### Windows
```bash
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR

# Install Chrome
# Download from: https://www.google.com/chrome/
```

## Python Dependencies

### Install Evaluation System Dependencies
```bash
# From the main AgentSynth directory
cd agentsynth/evaluation_system
pip install -r requirements.txt
```

## Environment Setup

### 1. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Configure Tesseract (if needed)
```bash
# Find tesseract path
which tesseract

# Set environment variable if needed
export TESSDATA_PREFIX="/usr/share/tesseract-ocr/4.00/tessdata"
```

### 3. Verify Installation
```bash
cd evaluation_system/scripts
python3 run_agentsynth_evaluation.py --help
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   ```

2. **OpenCV import error**
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

3. **Selenium WebDriver issues**
   ```bash
   # Install ChromeDriver
   pip install webdriver-manager
   ```

4. **Permission errors**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.py
   ```

### Testing Dependencies

```bash
cd evaluation_system
python3 -c "
import cv2
import numpy as np
import pytesseract
from PIL import Image
import psutil
import requests
from selenium import webdriver
print('All dependencies installed successfully!')
"
```

## Usage

Once installed, you can use the evaluation system:

```bash
# Generate evaluation functions for all 23 dataset files
cd evaluation_system/scripts
python3 run_agentsynth_evaluation.py --comprehensive --max-tasks 10

# Generate for a single file
python3 run_agentsynth_evaluation.py --single-file ../../oai_data_files/openai_finetune_per_action_part_001.jsonl --max-tasks 5
```

## Notes

- The evaluation system depends on the main AgentSynth repository's `utils.py` for LLM integration
- All paths in the scripts are relative to the evaluation_system directory
- The system automatically extracts screenshots and task descriptions from the AgentSynth dataset format
