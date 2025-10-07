#!/usr/bin/env python3
"""
Reformat all training files using the official Mistral reformat_data.py script
"""

import os
import subprocess
from pathlib import Path

def reformat_all_files():
    """Reformat all JSONL files in the oai_data_files directory"""
    input_dir = "mistral_data_files"
    
    # Find all JSONL files
    jsonl_files = list(Path(input_dir).glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {input_dir}")
        return
    
    print(f"Found {len(jsonl_files)} files to reformat")
    
    for file_path in jsonl_files:
        print(f"Reformatting {file_path}...")
        try:
            # Run the reformat script
            result = subprocess.run([
                "python", "reformat_data.py", str(file_path)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully reformatted {file_path}")
                if result.stdout:
                    print(f"   Output: {result.stdout.strip()}")
            else:
                print(f"❌ Error reformatting {file_path}")
                print(f"   Error: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Exception while reformatting {file_path}: {e}")
    
    print("Reformatting complete!")

if __name__ == "__main__":
    reformat_all_files()
