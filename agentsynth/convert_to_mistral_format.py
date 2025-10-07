#!/usr/bin/env python3
"""
Convert OpenAI-style training files to Mistral format
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

def convert_openai_to_mistral_format(openai_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAI format to Mistral format
    """
    messages = openai_data.get("messages", [])
    
    # Extract system, user, and assistant messages
    system_content = ""
    user_content = ""
    assistant_content = ""
    
    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        
        if role == "system":
            # Extract text from system message
            for item in content:
                if item.get("type") == "input_text":
                    system_content = item.get("text", "")
                    break
        elif role == "user":
            # Extract text and image from user message
            text_parts = []
            for item in content:
                if item.get("type") == "input_text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "input_image":
                    # Keep image URL as is
                    text_parts.append(f"[IMAGE: {item.get('image_url', '')}]")
            user_content = " ".join(text_parts)
        elif role == "assistant":
            # Extract text from assistant message
            for item in content:
                if item.get("type") == "output_text":
                    assistant_content = item.get("text", "")
                    break
    
    # Create Mistral format
    mistral_format = {
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user", 
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ]
    }
    
    return mistral_format

def convert_file(input_file: str, output_file: str):
    """Convert a single JSONL file from OpenAI to Mistral format"""
    print(f"Converting {input_file} to {output_file}")
    
    converted_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the OpenAI format
                openai_data = json.loads(line.strip())
                
                # Convert to Mistral format
                mistral_data = convert_openai_to_mistral_format(openai_data)
                
                # Write the converted data
                outfile.write(json.dumps(mistral_data, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
                continue
    
    print(f"Converted {converted_count} lines, {error_count} errors")

def main():
    """Convert all training files to Mistral format"""
    input_dir = "oai_data_files"
    output_dir = "mistral_data_files"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all input files
    input_files = list(Path(input_dir).glob("*.jsonl"))
    
    if not input_files:
        print(f"No JSONL files found in {input_dir}")
        return
    
    print(f"Found {len(input_files)} files to convert")
    
    for input_file in input_files:
        # Create output filename
        output_file = Path(output_dir) / f"mistral_{input_file.name}"
        
        # Convert the file
        convert_file(str(input_file), str(output_file))
    
    print(f"Conversion complete! Files saved to {output_dir}")

if __name__ == "__main__":
    main()
