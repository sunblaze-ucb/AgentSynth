#!/usr/bin/env python3
"""
Test script for JSON parsing improvements.
"""

import sys
import json
import re
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

def fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues."""
    
    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix missing commas between object properties (more comprehensive)
    # Pattern: "key": value "next_key" -> "key": value, "next_key"
    json_str = re.sub(r'("\s*:\s*[^,}]+)\s+(")', r'\1, \2', json_str)
    
    # Fix missing commas between array elements
    json_str = re.sub(r'(\])\s*(\[)', r'\1, \2', json_str)
    
    # Fix missing commas between object properties on new lines
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    
    # Fix missing commas after values before closing braces
    json_str = re.sub(r'([^,}\]])\s*(\n\s*[}\]])', r'\1,\2', json_str)
    
    return json_str

def parse_llm_response(response: str) -> dict:
    """Parse LLM response with robust JSON handling."""
    
    try:
        # First try standard JSON parsing
        result = json.loads(response)
        if result:
            return result
    except Exception as e:
        print(f"Standard JSON parsing failed: {e}")
    
    # Try to extract JSON from markdown code blocks
    try:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            result = json.loads(json_str)
            if result:
                return result
    except Exception as e:
        print(f"Markdown JSON extraction failed: {e}")
    
    # Try to find and fix common JSON issues
    try:
        # Remove any text before the first {
        start_idx = response.find('{')
        if start_idx != -1:
            json_str = response[start_idx:]
            
            # Try to find the matching closing brace
            brace_count = 0
            end_idx = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx != -1:
                json_str = json_str[:end_idx]
                
                # Try to fix common issues
                json_str = fix_common_json_issues(json_str)
                
                result = json.loads(json_str)
                if result:
                    return result
    except Exception as e:
        print(f"JSON repair attempt failed: {e}")
    
    # Fallback
    return {"error": "Failed to parse JSON"}

def test_json_parsing():
    """Test various JSON parsing scenarios."""
    
    test_cases = [
        # Valid JSON
        '{"task_type": "web_navigation", "verification_methods": ["verify_web_page_state"]}',
        
        # Trailing comma
        '{"task_type": "web_navigation", "verification_methods": ["verify_web_page_state"],}',
        
        # Missing comma
        '{"task_type": "web_navigation" "verification_methods": ["verify_web_page_state"]}',
        
        # In markdown
        '```json\n{"task_type": "web_navigation", "verification_methods": ["verify_web_page_state"]}\n```',
        
        # With extra text
        'Here is the analysis:\n{"task_type": "web_navigation", "verification_methods": ["verify_web_page_state"]}\nEnd of analysis',
        
        # Complex nested JSON
        '''{
            "task_type": "web_navigation",
            "verification_methods": ["verify_web_page_state"],
            "expected_outcome": {
                "url_verification": {
                    "expected_url": "https://example.com"
                }
            }
        }''',
        
        # Malformed complex JSON (like what we saw in the terminal)
        '''{
            "task_type": "data_extraction",
            "verification_methods": [
                "verify_web_page_state",
                "analyze_screenshot_for_elements"
            ],
            "expected_outcome": {
                "url_verification": {
                    "expected_url_pattern": "https://www.amazon.com/*/dp/B0D6NSD57S*",
                    "http_status": 200,
                    "page_title_contains": "Sunrise on the Reaping"
                }
            }
        }''',
    ]
    
    print("Testing JSON parsing improvements...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}:")
        print(f"Input: {test_case[:100]}{'...' if len(test_case) > 100 else ''}")
        
        result = parse_llm_response(test_case)
        
        if "error" in result:
            print(f"❌ Failed: {result['error']}")
        else:
            print(f"✅ Success: {result.get('task_type', 'unknown')} task")
            print(f"   Methods: {result.get('verification_methods', [])}")

if __name__ == "__main__":
    test_json_parsing()
