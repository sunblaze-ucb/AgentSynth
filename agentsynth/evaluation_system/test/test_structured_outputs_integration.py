#!/usr/bin/env python3
"""
Integration test for structured outputs with real API calls.
"""

import sys
import os
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

def test_structured_outputs_integration():
    """Test structured outputs with a real API call if credentials are available."""
    
    print("Testing Structured Outputs Integration")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  No OpenAI API key found. Skipping integration test.")
        print("To test with real API calls, set OPENAI_API_KEY environment variable.")
        return
    
    print("✅ OpenAI API key found. Testing with real API call...")
    
    try:
        from generate_evaluation_functions import EvaluationFunctionGenerator
        
        # Create generator
        generator = EvaluationFunctionGenerator(model_name='gpt-4o')
        
        # Create test task data
        test_task_data = {
            'id': 'test_task_001',
            'messages': [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]
                },
                {
                    'role': 'user', 
                    'content': [
                        {'type': 'text', 'text': 'Given the task: Navigate to Amazon and search for "The Hunger Games" book. Find the book and add it to your cart.'},
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='}
                    ]
                }
            ]
        }
        
        print("🧪 Testing structured outputs with sample task...")
        
        # Test the analysis
        result = generator.analyze_task_and_generate_evaluator(test_task_data)
        
        if 'error' in result:
            print(f"❌ Error in analysis: {result['error']}")
            return
        
        print("✅ Structured outputs analysis completed successfully!")
        print(f"Task ID: {result['task_id']}")
        print(f"Task Description: {result['task_description'][:100]}...")
        
        analysis = result['analysis_result']
        print(f"Task Type: {analysis.get('task_type', 'unknown')}")
        print(f"Verification Methods: {analysis.get('verification_methods', [])}")
        print(f"Confidence: {analysis.get('confidence', 0.0):.2f}")
        print(f"Reasoning: {analysis.get('reasoning', 'No reasoning provided')[:100]}...")
        
        # Check if evaluation function was generated
        if 'evaluation_function' in result:
            print("✅ Evaluation function generated successfully!")
            print(f"Function length: {len(result['evaluation_function'])} characters")
        else:
            print("⚠️  No evaluation function generated")
        
        print("\n🎯 Structured outputs integration test PASSED!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_structured_outputs_integration()
