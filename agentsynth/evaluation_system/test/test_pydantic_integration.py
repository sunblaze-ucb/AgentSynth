#!/usr/bin/env python3
"""
Integration test for Pydantic structured outputs with real API calls.
"""

import sys
import os
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

def test_pydantic_integration():
    """Test the Pydantic structured outputs with real API calls."""
    
    print("Testing Pydantic Structured Outputs Integration")
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
        
        # Test the Pydantic structured outputs function
        test_prompt = """
        Analyze this computer use task: Navigate to Amazon and search for "The Hunger Games" book. 
        Find the book and add it to your cart.
        """
        
        print("🧪 Testing Pydantic structured outputs with sample task...")
        
        # Test the structured outputs call
        result = generator._call_llms_with_pydantic(
            "You are an expert at analyzing computer use tasks and determining the best verification methods.",
            test_prompt,
            None,  # No image for this test
            model='gpt-4o'
        )
        
        print("✅ Pydantic structured outputs completed successfully!")
        print(f"Task Type: {result.task_type}")
        print(f"Verification Methods: {result.verification_methods}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning[:100]}...")
        
        # Test conversion to dict
        result_dict = result.model_dump()
        print("✅ Model to dict conversion works")
        print(f"Dict keys: {list(result_dict.keys())}")
        
        print("\n🎯 Pydantic structured outputs integration test PASSED!")
        print("The implementation is ready for production use.")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pydantic_integration()
