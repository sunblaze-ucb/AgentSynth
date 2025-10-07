#!/usr/bin/env python3
"""
Test script for Pydantic structured outputs implementation.
"""

import sys
import os
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

def test_pydantic_models():
    """Test the Pydantic models for structured outputs."""
    
    print("Testing Pydantic Models for Structured Outputs")
    print("=" * 50)
    
    from generate_evaluation_functions import TaskAnalysis, ExpectedOutcome, UrlVerification, UIElement
    
    # Test creating a TaskAnalysis model
    try:
        test_analysis = TaskAnalysis(
            task_type="web_navigation",
            verification_methods=["verify_web_page_state", "analyze_screenshot_for_elements"],
            expected_outcome=ExpectedOutcome(
                url_verification=UrlVerification(
                    expected_url="https://amazon.com/s?k=test",
                    expected_content=["Search results", "Product listings"]
                ),
                ui_elements=[
                    UIElement(type="text", text="Search results"),
                    UIElement(type="button", text="Add to cart")
                ]
            ),
            verification_parameters={
                "url": "https://amazon.com",
                "timeout": 30
            },
            success_criteria={
                "page_loaded": True,
                "search_results_found": True
            },
            custom_logic="Check for search results and product listings",
            confidence=0.85,
            reasoning="This is a web navigation task requiring page verification and UI element detection"
        )
        
        print("✅ TaskAnalysis model created successfully")
        print(f"   Task type: {test_analysis.task_type}")
        print(f"   Verification methods: {test_analysis.verification_methods}")
        print(f"   Confidence: {test_analysis.confidence}")
        print(f"   Expected URL: {test_analysis.expected_outcome.url_verification.expected_url}")
        
        # Test model serialization
        model_dict = test_analysis.model_dump()
        print("✅ Model serialization works")
        print(f"   Dict keys: {list(model_dict.keys())}")
        
        # Test model validation
        test_analysis_2 = TaskAnalysis.model_validate(model_dict)
        print("✅ Model validation works")
        print(f"   Recreated task type: {test_analysis_2.task_type}")
        
        print("\n🎯 Pydantic models test PASSED!")
        
    except Exception as e:
        print(f"❌ Pydantic models test failed: {e}")
        import traceback
        traceback.print_exc()

def test_pydantic_structured_outputs():
    """Test the Pydantic structured outputs functionality."""
    
    print("\nTesting Pydantic Structured Outputs Integration")
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
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pydantic_models()
    test_pydantic_structured_outputs()
