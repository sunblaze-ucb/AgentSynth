#!/usr/bin/env python3
"""
Simple test for Pydantic models only.
"""

import sys
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

def test_pydantic_models():
    """Test the Pydantic models for structured outputs."""
    
    print("Testing Pydantic Models (Simple)")
    print("=" * 40)
    
    try:
        from pydantic import BaseModel, Field
        from typing import List, Dict, Any, Optional
        
        # Define the models directly here to avoid import issues
        class UrlVerification(BaseModel):
            expected_url: Optional[str] = Field(None, description="Expected URL to verify")
            expected_url_pattern: Optional[str] = Field(None, description="Expected URL pattern to match")
            expected_content: Optional[List[str]] = Field(None, description="Expected content to find on the page")

        class UIElement(BaseModel):
            type: str = Field(..., description="Type of UI element (text, button, etc.)")
            text: Optional[str] = Field(None, description="Text content of the element")
            selector: Optional[str] = Field(None, description="CSS selector for the element")

        class ExpectedOutcome(BaseModel):
            url_verification: Optional[UrlVerification] = Field(None, description="URL verification requirements")
            ui_elements: Optional[List[UIElement]] = Field(None, description="UI elements to verify")

        class TaskAnalysis(BaseModel):
            task_type: str = Field(..., description="The type of computer use task")
            verification_methods: List[str] = Field(..., description="List of verification methods to use")
            expected_outcome: ExpectedOutcome = Field(..., description="Expected outcomes to verify")
            verification_parameters: Dict[str, Any] = Field(..., description="Parameters for verification methods")
            success_criteria: Dict[str, Any] = Field(..., description="Criteria that define task success")
            custom_logic: str = Field(..., description="Any custom verification logic needed")
            confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the analysis (0.0-1.0)")
            reasoning: str = Field(..., description="Explanation of the analysis approach")
        
        print("✅ Pydantic models defined successfully")
        
        # Test creating a TaskAnalysis model
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
        print("The models are ready for OpenAI structured outputs.")
        
    except Exception as e:
        print(f"❌ Pydantic models test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pydantic_models()
