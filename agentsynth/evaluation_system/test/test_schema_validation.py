#!/usr/bin/env python3
"""
Test script to validate the JSON schema for structured outputs.
"""

import json
import sys
from pathlib import Path

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent / "core"))

def test_schema_validation():
    """Test that the JSON schema is valid for OpenAI structured outputs."""
    
    print("Testing JSON Schema Validation")
    print("=" * 40)
    
    from generate_evaluation_functions import EvaluationFunctionGenerator
    
    generator = EvaluationFunctionGenerator()
    schema = generator.task_analysis_schema
    
    print("✅ Schema loaded successfully")
    
    # Test that the schema structure is correct
    assert schema["type"] == "json_schema", "Schema type should be 'json_schema'"
    assert "json_schema" in schema, "Schema should contain 'json_schema' key"
    assert "name" in schema["json_schema"], "Schema should have a name"
    assert "strict" in schema["json_schema"], "Schema should have strict mode"
    assert "schema" in schema["json_schema"], "Schema should contain the actual schema"
    
    print("✅ Schema structure is valid")
    
    # Test that all required fields are present
    actual_schema = schema["json_schema"]["schema"]
    assert "additionalProperties" in actual_schema, "Schema should have additionalProperties"
    assert actual_schema["additionalProperties"] == False, "additionalProperties should be False"
    assert "properties" in actual_schema, "Schema should have properties"
    assert "required" in actual_schema, "Schema should have required fields"
    
    print("✅ Schema has required additionalProperties: false")
    
    # Test that all nested objects have additionalProperties: false
    def check_additional_properties(obj, path=""):
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                if "additionalProperties" not in obj:
                    raise AssertionError(f"Object at {path} missing additionalProperties")
                if obj["additionalProperties"] != False:
                    raise AssertionError(f"Object at {path} has additionalProperties != False")
            
            for key, value in obj.items():
                check_additional_properties(value, f"{path}.{key}" if path else key)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_additional_properties(item, f"{path}[{i}]")
    
    check_additional_properties(actual_schema)
    print("✅ All nested objects have additionalProperties: false")
    
    # Test that the schema can be serialized to JSON
    json_str = json.dumps(schema, indent=2)
    print(f"✅ Schema serializes to JSON ({len(json_str)} characters)")
    
    # Test that the schema can be deserialized
    parsed_schema = json.loads(json_str)
    assert parsed_schema == schema, "Schema should be identical after JSON round-trip"
    print("✅ Schema survives JSON round-trip")
    
    print("\n🎯 Schema validation PASSED!")
    print("The schema is ready for OpenAI structured outputs.")
    
    # Print a summary of the schema
    print(f"\nSchema Summary:")
    print(f"- Name: {schema['json_schema']['name']}")
    print(f"- Strict mode: {schema['json_schema']['strict']}")
    print(f"- Required fields: {len(actual_schema['required'])}")
    print(f"- Properties: {len(actual_schema['properties'])}")
    print(f"- Task types: {len(actual_schema['properties']['task_type']['enum'])}")
    print(f"- Verification methods: {len(actual_schema['properties']['verification_methods']['items']['enum'])}")

if __name__ == "__main__":
    test_schema_validation()
