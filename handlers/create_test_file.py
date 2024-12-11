import pandas as pd
import json
import os


def create_test_files(excel_path: str, json_path: str):
    """Create example Excel and JSON files for testing."""
    
    # Create Excel data
    excel_data = {
        'Question': [
            "What are the main principles of clean code?",
            "How can I improve my code readability?",
            "What are best practices for code documentation?"
        ],
        'Model': ["accounts/fireworks/models/llama-v3-8b-instruct"] * 3,
        'Temperature': [0.7, 0.8, 0.9],
        'MaxTokens': [150, 200, 250],
        'SystemPrompt': [
            "You are an expert in software engineering best practices",
            "You are a coding mentor focused on code quality",
            "You are a technical documentation specialist"
        ]
    }
    
    # Create JSON data
    json_data = [
        {
            "question": "What makes code maintainable?",
            "model_name": "accounts/fireworks/models/llama-v3-8b-instruct",
            "temp": 0.7,
            "max_tok": 150,
            "sys_prompt": "You are a software architecture expert"
        },
        {
            "question": "How do you handle technical debt?",
            "model_name": "accounts/fireworks/models/llama-v3-8b-instruct",
            "temp": 0.8,
            "max_tok": 200,
            "sys_prompt": "You are an experienced tech lead"
        },
        {
            "question": "What are code review best practices?",
            "model_name": "accounts/fireworks/models/llama-v3-8b-instruct",
            "temp": 0.9,
            "max_tok": 250,
            "sys_prompt": "You are a senior code reviewer"
        }
    ]
    
    # Save Excel file
    df = pd.DataFrame(excel_data)
    df.to_excel(excel_path, index=False)
    print(f"Excel file saved to: {excel_path}")
    
    # Save JSON file
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON file saved to: {json_path}")

# Example usage
if __name__ == "__main__":
    create_test_files(
        excel_path="test_input.xlsx",
        json_path="test_input.json"
    )