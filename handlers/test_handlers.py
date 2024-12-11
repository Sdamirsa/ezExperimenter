import asyncio
import os
import pandas as pd
import json
import tempfile
from pydantic import BaseModel
from typing import Dict, Any

from handlers.excel_handler import async_handler_excel
from handlers.json_handler import async_handler_json

from generation.firework.async_generator import firework_async_generate

# Define constants
BASE_URL = "https://api.fireworks.ai/inference/v1"
API_KEY = os.getenv("FIREWORKS_API_KEY")
MODEL_NAME_LLM_TEST = "accounts/fireworks/models/llama-v3-8b-instruct"
GENERATE_FUNCTION_TEST = firework_async_generate


# Define your API credentials and paths
BASE_URL = "https://api.fireworks.ai/inference/v1"
API_KEY = os.getenv("FIREWORKS_API_KEY")

# Define input and output paths
INPUT_EXCEL_PATH = r"C:\Users\LEGION\Documents\GIT\LLM_is_all_you_need\ezExperimenter_0.1\handlers\test_input.xlsx"  # Update this
INPUT_JSON_PATH = r"C:\Users\LEGION\Documents\GIT\LLM_is_all_you_need\ezExperimenter_0.1\handlers\test_input.json"   # Update this
OUTPUT_EXCEL_PATH = r"C:\Users\LEGION\Documents\GIT\LLM_is_all_you_need\ezExperimenter_0.1\handlers\test_output.xlsx"  # Update this
OUTPUT_JSON_PATH = r"C:\Users\LEGION\Documents\GIT\LLM_is_all_you_need\ezExperimenter_0.1\handlers\test_output.json"  # Update this

async def test_excel_json_handlers():
    test_results = {
        'excel_handler': False,
        'json_handler': False
    }
    
    try:
        await run_excel_handler_test()
        test_results['excel_handler'] = True
        print("\n✅ Excel handler test completed successfully")
    except Exception as e:
        print(f"\n❌ Excel handler test failed: {str(e)}")
        
    try:
        await run_json_handler_test()
        test_results['json_handler'] = True
        print("\n✅ JSON handler test completed successfully")
    except Exception as e:
        print(f"\n❌ JSON handler test failed: {str(e)}")
    
    print("\n=== Test Summary ===")
    for test_name, passed in test_results.items():
        print(f"{test_name}: {'✅' if passed else '❌'}")

async def run_excel_handler_test():
    print("\n=== Excel Handler Test ===")
    
    # Define keymap
    keymap_excel = {
        'Question': 'prompt',
        'Model': 'model',
        'Temperature': 'temperature',
        'MaxTokens': 'max_tokens',
        'SystemPrompt': 'system_prompt'
    }
    
    try:
        # Process Excel file
        results_df = await async_handler_excel(
            excel_path=INPUT_EXCEL_PATH,
            keymap=keymap_excel,
            generator_function=GENERATE_FUNCTION_TEST,
            base_url=BASE_URL,
            api_key=API_KEY,
            batch_size=2,
            output_path=OUTPUT_EXCEL_PATH
        )
        
        # Validate and display results
        print("\nExcel Processing Results:")
        for idx, row in results_df.iterrows():
            print(f"\nQuestion {idx + 1}: {row['Question']}")
            print(f"Response: {row['clean_response'][:100]}...")  # First 100 chars
            print(f"Inference Time: {row['inference_time']:.2f}s")
            print(f"Total Tokens: {row['tokens']}")
        
    except Exception as e:
        print(f"Error in Excel processing: {str(e)}")
        raise

async def run_json_handler_test():
    print("\n=== JSON Handler Test ===")
    
    # Define keymap
    keymap_json = {
        'question': 'prompt',
        'model_name': 'model',
        'temp': 'temperature',
        'max_tok': 'max_tokens',
        'sys_prompt': 'system_prompt'
    }
    
    try:
        # Process JSON file
        results_json = await async_handler_json(
            json_path=INPUT_JSON_PATH,
            keymap=keymap_json,
            generator_function=GENERATE_FUNCTION_TEST,
            base_url=BASE_URL,
            api_key=API_KEY,
            batch_size=2,
            output_path=OUTPUT_JSON_PATH
        )
        
        # Display results
        print("\nJSON Processing Results:")
        for idx, item in enumerate(results_json):
            print(f"\nQuestion {idx + 1}: {item['question']}")
            print(f"Response: {item['clean_response'][:100]}...")  # First 100 chars
            print(f"Inference Time: {item['inference_time']:.2f}s")
            print(f"Total Tokens: {item['tokens']}")

    except Exception as e:
        print(f"Error in JSON processing: {str(e)}")
        raise

# Run tests
# if __name__ == "__main__":
#     asyncio.run(test_handlers())

# Run test in jupyter
# import asyncio
# from handlers.test_handlers import test_handlers

# await test_handlers()




#####################################
####   experiment_handler   #########
#####################################

import asyncio
import os
import pandas as pd
import tempfile
from pathlib import Path
from generation.firework.async_generator import firework_async_generate
from handlers.async_experiment_handler import run_multiple_experiments

# Define constants
BASE_URL = "https://api.fireworks.ai/inference/v1" # update this
API_KEY = os.getenv("FIREWORKS_API_KEY") # update this
MODEL_NAME_LLM_TEST = "accounts/fireworks/models/llama-v3-8b-instruct"  # update this

async def test_run_multiple_experiments():
    """Test the batch experiment handler with different configurations."""
    test_results = {
        'file_creation': False,
        'experiment_execution': False,
        'output_validation': False
    }
    
    try:
        # Create temporary test files
        temp_files = create_test_files()
        test_results['file_creation'] = True
        print("\n✅ Test files created successfully")
        
        # Run experiments
        results = await run_experiment_tests(temp_files)
        test_results['experiment_execution'] = True
        print("\n✅ Experiments executed successfully")
        
        # Validate outputs
        validate_experiment_results(results)
        test_results['output_validation'] = True
        print("\n✅ Output validation completed successfully")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
    finally:
        cleanup_test_files(temp_files)
    
    # Print test summary
    print("\n=== Test Summary ===")
    for test_name, passed in test_results.items():
        print(f"{test_name}: {'✅' if passed else '❌'}")

def create_test_files():
    """Create temporary test files with different configurations."""
    temp_files = []
    
    # Create first test file - basic prompts
    df1 = pd.DataFrame({
        'Prompt': [
            "What is clean code?",
            "Explain code refactoring.",
            "What are coding best practices?"
        ]
    })
    
    # Create second test file - with system prompts
    df2 = pd.DataFrame({
        'Question': [
            "How to improve code quality?",
            "What is technical debt?",
            "Explain unit testing."
        ],
        'SystemPrompt': [
            "You are a software engineering expert.",
            "You are a technical architect.",
            "You are a QA specialist."
        ]
    })
    
    # Save temporary files
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp2:
        df1.to_excel(tmp1.name, index=False)
        df2.to_excel(tmp2.name, index=False)
        temp_files.extend([tmp1.name, tmp2.name])
    
    return temp_files

async def run_experiment_tests(temp_files):
    """Run experiments with different configurations."""
    experiment_configs = [
        {
            'experiment_name': 'basic_test',
            'excel_path': temp_files[0],
            'prompt_column': 'Prompt',
            'temperature': 0.7,
            'model': MODEL_NAME_LLM_TEST,
            'max_tokens': 150,
            'batch_size': 2
        },
        {
            'experiment_name': 'advanced_test',
            'excel_path': temp_files[1],
            'prompt_column': 'Question',
            'temperature': 0.9,
            'model': MODEL_NAME_LLM_TEST,
            'max_tokens': 200,
            'batch_size': 2,
            'system_prompt': "You are an AI expert."
        }
    ]
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        results = await run_multiple_experiments(
            experiment_configs=experiment_configs,
            base_output_folder=temp_dir,
            generator_function=firework_async_generate,
            base_url=BASE_URL,
            api_key=API_KEY
        )
        return results

def validate_experiment_results(results):
    """Validate the experiment results."""
    for exp_name, config, result, error in results:
        if error:
            raise ValueError(f"Experiment {exp_name} failed: {error}")
        
        df_in, df_out = result
        
        # Validate input/output structure
        assert len(df_in) == len(df_out), f"Input/output length mismatch in {exp_name}"
        assert 'clean_response' in df_out.columns, f"Missing clean_response in {exp_name}"
        assert 'inference_time' in df_out.columns, f"Missing inference_time in {exp_name}"
        assert 'total_tokens' in df_out.columns, f"Missing total_tokens in {exp_name}"
        
        # Validate responses
        assert all(df_out['clean_response'].notna()), f"Found null responses in {exp_name}"
        assert all(df_out['inference_time'] > 0), f"Invalid inference times in {exp_name}"
        assert all(df_out['total_tokens'] > 0), f"Invalid token counts in {exp_name}"

def cleanup_test_files(temp_files):
    """Clean up temporary test files."""
    for file_path in temp_files:
        try:
            os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Could not delete temporary file {file_path}: {e}")

# Run test in regular Python
# if __name__ == "__main__":
#     asyncio.run(test_run_multiple_experiments())

# For Jupyter notebook:
# import nest_asyncio
# nest_asyncio.apply()
# await test_run_multiple_experiments()