import asyncio
from generation.firework.async_generator import firework_async_generate

from pydantic import BaseModel, Field
import os
import pandas as pd

# Define your API credentials
FIREWORK_BASE_URL = "https://api.fireworks.ai/inference/v1" # Update this
MY_FIREWORK_API_KEY = os.getenv("FIREWORKS_API_KEY") # Update this
assert MY_FIREWORK_API_KEY is not None, "API Key is not set!" # Update this
PATH_TO_IMAGE_TEST = r"C:\Users\LEGION\Documents\GIT\LLM_is_all_you_need\ezExperimenter_0.1\generation\duck_test.jpg" # Update this
assert os.path.exists(PATH_TO_IMAGE_TEST), "Image file does not exist!" # Update this
MODEL_NAME_LLM_TEST = "accounts/fireworks/models/llama-v3-8b-instruct" # Update this
MODEL_NAME_VLM_TEST = "accounts/fireworks/models/phi-3-vision-128k-instruct" # Update this


async def firework_async_generate_test():
    
    # 1. Simple text example
    try:
        text_result = await firework_async_generate(
            prompt="Explain what is the programmers duck.",
            model=MODEL_NAME_LLM_TEST,
            base_url=FIREWORK_BASE_URL,
            api_key=MY_FIREWORK_API_KEY,
            temperature=0.7,
            max_tokens=100
        )
        
        print("\n=== Text Example ===")
        print("Response:", text_result['clean_response'])
        print("\nInput Parameters:", text_result['input_params'])
        print("\nOutput Parameters:", text_result['output_params'])
        if text_result['output_params']['output_text']:
            print(f"✅ Text test passed --> {text_result['output_params']['output_text']}")
        else:
            print(f"❌ Text test failed. ")
            
            
    except Exception as e:
        print(f"❌ Text test failed: {str(e)}")
        raise
    
    # 2. Vision example
    try:
        vision_result = await firework_async_generate(
            prompt="What's happening in this image? Describe it in detail.",
            model=MODEL_NAME_VLM_TEST,
            base_url=FIREWORK_BASE_URL,
            api_key=MY_FIREWORK_API_KEY,
            image_paths=PATH_TO_IMAGE_TEST,
            temperature=0.7
        )
        
        print("\n=== Vision Example ===")
        print("Response:", vision_result['clean_response'])
        print("\nInput Parameters:", vision_result['input_params'])
        print("\nOutput Parameters:", vision_result['output_params'])
        if vision_result['output_params']['output_text']:
            print(f"✅ Vision test passed --> {vision_result['output_params']['output_text']}")
        else:
            print(f"❌ Vision test failed. ")
    except Exception as e:
        print(f"❌ Vision test failed: {str(e)}")
        raise        

    # 3. JSON mode with Pydantic example
    try:
        class Animal(BaseModel):
            name: str = Field(..., description="name of animal mentioned in this text")


        json_result = await firework_async_generate(
            prompt="One day, a programmer was challenged by the fact that a human who was harmed by other, can become the one who harms another. Then he turned to his yellow duck, and asked him about the soloution to break this circle of hatred.\n What is the name of the animal?",
            model=MODEL_NAME_VLM_TEST,
            base_url=FIREWORK_BASE_URL,
            api_key=MY_FIREWORK_API_KEY,
            image_paths=PATH_TO_IMAGE_TEST,
            pydantic_model=Animal,
            temperature=0.7
        )
        print("\n✅ JSON mode test passed")
        print("\n=== JSON Example ===")
        print("Structured Output:", json_result['structured_output'])
        print("\nInput Parameters:", json_result['input_params'])
        print("\nOutput Parameters:", json_result['output_params'])
        if json_result['output_params']['output_text']:
            print(f"✅ JSON mode test passed --> {json_result['output_params']['output_text']}")
        else:
            print(f"❌ JSON mode test failed. ")        
    except Exception as e:
        print(f"❌ JSON mode test failed: {str(e)}")
        raise
    
# # Run all tests
# async def main():
#     await firework_async_generate_test()

# # For Jupyter notebook
# await firework_async_generate_test()