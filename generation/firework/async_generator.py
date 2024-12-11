import openai
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import base64
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise

def process_image_paths(image_paths: str) -> List[str]:
    """
    Process the image_paths string, splitting by comma and removing white spaces.
    Returns a list of clean image paths.
    """
    if not image_paths:
        return []
    # Split by comma and strip white spaces
    return [path.strip() for path in image_paths.split(',') if path.strip()]

async def firework_async_generate(
    prompt: Union[str, List[Dict[str, Any]]],
    model: str,
    base_url: str,
    api_key: str,
    idx: int = 0,
    system_prompt: Optional[str] = None,
    pydantic_model: Optional[BaseModel] = None,
    image_paths: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process LLM/VLM requests with support for text, vision, and JSON modes.
    """
    if pydantic_model:
        prompt = prompt + "\nReturn your response in clean structured JSON."
        system_prompt = system_prompt + "\nReturn your response in clean structured JSON."
        
    start_time = datetime.now()
    logger.info(f"Starting LLM request {idx} at {start_time}")
    
    processed_image_paths = process_image_paths(image_paths) if image_paths else []
    
    if not api_key:
        logger.error("Please check your api key. It seems that it has less than 4 charectars, which is unusual.")
        raise ValueError("Please check your api key. You should save it in .env file and load it here.")
    
    if model.startswith("gpt-4o"): #this means that we can use STRUCTURED OUTPUT (i.e. directly providing pydantic)
        Just_JSON = False
    else:
        Just_JSON = True
    
    try:
        # Initialize client
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        # Track input parameters
        input_params = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "image_path": image_paths
        }
        if model_kwargs:
            input_params.update(model_kwargs)
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        # Handle different input types
        if isinstance(prompt, str):
            if processed_image_paths:
                try:
                    content = [{"type": "text", "text": prompt}]
                    for img_path in processed_image_paths:
                        image_base64 = encode_image(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        })
                    messages.append({
                        "role": "user",
                        "content": content
                    })
                except Exception as e:
                    logger.error(f"Error processing images: {e}")
                    raise
            else:
                messages.append({"role": "user", "content": prompt})
        else:
            messages.extend(prompt)
            
        # Prepare base parameters
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        if seed:
            params["seed"] = seed
        if pydantic_model:
            if Just_JSON:
                params["response_format"] = {
                    "type": "json_object",
                    "schema": pydantic_model.model_json_schema()
                }
            else:
                params["response_format"] = pydantic_model
                
        if model_kwargs:
            params.update(model_kwargs)
            
        # Process single request
        inference_start = time.time()
        if pydantic_model and not Just_JSON: 
            response = await asyncio.get_event_loop().run_in_executor(
                ThreadPoolExecutor(),
                lambda: client.beta.chat.completions.parse(**params)
            )
            inference_time = time.time() - inference_start
       
        else: 
            response = await asyncio.get_event_loop().run_in_executor(
                ThreadPoolExecutor(),
                lambda: client.chat.completions.create(**params)
            )
            inference_time = time.time() - inference_start 
        
        # Track output parameters
        output_params = {
            "inference_time": inference_time,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "stop_reason": response.choices[0].finish_reason,
            "input_prompt": prompt if isinstance(prompt, str) else str(prompt),
            "output_text": response.choices[0].message.content
        }
        
        result = {
            "idx": idx,
            "raw_response": response,
            "clean_response": response.choices[0].message.content,
            "input_params": input_params,
            "output_params": output_params
        }
        
        if pydantic_model:
            try:
                structured = pydantic_model.model_validate_json(
                    response.choices[0].message.content
                )
                result["structured_output"] = structured
            except Exception as e:
                logger.error(f"Error parsing structured output: {e}")
                result["structured_output"] = None
        
        end_time = datetime.now()
        logger.info(f"Completed LLM request {idx} at {end_time}. Duration: {end_time - start_time}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing LLM request {idx}: {e}")
        return {
            "idx": idx,
            "error": str(e),
            "raw_response": None,
            "clean_response": None,
            "structured_output": None,
            "input_params": input_params,
            "output_params": {
                "inference_time": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "stop_reason": None,
                "input_prompt": prompt if isinstance(prompt, str) else str(prompt),
                "output_text": None
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing LLM request {idx}: {e}")
        return {
            "idx": idx,
            "error": str(e),
            "raw_response": None,
            "clean_response": None,
            "structured_output": None,
            "input_params": input_params,
            "output_params": {
                "inference_time": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "stop_reason": None,
                "input_prompt": prompt if isinstance(prompt, str) else str(prompt),
                "output_text": None
            }
        }
        
        