import pandas as pd
import json
from typing import Dict, Optional, Any, List, Callable, Awaitable
import asyncio
from tqdm.notebook import tqdm
import logging

async def async_handler_json(
    json_path: str,
    keymap: Dict[str, str],
    generator_function: Callable[..., Awaitable[Dict[str, Any]]],
    base_url: str,
    api_key: str,
    output_path: Optional[str] = None,
    batch_size: int = 10,
    model_kwargs: Optional[Dict[str, Any]] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None
) -> List[Dict]:
    """
    Process multiple requests from JSON file using provided async generator function.
    
    Args:
        json_path: Path to JSON file containing list of dictionaries
        keymap: Dictionary mapping JSON keys to generator function parameters
        generator_function: Async function for generating responses
        base_url: API base URL
        api_key: API key
        output_path: Optional path to save results
        batch_size: Number of items to process in parallel
        model_kwargs: Additional model parameters
        start_idx: Starting index
        end_idx: Optional ending index
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of dictionaries")
    
    if end_idx is None:
        end_idx = len(data)
    data = data[start_idx:end_idx]
    
    # Validate keymap
    required_params = {'prompt', 'model'}
    if not all(param in keymap.values() for param in required_params):
        raise ValueError(f"Keymap must map to required parameters: {required_params}")
    
    async def process_batch(batch_data: List[Dict]) -> List[Dict]:
        tasks = []
        for item in batch_data:
            # Build parameters from keymap
            params = {
                'base_url': base_url,
                'api_key': api_key
            }
            
            for json_key, param_name in keymap.items():
                if json_key in item:
                    params[param_name] = item[json_key]
            
            if model_kwargs:
                params.update(model_kwargs)
            
            tasks.append(generator_function(**params))
        
        return await asyncio.gather(*tasks)
    
    # Process in batches
    results = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch_data = data[i:i + batch_size]
        batch_results = await process_batch(batch_data)
        
        # Add results to original data
        for j, result in enumerate(batch_results):
            data[i + j]['clean_response'] = result['clean_response']
            data[i + j]['inference_time'] = result['output_params']['inference_time']
            data[i + j]['tokens'] = result['output_params']['total_tokens']
            data[i + j]['stop_reason'] = result['output_params']['stop_reason']
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    return data