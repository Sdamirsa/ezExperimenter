import pandas as pd
import json
from typing import Dict, Optional, Any, List, Callable, Awaitable
import asyncio
from tqdm.notebook import tqdm
import logging

async def async_handler_excel(
    excel_path: str,
    keymap: Dict[str, str],
    generator_function: Callable[..., Awaitable[Dict[str, Any]]],
    base_url: str,
    api_key: str,
    output_path: Optional[str] = None,
    batch_size: int = 10,
    model_kwargs: Optional[Dict[str, Any]] = None,
    start_row: int = 0,
    end_row: Optional[int] = None
) -> pd.DataFrame:
    """
    Process multiple requests from Excel file using provided async generator function.
    
    Args:
        excel_path: Path to Excel file
        keymap: Dictionary mapping Excel columns to generator function parameters
        generator_function: Async function for generating responses
        base_url: API base URL
        api_key: API key
        output_path: Optional path to save results
        batch_size: Number of items to process in parallel
        model_kwargs: Additional model parameters
        start_row: Starting row index
        end_row: Optional ending row index
    """
    # Read Excel file
    df = pd.read_excel(excel_path)
    if end_row is None:
        end_row = len(df)
    df = df.iloc[start_row:end_row].copy()
    
    # Validate keymap
    required_params = {'prompt', 'model'}
    if not all(param in keymap.values() for param in required_params):
        raise ValueError(f"Keymap must map to required parameters: {required_params}")
    
    async def process_batch(batch_df: pd.DataFrame) -> List[Dict]:
        tasks = []
        for _, row in batch_df.iterrows():
            # Build parameters from keymap
            params = {
                'base_url': base_url,
                'api_key': api_key
            }
            
            for excel_col, param_name in keymap.items():
                if excel_col in row.index and not pd.isna(row[excel_col]):
                    params[param_name] = row[excel_col]
            
            if model_kwargs:
                params.update(model_kwargs)
            
            tasks.append(generator_function(**params))
        
        return await asyncio.gather(*tasks)
    
    # Process in batches
    results = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_results = await process_batch(batch_df)
        results.extend(batch_results)
    
    # Add results to DataFrame
    for i, result in enumerate(results):
        df.loc[start_row + i, 'clean_response'] = result['clean_response']
        df.loc[start_row + i, 'inference_time'] = result['output_params']['inference_time']
        df.loc[start_row + i, 'tokens'] = result['output_params']['total_tokens']
        df.loc[start_row + i, 'stop_reason'] = result['output_params']['stop_reason']
    
    if output_path:
        df.to_excel(output_path, index=False)
    
    return df
