import asyncio
from typing import List, Dict, Any, Optional, Tuple, Callable, Awaitable
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
import os
from tqdm.notebook import tqdm
from pydantic import BaseModel
import json
import signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flag for interrupt handling
interrupt_received = False

def signal_handler(signum, frame):
    global interrupt_received
    logger.info("\nInterrupt received. Will save after current batch completes...")
    interrupt_received = True

async def experiment_design_handler(
    excel_path: str,
    prompt_column: str,
    temperature: float,
    generator_function: Callable[..., Awaitable[Dict[str, Any]]],
    base_url: str,
    api_key: str,
    model: str,
    output_folder: str,
    experiment_name: str,
    batch_size: int = 10,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    image_column: Optional[str] = None,
    pydantic_model: Optional[BaseModel] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    start_row: int = 0,
    end_row: Optional[int] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle experiment design with input and output management.
    """
    print(f"\n----------------------------------------\n      ExP: {experiment_name}\n----------------------------------------")

    # Register signal handler for interrupts
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create output folder if it doesn't exist
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Read input Excel file
    df_input = pd.read_excel(excel_path)
    if end_row is None:
        end_row = len(df_input)
    df_input = df_input.iloc[start_row:end_row].copy()
    
    # Validate input
    if prompt_column not in df_input.columns:
        raise ValueError(f"Prompt column '{prompt_column}' not found in input file")
    
    # Create experiment parameters DataFrame
    df_experiment = df_input.copy()
    df_experiment['temperature'] = temperature
    df_experiment['model'] = model
    if max_tokens:
        df_experiment['max_tokens'] = max_tokens
    if system_prompt:
        df_experiment['system_prompt'] = system_prompt
    
    # Save input file
    input_path = output_folder / f"{experiment_name}_inputs.xlsx"
    df_experiment.to_excel(input_path, index=False)
    logger.info(f"Saved experiment inputs to: {input_path}")
    
    async def process_batch(batch_df: pd.DataFrame) -> List[Dict]:
        tasks = []
        for _, row in batch_df.iterrows():
            params = {
                'prompt': row[prompt_column],
                'model': model,
                'base_url': base_url,
                'api_key': api_key,
                'temperature': temperature,
            }
            
            if image_column and image_column in row:
                if pd.notna(row[image_column]):
                    image_paths = [path.strip() for path in str(row[image_column]).split(',') if path.strip()]
                    if image_paths:
                        params['image_paths'] = image_paths
            
            if max_tokens:
                params['max_tokens'] = max_tokens
            if system_prompt:
                params['system_prompt'] = system_prompt
            if pydantic_model:
                params['pydantic_model'] = pydantic_model
            if model_kwargs:
                params.update(model_kwargs)
            
            tasks.append(generator_function(**params))
        
        return await asyncio.gather(*tasks)

    def save_progress(df_output: pd.DataFrame, current_progress: int):
        """Helper function to save current progress"""
        output_path = output_folder / f"{experiment_name}_outputs.xlsx"
        json_output_path = output_folder / f"{experiment_name}_outputs.json"
        
        # Save Excel file
        df_output.to_excel(output_path, index=False)
        
        # Save JSON file
        json_output = df_output.to_dict(orient='records')
        with open(json_output_path, 'w') as f:
            json.dump(json_output, f, indent=2, default=str)
        
        logger.info(f"Progress saved ({current_progress}/{len(df_experiment)} rows processed)")
    
    # Initialize output DataFrame
    df_output = df_experiment.copy()
    
    # Helper function to safely serialize objects
    def safe_serialize(obj):
        try:
            if isinstance(obj, (dict, list)):
                return json.dumps(obj, default=str)
            return str(obj)
        except Exception as e:
            logger.warning(f"Error serializing object: {e}")
            return str(obj)
    
    try:
        # Process in batches
        for i in tqdm(range(0, len(df_experiment), batch_size)):
            if interrupt_received:
                logger.info("Interrupt received - saving progress and exiting...")
                break
            
            batch_df = df_experiment.iloc[i:i + batch_size]
            batch_results = await process_batch(batch_df)
            
            # Process batch results
            for j, result in enumerate(batch_results):
                current_idx = start_row + i + j
                
                # Basic parameters
                df_output.loc[current_idx, 'clean_response'] = result['clean_response']
                df_output.loc[current_idx, 'inference_time'] = result['output_params']['inference_time']
                df_output.loc[current_idx, 'prompt_tokens'] = result['output_params']['prompt_tokens']
                df_output.loc[current_idx, 'completion_tokens'] = result['output_params']['completion_tokens']
                df_output.loc[current_idx, 'total_tokens'] = result['output_params']['total_tokens']
                df_output.loc[current_idx, 'stop_reason'] = result['output_params']['stop_reason']
                
                # Save raw_response as serialized string
                if 'raw_response' in result:
                    raw_response_dict = {
                        'id': result['raw_response'].id,
                        'choices': [
                            {
                                'message': {
                                    'role': choice.message.role,
                                    'content': choice.message.content
                                },
                                'finish_reason': choice.finish_reason
                            } for choice in result['raw_response'].choices
                        ],
                        'usage': {
                            'prompt_tokens': result['raw_response'].usage.prompt_tokens,
                            'completion_tokens': result['raw_response'].usage.completion_tokens,
                            'total_tokens': result['raw_response'].usage.total_tokens
                        }
                    }
                    df_output.loc[current_idx, 'raw_response'] = safe_serialize(raw_response_dict)
                
                # Save structured_output if present
                if 'structured_output' in result and result['structured_output'] is not None:
                    if hasattr(result['structured_output'], 'model_dump'):
                        structured_dict = result['structured_output'].model_dump()
                    else:
                        structured_dict = result['structured_output']
                    df_output.loc[current_idx, 'structured_output'] = safe_serialize(structured_dict)
            
            # Save progress after each batch
            save_progress(df_output, i + len(batch_results))
    
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        save_progress(df_output, i)
        raise
    
    finally:
        # Final save
        output_path = output_folder / f"{experiment_name}_outputs.xlsx"
        json_output_path = output_folder / f"{experiment_name}_outputs.json"
        
        df_output.to_excel(output_path, index=False)
        json_output = df_output.to_dict(orient='records')
        with open(json_output_path, 'w') as f:
            json.dump(json_output, f, indent=2, default=str)
        
        logger.info(f"Final results saved to: {output_path}")
    
    return df_experiment, df_output

# The rest of the code (run_multiple_experiments function) remains unchanged
async def run_multiple_experiments(
    experiment_configs: List[Dict[str, Any]],
    base_output_folder: str,
    generator_function: Any,
    base_url: str,
    api_key: str,
    pydantic_models: Optional[Dict[str, BaseModel]] = None
) -> List[Tuple[str, Dict[str, Any], Optional[Tuple[pd.DataFrame, pd.DataFrame]], Optional[str]]]:
    """
    Run multiple experiments with different configurations.
    """

    # Rest of the function remains exactly the same
    base_output_folder = Path(base_output_folder)
    base_output_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = base_output_folder / f"batch_experiment_{timestamp}"
    experiment_folder.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for idx, config in enumerate(experiment_configs, 1):
        experiment_name = config.get('experiment_name', f'experiment_{idx}')
        logger.info(f"\nStarting experiment: {experiment_name} ({idx}/{len(experiment_configs)})")
        
        try:
            required_params = {'excel_path', 'prompt_column', 'temperature', 'model'}
            missing_params = required_params - set(config.keys())
            if missing_params:
                raise ValueError(f"Missing required parameters: {missing_params}")
            
            output_folder = experiment_folder / experiment_name
            
            pydantic_model = None
            if pydantic_models and experiment_name in pydantic_models:
                pydantic_model = pydantic_models[experiment_name]

            experiment_params = {
                'generator_function': generator_function,
                'base_url': base_url,
                'api_key': api_key,
                'output_folder': str(output_folder),
                'pydantic_model': pydantic_model,
                **config
            }
            
            df_in, df_out = await experiment_design_handler(**experiment_params)
            
            logger.info(f"Completed experiment: {experiment_name}")
            logger.info(f"Processed {len(df_out)} prompts")
            logger.info(f"Average inference time: {df_out['inference_time'].mean():.2f}s")
            logger.info(f"Average tokens: {df_out['total_tokens'].mean():.1f}")
            
            results.append((experiment_name, config, (df_in, df_out), None))
            
        except Exception as e:
            error_msg = f"Error in experiment {experiment_name}: {str(e)}"
            logger.error(error_msg)
            results.append((experiment_name, config, None, error_msg))
    
    summary_data = []
    for exp_name, config, result, error in results:
        summary = {
            'experiment_name': exp_name,
            'status': 'Success' if error is None else 'Failed',
            'error_message': error or 'None',
            'input_file': config.get('excel_path', ''),
            'temperature': config.get('temperature', ''),
            'model': config.get('model', ''),
            'prompt_column': config.get('prompt_column', '')
        }
        
        if result is not None:
            df_in, df_out = result
            summary.update({
                'prompts_processed': len(df_out),
                'avg_inference_time': df_out['inference_time'].mean(),
                'avg_total_tokens': df_out['total_tokens'].mean()
            })
        
        summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = experiment_folder / 'experiment_summary.xlsx'
    summary_df.to_excel(summary_path, index=False)
    logger.info(f"\nExperiment summary saved to: {summary_path}")
    
    return results