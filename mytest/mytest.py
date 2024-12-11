##########################
###   Test functions   ###
##########################
import os
import nest_asyncio
import asyncio
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

from handlers.async_experiment_handler import run_multiple_experiments
from generation.firework.async_generator import firework_async_generate
nest_asyncio.apply()



def test_llm_functions(base_url, api_key, model, use_image, test_excel_path, base_output_folder=r"test\test_history", ):
    print(f"excel for test: {Path(test_excel_path)}")
    if not Path(test_excel_path).exists():
        print("Test Excel file not found at the given path.")
        return
    df = pd.read_excel(test_excel_path)
    if "My_prompt" not in df.columns or "my_image_paths" not in df.columns:
        print("Test Excel file does not have required columns.")
        return
    if use_image:
        test_df = df  # All rows
    else:
        test_df = df[df["my_image_paths"].isna() | (df["my_image_paths"] == "")]
    if test_df.empty:
        print("No suitable rows found for testing given the conditions.")
        return
    experiment_config = {
        "experiment_name": "test_experiment",
        "excel_path": test_excel_path,
        "prompt_column": "My_prompt",
        "temperature": 0.7,
        "model": model,
        "max_tokens": 100,
        "system_prompt": None,
        "batch_size": 2,
        "start_row": 0,
        "end_row": None
    }
    if use_image:
        experiment_config["image_column"] = "my_image_paths"

    async def run_test():
        results = await run_multiple_experiments(
            experiment_configs=[experiment_config],
            base_output_folder=base_output_folder,
            generator_function=firework_async_generate,
            base_url=base_url,
            api_key=api_key,
            pydantic_models=None
        )
        return results

    loop = asyncio.get_event_loop()
    test_results = loop.run_until_complete(run_test())

    if test_results:
        for exp_name, config, result, error in test_results:
            if error:
                print(f"Test experiment failed with error: {error}")
            else:
                print("Test experiment succeeded!")
                df_in, df_out = result
                print('Test Input Summary:', df_in.head())
                print('Test Output Summary:', df_out.head())
    else:
        print("No results returned from test experiment.")