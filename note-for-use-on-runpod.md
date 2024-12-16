You should create your runpod account and get a machine with cuda12.4 (you can filter machines with this cuda version). For the ducker use Better Ollama CUDA12 (this one: madiator2011/better-ollama:cuda12.4). Set the Volume disk (workspace folder where you will download models) to the desired size. For example if I want to use two modesl with 40 and 60 GBs, I will set this to 100+something (120 for example). I assume docker will only contain the engine, but for safety I set this to the size of biggest model plus somethng (it will cost you ~0.05$/h). In the example I provided I will set this to 60+something (90 for example). The container I mentuined will give you Ollama API (the port is exposed so you can talk with it) and the code server which is something like VS code enviroment. 


1. Get the modules from my github: 
    !git clone https://github.com/Sdamirsa/ezExperimenter

Note that this will save the code in ezExperimenter in the main (and not the workspace)

2. You should change the run_multiple_experiments.py to use tdqm and note the notebook

change this:

from tqdm.notebook import tqdm

to this:

from tqdm import tqdm

3. Install libraries:

    pip install pandas openai openpyxl tdqm pydantic numpy nest-asyncio asyncio

4. Pull your models (and save it in the workspace)

    cd workspace
    ollama pull llama3.3:70b-instruct-q4_K_M

5. Define the experiment like this:

    experiment_configs = [  

        {
            'experiment_name': 'Llama-3.3-70b Q4', #🟢
            'excel_path': r"/workspace/data_to_runpod copy.xlsx",
                    'prompt_column': 'TEXT4WEB',
            'temperature': 0.7,
            'model':  'llama3.3:70b-instruct-q4_K_M',
            
            'max_tokens': 1024,
            'system_prompt': None, # "You are a helpful assistant."
            'batch_size': 1,
            # pydantic_model = ,
            # "model_kwargs" = {},
            # "start_row" = 0,
            # "end_row"=None,
            # image_column = ,
        },        
        # {
        #     'experiment_name': 'Llama-3.3-70b Q2', #🟢
        #     'excel_path': r"/workspace/data_to_runpod.xlsx",
            
        #     'prompt_column': 'TEXT4WEB',
        #     'temperature': 0.7,
        #     'model':  'llama3.3:70b-instruct-q2_K',
            
        #     'max_tokens': 1024,
        #     'system_prompt': None, # "You are a helpful assistant."
        #     'batch_size': 1,
        #     # pydantic_model = ,
        #     # "model_kwargs" = {},
        #     # "start_row" = 0,
        #     # "end_row"=None,
        #     # image_column = ,
        # },        
        # {
        #     'experiment_name': 'Llama-3.3-70b Q6', #🟢
        #     'excel_path': r"/workspace/data_to_runpod.xlsx",
            
        #     'prompt_column': 'TEXT4WEB',
        #     'temperature': 0.7,
        #     'model':  'llama3.3:70b-instruct-q6_K',
            
        #     'max_tokens': 1024,
        #     'system_prompt': None, # "You are a helpful assistant."
        #     'batch_size': 1,
        #     # pydantic_model = ,
        #     # "model_kwargs" = {},
        #     # "start_row" = 0,
        #     # "end_row"=None,
        #     # image_column = ,
        # },        

    ]



6. And the code to run is:


    import os
    import nest_asyncio
    import asyncio
    from ezExperimenter.handlers.async_experiment_handler import run_multiple_experiments
    from ezExperimenter.generation.firework.async_generator import firework_async_generate


    base_output_folder = r"/workspace"
    ollama_base_url='https://spuikknnebejf2-11434.proxy.runpod.net/v1'
    ollama_api_key = "ollama"


    results = await run_multiple_experiments(
        experiment_configs=experiment_configs,
        base_output_folder=base_output_folder,
        generator_function=firework_async_generate,
        base_url=ollama_base_url,
        api_key=ollama_api_key
    )


7. If you use multiple models, they will remain in memroy for 5 minutes after final use. This may cause error. See this: https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately

To solve this you can either stop the model and go for the next model, or use keep_alive  parameter to be set to 0.

Notes: Ollama can do prallel runs if you have sufficien GPU memory. See this: https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests

Solution for running multiple models:

    # pip install pandas openai openpyxl tdqm pydantic numpy nest-asyncio asyncio

    experiment_configs = [  

        {
            'experiment_name': 'Llama-3.3-70b Q4', #🟢
            'excel_path': r"/workspace/data_to_runpod copy.xlsx",
                    'prompt_column': 'TEXT4WEB',
            'temperature': 0.7,
            'model':  'llama3.3:70b-instruct-q4_K_M',
            
            'max_tokens': 1024,
            'system_prompt': None, # "You are a helpful assistant."
            'batch_size': 1,
            # pydantic_model = ,
            # "model_kwargs" = {},
            # "start_row" = 0,
            # "end_row"=None,
            # image_column = ,
        },        
        {
            'experiment_name': 'Llama-3.3-70b Q2', #🟢
            'excel_path': r"/workspace/data_to_runpod.xlsx",
            
            'prompt_column': 'TEXT4WEB',
            'temperature': 0.7,
            'model':  'llama3.3:70b-instruct-q2_K',
            
            'max_tokens': 1024,
            'system_prompt': None, # "You are a helpful assistant."
            'batch_size': 1,
            # pydantic_model = ,
            # "model_kwargs" = {},
            # "start_row" = 0,
            # "end_row"=None,
            # image_column = ,
        },        
        {
            'experiment_name': 'Llama-3.3-70b Q6', #🟢
            'excel_path': r"/workspace/data_to_runpod.xlsx",
            
            'prompt_column': 'TEXT4WEB',
            'temperature': 0.7,
            'model':  'llama3.3:70b-instruct-q6_K',
            
            'max_tokens': 1024,
            'system_prompt': None, # "You are a helpful assistant."
            'batch_size': 1,
            # pydantic_model = ,
            # "model_kwargs" = {},
            # "start_row" = 0,
            # "end_row"=None,
            # image_column = ,
        },        

    ]

    import os
    import nest_asyncio
    import asyncio
    from ezExperimenter.handlers.async_experiment_handler import run_multiple_experiments
    from ezExperimenter.generation.firework.async_generator import firework_async_generate


    base_output_folder = r"/workspace"
    ollama_base_url='https://spuikknnebejf2-11434.proxy.runpod.net/v1'
    ollama_api_key = "ollama"

    for experiment_config in experiment_configs:
        temp_experiment_configs = [experiment_config]

        results = await run_multiple_experiments(
            experiment_configs=temp_experiment_configs,
            base_output_folder=base_output_folder,
            generator_function=firework_async_generate,
            base_url=ollama_base_url,
            api_key=ollama_api_key
        )

        # Stop the specific Ollama model
        model_name = experiment_config['model']
        print(f"Stopping Ollama model: {model_name}")
        subprocess.run(f"ollama stop {model_name}", shell=True)

        # Sleep for 1 minute (60 seconds) between runs
        print(f"Waiting 1 minute after running {experiment_config['experiment_name']}...")
        await asyncio.sleep(60)