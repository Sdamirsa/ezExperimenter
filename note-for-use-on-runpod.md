# Setup Guide

You should create your runpod account and get a machine with cuda12.4 (you can filter machines with this cuda version). 

## chossing the machine

For the ducker use Better Ollama CUDA12 (this one: madiator2011/better-ollama:cuda12.4). Set the Volume disk (workspace folder where you will download models) to the desired size. For example if I want to use two modesl with 40 and 60 GBs, I will set this to 100+something (120 for example). I assume docker will only contain the engine, but for safety I set this to the size of biggest model plus somethng (it will cost you ~0.05$/h). In the example I provided I will set this to 60+something (90 for example). The container I mentuined will give you Ollama API (the port is exposed so you can talk with it) and the code server which is something like VS code enviroment. 

Use the PCIE GPU if you are using one GPU (one H100 PCIe etc), or use SXM for multiple GPU. Basically SXM means higher transfer rate between GPUs, and NVL means higher transfer rate to CPU (from GPU). So if you are using one GPU, go with PCIE

## 1. Get the modules from my github: 
    !git clone https://github.com/Sdamirsa/ezExperimenter

Note that this will save the code in ezExperimenter in the main (and not the workspace)

## 2. You should change the run_multiple_experiments.py to use tdqm and note the notebook

change this:

from tqdm.notebook import tqdm

to this:

from tqdm import tqdm

## 3. Install libraries:

    pip install pandas openai openpyxl tdqm pydantic numpy nest-asyncio asyncio

## 4. Pull your models (and save it in the workspace)

    cd workspace
    ollama pull llama3.3:70b-instruct-q4_K_M

## 5. Define the experiment like this:

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



## 6. And the code to run is:


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


## 5-6 for multiple models
If you use multiple models, they will remain in memroy for 5 minutes after final use. This may cause error. See this: https://github.com/ollama/ollama/blob/main/docs/faq.md#how-do-i-keep-a-model-loaded-in-memory-or-make-it-unload-immediately

To solve this you can either stop the model and go for the next model, or use keep_alive  parameter to be set to 0.

Notes: Ollama can do prallel runs if you have sufficien GPU memory. See this: https://github.com/ollama/ollama/blob/main/docs/faq.md#how-does-ollama-handle-concurrent-requests

Solution for running multiple models:


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
            'excel_path': r"/workspace/data_to_runpod copy.xlsx",
            
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
            'excel_path': r"/workspace/data_to_runpod copy.xlsx",
            
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

    import subprocess
    import os
    import nest_asyncio
    import asyncio
    from ezExperimenter.handlers.async_experiment_handler import run_multiple_experiments
    from ezExperimenter.generation.firework.async_generator import firework_async_generate
    import subprocess

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
        print(f"Waiting 30 sec after running {experiment_config['experiment_name']}...")
        await asyncio.sleep(30)

















# speed and result on A100:
INFO:ezExperimenter.handlers.async_experiment_handler:
Starting experiment: Llama-3.3-70b Q4 (1/1)
INFO:ezExperimenter.handlers.async_experiment_handler:Saved experiment inputs to: /workspace/batch_experiment_20241216_123817/Llama-3.3-70b Q4/Llama-3.3-70b Q4_inputs.xlsx

----------------------------------------
      ExP: Llama-3.3-70b Q4
----------------------------------------
  0%|          | 0/3 [00:00<?, ?it/s]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:38:17.473245
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:38:48.438912. Duration: 0:00:30.965667
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (1/3 rows processed)
 33%|███▎      | 1/3 [00:31<01:02, 31.00s/it]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:38:48.477210
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:39:27.344603. Duration: 0:00:38.867393
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (2/3 rows processed)
 67%|██████▋   | 2/3 [01:09<00:35, 35.66s/it]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:39:27.388897
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:40:11.091723. Duration: 0:00:43.702826
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (3/3 rows processed)
100%|██████████| 3/3 [01:53<00:00, 37.89s/it]
INFO:ezExperimenter.handlers.async_experiment_handler:Final results saved to: /workspace/batch_experiment_20241216_123817/Llama-3.3-70b Q4/Llama-3.3-70b Q4_outputs.xlsx
INFO:ezExperimenter.handlers.async_experiment_handler:Completed experiment: Llama-3.3-70b Q4
INFO:ezExperimenter.handlers.async_experiment_handler:Processed 3 prompts
INFO:ezExperimenter.handlers.async_experiment_handler:Average inference time: 37.81s
INFO:ezExperimenter.handlers.async_experiment_handler:Average tokens: 1110.3
INFO:ezExperimenter.handlers.async_experiment_handler:
Experiment summary saved to: /workspace/batch_experiment_20241216_123817/experiment_summary.xlsx
Stopping Ollama model: llama3.3:70b-instruct-q4_K_M
Waiting 30 sec after running Llama-3.3-70b Q4...
INFO:ezExperimenter.handlers.async_experiment_handler:
Starting experiment: Llama-3.3-70b Q2 (1/1)
INFO:ezExperimenter.handlers.async_experiment_handler:Saved experiment inputs to: /workspace/batch_experiment_20241216_124041/Llama-3.3-70b Q2/Llama-3.3-70b Q2_inputs.xlsx

----------------------------------------
      ExP: Llama-3.3-70b Q2
----------------------------------------
  0%|          | 0/3 [00:00<?, ?it/s]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:40:41.334073
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:41:51.285598. Duration: 0:01:09.951525
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (1/3 rows processed)
 33%|███▎      | 1/3 [01:09<02:19, 69.98s/it]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:41:51.318287
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:42:22.776988. Duration: 0:00:31.458701
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (2/3 rows processed)
 67%|██████▋   | 2/3 [01:41<00:47, 47.35s/it]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:42:22.823021
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:43:06.147211. Duration: 0:00:43.324190
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (3/3 rows processed)
100%|██████████| 3/3 [02:24<00:00, 48.29s/it]
INFO:ezExperimenter.handlers.async_experiment_handler:Final results saved to: /workspace/batch_experiment_20241216_124041/Llama-3.3-70b Q2/Llama-3.3-70b Q2_outputs.xlsx
INFO:ezExperimenter.handlers.async_experiment_handler:Completed experiment: Llama-3.3-70b Q2
INFO:ezExperimenter.handlers.async_experiment_handler:Processed 3 prompts
INFO:ezExperimenter.handlers.async_experiment_handler:Average inference time: 48.21s
INFO:ezExperimenter.handlers.async_experiment_handler:Average tokens: 979.3
INFO:ezExperimenter.handlers.async_experiment_handler:
Experiment summary saved to: /workspace/batch_experiment_20241216_124041/experiment_summary.xlsx
Stopping Ollama model: llama3.3:70b-instruct-q2_K
Waiting 30 sec after running Llama-3.3-70b Q2...
INFO:ezExperimenter.handlers.async_experiment_handler:
Starting experiment: Llama-3.3-70b Q6 (1/1)
INFO:ezExperimenter.handlers.async_experiment_handler:Saved experiment inputs to: /workspace/batch_experiment_20241216_124336/Llama-3.3-70b Q6/Llama-3.3-70b Q6_inputs.xlsx

----------------------------------------
      ExP: Llama-3.3-70b Q6
----------------------------------------
  0%|          | 0/3 [00:00<?, ?it/s]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:43:36.360219
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 524 "
INFO:openai._base_client:Retrying request to /chat/completions in 0.423824 seconds
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:46:32.717695. Duration: 0:02:56.357476
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (1/3 rows processed)
 33%|███▎      | 1/3 [02:56<05:52, 176.39s/it]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:46:32.754628
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:47:28.804954. Duration: 0:00:56.050326
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (2/3 rows processed)
 67%|██████▋   | 2/3 [03:52<01:45, 105.63s/it]INFO:ezExperimenter.generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 12:47:28.843899
INFO:httpx:HTTP Request: POST https://spuikknnebejf2-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:ezExperimenter.generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 12:48:18.787421. Duration: 0:00:49.943522
INFO:ezExperimenter.handlers.async_experiment_handler:Progress saved (3/3 rows processed)
100%|██████████| 3/3 [04:42<00:00, 94.16s/it] 
INFO:ezExperimenter.handlers.async_experiment_handler:Final results saved to: /workspace/batch_experiment_20241216_124336/Llama-3.3-70b Q6/Llama-3.3-70b Q6_outputs.xlsx
INFO:ezExperimenter.handlers.async_experiment_handler:Completed experiment: Llama-3.3-70b Q6
INFO:ezExperimenter.handlers.async_experiment_handler:Processed 3 prompts
INFO:ezExperimenter.handlers.async_experiment_handler:Average inference time: 94.08s
INFO:ezExperimenter.handlers.async_experiment_handler:Average tokens: 1087.0
INFO:ezExperimenter.handlers.async_experiment_handler:
Experiment summary saved to: /workspace/batch_experiment_20241216_124336/experiment_summary.xlsx
Stopping Ollama model: llama3.3:70b-instruct-q6_K
Waiting 30 sec after running Llama-3.3-70b Q6...



# Speed on H100
INFO:handlers.async_experiment_handler:
Starting experiment: Llama-3.2-90b Q4 (1/1)
INFO:handlers.async_experiment_handler:Saved experiment inputs to: /workspace/Result/batch_experiment_20241216_134344/Llama-3.2-90b Q4/Llama-3.2-90b Q4_inputs.xlsx

----------------------------------------
      ExP: Llama-3.2-90b Q4
----------------------------------------
  0%|          | 0/3 [00:00<?, ?it/s]INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:43:45.153813
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 524 "
INFO:openai._base_client:Retrying request to /chat/completions in 0.401330 seconds
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:46:32.744948. Duration: 0:02:47.591135
INFO:handlers.async_experiment_handler:Progress saved (1/3 rows processed)
 33%|███▎      | 1/3 [02:47<05:35, 167.69s/it]INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:46:32.841213
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:46:55.951972. Duration: 0:00:23.110759
INFO:handlers.async_experiment_handler:Progress saved (2/3 rows processed)
 67%|██████▋   | 2/3 [03:10<01:22, 82.68s/it] INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:46:56.017174
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:47:24.119266. Duration: 0:00:28.102092
INFO:handlers.async_experiment_handler:Progress saved (3/3 rows processed)
100%|██████████| 3/3 [03:39<00:00, 73.01s/it]
INFO:handlers.async_experiment_handler:Final results saved to: /workspace/Result/batch_experiment_20241216_134344/Llama-3.2-90b Q4/Llama-3.2-90b Q4_outputs.xlsx
INFO:handlers.async_experiment_handler:Completed experiment: Llama-3.2-90b Q4
INFO:handlers.async_experiment_handler:Processed 3 prompts
INFO:handlers.async_experiment_handler:Average inference time: 72.70s
INFO:handlers.async_experiment_handler:Average tokens: 897.0
INFO:handlers.async_experiment_handler:
Experiment summary saved to: /workspace/Result/batch_experiment_20241216_134344/experiment_summary.xlsx
Stopping Ollama model: llama3.2-vision:90b-instruct-q4_K_M
Waiting 30 sec after running Llama-3.2-90b Q4...
INFO:handlers.async_experiment_handler:
Starting experiment: Llama-3.2-11b Q8 (1/1)
INFO:handlers.async_experiment_handler:Saved experiment inputs to: /workspace/Result/batch_experiment_20241216_134754/Llama-3.2-11b Q8/Llama-3.2-11b Q8_inputs.xlsx

----------------------------------------
      ExP: Llama-3.2-11b Q8
----------------------------------------
  0%|          | 0/3 [00:00<?, ?it/s]INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:47:54.478183
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:48:34.339327. Duration: 0:00:39.861144
INFO:handlers.async_experiment_handler:Progress saved (1/3 rows processed)
 33%|███▎      | 1/3 [00:39<01:19, 39.93s/it]INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:48:34.412337
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:48:40.503600. Duration: 0:00:06.091263
INFO:handlers.async_experiment_handler:Progress saved (2/3 rows processed)
 67%|██████▋   | 2/3 [00:46<00:20, 20.07s/it]INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:48:40.575616
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:48:46.407688. Duration: 0:00:05.832072
INFO:handlers.async_experiment_handler:Progress saved (3/3 rows processed)
100%|██████████| 3/3 [00:51<00:00, 17.33s/it]
INFO:handlers.async_experiment_handler:Final results saved to: /workspace/Result/batch_experiment_20241216_134754/Llama-3.2-11b Q8/Llama-3.2-11b Q8_outputs.xlsx
INFO:handlers.async_experiment_handler:Completed experiment: Llama-3.2-11b Q8
INFO:handlers.async_experiment_handler:Processed 3 prompts
INFO:handlers.async_experiment_handler:Average inference time: 17.17s
INFO:handlers.async_experiment_handler:Average tokens: 893.7
INFO:handlers.async_experiment_handler:
Experiment summary saved to: /workspace/Result/batch_experiment_20241216_134754/experiment_summary.xlsx
Stopping Ollama model: llama3.2-vision:11b-instruct-q8_0
Waiting 30 sec after running Llama-3.2-11b Q8...
INFO:handlers.async_experiment_handler:
Starting experiment: Llama-3.2-11b Q4 (1/1)
INFO:handlers.async_experiment_handler:Saved experiment inputs to: /workspace/Result/batch_experiment_20241216_134916/Llama-3.2-11b Q4/Llama-3.2-11b Q4_inputs.xlsx

----------------------------------------
      ExP: Llama-3.2-11b Q4
----------------------------------------
  0%|          | 0/3 [00:00<?, ?it/s]INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:49:16.753854
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:49:38.914841. Duration: 0:00:22.160987
INFO:handlers.async_experiment_handler:Progress saved (1/3 rows processed)
 33%|███▎      | 1/3 [00:22<00:44, 22.24s/it]INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:49:38.997883
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:49:43.916498. Duration: 0:00:04.918615
INFO:handlers.async_experiment_handler:Progress saved (2/3 rows processed)
 67%|██████▋   | 2/3 [00:27<00:12, 12.13s/it]INFO:generation.firework.async_generator:Starting LLM request 0 at 2024-12-16 13:49:44.046799
INFO:httpx:HTTP Request: POST https://f80g6qtg9ni92w-11434.proxy.runpod.net/v1/chat/completions "HTTP/1.1 200 OK"
INFO:generation.firework.async_generator:Completed LLM request 0 at 2024-12-16 13:49:50.215333. Duration: 0:00:06.168534
INFO:handlers.async_experiment_handler:Progress saved (3/3 rows processed)
100%|██████████| 3/3 [00:33<00:00, 11.18s/it]
INFO:handlers.async_experiment_handler:Final results saved to: /workspace/Result/batch_experiment_20241216_134916/Llama-3.2-11b Q4/Llama-3.2-11b Q4_outputs.xlsx
INFO:handlers.async_experiment_handler:Completed experiment: Llama-3.2-11b Q4
INFO:handlers.async_experiment_handler:Processed 3 prompts
INFO:handlers.async_experiment_handler:Average inference time: 11.00s
INFO:handlers.async_experiment_handler:Average tokens: 873.0
INFO:handlers.async_experiment_handler:
Experiment summary saved to: /workspace/Result/batch_experiment_20241216_134916/experiment_summary.xlsx
Stopping Ollama model: llama3.2-vision:11b-instruct-q4_K_M
Waiting 30 sec after running Llama-3.2-11b Q4...



