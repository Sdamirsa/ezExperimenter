# Description

This is an all-in-one module for running LLM/VLM for generation, with or without structured output (Pydantic JSON).


🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩

🟩🟩 [Step-by-step instruction](https://youtu.be/NXbvN1i3x-g) 🟩🟩

🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩


# Setup
## 0- Python, VS code, and ezExperimenter code
- [Download and install python](https://www.python.org/downloads/)
- [Donwload and install VS code](https://code.visualstudio.com/download)
- Download this repo (go to [this link](https://download-directory.github.io/) and enter the url)
- Unzip and start the folder in VS code

## 1- Create virtual enviroment
    python -m venv venv

## 2- Activate your enviroment
    venv\Scripts\activate

## 3- install packages and libraries
    pip install -r requirements.txt

## 4- Get your api and base url. What is supported?
Here's the improved version of your guide with refined wording for clarity and consistency:

### 4. API and Model Deployment Guide

#### 4.a - OpenAI API
- **[Create an Account and Obtain an API Key](https://www.youtube.com/shorts/Hqyj_jJ4Lrk)**  
    - Ensure your account is funded to use the API.
- **Base URL:** `https://api.openai.com/v1/`
- **Models:** [OpenAI Models List](https://platform.openai.com/docs/models)  
    - Example: `"gpt-4o"` or `"gpt-4o-2024-11-20"`

#### 4.b - Fireworks for Open-Source Models (Serverless or Dedicated Deployment)
- **API:**  
    - Create an account and get an API key.  
    - For **serverless**, fund your account.  
    - For **dedicated deployment** (specifying a GPU and uploading/selecting your desired model), a credit card is required.
- **Base URL:** `https://api.fireworks.ai/inference/v1`
- **Models:**  
    - Use the URL at the top of each model card for serverless models.  
    - For models unavailable as serverless, configure a GPU and deploy the model before retrieving its name.  
    - Example: `'accounts/fireworks/models/llama-v3p2-3b-instruct'`

#### 4.c - Deploy Your Own Model on Runpod (Serverless or Dedicated Deployment)
- **Guide:** [Runpod Deployment Instructions](https://github.com/Sdamirsa/TiLense-4BlackBox-VLM)  
- **Note:** Some models may not be supported in vLLM, which can result in nonsensical outputs or failure to generate responses.

#### 4.d - Hugging Face (Supports Most Models with HF Inference)
- **API:** Obtain an access token from your [Profile Dashboard](https://huggingface.co/settings/tokens).  
    - A credit card is required to activate API access.
- **Base URL:** `https://api-inference.huggingface.co/v1/`
- **Models:**  
    - Any model supporting the Inference API (serverless) can be used.  
    - Navigate to your desired model, select the "DEPLOY" dropdown at the top-right, and verify the "Inference API (serverless)" option is available.  
    - Example: `Qwen/Qwen2.5-7B-Instruct`

#### 4.e - Ollama (Local Deployment of Open-Source Models)
- **Steps:**  
    1. Install Ollama on your local system.  
    2. Pull a model by running: `ollama pull <model-name:size>` (downloads the model locally).  
    - Note: Batch size must be set to 1 to avoid parallel computation issues on local machines.
- **Base URL:** `http://localhost:11434/v1/`
- **API Key:** `"ollama"`

#### 4.f - Other Platforms
- Many modern platforms now offer OpenAI-compatible APIs.  
- To integrate, define the platform-specific `base_url` and `api` parameters in your code.


## 5- Add your API to the .env file (you will read it from os.getenv, to have a safe use of API)

## 6- Then run the app
    streamlit run app.py

## 7- Use the app to get your code
The app will guide you step by step. You need to store all of your prompts (model inputs) and images (full path to images, separated by a comma) inside an Excel file so you can read from there, and create the experiment inputs and outputs in the desired destination folder.

# 8- run your code in notebook (Use.ipynb)

### To do:

<details>
<summary>Test functions</summary>

Add test to the app.py for testing the generation and handlers, for each selected api type. The current code only works with firework, so I hashtag it for now. 

</details>
