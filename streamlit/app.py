import streamlit as st
import os
import pandas as pd
import json
from pathlib import Path
from pydantic import create_model, Field
from typing import Any, Dict, Optional, List, Union
from typing import Literal


st.set_page_config(page_title="LLM Experiment Runner", page_icon="💻", layout="wide")


def standardize_name(name: str) -> str:
    """Convert a given name to a standard form (no spaces or special chars), using underscores."""
    return name.strip().replace(" ", "_")


def build_pydantic_model(objects_dict: Dict[str, List[Dict[str, Any]]], root_object_name: str):
    created_models = {}

    def create_model_recursive(obj_name: str):
        if obj_name in created_models:
            return created_models[obj_name]

        fields = objects_dict[obj_name]
        field_params = {}
        for f in fields:
            f_name = standardize_name(f["name"])
            f_type_str = f["type"]
            f_desc = f["description"]
            choices = f.get("choices", None)
            # Determine Python type
            if f_type_str == "string":
                if choices:
                    f_type = Literal[tuple(choices)]
                else:
                    f_type = str
            elif f_type_str == "int":
                if choices:
                    int_choices = []
                    for c in choices:
                        try:
                            int_choices.append(int(c))
                        except:
                            pass
                    if int_choices:
                        f_type = Literal[tuple(int_choices)]
                    else:
                        f_type = int
                else:
                    f_type = int
            elif f_type_str == "float":
                if choices:
                    float_choices = []
                    for c in choices:
                        try:
                            float_choices.append(float(c))
                        except:
                            pass
                    if float_choices:
                        f_type = Literal[tuple(float_choices)]
                    else:
                        f_type = float
                else:
                    f_type = float
            elif f_type_str == "object":
                ref_obj = f.get("ref_object")
                if not ref_obj or ref_obj not in objects_dict:
                    raise ValueError(f"Reference object '{ref_obj}' not defined.")
                f_type = create_model_recursive(ref_obj)
            else:
                f_type = str

            field_params[f_name] = (f_type, Field(description=f_desc))

        m = create_model(obj_name, **field_params)
        created_models[obj_name] = m
        return m

    return create_model_recursive(root_object_name)


def print_object_fields(objects_dict: Dict[str, List[Dict[str,Any]]], obj_name: str, level: int = 0):
    """Print object fields hierarchically up to two levels."""
    indent = ">" * level
    fields = objects_dict.get(obj_name, [])
    for f in fields:
        line = f"{indent}- {f['name']} ({f['type']}): {f['description']}"
        if f['type'] == 'object':
            ref_obj = f.get('ref_object')
            st.write(line)
            if ref_obj and ref_obj in objects_dict and level < 2:
                # Recursively print one more level
                print_object_fields(objects_dict, ref_obj, level+1)
        else:
            # If choices defined, append them
            if f.get("choices"):
                line += f" [Allowed: {', '.join(f['choices'])}]"
            st.write(line)


# -------------------------
# Sidebar Configuration
with st.sidebar:
    st.title("Global Configuration")

    # Define instructions and default base URLs for each API type
    api_instructions = {
        "OpenAI API": """Create an Account and Obtain an API Key
\nEnsure your account is funded to use the API.
\nBase URL: https://api.openai.com/v1/
\nModels: OpenAI Models List: https://platform.openai.com/docs/models
\nExample: "gpt-4o" or "gpt-4o-2024-11-20"
""",
        "Firework (open-source)": """API:
Create an account and get an API key.
For serverless, fund your account.
For dedicated deployment (specifying a GPU and uploading/selecting your desired model), a credit card is required.
\nBase URL: https://api.fireworks.ai/inference/v1
\nModels:
Use the URL at the top of each model card for serverless models. See models: https://fireworks.ai/models?infrastructure=serverless
\nFor models unavailable as serverless, configure a GPU and deploy the model before retrieving its name.
\nExample: 'accounts/fireworks/models/llama-v3p2-3b-instruct'
""",
        "Runpod (own model)": """Deploy Your Own Model on Runpod (Serverless or Dedicated Deployment)
Guide: Runpod Deployment Instructions
Note: Some models may not be supported in vLLM, which can result in nonsensical outputs or failure to generate responses.
""",
        "Hugging Face": """Hugging Face (Supports Most Models with HF Inference)
\nAPI: Obtain an access token from your Profile Dashboard.
A credit card is required to activate API access.
\nBase URL: https://api-inference.huggingface.co/v1/
\nModels:
Any model supporting the Inference API (serverless) can be used.
Navigate to your desired model, select the "DEPLOY" dropdown at the top-right, and verify the "Inference API (serverless)" option is available.
\nExample: Qwen/Qwen2.5-7B-Instruct
""",
        "Ollama (local)": """Ollama (Local Deployment of Open-Source Models)
\nSteps:
\n- Install Ollama on your local system.
\n- Pull a model by running: ollama pull <model-name:size> (downloads the model locally).
\nNote: Batch size must be set to 1 for local mode.
\nBase URL: http://localhost:11434/v1/
\nAPI Key: "ollama"
""",
        "Other (OpenAI compatible API)": """Other Platforms
Many modern platforms now offer OpenAI-compatible APIs.
To integrate, define the platform-specific base_url and api parameters in your code.
"""
    }

    # Define default base URLs for each option
    default_base_urls = {
        "OpenAI API": "https://api.openai.com/v1/",
        "Firework (open-source)": "https://api.fireworks.ai/inference/v1",
        "Runpod (own model)": "",
        "Hugging Face": "https://api-inference.huggingface.co/v1/",
        "Ollama (local)": "http://localhost:11434/v1/",
        "Other (OpenAI compatible API)": "https://api.openai.com/v1/"
    }
    default_env_api = {
        "OpenAI API": "OPENAI_API_KEY",
        "Firework (open-source)": "FIREWORKS_API_KEY",
        "Runpod (own model)": "RUNPOD_API_KEY",
        "Hugging Face": "HF_API_KEY",
        "Ollama (local)": "ollama",
        "Other (OpenAI compatible API)": "define your api key in .env"
    }
    
    api_type = st.selectbox(
        "Select API Type",
        options=list(api_instructions.keys()),
        help="Select which type of API you want to use for model inference."
    )

    # Automatically set guidance and default base URL based on the selected API type
    st.write("**API Guidance:**", api_instructions[api_type])

    openai_base_url = st.text_input(
        "Base URL",
        value=default_base_urls[api_type],
        help="Specify the base endpoint for the chosen API."
    )

    env_var_name = st.text_input(
        "API Key Environment Variable Name",
        value=default_env_api[api_type],
        help="Name of the environment variable that stores your API key. You should save this in .env file."
    )

    global_base_output_folder = st.text_input(
        "Base Output Folder",
        value=r"C:\path\to\output",
        help="Folder where results will be saved."
    )

st.title("ezExperiment Runner: LLM/VLM (w/wo Structured Output) ")
st.write("Configure multiple experiments. Each experiment can have a pydantic schema with nested objects and enumerations. After configuration, you can run them together and generate code.")

st.header("Experiments Setup")
num_experiments = st.number_input("Number of Experiments", min_value=1, value=1, step=1, help="Set how many experiments you want to configure and run.")

# Initialize experiments in session state
if "experiments" not in st.session_state:
    st.session_state["experiments"] = [{} for _ in range(num_experiments)]
elif len(st.session_state["experiments"]) != num_experiments:
    if len(st.session_state["experiments"]) < num_experiments:
        st.session_state["experiments"].extend([{} for _ in range(num_experiments - len(st.session_state["experiments"]))])
    else:
        st.session_state["experiments"] = st.session_state["experiments"][:num_experiments]

tabs = st.tabs([f"Experiment {i+1}" for i in range(num_experiments)])


for idx, tab in enumerate(tabs):
    with tab:
        st.subheader(f"Experiment {idx+1} Configuration")

        # Initialize keys for pydantic for this experiment
        if f"pydantic_objects_{idx}" not in st.session_state:
            st.session_state[f"pydantic_objects_{idx}"] = {}

        # Maintain a list of column names per experiment
        if f"columns_{idx}" not in st.session_state:
            st.session_state[f"columns_{idx}"] = []

        col1, col2 = st.columns(2)
        with col1:
            exp_name_raw = st.text_input("Experiment Name", value=f"my_experiment_{idx+1}", key=f"exp_name_{idx}",
                                         help="A unique name for this experiment.")
            exp_name = standardize_name(exp_name_raw)

            excel_path = st.text_input("Excel File Path", value="", key=f"excel_path_{idx}",
                                       help="Path to the Excel or JSON file containing the prompts/data.")
            # New button to read column names from the provided Excel path
            if st.button("Read Column Names", key=f"read_cols_{idx}"):
                if excel_path.strip() and Path(excel_path.strip()).exists():
                    try:
                        df_test = pd.read_excel(excel_path.strip())
                        columns = list(df_test.columns)
                        st.session_state[f"columns_{idx}"] = columns
                        st.success("Column names read successfully.")
                    except Exception as e:
                        st.warning(f"Failed to read file: {e}")
                else:
                    st.warning("Please provide a valid Excel file path before reading columns.")

            if st.session_state[f"columns_{idx}"]:
                prompt_column = st.selectbox("Prompt Column Name", st.session_state[f"columns_{idx}"], key=f"prompt_column_{idx}",
                                             help="Select the column that contains the text prompt.")
            else:
                prompt_column_raw = st.text_input("Prompt Column Name", value="TEXT4WEB", key=f"prompt_column_text_{idx}",
                                                  help="Specify the column in the input file that contains the text prompt.")
                prompt_column = standardize_name(prompt_column_raw)

            temperature = st.number_input("Temperature", value=1.0, min_value=0.0, max_value=2.0, step=0.1, key=f"temperature_{idx}",
                                          help="Adjust the model's creativity/variance. Higher values mean more randomness.")
            model = st.text_input("Model Name", value="gpt-4o-2024-08-06", key=f"model_{idx}",
                                  help="Name of the model to be used for generation.")

        with col2:
            default_batch_size = 1 if api_type == "Ollama (local)" else 30
            batch_size = st.number_input("Batch Size", value=default_batch_size, min_value=1, key=f"batch_size_{idx}",
                                         help="Number of prompts to send per batch for faster processing.")
            max_tokens = st.number_input("Max Tokens (optional)", value=1024, min_value=1, step=1, key=f"max_tokens_{idx}",
                                         help="The maximum number of tokens for the model's response (optional).")
            system_prompt = st.text_area("System Prompt (optional)", key=f"system_prompt_{idx}",
                                         help="A system-level prompt that sets the model's context or instructions.")
            start_row = st.number_input("Start Row (0-based)", value=0, min_value=0, key=f"start_row_{idx}",
                                        help="The first row in the input file to process (0-based indexing).")
            end_row_input = st.text_input("End Row (optional, leave empty for all)", value="", key=f"end_row_{idx}",
                                          help="The last row to process (exclusive). Leave blank to process all rows.")
            if end_row_input.strip() == "":
                end_row = None
            else:
                try:
                    end_row = int(end_row_input)
                except ValueError:
                    end_row = None

        use_image = st.checkbox("Use Image Column?", value=False, key=f"use_image_{idx}",
                                help="Check this if your input file has a column containing image paths/URLs.")
        image_column = None
        if use_image:
            if st.session_state[f"columns_{idx}"]:
                image_column = st.selectbox("Image Column Name", st.session_state[f"columns_{idx}"], key=f"image_column_{idx}",
                                            help="Select the column that contains image paths.")
            else:
                image_column_raw = st.text_input("Image Column Name", key=f"image_column_text_{idx}",
                                                 help="Specify the column in the input file that contains image paths.")
                image_column = standardize_name(image_column_raw)

        # Checkbox to enable pydantic structuring
        want_pydantic = st.checkbox("Do you want to structure the output (with pydantic)?", key=f"want_pydantic_{idx}",
                                    help="Check to define a structured output schema for the model's responses.")

        if want_pydantic:
            # Show pydantic configuration section
            left_col, mid_col = st.columns([1,4])
            with mid_col:
                st.subheader("Pydantic Object Configuration")
                st.write("Define multiple named objects. Fields can have allowed values (comma-separated) to restrict output.")

                with st.expander("Add a new pydantic object"):
                    new_object_name_raw = st.text_input("Object Name", key=f"new_object_name_{idx}",
                                                        help="Name of the new object you want to define (e.g., 'User').")
                    if st.button("Create Object", key=f"create_object_{idx}"):
                        new_object_name = standardize_name(new_object_name_raw)
                        if new_object_name.strip():
                            if new_object_name in st.session_state[f"pydantic_objects_{idx}"]:
                                st.warning("Object with this name already exists.")
                            else:
                                st.session_state[f"pydantic_objects_{idx}"][new_object_name] = []

                object_list = list(st.session_state[f"pydantic_objects_{idx}"].keys())
                if object_list:
                    selected_object = st.selectbox("Select Object to Edit", object_list, key=f"selected_object_{idx}",
                                                   help="Select an object to add/edit fields.")
                    if selected_object:
                        st.write(f"Editing fields for object: **{selected_object}**")
                        field_name_raw = st.text_input("Field Name", key=f"field_name_{idx}",
                                                       help="Name of the field.")
                        
                        field_name_std = standardize_name(field_name_raw)
                        field_type = st.selectbox("Field Type", ["string", "int", "float", "object"], key=f"field_type_{idx}",
                                                  help="Data type for this field. 'object' references another pydantic object.")
                        is_list = st.checkbox("Is this field a list?", key=f"field_is_list_{idx}", help="Check this if the field should be a list of items.")
                        field_desc = st.text_input("Field Description", "", key=f"field_desc_{idx}",
                                                   help="A description of what this field represents.")
                        choices_text = st.text_input("Allowed Values (optional, comma-separated)", "", key=f"choices_text_{idx}",
                                                     help="Restrict field values to these choices (for string/int/float).")

                        ref_object = None
                        if field_type == "object":
                            other_objects = [o for o in object_list if o != selected_object]
                            if not other_objects:
                                st.warning("No other objects defined. Please define another object first to reference it here.")
                            else:
                                ref_object = st.selectbox("Select Object to Reference", other_objects, key=f"ref_obj_sel_{idx}",
                                                          help="Select the object this field should reference.")

                        if st.button("Add Field", key=f"add_field_button_{idx}",
                                     help="Add this field to the current object."):
                            if not field_name_std.strip():
                                st.warning("Field name cannot be empty.")
                            else:
                                field_info = {
                                    "name": field_name_std.strip(),
                                    "type": field_type,
                                    "description": field_desc.strip(),
                                    "is_list": is_list
                                }
                                if field_type == "object" and ref_object:
                                    field_info["ref_object"] = ref_object

                                # Parse choices
                                if choices_text.strip():
                                    ch_list = [c.strip() for c in choices_text.split(",") if c.strip()]
                                    if ch_list:
                                        field_info["choices"] = ch_list

                                st.session_state[f"pydantic_objects_{idx}"][selected_object].append(field_info)

                        # Show current fields hierarchically
                        st.write("Current Fields (up to two-level hierarchy):")
                        print_object_fields(st.session_state[f"pydantic_objects_{idx}"], selected_object)

                        if st.button("Clear Fields", key=f"clear_obj_fields_{idx}",
                                     help="Remove all fields from this object."):
                            st.session_state[f"pydantic_objects_{idx}"][selected_object] = []
                else:
                    st.info("No objects defined yet. Add a new object above.")

                root_object = None
                if object_list:
                    root_object_selection = st.selectbox("Select Root Object (optional)", ["None"] + object_list, key=f"root_object_{idx}",
                                               help="Select which object will be the root level schema for the output.")
                    if root_object_selection == "None":
                        root_object = None
                    else:
                        root_object = root_object_selection
        else:
            # User doesn't want pydantic
            # Clear any existing objects
            st.session_state[f"pydantic_objects_{idx}"] = {}
            root_object = None

        # Store experiment parameters
        st.session_state["experiments"][idx] = {
            "experiment_name": exp_name,
            "excel_path": excel_path.strip(),
            "prompt_column": prompt_column,
            "temperature": temperature,
            "model": model,
            "max_tokens": max_tokens if max_tokens else None,
            "system_prompt": system_prompt.strip() if system_prompt.strip() else None,
            "batch_size": batch_size,
            "base_output_folder": global_base_output_folder,
            "openai_base_url": openai_base_url,
            "env_var_name": env_var_name,
            "start_row": start_row,
            "end_row": end_row,
            "use_image": use_image,
            "image_column": image_column,
            "pydantic_objects": st.session_state[f"pydantic_objects_{idx}"] if want_pydantic else {},
            "root_object": root_object if want_pydantic else None
        }

st.header("Run Experiments")

col_run, col_code = st.columns([1,1])
with col_run:
    run_all_button = st.button("Run All Experiments", help="Click to run all configured experiments in sequence.")

with col_code:
    generate_code_button = st.button("Provide the raw code", help="Generate Python code that reproduces these experiment configurations.")

if run_all_button:
    st.error("Sorry I made this option unavailble to avoid leaking of your IP. If you have required (minimum) coding skill, you can go to the code, and un-hashtag lines below this code. It is possible that you get some errors due to the challenge of asyncio in streamlit, but you can solve it : ) ")
    # # Validate unique experiment names
    # experiment_names = [exp["experiment_name"] for exp in st.session_state["experiments"]]
    # if len(experiment_names) != len(set(experiment_names)):
    #     st.error("Experiment names must be unique. Please change duplicate experiment names.")
    #     st.stop()

    # experiments_to_run = []
    # pydantic_models_map = {}
    # for idx, exp_params in enumerate(st.session_state["experiments"]):
    #     if not exp_params["excel_path"]:
    #         st.error(f"Please provide a valid Excel file path for Experiment {idx+1}.")
    #         st.stop()
    #     if not Path(exp_params["excel_path"]).exists():
    #         st.error(f"Excel file not found at provided path for Experiment {idx+1}.")
    #         st.stop()
    #     if not exp_params["prompt_column"]:
    #         st.error(f"Please specify the prompt column name for Experiment {idx+1}.")
    #         st.stop()
    #     if not exp_params["model"]:
    #         st.error(f"Please specify the model name for Experiment {idx+1}.")
    #         st.stop()
    #     env_var_name = exp_params["env_var_name"]
    #     openai_api_key = os.getenv("env_var_name")
    #     if not openai_api_key:
    #         st.error(f"Please provide a valid API key for Experiment {idx+1}.")
    #         st.stop()

    #     exp_config = {
    #         "experiment_name": exp_params["experiment_name"],
    #         "excel_path": str(exp_params["excel_path"]),
    #         "prompt_column": exp_params["prompt_column"],
    #         "temperature": exp_params["temperature"],
    #         "model": exp_params["model"],
    #         "max_tokens": exp_params["max_tokens"],
    #         "system_prompt": exp_params["system_prompt"],
    #         "batch_size": exp_params["batch_size"],
    #         "start_row": exp_params["start_row"],
    #         "end_row": exp_params["end_row"]
    #     }

    #     if exp_params["use_image"] and exp_params["image_column"]:
    #         exp_config["image_column"] = exp_params["image_column"]

    #     # If this experiment has a root_object, build a model and add to pydantic_models_map
    #     if exp_params["root_object"] and exp_params["pydantic_objects"]:
    #         try:
    #             pydantic_model = build_pydantic_model(exp_params["pydantic_objects"], exp_params["root_object"])
    #             pydantic_models_map[exp_params["experiment_name"]] = pydantic_model
    #         except Exception as e:
    #             st.error(f"Error building pydantic model for Experiment {idx+1}: {e}")
    #             st.stop()

    #     experiments_to_run.append((exp_config, exp_params["base_output_folder"], exp_params["openai_base_url"], exp_params["env_var_name"]))

    # st.write("Starting all experiments...")
    # progress_bar = st.progress(0)
    # status_text = st.empty()

    # async def run_all():
    #     results = await run_multiple_experiments(
    #         experiment_configs=[cfg for (cfg, _, _, _) in experiments_to_run],
    #         base_output_folder=experiments_to_run[0][1],
    #         generator_function=firework_async_generate,
    #         base_url=experiments_to_run[0][2],
    #         api_key=experiments_to_run[0][3],
    #         pydantic_models=pydantic_models_map if pydantic_models_map else None
    #     )
    #     return results

    # loop = asyncio.get_event_loop()
    # results = loop.run_until_complete(run_all())

    # for i in range(1, 101):
    #     progress_bar.progress(i)
    # status_text.write("All experiments complete!")

    # st.header("Experiments Results")
    # if results:
    #     for exp_name, config, result, error in results:
    #         st.subheader(f"Experiment: {exp_name}")
    #         if error:
    #             st.error(f"Experiment failed with error: {error}")
    #         else:
    #             st.success("Experiment succeeded!")
    #             df_in, df_out = result
    #             st.write("Input Summary:")
    #             st.write(df_in.head())
    #             st.write("Output Summary:")
    #             st.write(df_out.head())

    #             experiment_folder = Path(config.get("excel_path", "")).parent.parent
    #             summary_path = list(experiment_folder.glob("**/experiment_summary.xlsx"))
    #             if summary_path:
    #                 st.write(f"Experiment summary saved to: {summary_path[0]}")
    #             else:
    #                 st.write("No summary file found.")
    # else:
    #     st.error("No results returned.")


if generate_code_button:
    # Validate unique experiment names again
    experiment_names = [exp["experiment_name"] for exp in st.session_state["experiments"]]
    if len(experiment_names) != len(set(experiment_names)):
        st.error("Experiment names must be unique before generating code. Please fix duplicates.")
        st.stop()

    # Collect all objects defined across all experiments
    all_objects = {}
    root_objects_map = {}
    for exp in st.session_state["experiments"]:
        for obj_name, fields in exp["pydantic_objects"].items():
            all_objects[obj_name] = fields
        if exp['root_object'] and exp['pydantic_objects']:
            root_objects_map[exp['experiment_name']] = exp['root_object']

    def get_references(obj_name):
        refs = []
        for f in all_objects.get(obj_name, []):
            if f["type"] == "object" and "ref_object" in f:
                refs.append(f["ref_object"])
        return refs

    # Topological sort for object dependencies
    sorted_objects = []
    visited = {}
    def visit(obj):
        if obj in visited:
            return
        visited[obj] = "visiting"
        for r in get_references(obj):
            visit(r)
        visited[obj] = "done"
        sorted_objects.append(obj)

    for o in all_objects:
        visit(o)


    def class_name(obj_name):
        # Convert to a class-like name, just capitalize first letter of each word
        return "".join([word.capitalize() for word in obj_name.split("_")])

    code_lines = []
    code_lines.append("# Code generated by ezExperiment Runner")
    code_lines.append("")
    code_lines.append("import os")
    code_lines.append("import nest_asyncio")
    code_lines.append("import asyncio")
    code_lines.append("import pandas as pd")
    code_lines.append("from pydantic import BaseModel, Field")
    code_lines.append("from typing import Literal, List")
    code_lines.append("from pathlib import Path")
    code_lines.append("nest_asyncio.apply()")
    code_lines.append("")
    code_lines.append("from handlers.async_experiment_handler import run_multiple_experiments")
    code_lines.append("from generation.firework.async_generator import firework_async_generate")
    code_lines.append("from mytest.mytest import test_llm_functions")
    code_lines.append("")
    code_lines.append("# validate api")
    if env_var_name =="ollama":
        code_lines.append("api_key = \"ollama\"")
    else: 
        code_lines.append(f"api_key_var = {json.dumps(env_var_name)}")
        code_lines.append("api_key = os.getenv(api_key_var)")
        code_lines.append("if api_key is None:")
        code_lines.append("    raise ValueError(\"Please check your api key. You should save it in .env file and load it here.\")")
    
    code_lines.append("##########################")
    code_lines.append("###    ezExperiments   ###")
    code_lines.append("##########################")
    code_lines.append("experiment_configs = [")
    for idx, exp_params in enumerate(st.session_state["experiments"]):
        exp_dict = { 
            'experiment_name': exp_params['experiment_name'],
            'excel_path': exp_params['excel_path'],
            'prompt_column': exp_params['prompt_column'],
            'temperature': exp_params['temperature'],
            'model': exp_params['model'],
            'max_tokens': exp_params['max_tokens'],
            'system_prompt': exp_params['system_prompt'],
            'batch_size': exp_params['batch_size'],
            'start_row': exp_params['start_row'],
            'end_row': exp_params['end_row']
        }
        if exp_params['use_image'] and exp_params['image_column']:
            exp_dict['image_column'] = exp_params['image_column']
        if exp_params['root_object'] and exp_params['pydantic_objects']:
            exp_dict['pydantic_model'] = exp_params['root_object']

        code_lines.append("    {")
        for k, v in exp_dict.items():
            if k == 'pydantic_model':
                continue  # Skip this key
            if v is not None:
                code_lines.append(f"        '{k}': {json.dumps(v)},")
            else:
                code_lines.append(f"        '{k}': None,")
        code_lines.append("    },")
    code_lines.append("]")
    code_lines.append("")
    code_lines.append(f"base_output_folder = {json.dumps(global_base_output_folder)}")
    code_lines.append(f"base_url = {json.dumps(openai_base_url)}")
    code_lines.append(f"api_key = api_key")
    code_lines.append("")
    if all_objects:
        code_lines.append("# Pydantic models defined by user:")
        for obj_name in sorted_objects:
            fields = all_objects[obj_name]
            code_lines.append(f"class {class_name(obj_name)}(BaseModel):")
            if not fields:
                code_lines.append("    pass")
            else:
                for f in fields:
                    f_type = f["type"]
                    f_desc = f["description"]
                    ch = f.get("choices", [])
                    # Determine field type in code
                    if f_type == "string":
                        if ch:
                            vals = ", ".join([json.dumps(c) for c in ch])
                            py_type = f"Literal[{vals}]"
                        else:
                            py_type = "str"
                    elif f_type == "int":
                        if ch:
                            int_choices = [int(x) for x in ch if x.isdigit()]
                            if int_choices:
                                vals = ", ".join([str(x) for x in int_choices])
                                py_type = f"Literal[{vals}]"
                            else:
                                py_type = "int"
                        else:
                            py_type = "int"
                    elif f_type == "float":
                        if ch:
                            float_choices = []
                            for x in ch:
                                try:
                                    float_choices.append(float(x))
                                except:
                                    pass
                            if float_choices:
                                vals = ", ".join([str(x) for x in float_choices])
                                py_type = f"Literal[{vals}]"
                            else:
                                py_type = "float"
                        else:
                            py_type = "float"
                    elif f_type == "object":
                        ref_obj = f["ref_object"]
                        py_type = class_name(ref_obj)
                    else:
                        py_type = "str"

                    # If this field should be a list
                    if f.get("is_list", False):
                        py_type = f"List[{py_type}]"

                    code_lines.append(f"    {f['name']}: {py_type} = Field(..., description={json.dumps(f_desc)})")
            code_lines.append("")

    pydantic_map_needed = any(exp['root_object'] and exp['pydantic_objects'] for exp in st.session_state['experiments'])
    if pydantic_map_needed:
        code_lines.append("pydantic_models = {")
        for exp_params in st.session_state["experiments"]:
            if exp_params['root_object'] and exp_params['pydantic_objects']:
                code_lines.append(f"    '{exp_params['experiment_name']}': {class_name(exp_params['root_object'])},")
        code_lines.append("}")
    else:
        code_lines.append("pydantic_models = {}")
    # code_lines.append("")
    # code_lines.append("# First, test the setup with known test data:")
    # code_lines.append("test_llm_functions(base_url, api_key, experiment_configs[0]['model'], use_image=False, base_output_folder=base_output_folder)")
    code_lines.append("")
    code_lines.append("async def main():")
    code_lines.append("    results = await run_multiple_experiments(")
    code_lines.append("        experiment_configs=experiment_configs,")
    code_lines.append("        base_output_folder=base_output_folder,")
    code_lines.append("        generator_function=firework_async_generate,")
    code_lines.append("        base_url=base_url,")
    code_lines.append("        api_key=api_key,")
    if pydantic_map_needed:
        code_lines.append("        pydantic_models=pydantic_models")
    code_lines.append("    )")
    code_lines.append("    return results")
    code_lines.append("")
    code_lines.append("loop = asyncio.get_event_loop()")
    code_lines.append("final_results = loop.run_until_complete(main())")
    code_lines.append('print("All experiments complete!")')
    code_snippet = "\n".join(code_lines)
    st.code(code_snippet, language="python")
    st.info("Copy the above code and run it in your environment. Replace file paths, API keys, and ensure object names and references are suitable.")
