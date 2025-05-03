# app.py
# Streamlit app to design, run batch, and view GPT agents using the gpts library

import os
import yaml
import json
import streamlit as st
from gpts import gpt

# Retrieve available model constants from gpt module
model_options = [
    gpt.GPT_4_dot_1,
    gpt.GPT_4_dot_1_mini,
    gpt.GPT_4_dot_1_nano,
    gpt.GPT_4o,
    gpt.GPT_4o_mini
]

# Ensure directory exists
def ensure_config_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

# Page configuration
st.set_page_config(page_title="GPT Agent Creator", layout="wide")

# Main title
st.title("GPT Agent Creator")

# Sidebar mode selector
mode = st.sidebar.selectbox(
    "Mode", ["Design Agent", "Run Agent", "View Existing Agents"]
)

config_dir = "config"
output_dir = "output"

# --- Design Agent ---
if mode == "Design Agent":
    st.header("Design a New Agent Specification")
    # Initialize spec storage
    if 'spec' not in st.session_state:
        st.session_state['spec'] = None

    # Model selection
    default_idx = 0
    try:
        default_idx = model_options.index(gpt.default_model)
    except Exception:
        pass
    model_choice = st.selectbox("Select Model", model_options, index=default_idx)

    # Agent fields
    agent_name = st.text_input("Agent Name", "MyAgent")
    description = st.text_area("Task Description", "Summarize articles into bullet points.")
    placeholder = st.text_input("Input Placeholder", "{text}")

    # Generate specification
    if st.button("Design Agent", key="design_btn"):
        with st.spinner("Generating specification..."):
            st.session_state['spec'] = gpt.design_agent(
                agent_name=agent_name,
                task_description=description,
                input_placeholder=placeholder,
                model=model_choice
            )

    # Display and save
    if st.session_state['spec']:
        st.subheader("Agent Specification (YAML)")
        st.code(st.session_state['spec'])
        st.subheader("Save Agent Configuration")
        filename = st.text_input("Filename", value=f"{agent_name}.yaml", key="filename_input")
        if st.button("Save Agent", key="save_btn"):
            ensure_config_dir(config_dir)
            file_path = os.path.join(config_dir, filename)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(st.session_state['spec'])
                st.success(f"Agent configuration saved to {file_path}")
            except Exception as e:
                st.error(f"Failed to save file: {e}")

# --- Run Agent ---
elif mode == "Run Agent":
    st.header("Run Agent Batch")
    if not os.path.isdir(config_dir):
        st.warning(f"Config directory '{config_dir}' not found.")
    else:
        config_files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        if not config_files:
            st.info("No agent configuration files found.")
        else:
            selected_cfg = st.selectbox("Select an agent config", config_files)
            cfg_path = os.path.join(config_dir, selected_cfg)
            config = yaml.safe_load(open(cfg_path, encoding='utf-8'))

            st.subheader("Agent Settings")
            default_idx = 0
            try:
                default_idx = model_options.index(gpt.default_model)
            except Exception:
                pass
            model_choice = st.selectbox("Select Model", model_options, index=default_idx)
            max_tokens = st.number_input("Max Tokens", min_value=1, value=gpt.default_max_tokens)
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=gpt.default_temperature)
            json_format = st.text_input("JSON Format (optional)", value="")

            uploaded_file = st.file_uploader("Upload input JSONL file", type=["jsonl"])
            output_filename = st.text_input("Output filename", value="results.jsonl")

            if st.button("Run Batch", key="run_batch_btn"):
                if uploaded_file is None:
                    st.error("Please upload a JSONL file.")
                else:
                    lines = [l for l in uploaded_file.read().decode('utf-8').splitlines() if l.strip()]
                    total = len(lines)
                    if total == 0:
                        st.error("Uploaded file is empty.")
                    else:
                        agent_name = selected_cfg.rsplit('.', 1)[0]
                        agent = gpt.build(agent_name)
                        # Override settings
                        agent.gpt_model = model_choice
                        agent.max_tokens = max_tokens
                        agent.temperature = temperature
                        if json_format:
                            agent.json_format = json_format

                        ensure_config_dir(output_dir)
                        progress = st.progress(0)
                        results = []
                        for i, line in enumerate(lines):
                            try:
                                inp = json.loads(line)
                            except json.JSONDecodeError as e:
                                st.warning(f"Skipping invalid JSON on line {i+1}: {e}")
                                continue
                            try:
                                res = agent.run(inputs=inp)
                            except RuntimeError as e:
                                st.warning(f"Line {i+1} failed: {e}")
                                res = {"error": str(e)}
                            results.append(res)
                            progress.progress((i+1) / total)

                        out_path = os.path.join(output_dir, output_filename)
                        try:
                            with open(out_path, 'w', encoding='utf-8') as f:
                                for r in results:
                                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
                            st.success(f"Batch completed. Results saved to {out_path}")
                        except Exception as e:
                            st.error(f"Failed to save results: {e}")

# --- View Existing Agents ---
else:
    st.header("View Existing Agents")
    if not os.path.isdir(config_dir):
        st.warning(f"Config directory '{config_dir}' not found.")
    else:
        files = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        if not files:
            st.info("No agent configuration files found.")
        else:
            selected = st.selectbox("Select a config file", files)
            if st.button("Load Config", key="load_cfg_btn"):
                path = os.path.join(config_dir, selected)
                config = yaml.safe_load(open(path, encoding='utf-8'))
                st.subheader(f"Configuration: {selected}")
                st.json(config)
