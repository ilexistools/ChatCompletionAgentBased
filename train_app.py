# train_app.py
# Streamlit app to train/improve GPT agents using the gpts library

import os
import yaml
import json
import streamlit as st
from gpts import gpt

# Retrieve available model constants
model_options = [
    gpt.GPT_4_dot_1,
    gpt.GPT_4_dot_1_mini,
    gpt.GPT_4_dot_1_nano,
    gpt.GPT_4o,
    gpt.GPT_4o_mini
]

# Ensure directory exists
def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

# Page configuration
st.set_page_config(page_title="GPT Agent Trainer", layout="wide")

st.title("GPT Agent Trainer")

# Training method selection
method = st.sidebar.radio(
    "Training Method", ["Data-driven", "AI-driven", "Human-in-the-loop"]
)

# Load agent config
config_dir = "config"
if not os.path.isdir(config_dir):
    st.error(f"Config directory '{config_dir}' not found.")
    st.stop()
configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
if not configs:
    st.error("No agent configuration files in config/.")
    st.stop()
selected_cfg = st.selectbox("Select Agent Config", configs)
agent_name = selected_cfg.rsplit('.', 1)[0]
config = yaml.safe_load(open(os.path.join(config_dir, selected_cfg), encoding='utf-8'))
st.subheader("Agent Configuration")
st.json(config)

# Common settings
st.subheader("Trainer Settings")
default_idx = 0
try:
    default_idx = model_options.index(gpt.default_model)
except Exception:
    pass
trainer_model = st.selectbox("Select Trainer Model", model_options, index=default_idx)
verbose = st.checkbox("Verbose", value=False)

# Input uploads
st.subheader("Training Data Upload")
train_file = st.file_uploader("Upload training data JSONL", type=["jsonl"], key="train_data")
expected = None
if method == "Data-driven":
    expect_file = st.file_uploader("Upload expected outputs JSONL", type=["jsonl"], key="expected")

# Output config save
output_dir = "trained_config"
ensure_dir(output_dir)
out_filename = st.text_input("Output filename", value=f"{agent_name}_improved.yaml")

# Run training
if st.button("Start Training"):
    # Load training inputs
    if not train_file:
        st.error("Please upload training data file.")
        st.stop()
    train_lines = [l for l in train_file.read().decode('utf-8').splitlines() if l.strip()]
    training_data = []
    for i, line in enumerate(train_lines):
        try:
            training_data.append(json.loads(line))
        except Exception as e:
            st.warning(f"Skipping invalid JSON training line {i+1}: {e}")
    # Data-driven: load expected outputs
    if method == "Data-driven":
        if not expect_file:
            st.error("Please upload expected outputs file.")
            st.stop()
        exp_lines = [l for l in expect_file.read().decode('utf-8').splitlines() if l.strip()]
        expected = []
        for i, line in enumerate(exp_lines):
            try:
                expected.append(json.loads(line))
            except Exception:
                expected.append(line.strip())

    # Build agent
    agent = gpt.build(agent_name)
    agent.gpt_model = trainer_model
    # Invoke appropriate improvement
    st.info(f"Running {method} training...")
    if method == "Data-driven":
        improved = gpt.improve_gpt_prompt_by_data(
            agent,
            training_data,
            expected,
            trainer_model=trainer_model,
            verbose=verbose
        )
    elif method == "AI-driven":
        improved = gpt.improve_gpt_prompt_by_ai(
            agent,
            training_data,
            trainer_model=trainer_model,
            verbose=verbose
        )
    else:
        improved = gpt.improve_gpt_prompt_by_human(
            agent,
            training_data,
            trainer_model=trainer_model,
            verbose=verbose
        )

    # Display and save improved configuration
    st.subheader("Improved Configuration")
    st.code(improved)
    out_path = os.path.join(output_dir, out_filename)
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(improved)
        st.success(f"Improved configuration saved to {out_path}")
    except Exception as e:
        st.error(f"Failed to save improved config: {e}")
