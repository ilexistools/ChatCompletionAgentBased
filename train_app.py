# train_app.py
# Streamlit app to train/improve GPT agents using a unified ui_options dict
# now using ui_improve_gpt_prompt_by_ai to avoid overwriting the original

import os
import yaml
import json
import streamlit as st
from gpts import gpt
# import your renamed version; adjust the path if it's in a different module


# Ensure directory exists
def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

# Page configuration
st.set_page_config(page_title="GPT Agent Trainer", layout="wide")
st.title("GPT Agent Trainer")

# session state
if 'ui_options' not in st.session_state:
    st.session_state['ui_options'] = {}
ui_options = st.session_state['ui_options']

# — Select agent configuration —
config_dir = "config"
ensure_dir(config_dir)
configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and not f.startswith('_')]
if not configs:
    st.warning(f"No .yaml files found in '{config_dir}' – please add your agent configs there.")
    st.stop()
ui_options['agent_config'] = st.selectbox("Select Agent Config (.yaml)", configs)
ui_options['agent_name']   = ui_options['agent_config'].rsplit('.', 1)[0]

# Load and show selected config
config_path = os.path.join(config_dir, ui_options['agent_config'])
with open(config_path, encoding='utf-8') as f:
    config = yaml.safe_load(f)
st.subheader("Agent Configuration")
st.json(config)

# — Agent Settings —
st.subheader("Agent Settings")
model_choices = [
    gpt.GPT_4_dot_1,
    gpt.GPT_4_dot_1_mini,
    gpt.GPT_4_dot_1_nano,
    gpt.GPT_4o,
    gpt.GPT_4o_mini
]
default_model = getattr(gpt, 'default_model', model_choices[0])
ui_options['model'] = st.selectbox(
    "Select Model",
    model_choices,
    index=model_choices.index(default_model) if default_model in model_choices else 0
)
ui_options['max_tokens'] = st.number_input(
    "Max Tokens", min_value=1, value=getattr(gpt, 'default_max_tokens', 1000)
)
ui_options['temperature'] = st.slider(
    "Temperature", min_value=0.0, max_value=1.0, value=getattr(gpt, 'default_temperature', 0.2)
)

# — Training data upload —
st.subheader("Training Data Upload (.jsonl)")
train_file = st.file_uploader("Upload training data JSONL", type=["jsonl"])
if train_file:
    text = train_file.read().decode('utf-8')
    lines = [l for l in text.splitlines() if l.strip()]
    ui_options['training_data'] = []
    for i, line in enumerate(lines, start=1):
        try:
            ui_options['training_data'].append(json.loads(line))
        except json.JSONDecodeError:
            st.warning(f"Skipping invalid JSON on line {i}")

# — Trainer settings —
st.subheader("Trainer Settings")
default_trainer = getattr(gpt, 'default_model', model_choices[0])
ui_options['trainer_model'] = st.selectbox(
    "Select Trainer Model",
    model_choices,
    index=model_choices.index(default_trainer) if default_trainer in model_choices else 0
)

# — Run AI-driven improvement with progress UI —
if st.button("Run AI-driven Improvement"):
    if not ui_options.get('training_data'):
        st.error("Please upload valid training data JSONL before running improvement.")
    else:
        # placeholders
        phase_text    = st.empty()
        progress_text = st.empty()
        progress_bar  = st.progress(0)

        st.info("Building agent and running AI-driven prompt improvement…")
        agent = gpt.build(ui_options['agent_name'])
        agent.gpt_model   = ui_options['model']
        agent.max_tokens  = ui_options['max_tokens']
        agent.temperature = ui_options['temperature']

        # call the renamed function
        improved_prompt = gpt.ui_improve_gpt_prompt_by_ai(
            agent,
            ui_options['training_data'],
            ui_options['trainer_model'],
            verbose=False,
            on_phase=lambda name: phase_text.text(f"🔄 {name}…"),
            on_step = lambda idx, total: (
                progress_text.text(f"[{idx}/{total}] processing…"),
                progress_bar.progress(int(idx * 100 / total))
            )
        )
        ui_options['improved_prompt'] = improved_prompt
        st.success("✅ Completed!")
        st.subheader("Improved Agent Configuration")
        st.code(improved_prompt)

# — Save improved prompt back into config/ and rerun to reload —
if 'improved_prompt' in ui_options:
    ensure_dir(config_dir)
    filename = st.text_input(
        "Save As (filename.yaml)",
        value=f"{ui_options['agent_name']}_improved.yaml"
    )
    if st.button("Save Improved Configuration", key="save_config"):
        save_path = os.path.join(config_dir, filename)
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(ui_options['improved_prompt'])
            st.success(f"Improved configuration saved to `{save_path}`")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error saving file: {e}")
