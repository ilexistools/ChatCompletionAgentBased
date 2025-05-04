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

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

# Page configuration
st.set_page_config(page_title="GPT Agent Hub", layout="wide")

# Main title
st.title("GPT Agent Hub")

# Sidebar mode selector
mode = st.sidebar.selectbox(
    "Mode", [
        "Design Agent", 
        "Run Agent", 
        "View Agents", 
        "Train Agent"
    ]
)

# Common directories
config_dir = "config"
output_dir = "output"
ensure_dir(config_dir)
ensure_dir(output_dir)

# --- Design Agent ---
if mode == "Design Agent":
    st.header("Design a New Agent Specification")
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
    if st.button("Design Agent"):
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
        filename = st.text_input("Filename", value=f"{agent_name}.yaml")
        if st.button("Save Agent"):
            file_path = os.path.join(config_dir, filename)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(st.session_state['spec'])
                st.success(f"Agent configuration saved to {file_path}")
            except Exception as e:
                st.error(f"Failed to save file: {e}")

# --- Run Agent Batch ---
elif mode == "Run Agent":
    st.header("Run Agent Batch")
    configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and not f.startswith('_')]
    if not configs:
        st.info("No agent configuration files found. Please design or add configs in the 'config' folder.")
    else:
        selected_cfg = st.selectbox("Select an agent config", configs)
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

        uploaded_file = st.file_uploader("Upload input JSONL file", type=["jsonl"] )
        output_filename = st.text_input("Output filename", value="results.jsonl")

        if st.button("Run Batch"):
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
                    agent.gpt_model = model_choice
                    agent.max_tokens = max_tokens
                    agent.temperature = temperature
                    if json_format:
                        agent.json_format = json_format

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
elif mode == "View Agents":
    st.header("View Existing Agents")
    files = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and not f.startswith('_')]
    if not files:
        st.info("No agent configuration files found.")
    else:
        selected = st.selectbox("Select a config file", files)
        if st.button("Load Config"):
            path = os.path.join(config_dir, selected)
            config = yaml.safe_load(open(path, encoding='utf-8'))
            st.subheader(f"Configuration: {selected}")
            st.json(config)

# --- Train Agent ---
# --- Train Agent ---
elif mode == "Train Agent":
    st.header("Train / Improve Agent Prompt")

    # initialize ui_options in session state
    if 'ui_options' not in st.session_state:
        st.session_state['ui_options'] = {}
    ui_options = st.session_state['ui_options']

    # select agent configuration
    ensure_dir(config_dir)
    configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml') and not f.startswith('_')]
    if not configs:
        st.warning(f"No .yaml files found in '{config_dir}' – please add your agent configs there.")
        st.stop()
    ui_options['agent_config'] = st.selectbox("Select Agent Config (.yaml)", configs)
    ui_options['agent_name']   = ui_options['agent_config'].rsplit('.', 1)[0]

    # load and display the selected config
    config_path = os.path.join(config_dir, ui_options['agent_config'])
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)
    st.subheader("Agent Configuration")
    st.json(config)

    # agent settings
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

    # training data upload
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

    # trainer settings
    st.subheader("Trainer Settings")
    default_trainer = getattr(gpt, 'default_model', model_choices[0])
    ui_options['trainer_model'] = st.selectbox(
        "Select Trainer Model",
        model_choices,
        index=model_choices.index(default_trainer) if default_trainer in model_choices else 0
    )

    # run AI-driven improvement
    if st.button("Run AI-driven Improvement"):
        if not ui_options.get('training_data'):
            st.error("Please upload valid training data JSONL before running improvement.")
        else:
            phase_text    = st.empty()
            progress_text = st.empty()
            progress_bar  = st.progress(0)

            st.info("Building agent and running AI-driven prompt improvement…")
            agent = gpt.build(ui_options['agent_name'])
            agent.gpt_model   = ui_options['model']
            agent.max_tokens  = ui_options['max_tokens']
            agent.temperature = ui_options['temperature']

            improved_prompt = gpt.ui_improve_gpt_prompt_by_ai(
                agent,
                ui_options['training_data'],
                ui_options['trainer_model'],
                verbose=False,
                on_phase=lambda name: phase_text.text(f"🔄 {name}…"),
                on_step=lambda idx, total: (
                    progress_text.text(f"[{idx}/{total}] processing…"),
                    progress_bar.progress(int(idx * 100 / total))
                )
            )
            ui_options['improved_prompt'] = improved_prompt
            st.success("✅ Improvement completed!")
            st.subheader("Improved Agent Configuration")
            st.code(improved_prompt)

    # save improved prompt
    if 'improved_prompt' in ui_options:
        filename = st.text_input(
            "Save As (filename.yaml)",
            value=f"{ui_options['agent_name']}_improved.yaml"
        )
        if st.button("Save Improved Config", key="save_config"):
            save_path = os.path.join(config_dir, filename)
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(ui_options['improved_prompt'])
                st.success(f"Improved configuration saved to `{save_path}`")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error saving file: {e}")
