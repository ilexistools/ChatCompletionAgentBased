## Chat Completion Agent Based

A simple chat completion agent framework using LLMs.

This repository contains Python modules and configurations for creating, managing, and improving GPT-based agents for various tasks. It provides utility functions and example scripts to demonstrate how to build agents, apply them to data, and optimize their prompts via AI-driven, human-in-the-loop, or data-driven approaches.

## Limitations

* Sequential logic paths must be fully specified by the user.
* Custom GPT agents cannot directly handle external tools.

## Repository Contents

* `gpts.py`: Definitions and implementations of various agent functions:

  * `ask(prompt: str, **kwargs) -> any`
  * `design_agent(agent_name: str, task_description: str, input_placeholder: str, **kwargs) -> str`
  * `create_agent(role: str, goal: str, backstory: str, knowledge: str, **kwargs) -> GPTAgent`
  * `apply_agent_to_files(agent: GPTAgent, source_folder: str, **kwargs) -> list[dict]`
  * `improve_gpt_prompt_by_ai(agent: GPTAgent, training_data: List[dict], trainer_model=default_model, **kwargs) -> str`
  * `improve_gpt_prompt_by_human(agent: GPTAgent, training_data: List[dict], trainer_model=default_model, **kwargs) -> str`
  * `improve_gpt_prompt_by_data(agent: GPTAgent, training_data: List[dict], expected_outputs: List[str], trainer_model=default_model, **kwargs) -> str`
* `factory.py`: Responsible for creating and managing GPTAgent instances via `GPTFactory`.
* `util.py`: File-reading and writing utilities: `read_all_text`, `read_pdf`, `read_json_file`, `read_jsonl_file`, `read_tab_separated_file`, `write_all_text`.
* `ex01.py`, `ex02.py`, `ex03.py`: Example scripts demonstrating basic usage, parameter configuration, and JSON output formatting.
* `requirements.txt`: Dependency list.
* `env-examples`: Example environment variable specifications for model, base URL, and API key.
* `main.py`: Entry point for running experiments.

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## App

Use the command:

```bash
streamlit run app.py
```

## Usage

### Simple Prompt with `ask()`

```python
from gpts import gpt
result = gpt.ask("What are the days of the week?")
print(result)
```

Supported `ask` kwargs: `model`, `max_tokens`, `temperature`, `json_format`, `role`, `goal`, `knowledge`.

### Designing an Agent with `design_agent()`

```python
from gpts import gpt
spec = gpt.design_agent(
    agent_name="Summarizer",
    task_description="Summarize articles into key bullet points.",
    input_placeholder="Provide article text here"
)
print(spec)
```

Supported `design_agent` kwargs: `model`, `max_tokens`, `temperature`.

### Creating and Running an Agent

```python
from gpts import gpt
agent = gpt.create_agent(
    role="Translator",
    goal="Translate text to English and return results in JSON format.",
    backstory="The text to translate is: {text}",
    knowledge="Expert translator with context awareness. Language nuances and idioms."
)
# Optionally customize JSON output format
agent.json_format = "{'translation':str}"

result = agent.run(inputs={'text': 'Olá, mundo!'}, temperature=0.7)
print(result)
```

### Batch Processing Files with `apply_agent_to_files()`

```python
from gpts import gpt

agent = gpt.create_agent(
    role="Translator",
    goal="Translate text to English and return results in JSON format.",
    backstory="The text to translate is: {text}",
    knowledge="Expert translator with context awareness. Language nuances and idioms."
)
results = gpt.apply_agent_to_files(agent, 'data/source', verbose=True)
for entry in results:
    print(entry['filename'], entry['status'])
```

### Improving Prompts with `improve_gpt_prompt_by_data()`

```python
from gpts import gpt

agent = gpt.build('_critics_evaluator')
agent.gpt_model = 'gpt-4.1-nano'

training_data = [
    {'text': 'The film works as an entertaining spy thriller set in the corridors of the Vatican, but also as a serene reflection on commitment, power and faith.[Full review in Spanish]'},
    {'text': 'The movie was great until the turn no n the ending was such disgusting rewriting of anything that would happen in the conclave. Would give it a zero'},
    {'text': "f it weren't for the ending I've have given this a straight 5 stars. As it was, the twist feels unnecessary and detracts from everything that's gone before."},
    {'text': "Cage does go all in, but remains in control and never fully succumbs to overdone theatrics as in the recent past. It’s a performance that makes his unnamed “surfer” sad, tortured, pathetic, relatable and redeemable."},
    {'text': "Horrible movie. At times made no sense, abstract, inane. More like B movie. I don't think that I've seen that bad of a movie in years. Save your $$."}
]

expected_results= ['Professional', 'Audience', 'Audience', 'Professional', 'Audience']

gpt.improve_gpt_prompt_by_data(agent, training_data, expected_results, 'gpt-4.1-mini', verbose=True)
# gpt.improve_gpt_prompt_by_ai(agent, training_data, 'gpt-4.1-mini', verbose=True)
# gpt.improve_gpt_prompt_by_human(agent, training_data, 'gpt-4.1-mini', verbose=True)
```

## Configuration

Configure your OpenAI API key and optional defaults in environment variables or a config file as shown in `env-examples`.

## Dependencies

* click==8.1.8
* openai==1.77.0
* PyPDF2==3.0.1
* python-dotenv==1.1.0
* PyYAML==6.0.1
* PyYAML==6.0.2
* streamlit==1.45.0
* tiktoken==0.7.0

Install with:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

MIT License. See the LICENSE file for details.

## Contact

For questions or inquiries, please contact the repository maintainer.
