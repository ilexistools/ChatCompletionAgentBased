# Chat Completion Agent Based
A simple chat completion agent class for llms.

This repository contains Python scripts and configurations for creating and managing chat completions using large language models (LLMs). The provided scripts are examples of how to utilize OpenAI's API to create custom simple agents for specific sequential tasks. 

## Limitations
- You must provided all the sequential logic path for use.
- The custom GPT agent has no capabilities to handle tools.

## Repository Contents

- `agents.py`: This script contains the definitions and implementations of various AI agents.
- `factory.py`: This script is responsible for creating and managing instances of AI agents.
- `ex01.py`: Example script demonstrating the use of AI agents for a specific task.
- `ex02.py`: Another example script showcasing different functionalities of the AI agents.
- `requirements.txt`: List of dependencies required to run the scripts in this repository.

## Installation

To use the scripts in this repository, you need to install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

## How to use

### *Add Your GPT Agents Description*
Add your GPT agent descriptions to the config folder using YAML files. Template model:

```yaml

translator:
  role: >
    Portuguese Translator
  goal: >
    Translate text to Brazilian Portuguese.
  backstory: >
    As a translator, you must translate the {text} to Brazilian Portuguese.
  knowledge: >
    Word order in Portuguese is different from English.
```

### *Instantiate a New Factory*
In your code, import the GPTFactory and instantiate a new factory.

```python

from gpts.factory import GPTFactory
factory = GPTFactory()
```

### *Build the GPT Agent*
Build the GPT agent using the same name as in the YAML file.

```python

translator = factory.build('translator')
```

### *Run the GPT Agent*
Run the GPT agent, passing the required inputs. Example:

```python

text = 'This is the text to be translated'
results = translator.run(inputs={'text': text})
print(results)
```

The run function can take the following extra arguments:

temperature
max_tokens
model
Example with extra arguments:

```python

results = translator.run(inputs={'text': text}, temperature=0.7, max_tokens=150, model='text-davinci-003')
print(results)
```

## Configuration
Before running the scripts, ensure your environment variables and configuration files are set up properly. The scripts expect certain environment variables, such as your OpenAI API key.

## Dependencies
The repository requires the following dependencies, as specified in the requirements.txt file:

openai==1.37.0
python-dotenv==1.0.1
PyYAML==6.0.1
tiktoken==0.7.0
You can install these dependencies using:

```bash

pip install -r requirements.txt
```
## Contributing
Contributions are welcome. If you have any suggestions or improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or inquiries, please contact me.