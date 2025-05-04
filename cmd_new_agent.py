#!/usr/bin/env
"""
Command-line tool to design a GPT agent interactively.
Asks for the agent's name, the task description, the input description, and the model to use,
then invokes gpt.design_agent and saves the output in the config folder as a YAML file.
"""
import click
from gpts import gpt
from pathlib import Path
import yaml

@click.command()
@click.option(
    '--name', prompt='Agent name',
    help='Name of the agent to create'
)
@click.option(
    '--task', prompt='Agent task',
    help='Description of the task the agent should perform'
)
@click.option(
    '--input-description', 'input_description',
    prompt='Input placeholder',
    default='{text}',
    show_default=True,
    help='The placeholder for the input the agent should analyze'
)
@click.option(
    '--model', prompt='Model to use for design',
    default='gpt-4.1-nano',
    show_default=True,
    help='OpenAI model to use for agent design'
)
def main(name, task, input_description, model):
    """
    Interactive CLI entrypoint for designing an agent and saving its design as YAML.
    """
    # Create the agent using the gpts library
    result = gpt.design_agent(name, task, input_description, model=model, temperature=0.5)
    
    # Ensure config folder exists
    config_dir = Path('config')
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save the result to a YAML file
    filename = f"{name.replace(' ', '_').lower()}_design.yaml"
    filepath = config_dir / filename
    with open(filepath, 'w', encoding='utf-8') as fh:
        fh.write(result)

    # Notify the user
    print(f"\n=== Agent design saved to {filepath} ===")

if __name__ == '__main__':
    main()
