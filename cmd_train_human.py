#!/usr/bin/env python3
"""
CLI tool to improve a GPT agent's prompt using human-in-the-loop training data.
By default, training files reside in the `data/` folder, so you need only supply the filename.
"""
import os
import click
from gpts import gpt, util

# Available GPT model constants
MODEL_OPTIONS = {
    'GPT_4_dot_1': gpt.GPT_4_dot_1,
    'GPT_4_dot_1_mini': gpt.GPT_4_dot_1_mini,
    'GPT_4_dot_1_nano': gpt.GPT_4_dot_1_nano,
    'GPT_4o': gpt.GPT_4o,
    'GPT_4o_mini': gpt.GPT_4o_mini,
}
MODEL_KEYS = list(MODEL_OPTIONS.keys())

MODEL_PROMPT = "\n".join(f"{i+1}. {key}" for i, key in enumerate(MODEL_KEYS))

@click.command()
@click.option('--name', 'agent_name', prompt='Name of the agent to improve',
              help='Name of the existing agent to improve')
@click.option('--agent-model-index', 'agent_model_index',
              type=click.IntRange(1, len(MODEL_KEYS)),
              prompt=f'Select an agent model by number:\n{MODEL_PROMPT}',
              help='Numeric choice for the agent GPT model')
@click.option('--temperature', 'agent_temperature',
              type=float, prompt='Temperature of the agent to improve',
              help='Sampling temperature for the agent')
@click.option('--training-file', 'training_filename',
              type=str, prompt='Filename of training data (inside "data/")',
              help='Filename of JSONL training data (located in ./data/)')
@click.option('--trainer-model-index', 'trainer_model_index',
              type=click.IntRange(1, len(MODEL_KEYS)),
              prompt=f'Select a trainer model by number:\n{MODEL_PROMPT}',
              help='Numeric choice for the trainer GPT model')
def main(agent_name, agent_model_index, agent_temperature, training_filename, trainer_model_index):
    # Compose full path
    training_data_path = os.path.join('data', training_filename)

    if not os.path.exists(training_data_path):
        raise click.ClickException(f"Training file not found: {training_data_path}")

    # Load data
    training_data = util.read_jsonl_file(training_data_path)

    # Build agent and configure
    agent = gpt.build(agent_name)
    agent_model_key = MODEL_KEYS[agent_model_index - 1]
    trainer_model_key = MODEL_KEYS[trainer_model_index - 1]
    agent.gpt_model = MODEL_OPTIONS[agent_model_key]
    agent.temperature = agent_temperature

    # Improve prompt
    agent_prompt = gpt.improve_gpt_prompt_by_human(
        agent, training_data, MODEL_OPTIONS[trainer_model_key], verbose=True
    )

    click.echo("\n=== Improved Prompt ===\n")
    click.echo(agent_prompt)
    click.echo("=== End of Prompt ===\n")

if __name__ == '__main__':
    main()
