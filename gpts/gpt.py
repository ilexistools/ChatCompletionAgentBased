import os 
from gpts.factory import GPTFactory
from gpts.agents import GPTAgent
import tiktoken
from typing import List
import gpts.util as util 
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

# OpenAI models
GPT_4_dot_1='gpt-4.1'
GPT_4_dot_1_mini='gpt-4.1-mini'
GPT_4_dot_1_nano='gpt-4.1-nano'
GPT_4o = 'gpt-4o'
GPT_4o_mini = 'gpt-4o-mini'

# Default model and parameters
default_model = GPT_4_dot_1_nano
default_max_tokens = 1000
default_temperature = 0.2

def ask(prompt:str,**kwargs) -> any:
    model = kwargs.get('model', default_model)
    max_tokens = kwargs.get('max_tokens', default_max_tokens)
    temperature = kwargs.get('temperature', default_temperature)
    json_format = kwargs.get('json_format', None)
    role = kwargs.get('role', 'Assistant')
    goal = kwargs.get('goal', 'You are a helpful assistant.')
    knowledge = kwargs.get('knowledge', '')
    factory = GPTFactory()
    assistant = factory.build('assistant')
    assistant.gpt_model = model
    assistant.max_tokens = max_tokens 
    assistant.temperature = temperature
    assistant.json_format = json_format
    result = assistant.run(inputs={'role': role, 'goal': goal, 'prompt': prompt, 'knowledge': knowledge})
    return result

def build_agent(agent_name:str)-> GPTAgent:
    """
    Build and return an agent instance based on the provided agent name.

    Parameters:
        agent_name (str): The name of the agent to be built.

    Returns:
        object: An instance of the specified agent.
    """
    factory = GPTFactory()
    agent = factory.build(agent_name)
    return agent

def build(agent_name:str) -> GPTAgent:
    """
    Build and return an agent instance based on the provided agent name.

    Parameters:
        agent_name (str): The name of the agent to be built.

    Returns:
        object: An instance of the specified agent.
    """
    factory = GPTFactory()
    agent = factory.build(agent_name)
    return agent

def batch_run(agent:GPTAgent, inputs: list, **kwargs) -> list:
    """
    Run an agent on a batch of inputs and collect the responses.

    Parameters:
        agent: The agent instance to execute.
        inputs (list): A list of input dictionaries for the agent.
        verbose (bool, optional): If True, prints progress information. Defaults to False.

    Returns:
        list: A list containing the result for each input; if an error occurs, None is appended.
    """
    verbose = kwargs.get('verbose', False)
    results = []
    i = 0
    total = len(inputs)
    for input in inputs:
        if verbose:
            i += 1
            print(f"Processing {i} of {total}")
        try:
            result = agent.run(inputs=input)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            results.append(None)
    return results

def design_agent(agent_name:str, task_description:str, input_placeholder:str, **kwargs)->str:
    model = kwargs.get('model', default_model)
    max_tokens = kwargs.get('max_tokens', default_max_tokens)
    temperature = kwargs.get('temperature', default_temperature)
    factory = GPTFactory()
    designer = factory.build('_agent_designer')
    designer.gpt_model = model
    designer.max_tokens = max_tokens
    designer.temperature = temperature
    result = designer.run(inputs={'agent_name': agent_name, 'task_description': task_description, 'input_placeholder': input_placeholder})
    return result


def create_agent(role: str, goal: str, backstory: str, knowledge: str, **kwargs) -> GPTAgent:
    model = kwargs.get('model', default_model)
    max_tokens = kwargs.get('max_tokens', default_max_tokens)
    temperature = kwargs.get('temperature', default_temperature)
    json_format = kwargs.get('json_format', None)
    gpt = GPTAgent()
    gpt.role = role
    gpt.goal = goal 
    gpt.backstory = backstory
    gpt.knowledge = knowledge
    gpt.gpt_model = model 
    gpt.max_tokens = max_tokens
    gpt.temperature = temperature
    gpt.json_format = json_format
    return gpt 

def count_tokens(text: str, model: str = default_model) -> int:
    """
    Count the number of tokens in a given text using the specified GPT model.

    Parameters:
        text (str): The text whose tokens will be counted.
        model (str, optional): The GPT model identifier used for tokenization. Defaults to 'gpt-4o-mini'.

    Returns:
        int: The total token count of the text.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception as e:
        raise Exception(f"Token calculation error: {e}")

def split_text(text: str, chunk_size: int, model: str = default_model) -> List[str]:
    """
    Split `text` into a list of chunks, each of which contains at most
    `chunk_size` tokens according to the specified GPT `model`.

    Parameters:
        text (str): The full text to split.
        chunk_size (int): Maximum tokens allowed per chunk.
        model (str): GPT model identifier for tokenization.

    Returns:
        List[str]: A list of text chunks, each ≤ chunk_size tokens.
    """
    words = text.split()
    chunks: List[str] = []
    current_chunk = ""

    for word in words:
        # build the candidate chunk
        candidate = word if not current_chunk else f"{current_chunk} {word}"

        # if within chunk_size, extend; otherwise flush and start new
        if count_tokens(candidate, model) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # if a single word exceeds chunk_size, emit it alone
            if count_tokens(word, model) > chunk_size:
                chunks.append(word)
                current_chunk = ""
            else:
                current_chunk = word

    # append any remaining text
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def split_text_with_overlap(text: str, chunk_size: int, overlap: int, model: str = default_model) -> List[str]:
    """
    Split `text` into token-based chunks of size `chunk_size`, with
    each chunk overlapping the previous by `overlap` tokens.

    Parameters:
        text (str): The full text to split.
        chunk_size (int): Number of tokens per chunk.
        overlap (int): Number of tokens to re-use at the start of each new chunk.
        model (str): GPT model identifier for tokenization.

    Returns:
        List[str]: A list of overlapping text chunks.

    Raises:
        ValueError: If overlap >= chunk_size.
    """
    if overlap >= chunk_size:
        raise ValueError("`overlap` must be smaller than `chunk_size`")

    enc = tiktoken.encoding_for_model(model)
    token_ids = enc.encode(text)
    total_tokens = len(token_ids)

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_ids = token_ids[start:end]
        chunks.append(enc.decode(chunk_ids))

        start += step

    return chunks

def apply_agent_to_files(agent:GPTAgent, source_folder:str, **kwargs) -> list[dict]:
    verbose = kwargs.get('verbose', False)
    files = os.listdir(source_folder)
    i = 0
    total = len(files)
    results = []
    for filename in files:
        if verbose:
            i += 1
            print(f"Processing {i} of {total}")
        file_path = os.path.join(source_folder, filename)
        if filename.lower().endswith('.txt'):
            text = util.read_all_text(file_path)
        elif filename.lower().endswith('.pdf'):
            text = util.read_pdf(file_path)
        elif filename.lower().endswith('.json'):
            text = util.read_json_file(file_path)
        elif filename.lower().endswith('.jsonl'):
            text = util.read_jsonl_file(file_path)
        elif filename.lower().endswith('.tsv'):
            text = util.read_tab_separated_file(file_path)
        elif filename.lower().endswith('.tab'):
            text = util.read_tab_separated_file(file_path)
        else:
            try:
                text = util.read_all_text(file_path)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        if text:
            try:
                result = agent.run(inputs={'text': text})
                results.append({
                        'filename': filename,
                        'text': text,
                        'result': result,
                        'status': 'success'
                    })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results.append({
                        'filename': filename,
                        'text': text,
                        'result': None,
                        'status': 'error',
                    })
    return results

def improve_gpt_prompt(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model=default_model,
    **kwargs
) -> str:
    verbose = kwargs.get("verbose", False)
    # Função de log condicional
    log = print if verbose else lambda *args, **kw: None

    total = len(training_data)
    results = []
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens
    # Fase de coleta de resultados e avaliações humanas
    for idx, instance in enumerate(training_data, start=1):
        log(f"Training {idx} of {total}")
        try:
            result = agent.run(inputs=instance)
            log(f"Instance data: {instance}")
            log(f"Training result: {result}")
            evaluation = evaluator.run(inputs={'promnpt': agent.get_description(), 'input': instance, 'output': result})
            results.append((instance, result, evaluation))
            log(f"AI evaluation: {evaluation}")
            log("")  # linha em branco
        except Exception as e:
            print(f"Error during training: {e}")
            # continua para o próximo caso

    log("\nTraining completed. Evaluating results...")

    factory = GPTFactory()

    # Helper para instanciar e configurar os trainers
    def make_trainer(name: str):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = default_max_tokens
        return t

    # Gera instruções individuais com base nas avaliações
    trainer = make_trainer("_trainer")
    instructions = [
        trainer.run(inputs={
            "agent_description": agent.get_description(),
            "instance_data": str(instance),
            "result": str(result),
            "human_evaluation": evaluation,
        })
        for instance, result, evaluation in results
    ]
    log("Training instructions generated.")

    # Sintetiza instruções finais
    synthesizer = make_trainer("_trainer_synthesizer")
    final_instructions = synthesizer.run(inputs={
        "agent_description": agent.get_description(),
        "training_instructions": "\n".join(instructions),
    })
    log("\nFinal training instructions generated.")

    log(f"\n{final_instructions}")

    # Extrai JSON com a nova descrição do agente
    json_extractor = factory.build("_json_extractor")
    json_extractor.gpt_model = trainer_model
    json_extractor.json_format = "{role: str, goal: str, backstory: str, knowledge: str}"
    json_extractor.max_tokens = default_max_tokens

    json_description = json_extractor.run(inputs={"text": final_instructions})

    # Cria o novo agente e o testa em todos os dados de treino
    new_agent = create_agent(
        json_description["role"],
        json_description["goal"],
        json_description["backstory"],
        json_description["knowledge"],
    )
    log("\nTesting new agent with training data...")
    for instance in training_data:
        print(new_agent.run(inputs=instance))
    
    with open('config/improved_agent.yaml', 'w') as f:
        f.write(final_instructions)

    return final_instructions

def improve_gpt_prompt_by_human(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model=default_model,
    **kwargs
) -> str:
    verbose = kwargs.get("verbose", False)
    # Função de log condicional
    log = print if verbose else lambda *args, **kw: None

    total = len(training_data)
    results = []

    # Fase de coleta de resultados e avaliações humanas
    for idx, instance in enumerate(training_data, start=1):
        log(f"Training {idx} of {total}")
        try:
            result = agent.run(inputs=instance)
            log(f"Instance data: {instance}")
            log(f"Training result: {result}")
            evaluation = input("Human evaluation: ")
            results.append((instance, result, evaluation))
            log("")  # linha em branco
        except Exception as e:
            print(f"Error during training: {e}")
            # continua para o próximo caso

    log("Training completed. Evaluating results...")

    factory = GPTFactory()

    # Helper para instanciar e configurar os trainers
    def make_trainer(name: str):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = default_max_tokens
        return t

    # Gera instruções individuais com base nas avaliações
    trainer = make_trainer("_trainer")
    instructions = [
        trainer.run(inputs={
            "agent_description": agent.get_description(),
            "instance_data": str(instance),
            "result": str(result),
            "human_evaluation": evaluation,
        })
        for instance, result, evaluation in results
    ]
    log("Training instructions generated.")

    # Sintetiza instruções finais
    synthesizer = make_trainer("_trainer_synthesizer")
    final_instructions = synthesizer.run(inputs={
        "agent_description": agent.get_description(),
        "training_instructions": "\n".join(instructions),
    })
    log("Final training instructions generated.")

    print(f"Final instructions: {final_instructions}")

    # Extrai JSON com a nova descrição do agente
    json_extractor = factory.build("_json_extractor")
    json_extractor.gpt_model = trainer_model
    json_extractor.json_format = "{role: str, goal: str, backstory: str, knowledge: str}"
    json_extractor.max_tokens = default_max_tokens

    json_description = json_extractor.run(inputs={"text": final_instructions})

    # Cria o novo agente e o testa em todos os dados de treino
    new_agent = create_agent(
        json_description["role"],
        json_description["goal"],
        json_description["backstory"],
        json_description["knowledge"],
    )
    log("Testing new agent with training data...")
    for instance in training_data:
        print(new_agent.run(inputs=instance))
    
    with open('config/improved_agent.yaml', 'w') as f:
        f.write(final_instructions)

    return final_instructions


    