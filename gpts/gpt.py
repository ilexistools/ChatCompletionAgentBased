import os 
from gpts.factory import GPTFactory
from gpts.agents import GPTAgent
import tiktoken
from typing import List
import gpts.util as util 
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel


class ResultSchema(BaseModel):
    result: int

class AgentConfigSchema(BaseModel):
    role: str
    goal: str
    backstory: str
    knowledge: str

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
    """
    Send a prompt to a GPT-based assistant built by GPTFactory and return its response.

    Parameters:
        prompt (str): The input prompt to send to the assistant.
        **kwargs: Optional keyword arguments to configure the assistant:
            model (str): Identifier of the GPT model to use (default is default_model).
            max_tokens (int): Maximum number of tokens for the response (default is default_max_tokens).
            temperature (float): Sampling temperature for response generation (default is default_temperature).
            output_schema (Any): Optional format for structured output.
            role (str): Role name for the assistant (default is 'Assistant').
            goal (str): Instruction or goal guiding the assistant (default is 'You are a helpful assistant.').
            knowledge (str): Additional context or knowledge to supply to the assistant.
    Returns:
        Any: The result returned by the assistant's run method.
    """
    model = kwargs.get('model', default_model)
    max_tokens = kwargs.get('max_tokens', default_max_tokens)
    temperature = kwargs.get('temperature', default_temperature)
    output_schema = kwargs.get('output_schema', None)
    role = kwargs.get('role', 'Assistant')
    goal = kwargs.get('goal', 'You are a helpful assistant.')
    knowledge = kwargs.get('knowledge', '')
    factory = GPTFactory()
    assistant = factory.build('assistant')
    assistant.gpt_model = model
    assistant.max_tokens = max_tokens 
    assistant.temperature = temperature
    assistant.output_schema = output_schema
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
    """
    Generate a design specification for an AI agent using a GPT-based agent designer.

    Parameters:
        agent_name (str): The name to assign to the designed agent.
        task_description (str): A description of the tasks and responsibilities of the agent.
        input_placeholder (str): Placeholder text indicating the expected input format for the agent.
        **kwargs: Optional keyword arguments to configure the designer:
            model (str): Identifier of the GPT model to use (default is default_model).
            max_tokens (int): Maximum number of tokens for the design output (default is default_max_tokens).
            temperature (float): Sampling temperature for the design generation (default is default_temperature).

    Returns:
        str: The generated agent design specification.
    """
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
    """
    Create and configure a GPTAgent instance with specified attributes.

    Parameters:
        role (str): Role assigned to the agent.
        goal (str): Primary objective the agent aims to achieve.
        backstory (str): Background narrative for the agent.
        knowledge (str): Preloaded knowledge or context for the agent.
        **kwargs: Optional keyword arguments to configure the agent:
            model (str): Identifier of the GPT model to use (default is default_model).
            max_tokens (int): Maximum number of tokens for the agent's responses (default is default_max_tokens).
            temperature (float): Sampling temperature for response generation (default is default_temperature).
            output_schema (Any): Optional format specification for structured output.

    Returns:
        GPTAgent: A configured GPTAgent instance.
    """
    model = kwargs.get('model', default_model)
    max_tokens = kwargs.get('max_tokens', default_max_tokens)
    temperature = kwargs.get('temperature', default_temperature)
    output_schema = kwargs.get('output_schema', None)
    gpt = GPTAgent()
    gpt.role = role
    gpt.goal = goal 
    gpt.backstory = backstory
    gpt.knowledge = knowledge
    gpt.gpt_model = model 
    gpt.max_tokens = max_tokens
    gpt.temperature = temperature
    gpt.output_schema = output_schema
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
    """
    Apply a GPTAgent to every supported file in a directory and return processing results.

    Parameters:
        agent (GPTAgent): The agent used to process file contents.
        source_folder (str): Path to the directory containing files to process.
        **kwargs: Optional keyword arguments:
            verbose (bool): If True, prints progress updates to the console.

    Returns:
        list[dict]: A list of dictionaries for each file with the keys:
            'filename' (str): Name of the file.
            'text' (str): Extracted text content from the file.
            'result' (Any): Output returned by the agent for the file.
            'status' (str): 'success' if processing succeeded, otherwise 'error'.
    """
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


def improve_gpt_prompt_by_ai(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model=default_model,
    **kwargs
) -> str:
    """
    Automatically improve a GPT agent’s prompt via AI-driven training, evaluation, and synthesis.
    """
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
    }
    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    verbose = kwargs.get('verbose', False)
    log = print if verbose else lambda *args, **kwargs: None

    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if output_schema is not None:
            t.output_schema = output_schema
        return t

    log(color_text('--- Starting initial evaluation ---', 'blue'))
    results = []
    total = len(training_data)
    for idx, instance in enumerate(training_data, start=1):
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            results.append((instance, output, eval_resp))
            log(color_text('✔ Output successfully obtained', 'green'))
        except Exception as e:
            log(color_text(f"✖ Error on instance {idx}: {e}", 'red'))

    log(color_text('--- Initial evaluation completed ---', 'blue'))

    log(color_text('--- Generating fine-tuning instructions ---', 'blue'))
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, eval_resp in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': eval_resp,
        })
        instructions.append(instr)
    log(color_text('✔ Individual instructions generated', 'green'))

    log(color_text('--- Validating instructions ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        output_schema=ResultSchema
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).result
        for instr in instructions
    ]
    log(color_text('✔ Validation complete', 'green'))

    log(color_text('--- Synthesizing final instructions ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Final instructions synthesized', 'green'))

    log(color_text('--- Extracting improved agent configuration ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        output_schema=AgentConfigSchema
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config.role,
        config.goal,
        config.backstory,
        config.knowledge,
    )
    log(color_text('✔ New agent created', 'green'))

    log(color_text('--- Testing improved agent ---', 'blue'))
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).result
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Validated result: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Error testing instance {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Final precision: {precision:.2f}", 'bold'))

    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\n{out}" for i, (inst, out, _) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions


def improve_gpt_prompt_by_human(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model=default_model,
    **kwargs
) -> str:
    """
    Interactively improve a GPT agent’s prompt using human-in-the-loop evaluation and AI-driven synthesis.
    """
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
    }

    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    verbose = kwargs.get('verbose', False)
    log = print if verbose else lambda *args, **kwargs: None

    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()

    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if output_schema is not None:
            t.output_schema = output_schema
        return t

    log(color_text('--- Starting human evaluation via input() ---', 'blue'))
    results = []
    total = len(training_data)
    for idx, instance in enumerate(training_data, start=1):
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            print(color_text(f"Instance: {instance}", 'yellow'))
            print(color_text(f"Output: {output}", 'yellow'))
            comment = input(color_text("Comment on output: ", 'bold'))
            results.append((instance, output, comment))
            log(color_text(f"Comment received: {comment}", 'green'))
        except Exception as e:
            log(color_text(f"✖ Error on instance {idx}: {e}", 'red'))

    log(color_text('--- Human evaluation completed ---', 'blue'))

    log(color_text('--- Generating fine-tuning instructions ---', 'blue'))
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, human_comment in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': human_comment,
        })
        instructions.append(instr)
    log(color_text('✔ Individual instructions generated', 'green'))

    log(color_text('--- Validating instructions ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        output_schema=ResultSchema
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).result
        for instr in instructions
    ]
    log(color_text('✔ Validation complete', 'green'))

    log(color_text('--- Synthesizing final instructions ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Final instructions synthesized', 'green'))

    log(color_text('--- Extracting improved agent configuration ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        output_schema=AgentConfigSchema
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config.role,
        config.goal,
        config.backstory,
        config.knowledge,
    )
    log(color_text('✔ New agent created', 'green'))

    log(color_text('--- Testing improved agent ---', 'blue'))
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).result
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Validated result: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Error testing instance {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Final precision: {precision:.2f}", 'bold'))

    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\n{out}" for i, (inst, out, _) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions


def improve_gpt_prompt_by_data(
    agent: GPTAgent,
    training_data: List[dict],
    expected_outputs: List[str],
    trainer_model=default_model,
    **kwargs
) -> str:
    """
    Automatically improve a GPT agent’s prompt using data-driven training, evaluation, and synthesis.
    """
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
    }

    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    verbose = kwargs.get('verbose', False)
    log = print if verbose else lambda *args, **kwargs: None

    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()

    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if output_schema is not None:
            t.output_schema = output_schema
        return t

    # Phase 1: Execution and collection
    log(color_text('--- Starting evaluation by expected data ---', 'blue'))
    results = []
    total = len(training_data)
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            log(color_text(f"Output: {output}", 'yellow'))
            log(color_text(f"Expected: {expected}", 'yellow'))
            results.append((instance, output, expected))
        except Exception as e:
            log(color_text(f"✖ Error on instance {idx}: {e}", 'red'))

    log(color_text('--- Results collection completed ---', 'blue'))

    # Phase 2: Fine-tuning instructions
    log(color_text('--- Generating fine-tuning instructions ---', 'blue'))
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, expected in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': expected,
        })
        instructions.append(instr)
    log(color_text('✔ Individual instructions generated', 'green'))

    # Phase 3: Binary validation
    log(color_text('--- Validating instructions ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        output_schema=ResultSchema
    )
    instr_scores = [
        validator.run(inputs={'evaluation': instr}).result
        for instr in instructions
    ]
    log(color_text('✔ Validation complete', 'green'))

    # Phase 4: Synthesis
    log(color_text('--- Synthesizing final instructions ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Final instructions synthesized', 'green'))

    # Phase 5: Extracting new config
    log(color_text('--- Extracting improved agent configuration ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        output_schema=AgentConfigSchema
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config.role,
        config.goal,
        config.backstory,
        config.knowledge,
    )
    log(color_text('✔ New agent created', 'green'))

    # Phase 6: Testing improved agent
    log(color_text('--- Testing improved agent ---', 'blue'))
    test_scores = []
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).result
            test_scores.append((instance, output, valid, expected))
            log(color_text(f"[{idx}] Validated result: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Error testing instance {idx}: {e}", 'red'))

    tp = sum(1 for (_, _, valid, exp), _ in zip(test_scores, results) if exp and valid == 1)
    fp = sum(1 for (_, _, valid, exp), _ in zip(test_scores, results) if not exp and valid == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Final precision: {precision:.2f}", 'bold'))

    # Phase 7: Saving results
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\nExpected: {exp}\nGot: {out}" for i, (inst, out, _, exp) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions


from typing import Callable, List, Optional, Type

def ui_improve_gpt_prompt_by_ai(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model: str = default_model,
    *,
    verbose: bool = False,
    on_phase: Callable[[str], None] = None,
    on_step: Callable[[int, int], None] = None,
    **kwargs
) -> str:
    """
    Automatically improve a GPT agent’s prompt via AI-driven training, evaluation, and synthesis,
    with optional callbacks for phase updates and per-step progress.
    """
    COLORS = {
        'reset': '\033[0m', 'bold': '\033[1m',
        'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m',
    }

    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    log = print if verbose else lambda *a, **k: None

    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()

    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if output_schema is not None:
            t.output_schema = output_schema
        return t

    total = len(training_data)
    results = []

    # Phase 1: initial evaluation
    if on_phase: on_phase("Initial evaluation")
    for idx, instance in enumerate(training_data, start=1):
        if on_step: on_step(idx, total)
        log(color_text(f"[{idx}/{total}] Executing agent...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            results.append((instance, output, eval_resp))
            log(color_text("✔ Output successfully obtained", 'green'))
        except Exception as e:
            log(color_text(f"✖ Error on instance {idx}: {e}", 'red'))

    # Phase 2: generate fine-tuning instructions
    if on_phase: on_phase("Generating fine-tuning instructions")
    trainer = make_trainer('_trainer')
    instructions = []
    for i, (instance, output, eval_resp) in enumerate(results, start=1):
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': eval_resp,
        })
        instructions.append(instr)
        if on_step: on_step(i, len(results))
    log(color_text("✔ Individual instructions generated", 'green'))

    # Phase 3: binary validation
    if on_phase: on_phase("Validating instructions")
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        output_schema=ResultSchema
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).result
        for instr in instructions
    ]
    log(color_text("✔ Validation complete", 'green'))

    # Phase 4: synthesize final instructions
    if on_phase: on_phase("Synthesizing final instructions")
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text("✔ Final instructions synthesized", 'green'))

    # Phase 5: extract new agent config
    if on_phase: on_phase("Extracting final configuration")
    extractor = make_trainer(
        '_json_extractor',
        output_schema=AgentConfigSchema
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config.role,
        config.goal,
        config.backstory,
        config.knowledge,
    )
    log(color_text("✔ New agent created", 'green'))

    # Phase 6: test improved agent
    if on_phase: on_phase("Testing improved agent")
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        if on_step: on_step(idx, len(training_data))
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).result
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Validated result: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Error testing instance {idx}: {e}", 'red'))

    # Phase 7: calculating precision
    if on_phase: on_phase("Calculating precision")
    tp = sum(1 for hs, ts in zip(human_scores, test_scores) if hs == 1 and ts[2] == 1)
    fp = sum(1 for hs, ts in zip(human_scores, test_scores) if hs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    log(color_text(f"Final precision: {precision:.2f}", 'bold'))
    if on_phase: on_phase(f"Final precision: {precision:.2f}")

    # Phase 8: save results
    if on_phase: on_phase("Saving results")
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(f"{i+1}. {inst}\n{out}" for i, (inst, out, _) in enumerate(test_scores))
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions


def ui_improve_gpt_prompt_by_human(
    agent: GPTAgent,
    training_data: List[dict],
    trainer_model: str = default_model,
    *,
    verbose: bool = False,
    on_phase: Callable[[str], None] = None,
    on_step: Callable[[int, int], None] = None,
    **kwargs
) -> str:
    """
    Interactively improve a GPT agent’s prompt using human-in-the-loop evaluation and AI-driven synthesis,
    with optional callbacks for phase updates and per-step progress.
    """
    COLORS = {
        'reset': '\033[0m', 'bold': '\033[1m',
        'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m',
    }

    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    log = print if verbose else lambda *a, **k: None

    def _phase(name: str):
        if on_phase:
            on_phase(name)
        log(color_text(f"--- {name} ---", 'blue'))

    def _step(idx: int, total: int):
        if on_step:
            on_step(idx, total)

    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()

    def make_trainer(name: str, max_tokens: int = default_max_tokens,
                     temperature: float = None, output_schema: Optional[Type[BaseModel]] = None):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if output_schema is not None:
            t.output_schema = output_schema
        return t

    total = len(training_data)
    results = []

    # Phase 1: Human evaluation
    _phase("Human evaluation via input()")
    for idx, instance in enumerate(training_data, start=1):
        _step(idx, total)
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            print(color_text(f"Instance: {instance}", 'yellow'))
            print(color_text(f"Output: {output}", 'yellow'))
            comment = input(color_text("Comment on output: ", 'bold'))
            results.append((instance, output, comment))
            log(color_text(f"Comment received: {comment}", 'green'))
        except Exception as e:
            log(color_text(f"✖ Error on instance {idx}: {e}", 'red'))
    _phase("Human evaluation completed")

    # Phase 2: Generating instructions
    _phase("Generating fine-tuning instructions")
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, human_comment in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': human_comment,
        })
        instructions.append(instr)
    log(color_text('✔ Individual instructions generated', 'green'))

    # Phase 3: Binary validation
    _phase("Validating instructions")
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        output_schema=ResultSchema
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).result
        for instr in instructions
    ]
    log(color_text('✔ Validation complete', 'green'))

    # Phase 4: Synthesis
    _phase("Synthesizing final instructions")
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Final instructions synthesized', 'green'))

    # Phase 5: Extracting config
    _phase("Extracting improved agent configuration")
    extractor = make_trainer(
        '_json_extractor',
        output_schema=AgentConfigSchema
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config.role,
        config.goal,
        config.backstory,
        config.knowledge,
    )
    log(color_text('✔ New agent created', 'green'))

    # Phase 6: Testing and calculating precision
    _phase("Testing improved agent")
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        _step(idx, total)
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).result
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Validated result: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Error testing instance {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    _phase(f"Final precision: {precision:.2f}")

    # Phase 7: Saving results
    _phase("Saving results")
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\n{out}"
            for i, (inst, out, _) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions


from typing import Callable, List, Optional, Type

def ui_improve_gpt_prompt_by_data(
    agent: GPTAgent,
    training_data: List[dict],
    expected_outputs: List[str],
    trainer_model: str = default_model,
    *,
    verbose: bool = False,
    on_phase: Callable[[str], None] = None,
    on_step: Callable[[int, int], None] = None,
    **kwargs
) -> str:
    """
    Automatically improve a GPT agent’s prompt using data-driven training, evaluation, and synthesis,
    with optional callbacks for phase updates and per-step progress.
    """
    COLORS = {
        'reset': '\033[0m', 'bold': '\033[1m',
        'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m',
    }

    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    log = print if verbose else lambda *a, **k: None

    def _phase(name: str):
        if on_phase:
            on_phase(name)
        log(color_text(f"--- {name} ---", 'blue'))

    def _step(idx: int, total: int):
        if on_step:
            on_step(idx, total)

    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if output_schema is not None:
            t.output_schema = output_schema
        return t

    total = len(training_data)
    results = []

    # Phase 1: execution and collection of pairs (output, expected)
    _phase("Evaluation by expected data")
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        _step(idx, total)
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            log(color_text(f"Output: {output}", 'yellow'))
            log(color_text(f"Expected: {expected}", 'yellow'))
            results.append((instance, output, expected))
        except Exception as e:
            log(color_text(f"✖ Error on instance {idx}: {e}", 'red'))
    _phase("Results collection completed")

    # Phase 2: Generating instructions per instance
    _phase("Generating fine-tuning instructions")
    trainer = make_trainer('_trainer')
    instructions = []
    for instance, output, expected in results:
        instr = trainer.run(inputs={
            'agent_description': agent.get_description(),
            'instance_data': instance,
            'result': output,
            'human_evaluation': expected,
        })
        instructions.append(instr)
    log(color_text('✔ Individual instructions generated', 'green'))

    # Phase 3: Binary validation of instructions
    _phase("Validating instructions")
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        output_schema=ResultSchema
    )
    instr_scores = [
        validator.run(inputs={'evaluation': instr}).result
        for instr in instructions
    ]
    log(color_text('✔ Validation complete', 'green'))

    # Phase 4: Synthesis of final instructions
    _phase("Synthesizing final instructions")
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Final instructions synthesized', 'green'))

    # Phase 5: extracting the new config JSON
    _phase("Extracting improved agent configuration")
    extractor = make_trainer(
        '_json_extractor',
        output_schema=AgentConfigSchema
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config.role,
        config.goal,
        config.backstory,
        config.knowledge,
    )
    log(color_text('✔ New agent created', 'green'))

    # Phase 6: testing new agent and precision calculation
    _phase("Testing improved agent")
    test_scores = []
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        _step(idx, total)
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).result
            test_scores.append((instance, output, valid, expected))
            log(color_text(f"[{idx}] Validated result: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Error testing instance {idx}: {e}", 'red'))

    tp = sum(1 for (_, _, valid, exp) in test_scores if exp and valid == 1)
    fp = sum(1 for (_, _, valid, exp) in test_scores if not exp and valid == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    _phase(f"Final precision: {precision:.2f}")

    # Phase 7: Saving results
    _phase("Saving results")
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\nExpected: {exp}\nGot: {out}"
            for i, (inst, out, _, exp) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions

