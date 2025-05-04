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
    """
    Send a prompt to a GPT-based assistant built by GPTFactory and return its response.

    Parameters:
        prompt (str): The input prompt to send to the assistant.
        **kwargs: Optional keyword arguments to configure the assistant:
            model (str): Identifier of the GPT model to use (default is default_model).
            max_tokens (int): Maximum number of tokens for the response (default is default_max_tokens).
            temperature (float): Sampling temperature for response generation (default is default_temperature).
            json_format (Any): Optional format for JSON output.
            role (str): Role name for the assistant (default is 'Assistant').
            goal (str): Instruction or goal guiding the assistant (default is 'You are a helpful assistant.').
            knowledge (str): Additional context or knowledge to supply to the assistant.
    Returns:
        Any: The result returned by the assistant's run method.
    """
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
            json_format (Any): Optional format specification for JSON output.

    Returns:
        GPTAgent: A configured GPTAgent instance.
    """
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

    Parameters:
        agent (GPTAgent): The initial agent whose prompt and configuration will be optimized.
        training_data (List[dict]): A collection of input instances and expected outputs for training.
        trainer_model (str): Identifier of the GPT model to use for training and evaluation (default is default_model).
        **kwargs:
            verbose (bool): If True, prints colorized progress and debug messages to the console.

    Returns:
        str: The final synthesized prompt instructions for the improved agent.
    """

    # Definições de cores ANSI para realce
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
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    # Fase 1: Execução e avaliação
    log(color_text('--- Iniciando avaliação inicial ---', 'blue'))
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
            log(color_text('✔ Output obtido com sucesso', 'green'))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))

    log(color_text('--- Avaliação inicial concluída ---', 'blue'))

    # Fase 2: Geração de instruções por instância
    log(color_text('--- Gerando instruções de fine-tuning ---', 'blue'))
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
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: Validação binária das instruções
    log(color_text('--- Validando instruções ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: Síntese de instruções finais
    log(color_text('--- Sintetizando instruções finais ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: Extração de novo config JSON
    log(color_text('--- Extraindo configuração do agente melhorado ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: Teste do novo agente e cálculo de precisão
    log(color_text('--- Testando agente melhorado ---', 'blue'))
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Precision final: {precision:.2f}", 'bold'))

    # Fase 7: Salvando resultados
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

    Parameters:
        agent (GPTAgent): The initial agent whose prompt and configuration will be optimized.
        training_data (List[dict]): A collection of input instances and their outputs for human review.
        trainer_model (str): Identifier of the GPT model to use for instruction generation and validation (default is default_model).
        **kwargs:
            verbose (bool): If True, prints colorized progress and debug messages to the console.

    Returns:
        str: The final synthesized prompt instructions for the improved agent.
    """

    # Definições de cores ANSI para realce
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

    # Avaliador automático (usado apenas na fase de teste)
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    # Fase 1: Execução e avaliação HUMANA via input() com comentário
    log(color_text('--- Iniciando avaliação humana via input() ---', 'blue'))
    results = []
    total = len(training_data)
    for idx, instance in enumerate(training_data, start=1):
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            # Exibir instância e saída para avaliação manual
            print(color_text(f"Instância: {instance}", 'yellow'))
            print(color_text(f"Saída: {output}", 'yellow'))
            # Solicitar comentário livre
            comment = input(color_text("Comentário sobre a saída: ", 'bold'))
            results.append((instance, output, comment))
            log(color_text(f"Comentário recebido: {comment}", 'green'))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))

    log(color_text('--- Avaliação humana concluída ---', 'blue'))

    # Fase 2: Geração de instruções por instância
    log(color_text('--- Gerando instruções de fine-tuning ---', 'blue'))
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
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: Validação binária das instruções
    log(color_text('--- Validando instruções ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: Síntese de instruções finais
    log(color_text('--- Sintetizando instruções finais ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: Extração de novo config JSON
    log(color_text('--- Extraindo configuração do agente melhorado ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: Teste do novo agente e cálculo de precisão
    log(color_text('--- Testando agente melhorado ---', 'blue'))
    test_scores = []
    for idx, instance in enumerate(training_data, start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Precision final: {precision:.2f}", 'bold'))

    # Fase 7: Salvando resultados
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

    Parameters:
        agent (GPTAgent): The initial agent whose prompt and configuration will be optimized.
        training_data (List[dict]): A collection of input instances to feed the agent.
        expected_outputs (List[str]): The corresponding expected outputs for each training instance.
        trainer_model (str): Identifier of the GPT model to use for training and evaluation (default is default_model).
        **kwargs:
            verbose (bool): If True, prints colorized progress and debug messages to the console.

    Returns:
        str: The final synthesized prompt instructions for the improved agent.
    """

    # Definições de cores ANSI para realce
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

    # Avaliador automático (usado apenas na fase de teste)
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    # Fase 1: Execução e coleta de pares (output, expected)
    log(color_text('--- Iniciando avaliação por dados esperados ---', 'blue'))
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
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))

    log(color_text('--- Coleta de resultados concluída ---', 'blue'))

    # Fase 2: Geração de instruções por instância
    log(color_text('--- Gerando instruções de fine-tuning ---', 'blue'))
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
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: Validação binária das instruções
    log(color_text('--- Validando instruções ---', 'blue'))
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    instr_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: Síntese de instruções finais
    log(color_text('--- Sintetizando instruções finais ---', 'blue'))
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: Extração de novo config JSON
    log(color_text('--- Extraindo configuração do agente melhorado ---', 'blue'))
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})

    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: Teste do novo agente e cálculo de precisão
    log(color_text('--- Testando agente melhorado ---', 'blue'))
    test_scores = []
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        try:
            output = new_agent.run(inputs=instance)
            eval_resp = evaluator.run(inputs={
                'prompt': agent.get_description(),
                'input': instance,
                'output': output,
            })
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid, expected))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    tp = sum(1 for (_, _, valid, exp), inst in zip(test_scores, results) if exp and valid == 1)
    fp = sum(1 for (_, _, valid, exp), inst in zip(test_scores, results) if not exp and valid == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    print(color_text(f"Precision final: {precision:.2f}", 'bold'))

    # Fase 7: Salvando resultados
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\nExpected: {exp}\nGot: {out}" for i, (inst, out, _, exp) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions

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

    Parameters:
        agent (GPTAgent): The initial agent to optimize.
        training_data (List[dict]): Input instances + expected outputs.
        trainer_model (str): GPT model to use for training/evaluation.
        verbose (bool): If True, prints ANSI-colored logs to console.
        on_phase (callable): Called as on_phase(phase_name) at start of each phase.
        on_step  (callable): Called as on_step(idx, total) each loop in phase 1.
        **kwargs: Ignored.
    Returns:
        str: The final synthesized prompt instructions.
    """

    # ANSI colors for console (used if verbose=True)
    COLORS = {
        'reset': '\033[0m', 'bold': '\033[1m',
        'red': '\033[31m', 'green': '\033[32m',
        'yellow': '\033[33m', 'blue': '\033[34m',
    }
    def color_text(text: str, color: str = 'reset') -> str:
        return f"{COLORS.get(color, COLORS['reset'])}{text}{COLORS['reset']}"

    log = print if verbose else lambda *a, **k: None

    # Build evaluator
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    # Factory for trainers
    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
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
    i = 0
    for instance, output, eval_resp in results:
        i += 1
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
        json_format='{result: int}'
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
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
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
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
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
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

    if on_phase: on_phase(f"Final precision: {precision:.2f}")

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
    # ANSI colors para realce
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

    # avaliador automático (usado na fase de teste)
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(name: str, max_tokens: int = default_max_tokens,
                     temperature: float = None, json_format: str = None):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    total = len(training_data)
    results = []

    # Fase 1: avaliação humana
    _phase("Avaliação humana via input()")
    for idx, instance in enumerate(training_data, start=1):
        _step(idx, total)
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            print(color_text(f"Instância: {instance}", 'yellow'))
            print(color_text(f"Saída: {output}", 'yellow'))
            comment = input(color_text("Comentário sobre a saída: ", 'bold'))
            results.append((instance, output, comment))
            log(color_text(f"Comentário recebido: {comment}", 'green'))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))
    _phase("Avaliação humana concluída")

    # Fase 2: geração de instruções
    _phase("Gerando instruções de fine-tuning")
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
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: validação binária
    _phase("Validando instruções")
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    human_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: síntese
    _phase("Sintetizando instruções finais")
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: extração de config
    _phase("Extraindo configuração do agente melhorado")
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: teste e cálculo de precisão
    _phase("Testando agente melhorado")
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
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    tp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 1 and ts[2] == 1)
    fp = sum(1 for gs, ts in zip(human_scores, test_scores) if gs == 0 and ts[2] == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    _phase(f"Precision final: {precision:.2f}")

    # Fase 7: salvando resultados
    _phase("Salvando resultados")
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\n{out}"
            for i, (inst, out, _) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions

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
    # ANSI colors para realce
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

    # avaliador automático (usado na fase de teste)
    evaluator = build('_trainer_evaluator')
    evaluator.gpt_model = trainer_model
    evaluator.max_tokens = default_max_tokens

    factory = GPTFactory()
    def make_trainer(
        name: str,
        max_tokens: int = default_max_tokens,
        temperature: float = None,
        json_format: str = None,
    ):
        t = factory.build(name)
        t.gpt_model = trainer_model
        t.max_tokens = max_tokens
        if temperature is not None:
            t.temperature = temperature
        if json_format is not None:
            t.json_format = json_format
        return t

    total = len(training_data)
    results = []

    # Fase 1: execução e coleta de pares (output, expected)
    _phase("Avaliação por dados esperados")
    for idx, (instance, expected) in enumerate(zip(training_data, expected_outputs), start=1):
        _step(idx, total)
        log(color_text(f"[{idx}/{total}] Executando agente...", 'yellow'))
        try:
            output = agent.run(inputs=instance)
            log(color_text(f"Output: {output}", 'yellow'))
            log(color_text(f"Expected: {expected}", 'yellow'))
            results.append((instance, output, expected))
        except Exception as e:
            log(color_text(f"✖ Erro na instância {idx}: {e}", 'red'))
    _phase("Coleta de resultados concluída")

    # Fase 2: geração de instruções por instância
    _phase("Gerando instruções de fine-tuning")
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
    log(color_text('✔ Instruções individuais geradas', 'green'))

    # Fase 3: validação binária das instruções
    _phase("Validando instruções")
    validator = make_trainer(
        '_trainer_binary_validator',
        max_tokens=200,
        temperature=0.2,
        json_format='{result: int}'
    )
    instr_scores = [
        validator.run(inputs={'evaluation': instr}).get('result', 0)
        for instr in instructions
    ]
    log(color_text('✔ Validação completa', 'green'))

    # Fase 4: síntese de instruções finais
    _phase("Sintetizando instruções finais")
    synthesizer = make_trainer('_trainer_synthesizer')
    final_instructions = synthesizer.run(inputs={
        'agent_description': agent.get_description(),
        'training_instructions': '\n'.join(instructions),
    })
    log(color_text('✔ Instruções finais sintetizadas', 'green'))

    # Fase 5: extração de novo config JSON
    _phase("Extraindo configuração do agente melhorado")
    extractor = make_trainer(
        '_json_extractor',
        json_format='{role: str, goal: str, backstory: str, knowledge: str}'
    )
    config = extractor.run(inputs={'text': final_instructions})
    new_agent = create_agent(
        config['role'],
        config['goal'],
        config['backstory'],
        config['knowledge'],
    )
    log(color_text('✔ Novo agente criado', 'green'))

    # Fase 6: teste do novo agente e cálculo de precisão
    _phase("Testando agente melhorado")
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
            valid = validator.run(inputs={'evaluation': eval_resp}).get('result', 0)
            test_scores.append((instance, output, valid, expected))
            log(color_text(f"[{idx}] Resultado validado: {valid}", 'yellow'))
        except Exception as e:
            log(color_text(f"✖ Erro no teste da instância {idx}: {e}", 'red'))

    # cálculo de precisão
    tp = sum(1 for (_, _, valid, exp) in test_scores if exp and valid == 1)
    fp = sum(1 for (_, _, valid, exp) in test_scores if not exp and valid == 1)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    _phase(f"Precision final: {precision:.2f}")

    # Fase 7: salvando resultados
    _phase("Salvando resultados")
    util.write_all_text(
        'improved_results.txt',
        '\n\n'.join(
            f"{i+1}. {inst}\nExpected: {exp}\nGot: {out}"
            for i, (inst, out, _, exp) in enumerate(test_scores)
        )
    )
    util.write_all_text('config/improved_agent.yaml', final_instructions)

    return final_instructions






