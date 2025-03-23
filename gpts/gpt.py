from gpts.factory import GPTFactory
from gpts.agents import GPTAgent
from gpts.splitter import TextSplitter

def ask(prompt, model='gpt-4o-mini', max_tokens=300):
    """
    Send a prompt to an assistant model and return its response.

    Parameters:
        prompt (str): The input prompt to send to the assistant.
        model (str, optional): The identifier of the GPT model to use. Defaults to 'gpt-4o-mini'.
        max_tokens (int, optional): The maximum number of tokens allowed in the response. Defaults to 300.

    Returns:
        str: The assistant's generated response.
    """
    factory = GPTFactory()
    assistant = factory.build('assistant')
    assistant.gpt_model = model
    assistant.max_tokens = max_tokens 
    goal = 'You are a helpful assistant.' 
    knowledge = ''
    result = assistant.run(inputs={'goal': goal, 'prompt': prompt, 'knowledge': knowledge})
    return result

def build_agent(agent_name):
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

def batch_run(agent, inputs: list, verbose=False):
    """
    Run an agent on a batch of inputs and collect the responses.

    Parameters:
        agent: The agent instance to execute.
        inputs (list): A list of input dictionaries for the agent.
        verbose (bool, optional): If True, prints progress information. Defaults to False.

    Returns:
        list: A list containing the result for each input; if an error occurs, None is appended.
    """
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

def create_agent(role: str, goal: str, backstory: str, knowledge: str = '', model='gpt-4o-mini'):
    """
    Create and configure a GPTAgent with the provided role, goal, backstory, and optional knowledge.

    Parameters:
        role (str): The role to assign to the agent.
        goal (str): The objective or goal for the agent.
        backstory (str): Background information about the agent.
        knowledge (str, optional): Additional contextual knowledge for the agent. Defaults to ''.
        model (str, optional): The identifier of the GPT model to use. Defaults to 'gpt-4o-mini'.

    Returns:
        GPTAgent: A configured GPTAgent instance.
    """
    max_tokens = 1000
    temperature = 0.2
    gpt = GPTAgent()
    gpt.role = role
    gpt.goal = goal 
    gpt.backstory = backstory
    gpt.knowledge = knowledge
    gpt.gpt_model = model 
    gpt.max_tokens = max_tokens
    gpt.temperature = temperature
    return gpt 

def count_tokens(text: str, model: str = 'gpt-4o-mini') -> int:
    """
    Count the number of tokens in a given text using the specified GPT model.

    Parameters:
        text (str): The text whose tokens will be counted.
        model (str, optional): The GPT model identifier used for tokenization. Defaults to 'gpt-4o-mini'.

    Returns:
        int: The total token count of the text.
    """
    ts = TextSplitter()
    tokens_count = ts.count_tokens(text, model)
    return tokens_count

def split_text(text: str, max_tokens: int = 100000, model: str = 'gpt-4o-mini', overlap: int = 100) -> list:
    """
    Split the input text into chunks based on a maximum token limit and an optional overlap.

    Parameters:
        text (str): The text to be split into chunks.
        max_tokens (int, optional): The maximum number of tokens per chunk. Defaults to 100000.
        model (str, optional): The GPT model identifier used for splitting the text. Defaults to 'gpt-4o-mini'.
        overlap (int, optional): The number of tokens to overlap between consecutive chunks. Defaults to 100.

    Returns:
        list: A list of text chunks.
    """
    ts = TextSplitter()
    chunks = ts.split_text(text, model_name=model, input_max_tokens=max_tokens, overlap=overlap)
    return chunks

def run_slicing(initial_prompt_model: str, final_prompt_model: str, max_tokens_output: int, text: str, chunk_size: int, model: str = 'gpt-4o-mini', overlap: int = 100, verbose: bool = False) -> str:
    """
    Process text by splitting it into chunks, applying an initial prompt to each chunk, and then combining
    the partial results using a final prompt to generate a final output.

    Parameters:
        initial_prompt_model (str): The prompt prefix used for processing each chunk.
        final_prompt_model (str): The prompt prefix used to combine the partial results.
        max_tokens_output (int): Maximum number of tokens for each assistant response.
        text (str): The text to be processed.
        chunk_size (int): The size limit for each text chunk (note: not directly used in splitting in this function).
        model (str, optional): The GPT model identifier to use. Defaults to 'gpt-4o-mini'.
        overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 100.
        verbose (bool, optional): If True, prints progress information. Defaults to False.

    Returns:
        str: The final combined result after slicing and processing.
    """
    factory = GPTFactory()
    assistant = factory.build('assistant')
    assistant.gpt_model = model
    assistant.max_tokens = max_tokens_output
    goal = 'You are a helpful assistant.' 
    knowledge = ''
    ts = TextSplitter()
    # Note: The parameter input_max_tokens is set to a string ('gpt-4o-mini') per original code.
    chunks = ts.split_text(text, model_name=model, input_max_tokens='gpt-4o-mini', overlap=100)
    total_chunks = len(chunks)
    results = []
    i = 0
    for chunk in chunks:
        i += 1
        if verbose:
            print(f"Processing {i} of {total_chunks}")
        try:
            result = assistant.run(inputs={'goal': goal, 'prompt': initial_prompt_model + text, 'knowledge': knowledge})
            results.append(str(result))
        except Exception as e:
            print(f"Error: {e}")
    result = assistant.run(inputs={'goal': goal, 'prompt': final_prompt_model + '\n'.join(results), 'knowledge': knowledge})
    return result

def extract_information(information: str, max_tokens_output: int, text: str, chunk_size: int = 100000, model: str = 'gpt-4o-mini', overlap: int = 100, verbose: bool = False) -> str:
    """
    Extract specific information from unstructured text by processing it in chunks and then combining the results.

    Parameters:
        information (str): A description of the information to extract.
        max_tokens_output (int): Maximum number of tokens for each assistant response.
        text (str): The unstructured text to process.
        chunk_size (int, optional): Maximum token size for each chunk. Defaults to 100000.
        model (str, optional): The GPT model identifier to use. Defaults to 'gpt-4o-mini'.
        overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 100.
        verbose (bool, optional): If True, prints processing progress. Defaults to False.

    Returns:
        str: A structured result containing the extracted information.
    """
    factory = GPTFactory()
    assistant = factory.build('assistant')
    assistant.gpt_model = model
    assistant.max_tokens = max_tokens_output
    ts = TextSplitter()
    chunks = ts.split_text(text, model_name=model, input_max_tokens=chunk_size, overlap=overlap)
    total_chunks = len(chunks)
    if verbose:
        print(f"Total chunks: {total_chunks}")
    if total_chunks > 1:
        goal = 'Extract specific information from unstructured text.' 
        prompt = 'Read the text and extract the following set of information: ' + information + '\n\n' + 'The text is: '
        knowledge = ''
        results = []
        i = 0
        for chunk in chunks:
            i += 1
            if verbose:
                print(f"Processing {i} of {total_chunks}")
            try:
                result = assistant.run(inputs={'goal': goal, 'prompt': prompt + chunk, 'knowledge': knowledge})
                results.append(str(result))
            except Exception as e:
                print(f"Error: {e}")
        goal = 'Extract specific information from a set of extracted information results. Key information: ' + information
        prompt = ('Extract information from multiple segments into a single structured result.' +
                  '\n\n' + 'The multiple results are: ' + '\n'.join(results))
        knowledge = 'You must ensure that no information is lost and deliver a clean final output.'
        result = assistant.run(inputs={'goal': goal, 'prompt': prompt, 'knowledge': knowledge})
    else:
        goal = 'Extract specific information from unstructured text.'
        prompt = 'Read the text and extract the following set of information: ' + '\n'.join(information) + '\n\n' + 'The text is: '
        knowledge = ''
        try:
            result = assistant.run(inputs={'goal': goal, 'prompt': prompt + chunks[0], 'knowledge': knowledge})
        except Exception as e:
            print(f"Error: {e}")
            result = ""
    return result

def summarize(max_tokens_output: int, text: str, chunk_size: int = 100000, model: str = 'gpt-4o-mini', overlap: int = 100, verbose: bool = False) -> str:
    """
    Generate a summary of the provided text by processing it in chunks and combining partial summaries.

    Parameters:
        max_tokens_output (int): Maximum number of tokens for each assistant response.
        text (str): The text to be summarized.
        chunk_size (int, optional): Maximum token size for each chunk. Defaults to 100000.
        model (str, optional): The GPT model identifier to use. Defaults to 'gpt-4o-mini'.
        overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 100.
        verbose (bool, optional): If True, prints processing progress. Defaults to False.

    Returns:
        str: The final summarized text.
    """
    factory = GPTFactory()
    assistant = factory.build('assistant')
    assistant.gpt_model = model
    assistant.max_tokens = max_tokens_output
    ts = TextSplitter()
    chunks = ts.split_text(text, model_name=model, input_max_tokens=chunk_size, overlap=overlap)
    total_chunks = len(chunks)
    if verbose:
        print(f"Total chunks: {total_chunks}")
    if total_chunks > 1:
        goal = 'Write a summary of the provided text.'
        prompt = 'Write a summary of the following text:'
        knowledge = ''
        results = []
        i = 0
        for chunk in chunks:
            i += 1
            if verbose:
                print(f"Processing {i} of {total_chunks}")
            try:
                result = assistant.run(inputs={'goal': goal, 'prompt': prompt + chunk, 'knowledge': knowledge})
                results.append(str(result))
            except Exception as e:
                print(f"Error: {e}")
        goal = 'Write a summary of the provided text.'
        prompt = ('Write a final summary based on the following summaries from parts of the whole text:' +
                  '\n'.join(results))
        knowledge = 'You must ensure that no information is lost and deliver a clean final summary.'
        result = assistant.run(inputs={'goal': goal, 'prompt': prompt, 'knowledge': knowledge})
    else:
        goal = 'Write a summary of the provided text.'
        prompt = 'Write a summary of the following text:'
        knowledge = ''
        try:
            result = assistant.run(inputs={'goal': goal, 'prompt': prompt + chunks[0], 'knowledge': knowledge})
        except Exception as e:
            print(f"Error: {e}")
            result = ""
    return result

def answer_questions(questions: str, max_tokens_output: int, text: str, chunk_size: int = 100000, model: str = 'gpt-4o-mini', overlap: int = 100, verbose: bool = False) -> str:
    """
    Answer specific questions based on the provided text by processing it in chunks and consolidating the answers.

    Parameters:
        questions (str): The questions to be answered.
        max_tokens_output (int): Maximum number of tokens for each assistant response.
        text (str): The text containing information to answer the questions.
        chunk_size (int, optional): Maximum token size for each chunk. Defaults to 100000.
        model (str, optional): The GPT model identifier to use. Defaults to 'gpt-4o-mini'.
        overlap (int, optional): The number of tokens to overlap between chunks. Defaults to 100.
        verbose (bool, optional): If True, prints processing progress. Defaults to False.

    Returns:
        str: The final consolidated answer to the questions.
    """
    factory = GPTFactory()
    assistant = factory.build('assistant')
    assistant.gpt_model = model
    assistant.max_tokens = max_tokens_output
    ts = TextSplitter()
    chunks = ts.split_text(text, model_name=model, input_max_tokens=chunk_size, overlap=overlap)
    total_chunks = len(chunks)
    if verbose:
        print(f"Total chunks: {total_chunks}")
    if total_chunks > 1:
        goal = 'Answer specific questions from unstructured text.'
        prompt = 'Read the text and answer the following questions: ' + questions + '\n\n' + 'The text is: '
        knowledge = ''
        results = []
        i = 0
        for chunk in chunks:
            i += 1
            if verbose:
                print(f"Processing {i} of {total_chunks}")
            try:
                result = assistant.run(inputs={'goal': goal, 'prompt': prompt + chunk, 'knowledge': knowledge})
                results.append(str(result))
            except Exception as e:
                print(f"Error: {e}")
        goal = 'Answer specific questions from unstructured text.'
        prompt = ('Answer questions from multiple answers from segments of texts into a final set of answers. Questions: ' +
                  questions + '\n\n' + 'The multiple answers: ' + '\n'.join(results))
        knowledge = 'You must ensure that no information is lost and deliver a clean final output.'
        result = assistant.run(inputs={'goal': goal, 'prompt': prompt, 'knowledge': knowledge})
    else:
        goal = 'Answer specific questions from unstructured text.'
        prompt = 'Read the text and answer the following questions: ' + questions + '\n\n' + 'The text is: '
        knowledge = ''
        try:
            result = assistant.run(inputs={'goal': goal, 'prompt': prompt + chunks[0], 'knowledge': knowledge})
        except Exception as e:
            print(f"Error: {e}")
            result = ""
    return result
