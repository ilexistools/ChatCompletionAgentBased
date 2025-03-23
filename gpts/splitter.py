import os 
from dotenv import load_dotenv, find_dotenv
import tiktoken
from gpts.factory import GPTFactory
import math 

class TextSplitter:
    """
    A class for splitting and summarizing text using GPT models.

    Attributes:
        api_key (str): API key for OpenAI.
        api_url_base (str): Base URL for the OpenAI API.
        api_model (str): Model name for the OpenAI API.
        GPT1o_name (str): Name for the GPT1o model.
        GPT1o_max_tokens (int): Maximum tokens for the GPT1o model.
        GPT4o_name (str): Name for the GPT-4o model.
        GPT4o_max_tokens (int): Maximum tokens for the GPT-4o model.
        GPT4o_mini_name (str): Name for the GPT-4o-mini model.
        GPT4o_mini_max_tokens (int): Maximum tokens for the GPT-4o-mini model.
        splitter_summarizer: Instance for summarizing text.
        splitter_refiner: Instance for refining summaries.
    """

    def __init__(self):
        """
        Initialize the TextSplitter instance by loading environment variables and setting up GPT model configurations.
        """
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url_base = os.getenv("OPENAI_API_BASE")
        self.api_model = os.getenv("OPENAI_MODEL_NAME")
        # GPT models info
        self.GPT1o_name = 'o1'
        self.GPT1o_max_tokens = 200000
        self.GPT4o_name = 'gpt-4o'
        self.GPT4o_max_tokens = 128000
        self.GPT4o_mini_name = 'gpt-4o-mini'
        self.GPT4o_mini_max_tokens = 128000
        self.splitter_summarizer = GPTFactory().build('splitter_summarizer')
        self.splitter_refiner = GPTFactory().build('splitter_refiner')

    def count_tokens(self, text, model_name="gpt-4o-mini"):
        """
        Count the number of tokens in a given text using the specified model's encoding.

        Args:
            text (str): The text to tokenize.
            model_name (str): The name of the model to determine encoding (default 'gpt-4o-mini').

        Returns:
            int: The number of tokens in the text.

        Raises:
            Exception: If token calculation fails.
        """
        try:
            enc = tiktoken.encoding_for_model(model_name)
            return len(enc.encode(text))
        except Exception as e:
            raise Exception(f"Token calculation error: {e}")

    def calculate_tokens(self, max_tokens=128000, input_percentage=75, prompt_percentage=5, response_percentage=20):
        """
        Calculate the token allocation for input, prompt, and response based on given percentages.

        Args:
            max_tokens (int): Maximum number of tokens available.
            input_percentage (int): Percentage of tokens allocated for input text.
            prompt_percentage (int): Percentage of tokens allocated for the prompt.
            response_percentage (int): Percentage of tokens allocated for the response.

        Returns:
            dict: A dictionary with keys 'input_tokens', 'prompt_tokens', and 'response_tokens'.

        Raises:
            ValueError: If percentages do not sum to 100 or if any percentage is negative.
        """
        total_percentage = input_percentage + prompt_percentage + response_percentage
        if total_percentage != 100:
            raise ValueError(f"The percentages must sum up to 100%. Currently, they sum up to {total_percentage}%.")
        if input_percentage < 0 or prompt_percentage < 0 or response_percentage < 0:
            raise ValueError("All percentages must be positive values.")

        input_tokens = int(max_tokens * (input_percentage / 100))
        prompt_tokens = int(max_tokens * (prompt_percentage / 100))
        response_tokens = int(max_tokens * (response_percentage / 100))

        calculated_total = input_tokens + prompt_tokens + response_tokens
        difference = max_tokens - calculated_total
        if difference != 0:
            # Adjust the response tokens to compensate for the difference
            response_tokens += difference
        return {
            "input_tokens": input_tokens,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens
        }

    def calculate_tokens_percentage(self, max_tokens, tokens):
        """
        Calculate the percentage of tokens relative to the maximum tokens available.

        Args:
            max_tokens (int or float): The maximum number of tokens.
            tokens (int or float): The current number of tokens.

        Returns:
            float: The percentage of tokens used, rounded to two decimal places.

        Raises:
            TypeError: If max_tokens or tokens are not numeric.
            ValueError: If max_tokens is not positive, tokens is negative, or tokens exceed max_tokens.
        """
        if not isinstance(max_tokens, (int, float)):
            raise TypeError("max_tokens must be an integer or float.")
        if not isinstance(tokens, (int, float)):
            raise TypeError("tokens must be an integer or float.")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        if tokens < 0:
            raise ValueError("The number of tokens cannot be negative.")
        if tokens > max_tokens:
            raise ValueError("The number of tokens cannot exceed max_tokens.")

        percentage = (tokens / max_tokens) * 100
        return round(percentage, 2)

    def split_text(self, text, model_name='gpt-4o-mini', input_max_tokens=115000, overlap=200):
        """
        Split the text into manageable chunks based on the token limit and overlap.

        Args:
            text (str): The text to split.
            model_name (str): The model name for tokenization (default 'gpt-4o-mini').
            input_max_tokens (int): Maximum tokens allowed per chunk.
            overlap (int): Number of tokens to overlap between consecutive chunks.

        Returns:
            list: A list of text chunks.

        Raises:
            Exception: If an error occurs during splitting.
        """
        try:
            chunks = self.split_into_chunks(text, model_name, input_max_tokens, overlap)
            return chunks
        except Exception as e:
            raise Exception(f"Error splitting text: {e}")

    def split_into_chunks(self, text, model_name='gpt-4o-mini', max_tokens=115000, overlap=200):
        """
        Split a text into chunks of tokens with a specified overlap.

        Args:
            text (str): The text to be split.
            model_name (str): The model name for tokenization (default 'gpt-4o-mini').
            max_tokens (int): Maximum number of tokens per chunk.
            overlap (int): Number of tokens to overlap between chunks.

        Returns:
            list: A list of decoded text chunks.

        Raises:
            Exception: If an error occurs during the tokenization or splitting process.
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            tokens = encoding.encode(text)
            chunks = []
            step = max_tokens - overlap  # Define the step based on the overlap
            for i in range(0, len(tokens), step):
                chunk = tokens[i:i + max_tokens]
                decoded_chunk = encoding.decode(chunk)
                chunks.append(decoded_chunk)
                if i + max_tokens >= len(tokens):
                    break
            return chunks
        except Exception as e:
            raise Exception(f"Error splitting text into chunks: {e}")
    
    def refine(self, chunks, verbose=False):
        """
        Refine the summaries by processing each chunk iteratively using a summarizer and a refiner.

        Args:
            chunks (list): List of text chunks.
            verbose (bool): If True, print detailed debug statements.

        Returns:
            str: A refined summary combining the processed chunks.
        """
        if verbose:
            print('Refining chunks')
            print('Number of chunks:', len(chunks))
        final_summary = []
        i = 0
        for chunk in chunks:
            i += 1
            if verbose:
                print('Processing chunk', i)
            if len(final_summary) == 0:
                new_summary = self.splitter_summarizer.run(inputs={'text': chunk})
                final_summary.append(new_summary)
                if verbose:
                    tokens_count = self.count_tokens('\n'.join(final_summary))
                    print(f"Final summary length: {tokens_count} tokens")
            else:
                new_summary = self.splitter_refiner.run(inputs={'summary': '\n'.join(final_summary), 'text': chunk})
                final_summary.append(new_summary)
                if verbose:
                    tokens_count = self.count_tokens('\n'.join(final_summary))
                    print(f"Final summary length: {tokens_count} tokens")
        return '\n'.join(final_summary)

    def map_reduce(self, chunks, max_tokens=115000, verbose=False, depth=0, max_depth=10):
        """
        Perform a map-reduce style summarization on the provided chunks.
        Each chunk is summarized individually, and the summaries are combined.
        If the combined summary exceeds the token limit, it is further summarized recursively.

        Args:
            chunks (list): List of text chunks.
            max_tokens (int): Maximum tokens allowed for the combined summary.
            verbose (bool): If True, print detailed debug information.
            depth (int): Current recursion depth.
            max_depth (int): Maximum allowed recursion depth.

        Returns:
            str: A combined summary of all chunks.

        Raises:
            RecursionError: If maximum recursion depth is reached.
            Exception: If an error occurs during summarization.
        """
        if depth > max_depth:
            raise RecursionError("Maximum recursion depth reached in map_reduce.")

        summaries = []
        for i, chunk in enumerate(chunks, 1):
            if verbose:
                print(f"Processing chunk {i} of {len(chunks)}")
            try:
                summary = self.splitter_summarizer.run(inputs={'text': chunk})
                summary_text = summary
            except Exception as e:
                raise Exception(f"Error during summarization of chunk {i}: {e}")

            if verbose:
                chunk_token_count = self.count_tokens(chunk, model_name='gpt-4o-mini')
                summary_token_count = self.count_tokens(summary_text, model_name='gpt-4o-mini')
                print(f"Chunk size: {chunk_token_count} tokens, Summary size: {summary_token_count} tokens")

            summaries.append(summary_text)

        combined_summary = '\n'.join(summaries)
        combined_token_count = self.count_tokens(combined_summary, model_name='gpt-4o-mini')

        if combined_token_count > max_tokens:
            if verbose:
                print('Summaries too long, further summarizing...')
            try:
                new_chunks = self.split_text(combined_summary, model_name='gpt-4o-mini', input_max_tokens=max_tokens, overlap=0)
            except Exception as e:
                raise Exception(f"Error during splitting combined summaries: {e}")
            return self.map_reduce(new_chunks, max_tokens=max_tokens, verbose=verbose, depth=depth + 1, max_depth=max_depth)
        else:
            if verbose:
                print(f'Final summary length: {combined_token_count} tokens')
            return combined_summary
    
    def summarize_to_fit(self, text, max_tokens, max_summary_tokens=1000, verbose=False):
        """
        Summarize the input text to fit within a specified token limit using map-reduce summarization.

        Args:
            text (str): The text to be summarized.
            max_tokens (int): Maximum tokens allowed for the final summary.
            max_summary_tokens (int): Maximum tokens allowed for each intermediate summary chunk.
            verbose (bool): If True, print detailed debug information.

        Returns:
            str: A summary of the input text.
        """
        input_tokens = self.count_tokens(text)
        n_chunks = math.floor(max_tokens / max_summary_tokens)
        n_chunks = max(1, n_chunks)  # Ensure at least one chunk
        chunk_size = math.ceil(input_tokens / n_chunks)
        chunks = self.split_into_chunks(text, model_name='gpt-4o-mini', max_tokens=chunk_size, overlap=100)
        summary = self.map_reduce(chunks, max_tokens=chunk_size, verbose=verbose)
        return summary
    
    def summarize_to_fit_refined(self, text, max_tokens, max_summary_tokens=1000, verbose=False):
        """
        Summarize the input text to fit within a specified token limit using a refined summarization approach.

        Args:
            text (str): The text to be summarized.
            max_tokens (int): Maximum tokens allowed for the final summary.
            max_summary_tokens (int): Maximum tokens allowed for each intermediate summary chunk.
            verbose (bool): If True, print detailed debug information.

        Returns:
            str: A refined summary of the input text.
        """
        input_tokens = self.count_tokens(text)
        n_chunks = math.floor(max_tokens / max_summary_tokens)
        n_chunks = max(1, n_chunks)  # Ensure at least one chunk
        chunk_size = math.ceil(input_tokens / n_chunks)
        chunks = self.split_into_chunks(text, model_name='gpt-4o-mini', max_tokens=chunk_size, overlap=100)
        summary = self.refine(chunks, verbose=verbose)
        return summary
