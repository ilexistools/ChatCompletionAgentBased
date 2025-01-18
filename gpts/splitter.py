import os 
from dotenv import load_dotenv, find_dotenv
import tiktoken
from gpts.factory import GPTFactory
import math 

class TextSplitter:
    def __init__(self):
        # Load environment variables
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
        try:
            enc = tiktoken.encoding_for_model(model_name)
            return len(enc.encode(text))
        except Exception as e:
            raise Exception(f"Token calculation error: {e}")

    def calculate_tokens(self, max_tokens=128000, input_percentage=75, prompt_percentage=5, response_percentage=20):
        # Validate the percentages
        total_percentage = input_percentage + prompt_percentage + response_percentage
        if total_percentage != 100:
            raise ValueError(f"The percentages must sum up to 100%. Currently, they sum up to {total_percentage}%.")
        if input_percentage < 0 or prompt_percentage < 0 or response_percentage < 0:
            raise ValueError("All percentages must be positive values.")

        # Calculate the tokens
        input_tokens = int(max_tokens * (input_percentage / 100))
        prompt_tokens = int(max_tokens * (prompt_percentage / 100))
        response_tokens = int(max_tokens * (response_percentage / 100))

        # Adjust to ensure the total equals max_tokens
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
        # Validate input types
        if not isinstance(max_tokens, (int, float)):
            raise TypeError("max_tokens must be an integer or float.")
        if not isinstance(tokens, (int, float)):
            raise TypeError("tokens must be an integer or float.")

        # Validate input values
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")
        if tokens < 0:
            raise ValueError("The number of tokens cannot be negative.")
        if tokens > max_tokens:
            raise ValueError("The number of tokens cannot exceed max_tokens.")

        # Calculate the percentage
        percentage = (tokens / max_tokens) * 100
        return round(percentage, 2)

    def split_text(self, text, model_name='gpt-4o-mini', input_max_tokens=115000, overlap=200):
        try:
            # Split the text into chunks
            chunks = self.split_into_chunks(text, model_name, input_max_tokens, overlap)
            return chunks
        except Exception as e:
            raise Exception(f"Error splitting text: {e}")

    def split_into_chunks(self, text, model_name='gpt-4o-mini', max_tokens=115000, overlap=200):
        try:
            encoding = tiktoken.encoding_for_model(model_name)
            tokens = encoding.encode(text)
            chunks = []
            step = max_tokens - overlap  # Define the step based on the overlap
            for i in range(0, len(tokens), step):
                chunk = tokens[i:i + max_tokens]
                decoded_chunk = encoding.decode(chunk)
                chunks.append(decoded_chunk)
                # If the next chunk exceeds the total number of tokens, adjust the step
                if i + max_tokens >= len(tokens):
                    break
            return chunks
        except Exception as e:
            raise Exception(f"Error splitting text into chunks: {e}")
    
    
    def refine(self, chunks,verbose=False):
        if verbose:
            print('Refining chunks')
            print('Number of chunks:', len(chunks))
        final_summary = []
        i = 0
        for chunk in chunks:
            i+=1
            if verbose:
                print('Processing chunk', i)
            if len(final_summary) == 0:
                new_summary = self.splitter_summarizer.run(inputs={'text': chunk})
                final_summary.append(new_summary)
                if verbose:
                    print(f'Final summary length: {self.count_tokens('\n'.join(final_summary))} tokens')
            else:
                new_summary = self.splitter_refiner.run(inputs={'summary': '\n'.join(final_summary), 'text': chunk})
                final_summary.append(new_summary)
                if verbose:
                    print(f'Final summary length: {self.count_tokens('\n'.join(final_summary))} tokens')
        return '\n'.join(final_summary)


    def map_reduce(self, chunks, max_tokens=115000, verbose=False, depth=0, max_depth=10):
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
        # Tokenizar o texto de entrada
        input_tokens = self.count_tokens(text)
       
        # Calcular o número máximo de chunks
        n_chunks = math.floor(max_tokens / max_summary_tokens)
        n_chunks = max(1, n_chunks)  # Garantir pelo menos um chunk

        # Calcular o tamanho de cada chunk
        chunk_size = math.ceil(input_tokens / n_chunks)

        # Dividir o texto em chunks
        chunks = self.split_into_chunks(text, model_name='gpt-4o-mini', max_tokens=chunk_size, overlap=100)

        # summarize 
        summary = self.map_reduce(chunks, max_tokens=chunk_size, verbose=verbose)

        return summary
    
    def summarize_to_fit_refined(self,  text, max_tokens, max_summary_tokens=1000, verbose=False):
        # Tokenizar o texto de entrada
        input_tokens = self.count_tokens(text)
       
        # Calcular o número máximo de chunks
        n_chunks = math.floor(max_tokens / max_summary_tokens)
        n_chunks = max(1, n_chunks)  # Garantir pelo menos um chunk

        # Calcular o tamanho de cada chunk
        chunk_size = math.ceil(input_tokens / n_chunks)

        # Dividir o texto em chunks
        chunks = self.split_into_chunks(text, model_name='gpt-4o-mini', max_tokens=chunk_size, overlap=100)

        # summarize
        summary = self.refine(chunks, verbose=verbose)

        return summary
        
       

                
   