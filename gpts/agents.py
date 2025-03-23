import os
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import tiktoken

class GPTAgent:
    def __init__(self, **kwargs):
        # Load environment variables
        load_dotenv(find_dotenv())
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url_base = os.getenv("OPENAI_API_BASE")
        self.api_model = os.getenv("OPENAI_MODEL_NAME")

        # Validate environment variables
        if not all([self.api_key, self.api_model]):
            raise ValueError("Missing required OpenAI environment variables.")

        # Set client
        self.client = OpenAI(base_url=self.api_url_base, api_key=self.api_key)

        # Set defaults
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.gpt_model = kwargs.get('model', self.api_model)
        self.temperature = kwargs.get('temperature', 0.2)
        self.json_format = kwargs.get('json_format', None)
        self.verbose = kwargs.get('verbose', False)

        # GPT agent variables
        self.role = kwargs.get('role', '')
        self.goal = kwargs.get('goal', '')
        self.backstory = kwargs.get('backstory', '')
        self.knowledge = kwargs.get('knowledge', '')
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def clear_messages(self):
        self.messages = []

    def _calculate_tokens(self, prompt, model="gpt2"):
        try:
            enc = tiktoken.encoding_for_model(model)
            return len(enc.encode(prompt))
        except Exception as e:
            if self.verbose:
                print(f"Token calculation error: {e}")
            return 0

    def run(self, **kwargs):
        # Update parameters dynamically
        self.verbose = kwargs.get('verbose', self.verbose)
        self.max_tokens = kwargs.get('max_tokens', self.max_tokens)
        self.gpt_model = kwargs.get('model', self.gpt_model)
        self.temperature = kwargs.get('temperature', self.temperature)
        inputs = kwargs.get('inputs', None)

        # Prepare backstory with placeholders replaced
        backstory = self.backstory
        if inputs:
            for k, v in inputs.items():
                backstory = backstory.replace(f"{{{k}}}", str(v))
        
        # Prepare knowledge with placeholders replaced
        knowledge = self.knowledge
        if inputs:
            for k, v in inputs.items():
                knowledge = knowledge.replace(f"{{{k}}}", str(v))

        # Build conversation messages
        if self.gpt_model in ['o1', 'o1-mini', 'o3-mini']:
            sys_role = 'assistant'
        else:
            sys_role = 'system' 

        self.clear_messages()
        self.add_message(sys_role, self.role)
        self.add_message(sys_role, self.goal)
        if knowledge:
            self.add_message(sys_role, f"Use this information as knowledge base: {knowledge}")
        if self.json_format:
            self.add_message(sys_role, f"JSON format set to: {self.json_format}")
            self.add_message("user", "Return response as JSON.")
        self.add_message("user", backstory)

        if self.verbose:
            print(f"Role: {self.role}")
            print(f"Goal: {self.goal}")
            print(f"Max tokens: {self.max_tokens}")
            print(f"Temperature: {self.temperature}")
            print(f"Messages: {self.messages}")

        # Make API call
        try:
            if self.gpt_model in ['o1', 'o1-mini', 'o3-mini']:
                # Convert messages to a single prompt string
                completion = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=self.messages,
                    response_format={"type": "json_object"} if self.json_format else None
                )
                # Extract response using .text for non-chat models
                response = completion.choices[0].message.content
            else:
                completion = self.client.chat.completions.create(
                    model=self.gpt_model,
                    messages=self.messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"} if self.json_format else None
                )
                response = completion.choices[0].message.content
            if self.verbose:
                print(f"Response: {response}")
            return json.loads(response) if self.json_format else response
        except Exception as e:
            print(f"API call error: {e}")
            return None
