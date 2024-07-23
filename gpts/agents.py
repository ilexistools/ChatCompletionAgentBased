import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import tiktoken 
import ast 

class GPTAgent:

    def __init__(self, **kwargs):
        # Load environment variables
        load_dotenv(find_dotenv())
        api_key=os.getenv("OPENAI_API_KEY")
        api_url_base = os.getenv("OPENAI_API_BASE")
        api_model = os.getenv("OPENAI_MODEL_NAME")
        # Set client
        if api_key == 'lm-studio':
         self.client = OpenAI(base_url=api_url_base, api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key)
        # Set default
        self.max_tokens = kwargs.get('max_tokens', 1000)
        self.gpt_model = kwargs.get('model', api_model)
        # GPT agent variables
        self.role = ''
        self.goal = ''
        self.backstory = ''
        self.knowledge = ''
        self.task = ''
        # Instantiate messages
        self.messages = []
        # format
        self.json_format = kwargs.get('json_format', None)
        # verbose
        self.verbose = False
    
    def add_system_message(self, content):
        message = {"role": "system", "content": content}
        self.messages.append(message)
    
    def add_user_message(self, content):
        message = {"role": "user", "content": content}
        self.messages.append(message)
    
    def clear_messages(self):
        self.messages = []
    
    def __calculate_tokens(prompt, model="gpt2"):
        # Escolha o codificador apropriado para o modelo
        enc = tiktoken.encoding_for_model(model)
        # Encode o prompt para obter os tokens
        tokens = enc.encode(prompt)
        # Calcule o número de tokens
        num_tokens = len(tokens)
        return num_tokens

    def get_total_tokens_in_messasges(self):
        texts = []
        for dic in self.messages:
            for k, v in dic.items():
                texts.append( k + v)
        return self.__calculate_tokens('\n'.join(texts))

    
    def run(self,**kwargs):
        self.clear_messages()
        verbose = kwargs.get('verbose', None)
        if verbose != None:
            self.verbose = verbose 
        inputs = kwargs.get('inputs', None)
        model = kwargs.get('model', None)
        if model != None:
            self.gpt_model = model
        max_tokens = kwargs.get('max_tokens', None)
        if max_tokens != None:
            self.max_tokens = max_tokens
        
        backstory = self.backstory
        if inputs != None:
            for k,v in inputs.items():
                backstory = self.backstory.replace('{' + str(k) + '}', str(v))

        self.add_system_message(self.role)
        self.add_system_message(self.goal)
        if len(self.knowledge) != 0:
            self.add_system_message('Use this information as knowledge base: ' + self.knowledge)
        if self.json_format is not None:
            self.add_system_message(f"JSON format set to: {self.json_format}")
            self.add_user_message('Return response as json.')
        self.add_user_message(backstory)
        

        if self.verbose == True:
            #tokens = self.get_total_tokens_in_messasges()
            print('Role: ' + self.role)
            print('Goal: ' + self.goal)
            print('Backstory: ' + self.backstory)
            print('Max tokens: ' + str(self.max_tokens)) 
            #print('Tokens in request: ' + str(tokens))

        completion = self.client.chat.completions.create(
            model=self.gpt_model,
            messages=self.messages,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"} if self.json_format else None

        )


        if self.json_format != None:
            try:
                return ast.literal_eval(completion.choices[0].message.content)
            except:
                return completion.choices[0].message.content
        else:
            return completion.choices[0].message.content

"""
mygpt = GptAgent()
mygpt.role = 'Assistant'
mygpt.goal = 'Answer questions'
mygpt.backstory = 'You answer questions briefly and concise.'
mygpt.knowledge = 'Humans like cats very much. They have them in their houses.'
mygpt.json_format = "{'result':str}"
result = mygpt.run('Qual a relação dos humanos com os gatos.')
"""
