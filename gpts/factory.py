import yaml
import os 
from gpts.agents import GPTAgent

class GPTFactory:

    def __init__(self):
        files = os.listdir('./config/')
        self.config = {}
        for filename in files:
            data = self.__load_from_yaml('./config/' + filename)
            self.config[data[0]] = data[1]

    def __load_from_yaml(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
        first_key = next(iter(data))
        dic = data[first_key]
        return (first_key, dic)
    
    def build(self, name,**kwargs):
        model = kwargs.get('model', None)
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', None)
        data = self.config[name]
        gpt = GPTAgent()
        gpt.role = data['role']
        gpt.goal = data['goal']
        gpt.backstory = data['backstory']
        gpt.knowledge = data['knowledge']
        if model != None:
            gpt.gpt_model = model 
        if max_tokens != None:
            gpt.max_tokens = max_tokens
        if temperature != None:
            gpt.temperature = temperature
        return gpt 

