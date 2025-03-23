from gpts.factory import GPTFactory
from gpts.agents import GPTAgent

def ask(prompt, model='gpt-4o-mini', max_tokens=300):
    factory = GPTFactory()
    assistant = factory.build('assistant')
    assistant.gpt_model = model
    assistant.max_tokens = max_tokens 
    goal = 'You are a helpful assistant.' 
    knowledge = ''
    result = assistant.run(inputs={'goal': goal, 'prompt': prompt, 'knowledge':knowledge})
    return result

def build_agent(agent_name):
    factory = GPTFactory()
    agent = factory.build(agent_name)
    return agent

def batch_run(agent,inputs:list, verbose=False):
    results = []
    i = 0
    total = len(inputs)
    for input in inputs:
        if verbose:
            i+=1
            print(f"Processing {i} of {total}")
        try:
            result = agent.run(inputs=input)
            results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            results.append(None)
    return results

def create_agent(role:str, goal:str, backstory:str, knowledge:str='', model='gpt-4o-mini'):
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
    