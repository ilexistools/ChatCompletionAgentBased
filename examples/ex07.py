import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__).replace('examples', '')))
sys.path.append(lib_path)
from gpts.factory import GPTFactory

factory = GPTFactory()

assistant = factory.build('assistant')
assistant.gpt_model = 'gpt-4o-mini'
assistant.max_tokens = 500


goal = 'Translate text.'
prompt = 'Translate the sentence to English: Programa é fácil.' 
knowledge = 'Brazilian portuguese'

result = assistant.run(inputs={'goal': goal, 'prompt': prompt, 'knowledge':knowledge})

print(result)