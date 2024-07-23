import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__).replace('examples', '')))
sys.path.append(lib_path)
from gpts.factory import GPTFactory

factory = GPTFactory()
writer = factory.build('writer', temperature=0.7, max_tokens=100)
translator = factory.build('translator', temperature=0.1, max_tokens=150)
translator.json_format="{'translation':str}"

text = writer.run(inputs={'sign': 'Leo'})
result = translator.run(inputs={'text': text})

print(result)
