import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__).replace('examples', '')))
sys.path.append(lib_path)
from gpts.factory import GPTFactory

factory = GPTFactory()
writer = factory.build('writer')
translator = factory.build('translator')

text = writer.run(inputs={'sign': 'Leo'})
result = translator.run(inputs={'text': text})

print(result)
