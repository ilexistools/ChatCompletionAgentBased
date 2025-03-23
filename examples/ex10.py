import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__).replace('examples', '')))
sys.path.append(lib_path)
from gpts import gpt

with open('examples/text.txt', 'r', encoding='utf-8') as fh:
    text = fh.read()

total_tokens = gpt.count_tokens(text)
print(f"Total tokens: {total_tokens}")

#result = gpt.summarize(max_tokens_output=1000, text=text,chunk_size=10000, model='gpt-4o-mini',overlap=100,verbose=True)

#result = gpt.answer_questions('What is the name of the monster? What is the name of the female main character?', max_tokens_output=1000, text=text,chunk_size=10000, model='gpt-4o-mini',overlap=100,verbose=True)

result = gpt.extract_information('Main characters and places.', max_tokens_output=1000, text=text,chunk_size=10000, model='gpt-4o-mini',overlap=100,verbose=True)

print(result)

