import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__).replace('examples', '')))
sys.path.append(lib_path)
from gpts import gpt

translator = gpt.create_agent('Translator', 'Translate words to English.', 'Translate the word: {word}')
translator.json_format = "{'translation':str}"
words = [
    {'word': 'computador'},
    {'word': 'eletricidade'},
    {'word': 'dados'}
]

results = gpt.batch_run(translator,words, verbose=True)


print("\nResults:\n")

for result in results:
    print(result)