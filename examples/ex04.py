import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__).replace('examples', '')))
sys.path.append(lib_path)
from gpts.splitter import TextSplitter

ts = TextSplitter()

with open('examples/text.txt', 'r', encoding='utf-8') as fh:
    text = fh.read()
    
print('Text size: %s' % (ts.count_tokens(text, model_name="gpt-4o-mini")))

chunks = ts.split_text(text, model_name="gpt-4o-mini", input_max_tokens=8000, overlap=100)
print('Number of chunks: %s' % len(chunks))


