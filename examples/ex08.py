import sys
import os
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__).replace('examples', '')))
sys.path.append(lib_path)
from gpts import gpt

result = gpt.ask("What are the days of the week?")


print(result)



