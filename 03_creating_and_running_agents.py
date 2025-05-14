from gpts import gpt
from pydantic import BaseModel

class Translation(BaseModel):
    text: str
  
agent = gpt.create_agent(
    role="You are a Translator",
    goal="You translate text to English.",
    backstory="{text}",
    knowledge="Expert translator with context awareness. Language nuances and idioms."
)
# Optionally customize structured output format
agent.output_schema = Translation


result = agent.run(inputs={'text': 'Olá, mundo!'}, temperature=0.7)

print(result.text)