from gpts import gpt
agent = gpt.create_agent(
    role="Translator",
    goal="Translate text to English and return results in JSON format.",
    backstory="The text to translate is: {text}",
    knowledge="Expert translator with context awareness. Language nuances and idioms."
)
# Optionally customize JSON output format
agent.json_format = "{'translation':str}"

result = agent.run(inputs={'text': 'Olá, mundo!'}, temperature=0.7)

print(result)