from gpts import gpt
agent = gpt.create_agent(
    role="You are a Translator",
    goal="You translate text to English.",
    backstory="{text}",
    knowledge="Expert translator with context awareness. Language nuances and idioms."
)
# Optionally customize JSON output format
agent.json_format = "{'translation':str}"

result = agent.run(inputs={'text': 'Olá, mundo!'}, temperature=0.7)

print(result)