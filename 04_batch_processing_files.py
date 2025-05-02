from gpts import gpt

agent = gpt.create_agent(
    role="Translator",
    goal="Translate text to English and return results in JSON format.",
    backstory="The text to translate is: {text}",
    knowledge="Expert translator with context awareness. Language nuances and idioms."
)
results = gpt.apply_agent_to_files(agent, 'data/source', verbose=True)
for entry in results:
    print(entry['filename'], entry['status'])