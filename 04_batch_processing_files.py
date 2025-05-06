from gpts import gpt

agent = gpt.create_agent(
    role="You are a Translator",
    goal="You translate text to English.",
    backstory="{text}",
    knowledge="Expert translator with context awareness. Language nuances and idioms."
)
results = gpt.apply_agent_to_files(agent, 'data/source', verbose=True)
for entry in results:
    print(entry['filename'], entry['status'])