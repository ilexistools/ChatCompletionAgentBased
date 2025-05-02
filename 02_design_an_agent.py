from gpts import gpt
spec = gpt.design_agent(
    agent_name="Summarizer",
    task_description="Summarize articles into key bullet points.",
    input_placeholder="Provide article text here"
)
print(spec)