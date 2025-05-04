from gpts import gpt 

# Options from the UI
ui_options = {}

# Get the options from the UI
# --- complete the code here ---

# Build and config the agent
agent = gpt.build(ui_options['agent_name'])
agent.gpt_model = ui_options['model']
agent.max_tokens = ui_options['max_tokens']
agent.temperature = ui_options['temperature']

# Execute the training method 
improved_agent_prompt = gpt.improve_gpt_prompt_by_ai(agent, ui_options['training_data'], ui_options['trainer_model'], verbose=True)

# Show the improved configuration
# --- complete the code here ---

# Add  button to save the improved configuration
# --- complete the code here ---


#gpt.improve_gpt_prompt_by_data(agent, ui_options['training_data'], expected_results,'gpt-4.1-mini', verbose=True)
#gpt.improve_gpt_prompt_by_ai(agent, training_data, 'gpt-4.1-mini', verbose=True)
#gpt.improve_gpt_prompt_by_human(agent, training_data, 'gpt-4.1-mini', verbose=True)



