from gpts import gpt 

translator = gpt.build('translator')


training_data = [
    {'text': 'She is a girl.'},
    {'text': 'You are my girlfriend.'},
    {'text': 'The fair is in town today.'},
    
]

gpt.improve_gpt_prompt_by_human( translator, training_data, 'gpt-4.1-mini', verbose=True)



