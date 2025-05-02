from gpts import gpt 

ai = gpt.build('critics_evaluator')
ai.gpt_model = 'gpt-4.1-nano'

training_data = [
    {'text': 'The film works as an entertaining spy thriller set in the corridors of the Vatican, but also as a serene reflection on commitment, power and faith.[Full review in Spanish]'},
    {'text': 'The movie was great until the turn no n the ending was such disgusting rewriting of anything that would happen in the conclave. Would give it a zero'},
    {'text': "f it weren't for the ending I've have given this a straight 5 stars. As it was, the twist feels unnecessary and detracts from everything that's gone before."},
    
]

expected_results= ['Profissional', 'Público', 'Público']

#gpt.improve_gpt_prompt_by_human( ai, training_data, 'gpt-4.1-mini', verbose=True)
gpt.improve_gpt_prompt_by_data( ai, training_data, expected_results,'gpt-4.1-mini', verbose=True)


