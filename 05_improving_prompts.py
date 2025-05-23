from gpts import gpt 

agent = gpt.build('critics_evaluator')
agent.gpt_model = 'gpt-4.1-nano'

training_data = [
    {'text': 'The film works as an entertaining spy thriller set in the corridors of the Vatican, but also as a serene reflection on commitment, power and faith.[Full review in Spanish]'},
    {'text': 'The movie was great until the turn no n the ending was such disgusting rewriting of anything that would happen in the conclave. Would give it a zero'},
    {'text': "f it weren't for the ending I've have given this a straight 5 stars. As it was, the twist feels unnecessary and detracts from everything that's gone before."},
    {'text': "Cage does go all in, but remains in control and never fully succumbs to overdone theatrics as in the recent past. It’s a performance that makes his unnamed “surfer” sad, tortured, pathetic, relatable and redeemable."},
    {'text': "Horrible movie. At times made no sense, abstract, inane. More like B movie. I don't think that I've seen that bad of a movie in years. Save your $$."}
]

expected_results= ['Professional', 'Audience', 'Audience', 'Professional', 'Audience']

gpt.improve_gpt_prompt_by_data(agent, training_data, expected_results,'gpt-4.1-mini', verbose=True)
#gpt.improve_gpt_prompt_by_ai(agent, training_data, 'gpt-4.1-mini', verbose=True)
#gpt.improve_gpt_prompt_by_human(agent, training_data, 'gpt-4.1-mini', verbose=True)





