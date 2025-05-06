from gpts import gpt 

translator = gpt.build('translator')
translator.json_format = "{'translation': str}"
text = 'This is the text to be translated'
results = translator.run(inputs={'text': text}, verbose=True)
print(results)
