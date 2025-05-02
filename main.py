from gpts import gpt 

translator = gpt.build('translator')
text = 'This is the text to be translated'
results = translator.run(inputs={'text': text})
print(results)
