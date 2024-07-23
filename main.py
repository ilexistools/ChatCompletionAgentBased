from gpts.factory import GPTFactory

factory = GPTFactory()
translator = factory.build('translator')
text = 'This is the text to be translated'
results = translator.run(inputs={'text': text})
print(results)
