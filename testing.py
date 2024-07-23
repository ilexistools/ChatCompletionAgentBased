from gpts.factory import GPTFactory

factory = GPTFactory()
writer = factory.build('writer')
translator = factory.build('translator')

text = writer.run(inputs={'sign': 'Leo'})
result = translator.run(inputs={'text': text})

print(result)

