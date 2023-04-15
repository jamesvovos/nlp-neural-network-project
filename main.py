import spacy
from chat import ChatBot

nlp = spacy.load('en_core_web_md')

# NOTE: toggle to true you want to retrain the model
training_required: bool = True
npc_id: int = 1

# setup chatbot interface
chatbot = ChatBot(training_required, npc_id)
chatbot.setup()

print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    resp = chatbot.get_response(sentence)
    print(resp)
