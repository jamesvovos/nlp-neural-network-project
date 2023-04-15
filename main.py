import spacy
from processor import DataProcessor
from chat import ChatBot
import torch

FILE = "data.pth"
trainingData = torch.load(FILE)

nlp = spacy.load('en_core_web_md')

# NOTE: toggle if you want to retrain the model
training_required: bool = False

# create NLP model object and pass it the text from file
data = DataProcessor(training_required)

# perform NLP pipeline process and return the updated data processor object
data_processor = data.initialise_data()

# setup chatbot interface
chatbot = ChatBot(data_processor, trainingData)


print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    resp = chatbot.get_response(sentence)
    print(resp)
