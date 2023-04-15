import spacy
import random
import json
import requests
import torch
from model import NeuralNet
from processor import DataProcessor

nlp = spacy.load('en_core_web_md')

""" AI chat interface """


class ChatBot(object):
    def __init__(self, training_required: bool, npc_id: int):
        self.npc_id = npc_id
        self.file = "data.pth"
        self.data = torch.load(self.file)
        self.dp = DataProcessor(training_required)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.response = requests.get(
            "http://127.0.0.1:8000/npcs/intents/" + str(npc_id))
        self.intents = json.loads(self.response.text)
        self.input_size = self.data["input_size"]
        self.hidden_size = self.data["hidden_size"]
        self.output_size = self.data["output_size"]
        self.tokenized_words = self.data["tokenized_words"]
        self.tags = self.data["tags"]
        self.model_state = self.data["model_state"]

    def setup(self):
        self.dp.initialise_data(self.npc_id)

    # function to get response from chatbot
    def get_response(self, msg):

        model = NeuralNet(self.input_size, self.hidden_size,
                          self.output_size).to(self.device)
        model.load_state_dict(self.model_state)
        model.eval()

        # sentence = "do you use credit cards?"
        sentence = msg
        self.dp.chatbot_text = nlp(sentence)

        sentence = self.dp.tokenize(sentence)
        x = self.dp.bag_of_words(sentence, self.tokenized_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(self.device)

        output = model(x)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        # check probabilities using softmax
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in self.intents:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        else:
            return "I do not understand..."
