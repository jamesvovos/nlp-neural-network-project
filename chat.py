import spacy
import random
import json
import requests
import torch
from model import NeuralNet

nlp = spacy.load('en_core_web_md')

""" AI chat interface """


class ChatBot(object):
    def __init__(self, dp, data):
        # NOTE: passing data processor object to perform tokenization/bag of words functions
        self.dp = dp
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.response = requests.get("http://127.0.0.1:8000/intents/")
        self.intents = json.loads(self.response.text)
        self.bot_name = "James's AI"
        self.input_size = data["input_size"]
        self.hidden_size = data["hidden_size"]
        self.output_size = data["output_size"]
        self.tokenized_words = data["tokenized_words"]
        self.tags = data["tags"]
        self.model_state = data["model_state"]

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
                    return random.choice(intent['responses'])['text']
        else:
            return "I do not understand..."
