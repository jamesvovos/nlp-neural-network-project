import spacy
import random
import json
import requests
import torch
from model import NeuralNet

nlp = spacy.load('en_core_web_md')

""" AI chat interface """


class ChatBot(object):
    def __init__(self, dp):
        # NOTE: passing data processor object to perform tokenization/bag of words functions
        self.dp = dp
        self.bot_name = "James's AI"

    # function to setup chat bot
    def create_chat(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        response = requests.get("http://127.0.0.1:8000/intents/")
        intents = json.loads(response.text)

        FILE = "data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        tokenized_words = data["tokenized_words"]
        tags = data["tags"]
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

        print("Let's chat! (type 'quit' to exit)")
        while True:
            # sentence = "do you use credit cards?"
            sentence = input("You: ")
            self.dp.chatbot_text = nlp(sentence)
            if sentence == "quit":
                break

            sentence = self.dp.tokenize(sentence)
            x = self.dp.bag_of_words(sentence, tokenized_words)
            x = x.reshape(1, x.shape[0])
            x = torch.from_numpy(x).to(device)

            output = model(x)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            # check probabilities using softmax
            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            if prob.item() > 0.75:
                for intent in intents:
                    if tag == intent["tag"]:
                        print(
                            f"{self.bot_name}: {random.choice(intent['responses'])['text']}")
            else:
                print(f"{self.bot_name}: I do not understand...")
