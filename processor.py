import spacy
from spacy import displacy
import pandas as pd
import numpy as np
import requests
import json
from pipeline import CustomPipeline

#nlp = spacy.load("en_core_web_sm")
nlp = spacy.load('en_core_web_md')
#nlp = spacy.blank("en")

""" NLP data processing helper functions """


class DataProcessor(object):
    def __init__(self, training_required):
        self.training_required = training_required
        self.tokenized_words = []
        self.lemmatized_words = []
        self.intents = []
        self.tags = []
        self.patterns = []
        self.responses = []
        self.ner_tags = []  # tag : pattern
        self.xy = []  # pattern : tag
        self.X_train = []
        self.y_train = []

    # tokenization
    def tokenize(self, sentence):
        stop_words = ['?', '!', '.', ',']
        doc = nlp(sentence)
        return [token.text.lower() for token in doc if token.text.lower() not in stop_words]

    # lemmatization
    def lemmatize(self, word):
        doc = nlp(word)
        return doc[0].lemma_

    # bag of words
    def bag_of_words(self, tokenized_sentence, words):
        """
        return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bow   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
        """
        # lemmatize each word
        tokenized_sentence = [self.lemmatize(
            word) for word in tokenized_sentence]
        # initialize bag with 0 for each word
        bag = np.zeros(len(words), dtype=np.float32)
        # grabs the index and the word from words array
        for i, word in enumerate(words):
            if word in tokenized_sentence:
                bag[i] = 1

        return bag

    # calculate the similarity between input a and input b using word vectors
    def calculate_word_vectors(self, a, b):
        return (a, "<->", b, a.similarity(b))

    # named entity recognition (NER)
    def named_entity_recognition(self, text):
        doc = nlp(text)
        for ent in doc.ents:
            print(ent.text, ent.label_)

    # visualizes the raw data using tags in browser
    def visualize_data(self, text):
        doc = nlp(text)
        displacy.serve(doc, style="ent")

    # get JSON file training data and create the NLP pipeline
    def initialise_data(self, npc_id: int):
        response = requests.get(
            "http://127.0.0.1:8000/npcs/intents/" + str(npc_id))
        intents = json.loads(response.text)

        # Convert list of dictionaries to JSON string
        intents_json = json.dumps(intents)

        # Read data from JSON string using pandas
        df = pd.read_json(intents_json)

        # NOTE: useful to debug dataframe data
        # print(df)

        # data we want to extract
        tags = df['tag'].tolist()
        patterns = [pattern for p in df['patterns'] for pattern in p]
        responses = [response for r in df['responses'] for response in r]

        # add intents
        for intent in intents:
            self.intents.append(intent)

        # add tags
        for tag in tags:
            self.tags.append(tag)

        # add patterns
        for pattern in patterns:
            self.patterns.append(pattern)

        # add responses
        for response in responses:
            self.responses.append(response)

        # setup the NLP pipeline
        pipeline = CustomPipeline(self)
        pipeline.create_pipeline()
        # pipeline.add_custom_components()

        return self
