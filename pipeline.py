import spacy
from spacy import displacy
import numpy as np
from training import TrainingModel

nlp = spacy.load('en_core_web_md')

""" NLP custom pipeline """


class CustomPipeline(object):
    def __init__(self, dp):
        # NOTE: passing the pipeline obj the data processor so it can process text input
        self.dp = dp

    # create NLP pipeline
    def create_pipeline(self):
        # NOTE: tokenization pipeline stage
        for intent in self.dp.intents:
            tag = intent['tag']
            for pattern in intent['patterns']:
                token = self.dp.tokenize(pattern)
                self.dp.tokenized_words.extend(token)
                # add named entity recognition (NER)
                self.dp.ner_tags.append(
                    {"label": tag.upper(), "pattern": pattern})
                # pattern : tag for XY training data
                self.dp.xy.append((token, tag))
        # inject the custom component into the NLP pipeline
        self.add_custom_components("Hey there")

        # NOTE: lemmatization pipeline stage
        self.dp.lemmatized_words = self.dp.tokenized_words
        # sort the array and ensure no duplicates (hence, the hashset)
        self.dp.lemmatized_words = sorted(set(self.dp.lemmatized_words))

        # NOTE: bag of words pipeline stage
        for (pattern_sentence, tag) in self.dp.xy:
            bag = self.dp.bag_of_words(
                pattern_sentence, self.dp.tokenized_words)
            self.dp.X_train.append(bag)

            label = self.dp.tags.index(tag)
            self.dp.y_train.append(label)  # CrossEntropyLoss

        # convert to numpy array
        self.dp.X_train = np.array(self.dp.X_train)
        self.dp.y_train = np.array(self.dp.y_train)

        # if training is toggled on -> train the model
        if self.dp.training_required == True:
            self.start_training()

    # add custom components to pipeline
    def add_custom_components(self, text):
        # create a new doc object to overrite current doc object
        updated_nlp = spacy.load('en_core_web_md')
        # create ruler
        ruler = updated_nlp.add_pipe("entity_ruler", before="ner")
        # add the tag : pattern
        ruler.add_patterns(self.dp.ner_tags)
        # return the new doc object we work with
        return updated_nlp(text)

    # view pipeline components
    def view_pipeline(self):
        print("Pipeline:", nlp.pipe_names)

    # train the model using the XY data
    def start_training(self):
        training = TrainingModel(self.dp)
        training.train()
