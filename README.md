# NLP Neural Network Research Project.
## _Implementation using PyTorch and spaCy._

Inspired by: https://www.youtube.com/watch?v=RpWeNzfSUHw&list=LL&index=37

## Installation Guide
PyTorch installation instructions [PyTorch website](https://pytorch.org/)
spaCy installation instructions [spaCy website](https://spacy.io/)

## 1. Install PyTorch
![PyTorchImage](https://imageio.forbes.com/specials-images/imageserve/60d815da0c030140b46c2abd/PyTorch-Facebook/960x0.jpg?format=jpg&width=960)

Install `PyTorch`
```sh
$pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```
## 2. Install spaCy and dependencies
![spaCyImage](https://spacy.io/_next/static/media/social_default.96b04585.jpg)
For spaCy NLP library installation:

Install setup tools:
```sh
$pip install -U pip setuptools wheel
```
Install `spaCy` NLP library:
```sh
$pip install -U spacy
```
Add model:
```sh
$python -m spacy download en_core_web_md
```

## Features

- Ability to pre-process text using NLP techniques such as tokenization, lemmatization, etc.
- Implementation of a feed forward neural network model using PyTorch.
- Modify JSON file or inject JSON via API to train model.
- Add custom tags and modify NLP pipeline using custom components and named entity recognition patterns (NER)
- Ability to trade out 'bag of words' and replace with word vectors, etc.
- Will be extending the use case to test different NLP techniques.
- Building API to create dynamic NPC characters in games.

## Tech Stack

Tech the project uses:

- [Python3] - High-level programming language
- [PyTorch] - Machine learning framework
- [spaCy.io] - Industrial-strength natural language processing library

## Customize
Have a look at [data/training-data.json](https://github.com/jamesvovos/nlp-neural-network-project/blob/master/data/training-data.json) . You can customize it according to your own use case. Just define a new `tag`, possible `patterns`, and possible `responses` for the AI to pick up on.

```sh
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey",
        "How are you",
        "Is anyone there?",
        "Hello",
        "Good day"
      ],
      "responses": [
        "Hey :-)",
        "Hello, thanks for visiting",
        "Hi there, what can I do for you?",
        "Hi there, how can I help?"
      ]
    },
    ...
  ]
}
```
You have to re-run the training whenever this file is modified. In the [main.py](https://github.com/jamesvovos/nlp-neural-network-project/blob/master/main.py) file just toggle the `training_required` bool to `True`
```sh
import spacy
from processor import DataProcessor
from chat import ChatBot

nlp = spacy.load('en_core_web_md')

# NOTE: toggle if you want to retrain the model
training_required: bool = False

# create NLP model object and pass it the text from file
data = DataProcessor(training_required)

# perform NLP pipeline process and return the updated data processor object
data_processor = data.initialise_data()

# setup chatbot interface
chatbot = ChatBot(data_processor)
chatbot.create_chat()
```

## License

MIT

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [spaCy.io]: <https://spacy.io/>
   [Python3]: <https://www.python.org/>
   [PyTorch]: <https://pytorch.org/>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
