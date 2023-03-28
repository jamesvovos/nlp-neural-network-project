import spacy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

nlp = spacy.load('en_core_web_md')

""" Train a neural network model using pre-processed dataset """


class TrainingModel(object):
    def __init__(self, dp):
        self.dp = dp  # model instance is passed a data processor object

    # train the neural network model
    def train(self):
        # Hyperparameters
        batch_size = 8
        hidden_size = 8
        output_size = len(self.dp.tags)
        input_size = len(self.dp.X_train[0])
        learning_rate = 0.001
        num_epochs = 1000

        dataset = ChatDataSet(self.dp)  # pass the data processor object
        train_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )

        # uses Nvidia cuda cores if available otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = NeuralNet(input_size, hidden_size, output_size).to(device)

        # loss and optimizer parameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for (words, labels) in train_loader:
                words = words.to(device)
                labels = labels.to(device, dtype=torch.int64)

                # forward
                outputs = model(words)
                loss = criterion(outputs, labels)

                # backward and optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

        print(f"final loss, loss={loss.item():.4f}")

        # save the data in a dictionary
        data = {
            "model_state": model.state_dict(),
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": hidden_size,
            "tokenized_words": self.dp.tokenized_words,
            "tags": self.dp.tags,
        }

        # saves and serializes the data
        FILE = "data.pth"
        torch.save(data, FILE)

        print(f"training complete, file saved to {FILE}")


class ChatDataSet(Dataset):
    def __init__(self, dp):
        self.n_samples = len(dp.X_train)
        self.x_data = dp.X_train
        self.y_data = dp.y_train

    # dataset [index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples
