import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from model import CNN
from utils import evaluate, get_data


def train():
    train_data, _ = get_data()
    model = CNN(1)

    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(2):
        for (inputs, targets) in train_data:
            model.zero_grad()
            yhat = model.forward(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "model_weight_dict.pth")


if __name__=="__main__":
    train()