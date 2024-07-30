import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from numpy import vstack
from numpy import argmax
from sklearn.metrics import accuracy_score

from matplotlib import pyplot

def evaluate(test_data, model):
    predictions, actuals = list(), list()

    with torch.no_grad():
        for (inputs, targets) in test_data:
            yhat = model(inputs)
            yhat = yhat.detach().numpy()
            actual = targets.numpy()

            yhat = argmax(yhat, axis=1)

            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))

            predictions.append(yhat)
            actuals.append(actual)
        
        predictions, actuals = vstack(predictions), vstack(actuals)
        acc = accuracy_score(actuals, predictions)

        return acc


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=32, shuffle=True)


def get_data():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False) 

    return train_data, test_data


if __name__=="__main__":
    train_data, test_data = get_data()

    i, (inputs, targets) = next(enumerate(train_data))

    r = 5
    c = 5

    fig, axs = pyplot.subplots(r, c)
    pyplot.subplots_adjust(hspace = 2.0)

    for r0 in range(r):
        for c0 in range(c):
            idx = r0 * c + c0
            axs[r0, c0].imshow(inputs[idx][0], cmap = 'gray')
            axs[r0, c0].set_title(f'{targets[idx]}')

    pyplot.show()
