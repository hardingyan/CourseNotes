import torch

from model import Net
from utils import evaluate, get_data


def train():
    train_data, test_data = get_data()
    net = Net()

    print("initial accuracy:", evaluate(test_data, net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        for (x, y) in train_data:
            net.zero_grad()
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))

    torch.save(net.state_dict(), "model_weight_dict.pth")
    torch.save(net, "model.pth")


if __name__=="__main__":
    train()