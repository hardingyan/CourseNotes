import torch
import matplotlib.pyplot as plt

from model import Net
from utils import evaluate, get_data

def infer():
    _ , test_data = get_data()

    net0 = Net()
    net0.load_state_dict(torch.load("model_weight_dict.pth"))
    net0.eval()
    print("infer accuracy:", evaluate(test_data, net0))

    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    net1 = torch.load("model.pth")
    net1.eval()
    print("infer accuracy:", evaluate(test_data, net1))

    for (n, (x, _)) in enumerate(test_data):
        if n >= 1:
            break
        predict = torch.argmax(net1.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()

if __name__=="__main__":
    infer()