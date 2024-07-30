import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

from PIL import Image

from model import CNN
from utils import evaluate, get_data


if __name__=="__main__":
    _ , test_data = get_data()

    model = CNN(1)
    model.load_state_dict(torch.load("model_weight_dict.pth"))
    model.eval()
    # print("infer accuracy:", evaluate(test_data, model))

    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        image = Image.open(image_path)
        image = image.convert('L')
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).float()
    else:
        image, _ = next(iter(test_data))
        image = image[0]
    
    predict = torch.argmax(model.forward(image))
    plt.imshow(image[0])
    plt.title("prediction: " + str(int(predict)))
    plt.show()