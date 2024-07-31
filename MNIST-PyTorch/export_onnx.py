import onnx 
import torch
import torchvision 

from model import CNN

model = CNN(1)
model.load_state_dict(torch.load("model_weight_dict.pth"))
model.eval()

torch.onnx.export(model,
                torch.randn(1, 1, 28, 28),
                "./model.onnx",
                verbose=False,
                input_names=["input"],
                output_names=["output"],
                opset_version=10,
                do_constant_folding=True,
                )