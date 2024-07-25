import onnx 
import torch
import torchvision 

net = torch.load("model.pth")
net.eval()

torch.onnx.export(net,  # 待转换的网络模型和参数
                torch.randn(1, 28*28), # 虚拟的输入，用于确定输入尺寸和推理计算图每个节点的尺寸
                "./model.onnx",  # 输出文件的名称
                verbose=False,      # 是否以字符串的形式显示计算图
                input_names=["input"],# + ["params_%d"%i for i in range(120)],  # 输入节点的名称，这里也可以给一个list，list中名称分别对应每一层可学习的参数，便于后续查询
                output_names=["output"], # 输出节点的名称
                opset_version=10,   # onnx 支持采用的operator set, 应该和pytorch版本相关，目前我这里最高支持10
                do_constant_folding=True, # 是否压缩常量
                )