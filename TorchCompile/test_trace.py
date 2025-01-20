import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
 
    def forward(self, x):
        x = x**2 
        x = x**3 

        return x

pattern = """ 
    graph(%x): 
        %const_2 = prim::Constant[value=2]() 
        %out = aten::pow(%x, %const_2) 
        return (%out) 
""" 
 
replacement = """ 
    graph(%x): 
        %out = aten::mul(%x, %x) 
        return (%out) 
""" 

if __name__ == '__main__':
    model = Model()

    dummy_input = torch.rand(4, 4) 

    with torch.no_grad():
        jit_model = torch.jit.trace(model, dummy_input)
    
    print(jit_model.graph)
    print(jit_model.code)

    torch._C._jit_pass_custom_pattern_based_rewrite_graph(pattern, replacement, 
                                                      jit_model.graph) 
 
    print(jit_model.graph) 
    print(jit_model.code)