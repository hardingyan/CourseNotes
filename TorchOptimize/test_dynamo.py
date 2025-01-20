from typing import List
import torch
import torch._dynamo as dynamo
from torch._dynamo import optimize
import torch._inductor.config
import dis

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("[dynamo] FX graph:")
    gm.graph.print_tabular()
    print(f"[dynamo] Code: {gm.code}")
    return gm.forward  # return a python callable

def func_add(x, y):
    return (x + y)

def func_if(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def test_dis():
    for k in ["co_names", "co_varnames", "co_consts"]:
        print(k, getattr(func_add.__code__, k))

    print("\nByte Code:\n")
    print(dis.dis(func_add))

def test_dynamo():
    func_add_opt = torch.compile(func_add, backend=my_compiler)

    a, b = torch.randn(10), torch.ones(10)
    c = func_add_opt(a, b)

def test_dynamo_explain():
    explanation = dynamo.explain(func_if)(torch.randn(10), torch.randn(10))
    print(explanation)

def test_dynamo_optimize():
    torch._inductor.config.debug = True
    func_add_opt = optimize("inductor")(func_add)

    a, b = torch.randn(10), torch.ones(10)
    c = func_add_opt(a, b)


if __name__ == "__main__":
    # test_dis()
    # test_dynamo()
    # test_dynamo_explain()
    test_dynamo_optimize()
