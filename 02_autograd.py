"""
@author  : zhangyifan
@date    : 2020-3-24 16:33
@brief   : pytorch自动求倒机制torch.autograd
"""
import torch
# 1. backward
# 2. grad

def auto_backward():
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    # (w+x)*(w+1)
    a = torch.add(w, x)
    b = torch.add(w, torch.tensor([1.]))
    y = torch.mul(a, b)
    torch.autograd.backward(y) # == y.backward() 该函数调用得Line 17
    print(w.grad)
    print(x.grad)

def auto_grad():
    # 常用于高阶求导
    x = torch.tensor([3.], requires_grad=True)
    y = torch.pow(x, 2) # y = x**2
    # output:y, input:x, 创建导数计算图用于高阶求导, 返回元组
    grad_1 = torch.autograd.grad(y, x, create_graph=True)
    print(grad_1)
    grad_2 = torch.autograd.grad(grad_1[0], x)
    print(grad_2)
    

auto_backward()
auto_grad()