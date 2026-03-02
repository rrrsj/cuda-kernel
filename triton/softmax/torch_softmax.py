import torch 
import torch.nn as nn
import triton 
import triton.language as tl 

def softmax(x:torch.Tensor,dim):
    x=x.transpose(dim,0)
    x_max=x.max(dim=0)[0]
    x=x-x_max[None,:]
    exp_x=torch.exp(x)
    sum_x=torch.sum(exp_x,dim=0)
    return (exp_x/sum_x[None,:]).transpose(dim,0)

x=torch.randn((2,2)).to('cuda')
torch_softmax=torch.softmax(x,dim=-1)
my_softmax=softmax(x,dim=-1)
print(torch_softmax)
print(my_softmax)
print((torch_softmax-my_softmax).max())