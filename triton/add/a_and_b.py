import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['all_number'], 
)
@triton.jit
def add_kernel(x_ptr,y_ptr,out_ptr,all_number,BLOCK_SIZE:tl.constexpr):
    now_id=tl.program_id(axis=0)
    now_start=now_id*BLOCK_SIZE
    now_element=now_start+tl.arange(0,BLOCK_SIZE)
    mask=now_element<all_number
    x=tl.load(x_ptr+now_element,mask=mask)
    y=tl.load(y_ptr+now_element,mask=mask)
    output=x+y
    tl.store(out_ptr+now_element,output,mask=mask)


def add_function(x:torch.Tensor,y:torch.Tensor):
    output=torch.empty_like(x)

    assert x.is_cuda and y.is_cuda and output.is_cuda 
    assert x.device==y.device

    all_element=output.numel()

    grid_map=lambda meta:(triton.cdiv(all_element,meta['BLOCK_SIZE']),)

    add_kernel[grid_map](x,y,output,all_element)
    return output


size=114514
x=torch.rand(size).to('cuda')
y=torch.rand(size).to('cuda')
torch_ans=x+y
triron_ans=add_function(x,y)
print(torch_ans)
print(triron_ans)
print((torch_ans-triron_ans).max())
    