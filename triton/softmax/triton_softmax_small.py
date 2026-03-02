import torch 
import torch.nn as nn
import triton 
import triton.language as tl 
from triton.runtime import driver

@triton.jit
def softmax_kernel(x_ptr,out_ptr,input_row_stride,input_col_stride,output_row_stride,output_col_stride, n_rows, n_cols,BLOCK_SIZE:tl.constexpr,parael_row:tl.constexpr):#是BLOCK级别的任务
    pro_start = tl.program_id(0)
    all_pro = tl.num_programs(0)
    for row_idx in tl.range(pro_start,n_rows,all_pro,num_stages=parael_row):
        row_start_ptr=x_ptr+row_idx*input_row_stride
        output_row_start_ptr=out_ptr+row_idx*output_row_stride
        load_element=tl.arange(0,BLOCK_SIZE)
        row_all_element=row_start_ptr+load_element*input_col_stride
        mask=load_element<n_cols
        row_data = tl.load(row_all_element, mask=mask, other=-float('inf'))
        row_new_data=row_data-tl.max(row_data, axis=0)
        exp_data=tl.exp(row_new_data)
        sum_exp_data=tl.sum(exp_data,axis=0)
        softmax_value=exp_data/sum_exp_data
        output_element=output_row_start_ptr+load_element*output_col_stride
        tl.store(output_element,softmax_value,mask=mask)



device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)#当前device的信息

NUM_SM = properties["multiprocessor_count"]#流处理器数量
NUM_REGS = properties["max_num_regs"]#一个SM寄存器数量限制
SIZE_SMEM = properties["max_shared_mem"]#共享内存容量
WARP_SIZE = properties["warpSize"]#一个 Warp 包含多少个线程
target = triton.runtime.driver.active.get_current_target()#告诉编译器我是哪种架构的 GPU，决定生成什么指令
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    num_warps = 8#多少个warp
    num_stages = 4 if SIZE_SMEM > 200000 else 2#同时处理多少个行
    y = torch.empty_like(x)


    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(x, y, x.stride(0),x.stride(1), y.stride(0),y.stride(1), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       parael_row=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs #每个线程使用的寄存器数量
        size_smem = kernel.metadata.shared#每个BLOCK的内存
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)#能启动多少个BLOCK，一个SM
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy#SM数量*BLOCK数量
        kernels[BLOCK_SIZE] = (kernel, num_programs)


    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](
        x, y, x.stride(0),x.stride(1), y.stride(0),y.stride(1), n_rows, n_cols,BLOCK_SIZE,num_stages
    )
    return y

torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name` `x_name` 的不同可能值
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot 参数名，其值对应于图表中不同线条
        line_vals=['triton', 'torch'],  # possible values for `line_arg`` `line_arg` 的可能值
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # line styles 线条的样式
        ylabel="GB/s",  # label name for the y-axis y 轴的标签名称
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot. 图表的名称，也用作保存图表的文件名
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name` `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)




benchmark.run(show_plots=False, print_data=True, save_path='./')