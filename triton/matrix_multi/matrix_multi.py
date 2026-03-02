import torch
import torch.nn as nn
import triton 
import triton.language as tl

@triton.jit
def matrix_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    x_row_stride,
    x_col_stride,
    y_row_stride,
    y_col_stride,
    out_row_stride,
    out_col_stride,
    x_row,
    x_col,
    y_row,
    y_col,
    out_row,
    out_col,
    BLOCK_SIZE_M:tl.constexpr,
    BLOCK_SIZE_N:tl.constexpr,
    BLOCK_SIZE_K:tl.constexpr,
    GROUP_SIZE_M:tl.constexpr,
    num_stage:tl.constexpr):
    #在这里GROUP_SIZE_M一定是小于all_row的
    #双重L2访存优化

    pid_row=tl.program_id(0)
    pid_col=tl.program_id(1)
    all_row = tl.num_programs(0)
    all_col = tl.num_programs(1)
    pid=pid_row+pid_col*all_col

    change_pro_number=all_col*GROUP_SIZE_M#需要这么多次变化一次row，等价于col再循环一次
    
    now_row_start=(pid//change_pro_number)*GROUP_SIZE_M*BLOCK_SIZE_M+(pid%GROUP_SIZE_M)*BLOCK_SIZE_M
    #当前在第几组pid//change_pro_number,每组处理GROUP_SIZE_M*BLOCK_SIZE_N个
    #pid%GROUP_SIZE_M当前在第几小组
    now_col_start=(pid-(pid//change_pro_number)*change_pro_number)%GROUP_SIZE_M*BLOCK_SIZE_N
    #tl.cdiv(pid,change_pro_number)当前是第几个循环
    #剩余部分%GROUP_SIZE_M表示当前在第几小组
    #当前小组的N是什么
    sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    