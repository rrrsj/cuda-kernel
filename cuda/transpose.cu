#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
using namespace std;
void cpu_transpose(float* a,float* b,int x_element,int y_element)
{
    for(int i=0;i<x_element;i++)
    {
        for(int j=0;j<y_element;j++)
        {
            b[j*x_element+i]=a[i*y_element+j];
        }
    }
    return ;
}

__global__ void gpu_transpose_kernel_write_merge(float *a,float *b,int x_element,int y_element)
{
    int thread_x_index=threadIdx.x;
    int thread_y_index=threadIdx.y;
    int grid_x_index=blockIdx.x;
    int grid_y_index=blockIdx.y;
    int thread_x_number=blockDim.x;
    int thread_y_number=blockDim.y;
    int now_x=thread_x_number*grid_x_index+thread_x_index;
    int now_y=thread_y_number*grid_y_index+thread_y_index;
    if(now_x<x_element && now_y<y_element)
    {
        b[now_y*x_element+now_x]=a[now_x*y_element+now_y];//随着x增加，b时连续的，所以是写连续的
    }
    return ;
}


__global__ void gpu_transpose_kernel_read_merge(float *a,float *b,int x_element,int y_element)
{
    int thread_x_index=threadIdx.x;
    int thread_y_index=threadIdx.y;
    int grid_x_index=blockIdx.x;
    int grid_y_index=blockIdx.y;
    int thread_x_number=blockDim.x;
    int thread_y_number=blockDim.y;
    int now_x=thread_x_number*grid_x_index+thread_x_index;
    int now_y=thread_y_number*grid_y_index+thread_y_index;
    if(now_x<x_element && now_y<y_element)
    {
        b[now_x*y_element+now_y]=a[now_y*x_element+now_x];//随着x增加，a是连续的，所以是读连续的
    }
    return ;
}

//对角映射
//让其横纵坐标都+1，让其尽量不访问同一个行或者不访问同一个列
int main()
{

    return 0;
}