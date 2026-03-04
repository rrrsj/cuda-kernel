#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
using namespace std;
//值得注意的是一个线程块必须在一个sm上运行，所以需要grid的合理设置
__global__ void matrix_sum(float* matrix_one,float* matrix_two,float*matrix_ans,int Element_x,int Element_y)
{
    int all_thread_x=blockDim.x;
    int all_thread_y=blockDim.y;
    int thread_x=threadIdx.x;
    int thread_y=threadIdx.y;
    int block_x=blockIdx.x;
    int block_y=blockIdx.y;
    int now_index=(block_x*all_thread_x+thread_x)*Element_y+block_y*all_thread_y+thread_y;
    if((block_x*all_thread_x+thread_x)<Element_x&&(block_y*all_thread_y+thread_y)<Element_y)
        matrix_ans[now_index]=matrix_one[now_index]+matrix_two[now_index];
}
void cpu_matrix_sum(float* matrix_one,float* matrix_two,float* matrix_ans,int Element_x,int Element_y)
{
    for(int i=0;i<Element_x*Element_y;i++)
    {
        matrix_ans[i]=matrix_one[i]+matrix_two[i];
    }    
    return ;
}

int main()
{
    int now_device=0;
    cudaSetDevice(now_device);
    int Element_x=1024;
    int Element_y=1024;
    float* matrix_one_host=new float[Element_x*Element_y];
    float* matrix_two_host=new float[Element_x*Element_y];
    float* matrix_ans_host=new float[Element_x*Element_y];
    float* matrix_ans=new float[Element_x*Element_y];
    float* matrix_one_gpu,*matrix_two_gpu,*matrix_ans_gpu;
    initialData(matrix_one_host,Element_x*Element_y);
    initialData(matrix_two_host,Element_x*Element_y);
    CHECK(cudaMalloc((float**)&matrix_one_gpu,sizeof(float)*Element_x*Element_y));
    CHECK(cudaMalloc((float**)&matrix_two_gpu,sizeof(float)*Element_x*Element_y));
    CHECK(cudaMalloc((float**)&matrix_ans_gpu,sizeof(float)*Element_x*Element_y));
    
    CHECK(cudaMemcpy(matrix_one_gpu,matrix_one_host,sizeof(float)*Element_x*Element_y,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(matrix_two_gpu,matrix_two_host,sizeof(float)*Element_x*Element_y,cudaMemcpyHostToDevice));
    int block_x=10;
    int block_y=10;
    int grid_x=(Element_x+block_x-1)/block_x;
    int grid_y=(Element_y+block_y-1)/block_y;
    dim3 block(block_x,block_y,1);
    dim3 grid(grid_x,grid_y,1);
    matrix_sum<<<grid,block>>>(matrix_one_gpu,matrix_two_gpu,matrix_ans_gpu,Element_x,Element_y);
    CHECK(cudaMemcpy(matrix_ans_host,matrix_ans_gpu,sizeof(float)*Element_x*Element_y,cudaMemcpyDeviceToHost));
    cpu_matrix_sum(matrix_one_host,matrix_two_host,matrix_ans,Element_x,Element_y);
    cout<<check_result(matrix_ans,matrix_ans_host,Element_x*Element_y);
    

    return 0;
}