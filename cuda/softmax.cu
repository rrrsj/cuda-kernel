#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
using namespace std;

void cpu_softmax(float*matrix_one,float*matrix_two,int n,int m)
{
    for(int i=0;i<n;i++)
    {
        float max_value=-10000;
        for(int j=0;j<m;j++)
        {
            max_value=max(max_value,matrix_one[i*m+j]);
        }
    }
}


__global__ void block_navie_softmax(float*input_matrix,float*output_matrix,int n,int m,int block_size)
{
    int thread_id_x=threadIdx.x;
    int thread_id_y=threadIdx.y;
    int block_id_x=blockIdx.x;
    int block_id_y=blockIdx.y;
    int all_thread_x=blockDim.x;
    int all_thread_y=blockDim.y;
    int now_x_ids=block_id_x*all_thread_x+thread_id_x;
    int now_y_ids=block_id_y*all_thread_y+thread_id_y;
    float block_max_value=-100000;
    extern __shared__ float row_value[];
    for(int i=0;i<block_size;i++)
    {
        block_max_value=fmaxf(block_max_value,input_matrix[now_x_ids*m+now_y_ids*block_size+i]);
    }
    row_value[now_y_ids]=block_max_value;
    __syncthreads();
    for(int i=(m/block_size)/2;i>=1;i/=2)
    {
        if(now_y_ids<i)
        {
            row_value[now_y_ids]=fmaxf(row_value[now_y_ids],row_value[now_y_ids+i]);
        }
        __syncthreads();
    }
    return ;
}


__device__ void warp_get_max(float now_value)
{
    for(int i=16;i>=1;i/=2)
    {
        now_value=fmaxf(now_value, __shfl_down_sync(0xffffffff, now_value, i));
    }
    return now_value;
}


__global__ void block_warp_softmax(float*input_matrix,float*output_matrix,int n,int m,int block_size)
{
    int thread_id_x=threadIdx.x;
    int thread_id_y=threadIdx.y;
    int block_id_x=blockIdx.x;
    int block_id_y=blockIdx.y;
    int all_thread_x=blockDim.x;
    int all_thread_y=blockDim.y;
    int now_x_ids=block_id_x*all_thread_x+thread_id_x;
    int now_y_ids=block_id_y*all_thread_y+thread_id_y;
    float block_max_value=-100000;
    extern __shared__ float row_value[];
    for(int i=0;i<block_size;i++)
    {
        block_max_value=fmaxf(block_max_value,input_matrix[now_x_ids*m+now_y_ids*block_size+i]);
    }

    
    row_value[now_y_ids]=block_max_value;
    __syncthreads();
    for(int i=(m/block_size)/2;i>=1;i/=2)
    {
        if(now_y_ids<i)
        {
            row_value[now_y_ids]=fmaxf(row_value[now_y_ids],row_value[now_y_ids+i]);
        }
        __syncthreads();
    }
    return ;

}

int main()
{
    int n=10,m=1024;
    float *matrix_one,*matrix_two;
    matrix_one=new float[n*m];
    matrix_two=new float[n*m];
    initialData(matrix_one,n*m);
    float *matrix_one_gpu,*matrix_two_gpu;
    CHECK(cudaMalloc(&matrix_one_gpu,sizeof(float)*n*m));
    CHECK(cudaMalloc(&matrix_two_gpu,sizeof(float)*n*m));
    CHECK(cudaMemcpy(matrix_one_gpu,matrix_one,sizeof(float)*n*m,cudaMemcpyHostToDevice));

    dim3 block(1,m/4,1);
    dim3 grid(n,1,1);
    size_t share_memory=sizeof(float)*m/4;
    SPEED((block_navie_softmax<<<grid,block,share_memory>>>(matrix_one_gpu,matrix_two_gpu,n,m,4)));


    return 0;
}