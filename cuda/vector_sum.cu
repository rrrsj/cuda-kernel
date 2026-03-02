#include<cuda_runtime.h>
#include<stdio.h>
#include"check_success.h"
#include<bits/stdc++.h>
using namespace std;

void cpu_sum_kernel(float*a,float*b,float*c,const int size)
{
    for(int i=0;i<size;i++)
    {
        c[i]=a[i]+b[i];
    }
    return ;
}

__global__ void gpu_sum_kernel(float*a,float*b,float*c)
{
    int now_thread_id = threadIdx.x;
    c[now_thread_id]=a[now_thread_id]+b[now_thread_id];
}

int main()
{
    int now_device=0;
    cudaSetDevice(now_device); //run in device 0
    int Element_number=128;
    float* p_vector_one=new float[Element_number];
    float* p_vector_two=new float[Element_number];
    float* p_ans=new float[Element_number];
    float* p_ans_from_gpu=new float[Element_number];
    memset(p_ans,0,sizeof(float)*Element_number);
    memset(p_ans_from_gpu,0,sizeof(float)*Element_number);
    float* gpu_vector_one,*gpu_vector_two,*gpu_ans;
    CHECK(cudaMalloc((float**)&gpu_vector_one,sizeof(float)*Element_number));
    CHECK(cudaMalloc((float**)&gpu_vector_two,sizeof(float)*Element_number));
    CHECK(cudaMalloc((float**)&gpu_ans,sizeof(float)*Element_number));
    initialData(p_vector_one,Element_number);
    initialData(p_vector_two,Element_number);
    CHECK(cudaMemcpy(gpu_vector_one,p_vector_one,sizeof(float)*Element_number,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(gpu_vector_two,p_vector_two,sizeof(float)*Element_number,cudaMemcpyHostToDevice));
    dim3 block(Element_number,1,1);//thread pre thread block 
    dim3 grid(Element_number/block.x);//thread block pre grid
    gpu_sum_kernel<<<grid,block>>>(gpu_vector_one,gpu_vector_two,gpu_ans);
    CHECK(cudaMemcpy(p_ans_from_gpu,gpu_ans,sizeof(float)*Element_number,cudaMemcpyDeviceToHost));
    cpu_sum_kernel(p_vector_one,p_vector_two,p_ans,Element_number);
    cout<<(check_result(p_ans,p_ans_from_gpu,Element_number))<<'\n';
    CHECK(cudaFree(gpu_vector_one));
    CHECK(cudaFree(gpu_vector_two));
    CHECK(cudaFree(gpu_ans));
    delete(p_vector_one);
    delete(p_vector_two);
    delete(p_ans_from_gpu);
    delete(p_ans);
    return 0;
}