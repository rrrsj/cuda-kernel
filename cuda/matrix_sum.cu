#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
using namespace std;

__global__ void matrix_sum(float* matrix_one,float* matrix_two,float*matrix_ans,int Element_x,int Element_y)
{
    int thread_x=threadIdx.x;
    int thread_y=threadIdx.y;
    int now_index=thread_x*Element_y+thread_y;
    if(thread_x<Element_x&&thread_y<Element_y)
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
    int Element_x=10;
    int Element_y=10;
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
    dim3 block(Element_x,Element_y,1);
    dim3 grid(1,1,1);
    matrix_sum<<<grid,block>>>(matrix_one_gpu,matrix_two_gpu,matrix_ans_gpu,Element_x,Element_y);
    CHECK(cudaMemcpy(matrix_ans_host,matrix_ans_gpu,sizeof(float)*Element_x*Element_y,cudaMemcpyDeviceToHost));
    cpu_matrix_sum(matrix_one_host,matrix_two_host,matrix_ans,Element_x,Element_y);
    cout<<check_result(matrix_ans,matrix_ans_host,Element_x*Element_y);
    

    return 0;
}