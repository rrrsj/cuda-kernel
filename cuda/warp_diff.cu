#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
using namespace std;

__global__ void slow_kernel(float* now)
{
    int now_thread_id=threadIdx.x;
    float a=0;
    if(now_thread_id%2==0)
    {
        a=1.0;
    }
    else
    {
        a=2.0;
    }
    now[now_thread_id]=a;
}

__global__ void fast_kernel(float*now)
{
    int now_thread_id=threadIdx.x;
    float a=0;
    if((now_thread_id/warpSize)%2==0)
    {
        a=1.0;
    }
    else
    {
        a=2.0;
    }
    now[now_thread_id]=a;
}//+rerank

int main()
{
    int now_device_id=0;
    cudaSetDevice(now_device_id);
    int Ele_number=1024;
    int thread_number=1024;
    int grid_number=1;

    float* ans=new float[Ele_number];
    dim3 thread_block(thread_number,1,1);
    dim3 grid(grid_number,1,1);
    float* gpu_ans;
    CHECK(cudaMalloc((float**)&gpu_ans,sizeof(float)*Ele_number));
    slow_kernel<<<grid,thread_block>>>(gpu_ans);
    CHECK(cudaMemcpy(ans,gpu_ans,sizeof(float)*Ele_number,cudaMemcpyDeviceToHost));
    for(int i=0;i<32;i++)
    {
        cout<<ans[i]<<' ';
    }
    return 0;
}