#include<stdio.h>
#include<cuda_runtime.h>
__global__ void hello_world()//global device could run
{
    printf("cuda: hello world.\n");
}
int main()
{
    printf("cpu: hello world.\n");
    hello_world<<<1, 10>>>();//<<>> thread map
    cudaDeviceReset();//device synchronize 
    
    return 0;
}