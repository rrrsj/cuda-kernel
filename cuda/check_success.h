#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void checkCudaError(cudaError_t cudaerror,const char* file, int line)
{
    if(cudaerror!=cudaSuccess)
    {
        fprintf(stderr, "CUDA Error at %s:%d: %s\n",file,line,cudaGetErrorString(cudaerror));
        exit(EXIT_FAILURE);
    }
}

#define CHECK(Function)\
do\
{\
    checkCudaError(Function, __FILE__, __LINE__);\
}while(0)

void initialData(float *ip, int size) {
    for (int i = 0; i < size; i++) {
        // RAND_MAX 是 rand() 能产生的最大整数
        ip[i] = ((float)rand() / (float)RAND_MAX) * 1.0f;
    }
}

#define SPEED(Function) \
do { \
    float _m = 0; \
    cudaEvent_t _st, _sp; \
    cudaEventCreate(&_st); \
    cudaEventCreate(&_sp); \
    cudaEventRecord(_st); \
    Function; \
    cudaError_t err = cudaGetLastError();\
    if (err != cudaSuccess) {\
        printf("CUDA Error (Startup): %s\n", cudaGetErrorString(err));\
    }\
    cudaEventRecord(_sp); \
    cudaEventSynchronize(_sp); \
    cudaEventElapsedTime(&_m, _st, _sp); \
    printf("Execution Time: %f ms\n", _m); \
    cudaEventDestroy(_st); \
    cudaEventDestroy(_sp); \
} while(0)

int check_result(float*a,float*b,const int size)
{
    int ok=1;
    for(int i=0;i<size;i++)
    {
        if(fabs(a[i]-b[i])>0.001)
        {
            return 0;
        }
    }
    return ok;
}
#endif