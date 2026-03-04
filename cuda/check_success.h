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
        ip[i] = (float)rand() / 1; 
    }
}

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