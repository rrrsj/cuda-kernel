#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
#include<cublas_v2.h>
using namespace std;

void cpu_matrix_multi(float*matrix_one,float*matrix_two,float*matrix_three,int n,int m,int k)
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<k;j++)
        {
            float ans_temp=0;
            for(int l=0;l<m;l++)
            {
                int index_one=i*m+l;
                int index_two=l*k+j;
                ans_temp+=matrix_one[index_one]*matrix_two[index_two];
            }
            int index_three=i*k+j;
            matrix_three[index_three]=ans_temp;
        }
    }
    return ;
}
void gpu_matrix_multi_blas(float*matrix_one,float*matrix_two,float* matrix_ans,int n,int m,int k,cublasHandle_t* now_handle)
{

    /*cublasStatus_t cublasSgemm(
        cublasHandle_t handle,           // [输入] cuBLAS 句柄
        cublasOperation_t transa,        // [输入] A 矩阵的操作（转置或不转置）
        cublasOperation_t transb,        // [输入] B 矩阵的操作
        int m,                           // [输入] op(A) 的行数，也是 C 的行数
        int n,                           // [输入] op(B) 的列数，也是 C 的列数
        int k,                           // [输入] op(A) 的列数，也是 op(B) 的行数
        const float *alpha,              // [输入] 缩放因子 alpha 的指针
        const float *A,                  // [输入] 矩阵 A 的显存地址
        int lda,                         // [输入] A 的 Leading Dimension (列主序下为行数)
        const float *B,                  // [输入] 矩阵 B 的显存地址
        int ldb,                         // [输入] B 的 Leading Dimension
        const float *beta,               // [输入] 缩放因子 beta 的指针
        float *C,                        // [输入/输出] 矩阵 C 的显存地址
        int ldc                          // [输入] C 的 Leading Dimension
    );*/
    const float a=1.0,b=0.0;
    cublasSgemm(*now_handle,CUBLAS_OP_N,CUBLAS_OP_N,k,n,m,&a,matrix_two,k,matrix_one,m,&b,matrix_ans,k);
    return ;
}

int main()
{
    const int n=10,m=10,k=10;
    float*matrix_one,*matrix_two,*matrix_ans;
    matrix_one=new float[n*m];
    matrix_two=new float[m*k];
    matrix_ans=new float[n*k];
    initialData(matrix_one,n*m);
    initialData(matrix_two,m*k);
    float* matrix_one_gpu,*matrix_two_gpu,*matrix_ans_gpu,*matrix_ans_from_gpu;
    matrix_ans_from_gpu=new float[n*k];
    CHECK(cudaMalloc(&matrix_one_gpu,sizeof(float)*n*m));
    CHECK(cudaMalloc(&matrix_two_gpu,sizeof(float)*m*k));
    CHECK(cudaMalloc(&matrix_ans_gpu,sizeof(float)*n*k));
    CHECK(cudaMemcpy(matrix_one_gpu,matrix_one,sizeof(float)*n*m,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(matrix_two_gpu,matrix_two,sizeof(float)*m*k,cudaMemcpyHostToDevice));
    cublasHandle_t now_handle;
    cublasCreate(&now_handle);
    cpu_matrix_multi(matrix_one,matrix_two,matrix_ans,n,m,k);
    gpu_matrix_multi_blas(matrix_one_gpu,matrix_two_gpu,matrix_ans_gpu,n,m,k,&now_handle);
    CHECK(cudaMemcpy(matrix_ans_from_gpu,matrix_ans_gpu,sizeof(float)*n*k,cudaMemcpyDeviceToHost));
    cublasDestroy(now_handle);
    cout<<check_result(matrix_ans_from_gpu,matrix_ans,n*k)<<'\n';
    return 0;
}