#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
#include<cublas_v2.h>
#define BM 4
#define BN 4
#define BK 4
#define TN 8
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

__global__ void navie_gpu_kernel(float* matrix_one,float* matrix_two,float* matrix_ans,int n,int m,int k)
{
    int all_thread_x=blockDim.x;
    int all_thread_y=blockDim.y;
    int thread_x_id=threadIdx.x;
    int thread_y_id=threadIdx.y;
    int grid_x_id=blockIdx.x;
    int grid_y_id=blockIdx.y;
    float now_ans=0;
    int row=grid_y_id*all_thread_y+thread_y_id;
    int col=grid_x_id*all_thread_x+thread_x_id;
    for(int i=0;i<m;i++)
    {
        now_ans+=matrix_one[row*m+i]*matrix_two[i*k+col];
    }
    matrix_ans[row*k+col]=now_ans;
    return;
}

__global__ void block_kernel(float* matrix_one,float* matrix_two,float* matrix_ans,int n,int m,int k)
{
    __shared__ float share_one[BN][BK+1];
    __shared__ float share_two[BK][BM+1];
    float now_ans[TN][TN]={0.0f};
    int block_id_y=blockIdx.x;
    int block_id_x=blockIdx.y;
    int thread_id_y=threadIdx.x;
    int thread_id_x=threadIdx.y;
    int all_thread_y=blockDim.x;
    int all_thread_x=blockDim.y;

    int one_position=(block_id_x*all_thread_x+thread_id_x)*TN;
    int two_position=(block_id_y*all_thread_y+thread_id_y)*TN;

    //每个thread负责 one_position -- one position + TN   two_position --- two position + TN
    //每个thread往share里写的位置应该是 thread_id_x*TN因为all thread x*TN=BN
    //每个thread读的位置应该是 one position+j i*BK+l
    #pragma unroll
    for(int i=0;i<m/BK;i++)
    {
        #pragma unroll
        for(int j=0;j<TN;j++)
        {
            #pragma unroll
            for(int l=0;l<BK;l++)
            {
                //one_position+j   l
                share_one[thread_id_x*TN+j][l]=matrix_one[(one_position+j)*m+i*BK+l];
            }
        }

        #pragma unroll
        for(int l=0;l<BK;l++)
        {
            #pragma unroll
            for(int j=0;j<TN;j++)
            {
                share_two[l][thread_id_y*TN+j]=matrix_two[(i*BK+l)*k+two_position+j];
            }
        }


        __syncthreads();
        
        #pragma unroll
        for (int l = 0; l < BK; l++) {
            
            float reg_one[TN];
            #pragma unroll
            for (int res_i = 0; res_i < TN; res_i++) {
                reg_one[res_i] = share_one[thread_id_x * TN + res_i][l];
            }
            float reg_two[TN];
            #pragma unroll
            for (int res_j = 0; res_j < TN; res_j++) {
                reg_two[res_j] = share_two[l][thread_id_y * TN + res_j];
            }

            #pragma unroll
            for (int res_i = 0; res_i < TN; res_i++) {
                #pragma unroll
                for (int res_j = 0; res_j < TN; res_j++) {
                    now_ans[res_i][res_j] += reg_one[res_i] * reg_two[res_j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int i=0;i<TN;i++)
    {
        #pragma unroll
        for(int j=0;j<TN;j++)
        {
            matrix_ans[(one_position+i)*n+two_position+j]=now_ans[i][j];
        }
    }
    return ;
    
}

int main()
{
    const int n=1024,m=1024,k=1024;
    float*matrix_one,*matrix_two,*matrix_ans;
    matrix_one=new float[n*m];
    matrix_two=new float[m*k];
    matrix_ans=new float[n*k];
    cublasHandle_t now_handle;

    float* matrix_one_gpu,*matrix_two_gpu,*matrix_ans_gpu,*matrix_ans_from_gpu;

    initialData(matrix_one,n*m);
    initialData(matrix_two,m*k);
    matrix_ans_from_gpu=new float[n*k];
    CHECK(cudaMalloc(&matrix_one_gpu,sizeof(float)*n*m));
    CHECK(cudaMalloc(&matrix_two_gpu,sizeof(float)*m*k));
    CHECK(cudaMalloc(&matrix_ans_gpu,sizeof(float)*n*k));
    CHECK(cudaMemcpy(matrix_one_gpu,matrix_one,sizeof(float)*n*m,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(matrix_two_gpu,matrix_two,sizeof(float)*m*k,cudaMemcpyHostToDevice));
    cublasCreate(&now_handle);


    cpu_matrix_multi(matrix_one,matrix_two,matrix_ans,n,m,k);
    
    SPEED(gpu_matrix_multi_blas(matrix_one_gpu,matrix_two_gpu,matrix_ans_gpu,n,m,k,&now_handle));
    CHECK(cudaMemcpy(matrix_ans_from_gpu,matrix_ans_gpu,sizeof(float)*n*k,cudaMemcpyDeviceToHost));
    cout<<check_result(matrix_ans_from_gpu,matrix_ans,n*k)<<'\n';

    dim3 block(32,32,1);
    dim3 grid(k/block.x,n/block.y,1);
    
    SPEED((navie_gpu_kernel<<<grid,block>>>(matrix_one_gpu,matrix_two_gpu,matrix_ans_gpu,n,m,k)));
    CHECK(cudaMemcpy(matrix_ans_from_gpu,matrix_ans_gpu,sizeof(float)*n*k,cudaMemcpyDeviceToHost));

    cout<<check_result(matrix_ans_from_gpu,matrix_ans,n*k)<<'\n';


    dim3 block_gpu(BM/TN,BN/TN);
    dim3 gird_gpu(k/BM,n/BN);
    SPEED((block_kernel<<<gird_gpu,block_gpu>>>(matrix_one_gpu,matrix_two_gpu,matrix_ans_gpu,n,m,k)));
    CHECK(cudaMemcpy(matrix_ans_from_gpu,matrix_ans_gpu,sizeof(float)*n*k,cudaMemcpyDeviceToHost));
    cout<<check_result(matrix_ans_from_gpu,matrix_ans,n*k)<<'\n';



    cublasDestroy(now_handle);
    return 0;
}