#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
using namespace std;
int main()
{
    int available_gpu_number=0;
    CHECK(cudaGetDeviceCount(&available_gpu_number));
    cout<<"now_available_gpu:   "<<available_gpu_number<<'\n';
    int now_gpu_id=0,driver_version=0,runtime_version=0;
    CHECK(cudaSetDevice(now_gpu_id));
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,now_gpu_id));
    cout<<"Device_name:    "<<deviceProp.name<<'\n';
    CHECK(cudaDriverGetVersion(&driver_version));
    CHECK(cudaRuntimeGetVersion(&runtime_version));
    cout<<"Driver Version:  "<<driver_version<<'\n';
    cout<<"Runtime Version:   "<<runtime_version<<'\n';
    cout<<"CUDA Capability Major/Minor version number:   "<< deviceProp.major << "." << deviceProp.minor << '\n';
    cout<<"Total amount of global memory:   "<< fixed << static_cast<double>(deviceProp.totalGlobalMem) / pow(1024.0, 3)<< " GBytes ("<< deviceProp.totalGlobalMem << " bytes)\n";
    cout<<"GPU Clock rate:    "<< deviceProp.clockRate * 1e-3f << " MHz ("<< deviceProp.clockRate * 1e-6f << " GHz)\n";
    cout<<"Memory Bus width:    "<< deviceProp.memoryBusWidth << "-bits\n";
    if (deviceProp.l2CacheSize)
    {
        cout<< "L2 Cache Size:    "<< deviceProp.l2CacheSize << " bytes\n";
    }
    cout<<"Max Texture Dimension Size (x,y,z)     "<< "1D=(" << deviceProp.maxTexture1D << "), 2D=(" << deviceProp.maxTexture2D[0] << ","<< deviceProp.maxTexture2D[1] << "), 3D=(" << deviceProp.maxTexture3D[0] << ","<< deviceProp.maxTexture3D[1] << ","<< deviceProp.maxTexture3D[2] << ")\n";
    cout<<"Max Layered Texture Size (dim) x layers:  1D=(" << deviceProp.maxTexture1DLayered[0] <<") x "<< deviceProp.maxTexture1DLayered[1] << ", 2D=(" << deviceProp.maxTexture2DLayered[0] << ","<< deviceProp.maxTexture2DLayered[1] << ") x "<< deviceProp.maxTexture2DLayered[2] << '\n';
    cout<<"Total amount of constant memory:    "<< deviceProp.totalConstMem << " bytes\n";
    cout<<"Total amount of shared memory per block:     "<< deviceProp.sharedMemPerBlock <<" bytes\n";
    cout<<"Total number of registers available per block:"<< deviceProp.regsPerBlock << '\n';
    cout<<"Warp size:     "<< deviceProp.warpSize << '\n';
    cout<<"Maximum number of threads per multiprocessor: "<< deviceProp.maxThreadsPerMultiProcessor << '\n';
    cout<<"Maximum number of threads per block:     "<< deviceProp.maxThreadsPerBlock << '\n';
    cout<<"Maximum size of each dimension of a block:    "<< deviceProp.maxThreadsDim[0] << " x "<< deviceProp.maxThreadsDim[1] << " x "<< deviceProp.maxThreadsDim[2] << '\n';
    cout<<"Maximum size of each dimension of a grid:     "<< deviceProp.maxGridSize[0] << " x "<< deviceProp.maxGridSize[1] << " x "<< deviceProp.maxGridSize[2] << '\n';
    cout<<"Maximum memory pitch:    "<< deviceProp.memPitch << " bytes\n";
    return 0;

    //L1 cache shared in only one sm
    //L2 cache shared in different sm
}