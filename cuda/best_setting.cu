#include<bits/stdc++.h>
#include<cuda_runtime.h>
#include"check_success.h"
//相比于cpu，gpu更加不关注延迟，更关注吞吐量，所以要尽可能打满gpu。但是cpu往往不能打满，因为这样延迟会比较高
//gpu的内存访问延迟比较大，但是需要更大的带宽，所以在访存指令阻塞的时候，必须有其他激活的线程正在执行，否则会浪费时间。这就要求单个线程不能需要太多的资源
//所需要的线程束=延迟*单位时间处理的指令数
//假设频率为H，显存带宽为W
//那么每个周期读取的数据为W/H
//假设每次读取读取D个数据，那么每个时间都需要W/H/D个线程在执行
//假设延迟为T，那么想把带宽打满，就需要W/H/D*T个线程不是阻塞状态，就可以打满带宽
using namespace std;
int main()
{
    int device_id=0;
    CHECK(cudaSetDevice(device_id));
    return 0;
}