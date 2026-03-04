#include<bits/stdc++.h>
#include<cuda_runtime.h>
using namespace std;
// __syncthreads();只能同步同一个 Block 内的线程。绝对不能用来同步不同 Block 之间的线程。不同的 Block 可能被调度到不同的 SM 上执行，甚至在不同时间执行。如果 Block A 的线程在等 Block B，而 Block B 还没开始跑（因为资源不够），就会造成死锁 (Deadlock)，整个程序挂起。
//if (threadId == 0) { __syncthreads(); // 只有线程 0 到了这里，其他线程没到！-> 死锁} else {// 做一些事// 其他线程永远到不了上面的同步点}
int main()
{

    return 0;
}