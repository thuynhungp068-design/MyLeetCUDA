template <const int kNumThreads = 256>
__global__ void block_reduce_all(float* a,float* y,int N){
    int tid=threadIdx.x;
    int idx=blockIdx.x*kNumThreads+tid;
    constexpr int kNumWarps=(kNumThreads+kWarpSize-1)/kWarpSize;
    __shared__ float shared[kNumWarps];

    float val=(idx<N)?a[idx]:0.0f;
    int warp=tid/kWarpSize;
    int lane=tid%kWarpSize;

    val=warp_reduce_sum<kWarpSize>(val);
    if(lane==0){
        shared[warp]=val;
        __syncthreads();
    }

    val=(lane<kNumWarps)?shared[lane]:0.0f;
    if(warp==0){
        val=warp_reduce_sum<kNumWarps>(val);
    }
    if(tid==0){
        atomic(y,val);
    }
}