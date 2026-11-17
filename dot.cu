template <const int kNumThreads = 256>
__global__ void dot(float* a,float* b,float* y,int N){
    int tid=threadIdx.x;
    int idx=blockIdx.x*kNumThreads+tid;
    constexpr int kNumWarps=(kNumThreads+kWarpSize-1)/kWarpSize;
    __shared__ float shared[kNumWarps];

    float prod=(idx<N)?a[idx]*b[idx]:0.0f;
    int warp=tid/kWarpSize;
    int lane=tid%kWarpSize;

    prod=warp_reduce_sum<kWarpSize>(prod);
    if(lane==0){
        shared[warp]=prod;
        __syncthreads();
    }

    prod=(lane<kNumWarps)?shared[lane]:0.0f;
    if(warp==0){
        prod=warp_reduce_sum<kNumWarps>(prod);
    }
    if(tid==0){
        atomic(y,prod);
    }
}