template <const int kNumThreads = 256>
__global__ void online_safe_softmax_per_token(const float* x,const float* y,int N){
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+tid;
    const int kNumWarps=(kNumThreads+kWarpSize-1)/kWarpSize;
    __shared__ MD shared[kNumWarps];

    float val=(idx<N)?x[idx]:-FLT_MAX;
    int warp=tid/kWarpSize;
    int lane=tid%kWarpSize;

    MD md;
    md.m=idx<N?val:-FLT_MAX;
    md.d=idx<N?1.0f:0.0f;

    md=warp_reduce_md<kWarpSize>(md);

    if(lane==0){
        shared[warp]=md;
        __syncthreads();
    }

    md=lane<kNumWarps?shared[lane]:MD{-FLT_MAX,0.0f};
    md=warp_reduce_md<kWarpSize>(md);

    float d_inv=__fdividef(1.0f,md.d);
    if(idx<N){
        y[idx]=__expf(val-md.m)*d_inv;
    }
}