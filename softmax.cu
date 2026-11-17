template <const int kNumThreads = 256>
__global__ void softmax_per_token(float* x,float* y,int N){
    const int tid=threadIdx.x;
    const int idx=blockIdx.x*blockDim.x+tid;

    float exp_val=(idx<N)?expf(x[idx]):0.0f;
    float exp_sum=block_reduce_sum<kNumThreads>(exp_val);
    if(idx<N){
        y[idx]=exp_val/exp_sum;
    }
}

template <const int kNumThreads = 256>
__global__ void safe_softmax_per_token(float* x,float* y,int N){
    const int tid=threadIdx.x;
    const int idx=blockIdx.x*blockDim.x+tid;

    float val=(idx<N)?x[idx]:-FLT_MAX;
    float max_val=block_reduce_max<kNumThreads>(val);

    float exp_val=(idx<N)?expf(x[idx]-max_val):0.0f;
    float exp_sum=block_reduce_sum<kNumThreads>(exp_val);

    if(idx<N){
        y[idx]=exp_val/exp_sum;
    }
}

struct __align__(8) MD{
    float m;
    float d;
};

template <const int kWarpWidth = kWarpSize>
__device__ __forceinline__ MD warp_reduce_md(MD md1){
#pragma unroll
    for(int mask=kWarpWidth>>1;mask>=1;mask>>=1){
        MD md2;
        md2.m=__shfl_xor_sync(0xffffffff,md1.m,mask);
        md2.d=__shfl_xor_sync(0xffffffff,md1.d,mask);
        
        MD b_m=(md1.m>md2.m)?md1:md2;
        MD s_m=(md1.m>md2.m)?md2:md1;

        md1.d=b_m.d+s_m.d*__expf(s_m.m-b_m.m);
        md1.m=b_m.m;
    }

    return md1;
}