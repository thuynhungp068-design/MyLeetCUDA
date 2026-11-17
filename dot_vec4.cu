template <const int kNumThreads = 256/4 >
__global__ void dot_vec4(float* a,float* b,float* y,int N){
    int tid=threadIdx.x;
    int idx=(blockIdx.x*kNumThreads+tid)*4;
    constexpr int kNumWarps=(kNumThreads+kWarpSize-1)/kWarpSize;
    __shared__ float shared[kNumWarps];

    float4 reg_a=FLOAT4(a[idx]);
    float4 reg_b=FLOAT4(b[idx]);
    float prod=(idx<N)?(reg_a.x*reg_b.x+reg_a.y*reg_b.y+reg_a.z*reg_b.z+reg_a.w*reg_b.w):0.0f;
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