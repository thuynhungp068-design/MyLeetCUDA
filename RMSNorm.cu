template <const int kNumThreads = 128>
__global__ void rms_norm(float* x,float* y,float g,int N,int K){
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    const float epsilon=1e-5f;

    __shared__ float s_variance;
    float value=(idx<N*K)?x[idx]:0.0f;
    float variance=value*value;
    variance=block_reduce_sum<kNumThreads>(variance);
    if(tid==0){
        s_variance=rsqrtf(variance/(float)K+epsilon);
        __syncthreads();
    }
    if(idx<N*K){
        y[idx]=(value*s_variance)*g;
    }
}

template <const int kNumThreads = 128/4>
__global__ void rms_norm_vec4(float* x,float* y,flaot g,int N,int K){
    int tid=threadIdx.x;
    int idx=(blockIdx.x*blockDim.x+threadIdx.x)*4;
    const float epsilon=1e-5f;

    __shared__ float s_variance;
    float4 reg_x=FLOAT4(x[idx]);
    float variance=(idx<N*K)?(reg_x.x*reg_x.x+reg_x.y*reg_x.y+reg_x.z*reg_x.z+reg_x.w*reg_x.w):0.0f;
    variance=block_reduce_sum<kNumThreads>(variance);
    if(tid==0){
        s_variance=rsqrtf(variance/(float)K+epsilon);
        __syncthreads();
    }
    float4 reg_y;
    reg_y.x = reg_x.x * s_variance * g;
    reg_y.y = reg_x.y * s_variance * g;
    reg_y.z = reg_x.z * s_variance * g;
    reg_y.w = reg_x.w * s_variance * g;
    if(idx<N*K){
        FLOAT4(y[idx])-reg_y;
    }
}