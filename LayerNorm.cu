template <const int kNumThreads = 128>
__global__ void layer_norm(float* x,float* y,float g,float b,int N,int K){
    int tid=threadIdx.x;
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    const float epsilon=1e-5f;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float value=(idx<N*K)?x[idx]:0.0f;

    float sum=block_reduce_sum<kNumThreads>(value);
    if(tid==0){
        s_mean=sum/(float)K;
        _syncthreads();
    }
    
    float variance=(value-s_mean)*(value-s_mean);
    variance=block_reduce_sum<kNumThreads>(variance);
    if(tid==0){
        s_variance=rsqrtf(variance/(float)K+epsilon);
        __syncthreads();
    }
    if(idx<N*K){
        y[idx]=((value-s_mean)*s_variance)*g+b;
    }
}

template <const int kNumThreads = 128 / 4>
__global__ void layer_norm_vec4(float *x, float *y, float g, float b, int N, int K) {
  int tid = threadIdx.x;
  int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
  const float epsilon = 1e-5f;

  __shared__ float s_mean;
  __shared__ float s_variance;
  float4 reg_x = FLOAT4(x[idx]);
  float value = (idx < N * K) ? (reg_x.x + reg_x.y + reg_x.z + reg_x.w) : 0.0f;

  float sum = block_reduce_sum<kNumThreads>(value);
  if (tid == 0)
    s_mean = sum / (float)K;
  __syncthreads();

  float4 reg_x_hat;
  reg_x_hat.x = reg_x.x - s_mean;
  reg_x_hat.y = reg_x.y - s_mean;
  reg_x_hat.z = reg_x.z - s_mean;
  reg_x_hat.w = reg_x.w - s_mean;
  float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y +
                   reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
  variance = block_reduce_sum<kNumThreads>(variance);
  if (tid == 0)
    s_variance = rsqrtf(variance / (float)K + epsilon);
  __syncthreads();

  float4 reg_y;
  reg_y.x = reg_x_hat.x * s_variance * g + b;
  reg_y.y = reg_x_hat.y * s_variance * g + b;
  reg_y.z = reg_x_hat.z * s_variance * g + b;
  reg_y.w = reg_x_hat.w * s_variance * g + b;
  if (idx < N * K)
    FLOAT4(y[idx]) = reg_y;
}