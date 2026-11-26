template <const int kRowPerWarp = 2>
__global__ void sgemv_k16(float* A,float* x,float* y,int M,int K){
    constexpr int kNumLanePerRow=(kWarpSize+kRowPerWarp-1)/kRowPerWarp;
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int lane=tx%kWarpSize;
    int k=lane%kNumLanePerRow;
    int m=(blockIdx.x*blockDim.y+ty)*kRowPerWarp+lane/kNumLanePerRow;
    if(m<M){
        float sum=A[m*K+k]*x[k];
        sum=warp_reduce_sum<kNumLanePerRow>(sum);
        if(k==0){
            y[m]=sum;
        }
    }
}