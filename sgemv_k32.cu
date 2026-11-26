__global__ void sgemv_k32(float* a,float* x,float* y,int M,int K){
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int lane=tx%kWarpSize;
    int m=blockIdx.x*blockDim.x+ty;
    if(m<M){
        float sum=0.0f;
        const int NUM_ITERS=(K+kWarpSize-1)/kWarpSize;
#pragma unroll
        for(int w=0;w<NUM_ITERS;w++){
            int k=w*kWarpSize+lane;
            sum+=a[m*K+k]*x[k];
        }
        sum=warp_reduce_sum<kWarpSize>(sum);
        if(lane==0){
            y[m]=sum;
        }
    }
}