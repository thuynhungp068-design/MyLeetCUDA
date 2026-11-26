__global__ void sgemv_k128(float* a,float* x,float* y,int M,int K){
    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int lane=tx%kWarpSize;
    int m=blockIdx.x*blockDim.y+ty;

    if(m<M){
        float sum=0.0f;
        const int NUM_TIERS=(((K+kWarpSize-1)/kWarpSize)+4-1)/4;
#pragma unroll
        for(int w=0;w<NUM_TIERS;w++){
            int k=(w*kWarpSize+lane)*4;
            float4 reg_x=FLOAT4(x[k]);
            float4 reg_a=FLOAT4(a[m*K+k]);
            sum+=(reg_a.x*reg_a.x+reg_a.y*reg_x.y+reg_a.z*reg_x.z+reg_a.w*reg_x.w);
        }
        sum=warp_reduce_sum<kWarpSize>(sum);
        if(lane==0){
            y[m]=sum;
        }
    }
}