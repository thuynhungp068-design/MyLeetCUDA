__device__ float warp_reduce_sum(float val){
    for(int offset=16;offset>0;offset>>=1){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}

__global__ void reduce_smem_shfl(float* g_in,float* g_out,int n){
    extern __shared__ float smem[];
    unsigned int tid=threadIdx.x;
    unsigned int gid=blockIdx.x*blockDim.x+threadIdx.x;

    smem[tid]=(gid<n)?g_in[gid]:0.0f;
    __syncthreads();

    for(unsigned int s=blockDim.x/2;s>=warpSize;s>>=1){// warpSize is 32
        if(tid<s){
            smem[tid]+=smem[tid+s];
        }
        __syncthreads();
    }

    float val=(tid<warpSize)?smem[tid]:0.0f;
    if(tid<warpSize){
        val=warp_reduce_sum(val);
    }

    if(tid==0){
        atomicAdd(g_out,val);
    }
}