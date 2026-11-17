__global__ void reduce_smem(float* g_in,float* g_out,int n){
    extern __shared__ float smem[];

    unsigned int tid=threadIdx.x;
    unsigned int gid=blockIdx.x*blockDim.x+threadIdx.x;

    smem[tid]=(gid<n)?g_in[gid]:0.0f;
    __syncthreads();

    for(unsigned int s=blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            smem[tid]+=smem[tid+s];
        }
        __syncthreads();
    }

    if(tid==0){
        atomicAdd(g_out,smem[0]);
    }
}