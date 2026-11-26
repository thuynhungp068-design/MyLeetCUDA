__global__ void sgemm(float* a,float* b,float* c,int M,int N,int K){
    constexpr int BM=32;
    constexpr int BN=32;
    constexpr int BK=32;
    __shared__ float s_a[BM][BK],s_b[BK][BN];

    int bx=blockIdx.x;
    int by=blockIdx.y;
    int tx=threadIdx.x;
    int tid=threadIdx.y*blockDim.x+ts;

    int load_smem_a_m=tid/32;
    int load_smem_a_k=tid%32;
    int load_smem_b_k=tid/32;
    int load_smem_b_n=tid%32;
    int load_gmem_a_m=by*BM+load_smem_a_m;
    int load_gmem_b_n=bx*BN+load_smem_b_n;

    float sum=0.0f;

    for(int bk=0;bk<(K+BK-1)/BK;bk++){
        int load_gmem_a_k=bk*BK+load_smem_a_k;
        int load_gmem_a_addr=load_gmem_a_m*K+load_gmem_a_k;
        s_a[load_smem_a_m][load_smem_a_k]=a[load_gmem_a_addr];
        int load_gmem_b_k=bk*BK+load_smem_b_k;
        int load_gmem_b_addr=load_gmem_b_k*N+load_gmem_b_n;
        s_b[load_smem_b_k][load_smem_b_n]=b[load_gmem_b_addr];
        __syncthreads();

#pragma unroll
        for(int k=0;k<BK;k++){
            int comp_smem_a_m=load_smem_a_m;
            int comp_smem_b_n=load_smem_b_n;
            sum+=s_a[comp_smem_a_m][k]*s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }

    int store_gmem_c_m=load_gmem_a_m;
    int store_gmem_c_n=load_gmem_b_n;
    int store_gmem_c_addr=store_gmem_c_m*N+store_gmem_c_n;
    c[store_gmem_c_addr]=sum;
}