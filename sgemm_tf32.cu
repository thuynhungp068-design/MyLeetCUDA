__global__ void f32x4_tf32x4_kernel(float *x,float *y,int N){
    int idx=(blockIdx.x*blockDim.x+threadIdx.x)*4;
    if(idx<N){
        float4 reg_x=FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x=wmma::__float_to_tf32(reg_x.x);
        reg_y.y=wmma::__float_to_tf32(reg_x.y);
        reg_y.z=wmma::__float_to_tf32(reg_x.z);
        reg_y.w=wmma::__float_to_tf32(reg_x.w);
        FLOAT4(y[idx])=reg_y;
    }
}

template <const int kWMmaM=16, const int kWMmaN=16, const int kWMmaK=8,
          const int kWMmaTileM=4, const int kWMmaTileN=2,
          const int kValTileM=2, const int kValTileN=4,
          const int kPadA=4, const int kPadB=4, const int kStages=2>
__global__ void sgemm_tf32(float* A,float* B,float* C,int M,int N,int K){
    const int bx=blockIdx.x;
    const int by=blockIdx.y;
    const int NUM_K_TILES=(K+kWMmaK-1)/kWMmaK;
    constexpr int BM=kWMmaM*kWMmaTileM*kValTileM;
    constexpr int BN=kWmmaN*kWMmaTileN*kValTileN;
    constexpr int BK=kWMmaK;

    extern __shared__ float smem_tf32[];
    float* s_a=smem_tf32;
    float* s_b=smem_tf32+kStages*BM*(BK+kPadA);
    constexpr int s_a_stage_offset=BM*(BK+kPadA);
    constexpr int s_b_stage_offset=BK*(BN+kPadB);

    const int tid=threadIdx.y*blockDim.x+threadIdx.x;
    const int warp_id=tid/32;
    const int warp_m=warp_id/2;
    const int warp_n=warp_id%2;
    
    int load_smem_a_m=tid/2;
    int load_smem_a_k=(tid%2==0)?0:4;
    int load_smem_b_k=tid/32;
    int load_smem_b_n=(tid%32)*4;

    int load_gmem_a_m=by*BM+load_smem_a_m;
    int load_gmem_b_n=bx*BN+load_smem_b_n;

    wmma::fragment<wmma::accumulator,kWMmaM,kWMmaN,kWMmaK,float> C_frag[kValTileM][kValTileN];

#pragma unroll
    for(int i=0;i<kValTileM;i++){
#pragma unroll
        for(int j=0;j<kValTileN;j++){
            wmma::fill_fragment(C_frag[i][j],0.0f);
        }
    }

    uint32_t smem_a_base_ptr=__cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr=__cvta_generic_to_shared(s_b);

#pragma unroll
    for(int k=0;k<kStages-1;k++){
        int load_gmem_a_k=k*kWMmaK+load_smem_a_k;
        int load_gmem_a_addr=load_gmem_a_m*K+load_gmem_a_k;
        int load_gmem_b_k=k*kWMmaK+load_smem_b_k;
        int load_gmem_b_addr=load_gmem_b_k*N+load_gmem_b_n;

        uint32_t load_smem_a_ptr=smem_a_base_ptr+(k*s_a_stage_offset+load_smem_a_m*(BK+kPadA)+load_smem_a_k)*sizeof(float);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr=smem_b_base_ptr+(k*s_b_stage_offset+load_smem_b_k*(BN+kPadB)+load_smem_b_n)*sizeof(float);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(kStages - 2); 
    __syncthreads();

#pragma unroll
    for(int k=kStages-1;k<NUM_K_TILES;k++){
        int smem_sel=(k+1)%kStages;
        int smem_sel_next=k%kStages;
        
        int load_gmem_a_k=k*kWMmaK+load_smem_a_k;
        int load_gmem_a_addr=load_gmem_a_m*K+load_gmem_a_k;
        int load_gmem_b_k=k*kWMmaK+load_smem_b_k;
        int load_gmem_b_addr=load_gmem_b_k*N+load_gmem_b_n;

        uint32_t load_smem_a_ptr=smem_a_base_ptr+(smem_sel_next*s_a_stage_offset+load_smem_a_m*(BK+kPadA)+load_smem_a_k)*sizeof(float);
        CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

        uint32_t load_smem_b_ptr=smem_b_base_ptr+(smem_sel_next*s_b_stage_offset+load_smem_b_k*(BN+kPadB)+load_smem_b_n)*sizeof(float);
        CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

        CP_ASYNC_COMMIT_GROUP();

        wmma::fragment<wmma::matrix_a,kWMmaM,kWMmaN,kWMmaK,wmma::precision::tf32,wmma::row_major> A_frag[kValTileM];
        wmma::fragment<wmma::matrix_b,kWMmaM,kWMmaN,kWMmaK,wmma::precision::tf32,wmma::row_major> B_frag[kValTileN];

#pragma unroll
        for(int i=0;i<kValTileM;i++){
            int warp_smem_a_m=warp_m*(kWMmaM*kValTileM)+i*kWMmaM;
            float* load_smem_a_frag_ptr=s_a+smem_sel*s_a_stage_offset+warp_smem_a_m*(BK+kPadA);
            wmma::load_matrix_sync(A_frag[i],load_smem_a_frag_ptr,BK+kPadA);
        }
#pragma unroll
        for(int j=0;j<kValTileN;j++){
            int warp_smem_b_n=warp_n*(kWMmaN*kValTIleN)+j*kWMmaN;
            float* load_smem_b_frag_ptr=s_b+smem_set*s_b_stage_offset+warp_smem_b_n;
            wmma::load_matrix_sync(B_frag[j],load_smem_b_frag_ptr,BN+kPadsB);
        }

#pragma unroll
        for(int i=0;i<kValTileM;i++){
#pragma unroll
            for(int j=0;j<kValTileN;j++){
                wmma::mma_sync(C_frag[i][j],A_frag[i],B_frag[j],C_frag[i][j]);
            }
        }

        CP_ASYNC_WAIT_GROUP(kStages-2);
        __syncthreads();
    }

    if(kStages-2>0){
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }
    {
#pragma unroll
        for(int k=0;k<kStages-1;k++){
            const int stage_sel=((NUM_K_TILES-(kStages-1)+k)%kStages);

            wmma::fragment<wmma::matrix_a,kWMmaM,kWMmaN,kWMmaK,wmma::precision::tf32,wmma::row_major> A_frag[kValTileM];
            wmma::fragment<wmma::matrix_b,kWMmaM,kWMmaN,kWMmaK,wmma::precision::tf32,wmma::row_major> B_frag[kValTileN];

#pragma unroll
            for(int i=0;i<kValTileM;i++){
                int warp_smem_a_m=warp_m*(kWMmaM*kValTileM)+i*kWMmaM;
                float* load_smem_a_frag_ptr=smem_tf32+stage_sel*s_a_stage_offset+warp_smem_a_m*(BK+kPadA);
                wmma::load_matrix_sync(A_frag[i],load_smem_a_frag_ptr,BK+kPadA);
            }

#pragma unroll
            for(int j=0;j<kValTIleN;j++){
                int warp_smem_b_n=warp_n*(kWMmaN*kValTileN)+j*kWMmaN;
                float* load_smem_b_frag_ptr=s_b+stage_sel*s_b_stage_offset+warp_smem_b_n;
                wmma::load_matrix_sync(B_frag[j],load_smem_b_frag_ptr,BN+kPadB);
            }

#pragma unroll
            for(int i=0;i<kValTileM;i++){
#pragma unroll
                for(int j=0;j<kValTileN;j++){
                    wmma::mma_sync(C_frag[i][j],A_frag[i],B_frag[j],C_frag[i][j]);
                }
            }
        }
    }

#pragma unroll
    for(int i=0;i<kValTileM;i++){
#pragma unroll
        for(int j=0;j<kValTileN;j++){
            int store_gmem_c_m=by*BM+warp_m*(kWMmaM*kValTileM)+i*kWMmaM;
            int store_gmem_c_b=bx*BM+warp_n*(kWMmaN*kValTileN)+j*kWMmaN;
            wmma::store_matrix_sync(C+store_gmem_c_m*N+store_gmem_c_n,C_frag[i][j],N,wmma::mem_row_major);
        }
    }
}