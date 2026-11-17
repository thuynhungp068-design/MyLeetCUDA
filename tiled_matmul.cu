#define TILE_SIZE 16

__global__ void matmul_tiled(const float* __restrict__ A,const float* __restrict__ B,float* C,int M,int K,int N){
    int row=blockIdx.y*TILE_SIZE+threadIdx.y;
    int col=blockIdx.x*TILE_SIZE+threadIdx.x;

    float sum=0.0f;

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int numTiles=(K+TILE_SIZE-1)/TILE_SIZE;

    for(int t=0;t<numTiles;t++){
        int a_col=t*TILE_SIZE+threadIdx.x;
        A_tile[threadIdx.y][threadIdx.x]=(row<M&&a_col<K)?A[row*K+a_col]:0.0f;

        int b_row=t*TILE_SIZE+threadIdx.y;
        B_tile[threadIdx.y][threadIdx.x]=(col<N&&b_row<K)?B[b_row*N+col]:0.0f;

        __syncthreads();

        for(int k=0;k<TILE_SIZE;k++){
            sum+=A_tile[threadIdx.y][k]*B_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if(row<M&&col<N){
        C[row*N+col]=sum;
    }

}