#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void tiledTranspose(const float* __restrict__ A,const float* __restrict__ B,int N){
    __shared__ float smem[TILE_DIM][TILE_DIM+1];

    int x=blockIdx.x*TILE_DIM+threadIdx.x;
    int y=blockIdx.y*TILE_DIM+threadIdx.y;

    for(int j=0;j<TILE_DIM;j+=BLOCK_ROWS){
        if((x<N)&&(y+j<N)){
            smem[threadIdx.y+j][threadIdx.x]=A[(y+j)*N+x];
        }
    }

    __syncthreads();

    x=blockIdx.y*TILE_DIM+threadIdx.x;
    y=blockIdx.x*TILE_DIM+threadIdx.y;

    for(int j=0;j<TILE_DIM;j+=BLOCK_ROWS){
        if((x<N)&&(y+j<N)){
            B[(y+j)*N+x]=smem[threadIdx.x][threadIdx.y+j];
        }
    }
}

void runTranspose(float* d_A,float* d_B,int N){
    dim3 block(TILE_DIM,BLOCK_ROWS);
    dim3 grid((N+TILE_DIM-1)/TILE_DIM,(N+TILE_DIM-1)/TILE_DIM);

    tiledTranspose<<<grid,block>>>(d_A,d_B,N);
}