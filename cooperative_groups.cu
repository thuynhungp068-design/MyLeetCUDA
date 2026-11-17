#include <cooperative_groups.h>
namespace cg=cooperative_groups;

__global__ void iterative_solver(float* data,int n,int max_iter){
    cg::grid_group grid=cg::this_grid();

    for(int iter=0;iter<max_iter;iter++){
        for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
            data[i]=compute_new_value(data[i],iter);
        }
        grid.sync();
    }
}

void launch_solver(float* d_data,int n,int max_iter){
    int threads=256;
    int blocks;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,iterrative_solver,threads,0);
    int device;
    cudaGetDevice(&device);
    cudeDeviceProp prop;
    cudaGetDeviceProperties(&prop,device);
    blocks*=prop.multiProcessorCount;

    void* args[]={&d_data,&n,&max_iter};
    cudaLaunchCooperativeKernel((void*)iterative_solver,blocks,threads,args,0,nullptr);
    cudaDeviceSynchronize();
}