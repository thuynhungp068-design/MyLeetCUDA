#include <cuda_runtime.h>
#include <cassert>

__global__ void scale(float* data,int n,float factor){
    int gid=blockIdx.x*blockDim.x+threadIdx.x;
    if(gid<n)data[gid]*=factor;
}

void pipeline_demo(int N,int NUM_STREAMS){
    int chunk=(N+NUM_STREAMS-1)/NUM_STREAMS;
    size_t bytes_chunk=chunk*sizeof(float);

    float *h_in,*h_out;
    cudaMallocHost(&h_in,N*sizeof(float));
    cudaMallocHost(&h_out,N*sizeof(float));

    for(int i=0;i<N;i++){
        h_in[i]=(float)i;
    }

    float **d_buf=new float*[NUM_STREAMS];
    for(int s=0;s<NUM_STREAMS;s++){
        cudaMalloc(&d_buf[s],bytes_chunk);
    }   

    cudaStream_t *streams=new cudaStream_t[NUM_STREAMS];
    for(int s=0;s<NUM_STREAMS;s++){
        cudaStreamCreate(&streams[s]);
    }

    int threads=256;

    for(int s=0;s<NUM_STREAMS;s++){
        int offset=s*chunk;
        int this_chunk=min(chunk,N-offset);
        if(this_chunk<=0)break;
        size_t this_bytes=this_chunk*sizeof(float);

        int blocks=(this_chunk+threads-1)/threads;

        cudaMemcpyAsync(d_buf[s],h_in+offset,this_bytes,cudaMemcpyHostToDevice,streams[s]);

        scale<<<blocks,threads,0,streams[s]>>>(d_buf[s],this_chunk,2.0f);

        cudaMemcpyAsync(h_out+offset,d_buf[s],this_bytes,cudaMemcpyDeviceToHost,streams[s]);
    }

    cudaDeviceSynchronize();

    for(int i=0;i<N;i++){
        assert(h_out[i]==h_in[i]*2.0f);
    }

    for(int s=0;s<NUM_STREAMS;s++){
        cudaFree(d_buf[s]);
        cudaStreamDestroy(streams[s]);
    }
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    delete[] streams;
    delete[] d_buf;
}