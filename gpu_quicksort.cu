#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ void partition(int* arr,int low,int high){
    int pivot=arr[high];
    int i=low-1;
    for(int j=low;j<high;j++){
        if(arr[j]<=pivot){
            i++;
            int tmp=arr[i];
            arr[i]=arr[j];
            arr[j]tmp;
        }
    }
    int tmp=arr[i+1];
    arr[i+1]=arr[high];
    arr[high]=tmp;
    return i+1;
}

__global__ void gpu_quicksort(int* arr,int low,int high){
    if(low<high){
        if(threadIdx.x==0){
            int pivot_idx=partition(arr,low,high);

            if(pivot_idx-1-low>16){
                gpu_quicksort<<<1,1>>>(arr,low,pivot_idx-1);
            }
            if(high-(pivot_idx+1)>16){
                gpu_quicksort<<<1,1>>>(arr,pivot_idx+1,high);
            }
        }
    }
}