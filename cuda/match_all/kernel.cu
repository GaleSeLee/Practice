#include <iostream>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__global__ void kernel(int *a) {
    int lane = threadIdx.x;
    int pred = 0;
    __match_all_sync(0xFFFFFFFF, a[lane], &pred);
}


int main(int argc, char **argv) {
    int *a_h = (int *)(std::malloc(sizeof(int) * 32));
    int *a_d ;
    cudaMalloc((void **)(&a_d), sizeof(int) * 32);
    
    for (int ii = 0; ii < 32; ii++) {
        a_h[ii] = 0;
    }

    cudaMemcpy(a_d, a_h, sizeof(int) * 32, cudaMemcpyHostToDevice);

    kernel<<<1,32>>>(a_d);

    std::free(a_h);
    cudaFree(a_d);
}
